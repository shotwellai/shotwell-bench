import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from shotwell_bench.boundary_score import score_boundaries
from shotwell_bench.alternatives import expand_alternatives
from shotwell_bench.providers import ContentBlocked, generate
from shotwell_bench.runner import SEG_SCHEMA, _done_ids, references, select_episodes, tasks_for_episode
from shotwell_bench.scoring import _f1


class ReferenceTests(unittest.TestCase):
    def test_reference_integrity(self):
        data = references()
        self.assertEqual(len(data['episodes']), 11)
        self.assertEqual(len(select_episodes(data, 'first5')), 5)
        for episode in data['episodes']:
            self.assertTrue(episode['video_url'].startswith('https://huggingface.co/'))
            boundaries = episode['boundaries']
            self.assertAlmostEqual(boundaries[0]['time'], 0)
            self.assertAlmostEqual(boundaries[-1]['time'], episode['duration'], places=2)
            self.assertTrue(all(a['time'] < b['time']
                                for a, b in zip(boundaries, boundaries[1:])))
            self.assertTrue(all(b['minimum'] <= b['time'] <= b['maximum']
                                for b in boundaries))
            golden = [b for b in boundaries if b['golden']]
            self.assertEqual(len(episode['segments']), len(golden) - 1)
            for segment, start, end in zip(episode['segments'], golden, golden[1:]):
                self.assertEqual(segment['start_boundary_id'], start['id'])
                self.assertEqual(segment['end_boundary_id'], end['id'])
                self.assertEqual(segment['start'], start['time'])
                self.assertEqual(segment['end'], end['time'])
            if episode['description_ready']:
                self.assertTrue(all(segment['required'] for segment in episode['segments']))

    def test_windows_cover_episode(self):
        episode = select_episodes(references(), 'first5')[0]
        windows = tasks_for_episode(episode, 'segmentation')
        self.assertEqual(windows[0].core_start, 0)
        self.assertAlmostEqual(windows[-1].core_end, episode['duration'])
        for left, right in zip(windows, windows[1:]):
            self.assertEqual(left.core_end, right.core_start)

    def test_optional_boundary_is_not_required_for_recall(self):
        refs = [{'time': 2, 'minimum': 1.9, 'maximum': 2.1,
                 'boundary_type': 'major'},
                {'time': 3, 'minimum': 2.9, 'maximum': 3.1,
                 'boundary_type': 'minor'}]
        result = score_boundaries(refs, [{'time': 2.0}])['summary']
        self.assertEqual(result['major_hits'], 1)
        self.assertEqual(result['unused_minors'], 1)
        self.assertEqual(result['extra_predictions'], 0)

    def test_f1(self):
        self.assertAlmostEqual(_f1(8, 2, 2)['f1'], .8)

    def test_bracketed_alternatives(self):
        self.assertEqual(expand_alternatives('[Rub / rotate] the glass with the left hand.'),
                         ['Rub the glass with the left hand.',
                          'rotate the glass with the left hand.'])
        self.assertEqual(expand_alternatives('Push the glass [and turn it] with the left hand.'),
                         ['Push the glass and turn it with the left hand.',
                          'Push the glass with the left hand.'])
        nested = expand_alternatives('[[Rub / Rinse] the spoon / IDLE]')
        self.assertEqual(nested, ['Rub the spoon', 'Rinse the spoon', 'IDLE'])

    def test_policy_refusal_is_not_a_generated_annotation(self):
        with patch('shotwell_bench.providers._vertex', side_effect=ContentBlocked('PROHIBITED_CONTENT')):
            result = generate('argon', 'Describe the video', [], SEG_SCHEMA)
        self.assertEqual(result['output'], {'segments': []})
        self.assertEqual(result['refusal_reason'], 'PROHIBITED_CONTENT')

    def test_retry_refusals_keeps_successful_tasks(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.jsonl'
            path.write_text(''.join(json.dumps(row) + '\n' for row in [
                {'task_id': 'a', 'status': 'done', 'refusal_reason': 'PROHIBITED_CONTENT'},
                {'task_id': 'b', 'status': 'done'},
            ]))
            self.assertEqual(_done_ids(path), {'a', 'b'})
            self.assertEqual(_done_ids(path, retry_refusals=True), {'b'})


if __name__ == '__main__':
    unittest.main()
