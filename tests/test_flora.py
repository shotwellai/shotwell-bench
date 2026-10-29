import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

from shotwell_bench.runner import references, select_episodes, tasks_for_episode

MODAL_AVAILABLE = importlib.util.find_spec('modal') is not None
if MODAL_AVAILABLE:
    from scripts.run_flora import INSTRUCTION, append_rows, clip_plan

@unittest.skipUnless(MODAL_AVAILABLE, 'FLORA tests require the optional modal package')
class FloraAdapterTests(unittest.TestCase):
    def test_clips_cover_every_golden_interval(self):
        self.assertLessEqual(len(INSTRUCTION), 2000)
        for episode in select_episodes(references(), 'completed'):
            clips, assignments = clip_plan(episode)
            self.assertTrue(all(end - start <= 120 for start, end in clips))
            assigned = []
            for clip_index, rows in assignments.items():
                start, end = clips[clip_index]
                for segment_index, local_start, local_end in rows:
                    assigned.append(segment_index)
                    self.assertGreaterEqual(local_start, 0)
                    self.assertLessEqual(local_end, end - start + 1e-4)
            self.assertEqual(sorted(assigned), list(range(len(episode['segments']))))

    def test_native_outputs_fit_standard_scoring_rows(self):
        episode = select_episodes(references(), 'homer_33')[0]
        result = {
            'predictions': [{'time': 4.25, 'description': 'Pick the onion.'}],
            'descriptions': {i: f'Description {i}' for i in range(len(episode['segments']))},
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'flora.jsonl'
            append_rows(path, episode, result)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(rows), len(tasks_for_episode(episode)))
        self.assertEqual(rows[0]['output']['segments'], [
            {'end': 4.25, 'description': 'Pick the onion.'},
        ])
        self.assertEqual(rows[-1]['output']['description'],
                         f'Description {len(episode["segments"]) - 1}')


if __name__ == '__main__':
    unittest.main()
