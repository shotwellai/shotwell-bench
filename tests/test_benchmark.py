import unittest
import base64
import hashlib
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from PIL import Image

from shotwell_bench.boundary_score import score_boundaries
from shotwell_bench.alternatives import expand_alternatives
from shotwell_bench.providers import ContentBlocked, _anthropic, _openrouter, _pack_openrouter_images, generate
from shotwell_bench.runner import SEG_SCHEMA, _done_ids, references, run, select_episodes, tasks_for_episode
from shotwell_bench.scoring import _f1, _judge_prompt, judgment_key


class ReferenceTests(unittest.TestCase):
    def test_reference_integrity(self):
        data = references()
        self.assertEqual(len(data['episodes']), 11)
        self.assertEqual(len(select_episodes(data, 'first5')), 5)
        self.assertEqual(len(select_episodes(data, 'completed')), 11)
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

    def test_judge_allows_compatible_specificity_but_not_new_actions(self):
        prompt = _judge_prompt(['Pour liquid into the glass with the right hand.'], [],
                               'Pour water from a jug into the glass with the right hand.')
        self.assertIn('Allow compatible extra specificity', prompt)
        self.assertIn('slight generalization', prompt)
        self.assertIn('a separate action absent from both', prompt)
        segment = {'required': ['Pour liquid into the glass with the right hand.'],
                   'optional': []}
        legacy = hashlib.sha256(json.dumps(
            [segment['required'], segment['optional'], 'Pour water.'],
            ensure_ascii=False).encode()).hexdigest()
        self.assertNotEqual(judgment_key('d:test:0', segment, 'Pour water.'),
                            f'd:test:0:{legacy}')
        self.assertNotEqual(judgment_key('d:test:0', segment, 'Pour water.'),
                            judgment_key('d:test:0', segment, 'Pour juice.'))

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

    def test_opus_uses_direct_anthropic_vision_format(self):
        with patch.dict('os.environ', {'ANTHROPIC_KEY': 'test-key'}), \
                patch('shotwell_bench.providers.requests.post') as post:
            post.return_value.json.return_value = {
                'content': [{'type': 'text', 'text': '{"description":"Pick the cup."}'}],
                'usage': {'input_tokens': 10},
            }
            output, usage = _anthropic('claude-opus-5-5', 'Describe', ['image-data'], {}, 500)
        payload = post.call_args.kwargs['json']
        self.assertEqual(payload['model'], 'claude-opus-5-5')
        self.assertNotIn('temperature', payload)
        self.assertEqual(payload['messages'][0]['content'][1]['source']['media_type'], 'image/jpeg')
        self.assertEqual(output['description'], 'Pick the cup.')
        self.assertEqual(usage['input_tokens'], 10)

    def test_anthropic_extracts_json_after_explanatory_text(self):
        with patch.dict('os.environ', {'ANTHROPIC_KEY': 'test-key'}), \
                patch('shotwell_bench.providers.requests.post') as post:
            post.return_value.json.return_value = {
                'content': [{'type': 'text', 'text':
                             'The action ends here.\n\n{"description":"Pick the cup."}'}],
            }
            output, _ = _anthropic('claude-sonnet-5-5', 'Describe', [],
                                    {'required': ['description']}, 500)
        self.assertEqual(output, {'description': 'Pick the cup.'})

    def test_qwen_uses_openrouter_images_and_schema(self):
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-key'}), \
                patch('shotwell_bench.providers.requests.post') as post:
            post.return_value.json.return_value = {
                'choices': [{'message': {'content': '{"description":"Pick the cup."}'}}],
                'usage': {'prompt_tokens': 10},
            }
            output, usage = _openrouter('qwen/qwen3-vl-235b-a22b-instruct', 'Describe',
                                        ['image-data'], {'type': 'object'}, 500)
        payload = post.call_args.kwargs['json']
        self.assertEqual(payload['model'], 'qwen/qwen3-vl-235b-a22b-instruct')
        self.assertEqual(payload['messages'][0]['content'][1]['image_url']['url'],
                         'data:image/jpeg;base64,image-data')
        self.assertEqual(payload['response_format']['json_schema']['schema'],
                         {'type': 'object'})
        self.assertEqual(output['description'], 'Pick the cup.')
        self.assertEqual(usage['prompt_tokens'], 10)

    def test_openrouter_packs_long_8fps_contact_sheet_series(self):
        image = Image.new('RGB', (16, 8), 'white')
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        encoded = base64.b64encode(buffer.getvalue()).decode()
        packed = _pack_openrouter_images([encoded] * 50)
        self.assertEqual(len(packed), 25)
        combined = Image.open(io.BytesIO(base64.b64decode(packed[0])))
        self.assertEqual(combined.size, (16, 16))
        long_packed = _pack_openrouter_images([encoded] * 112)
        self.assertEqual(len(long_packed), 28)
        long_combined = Image.open(io.BytesIO(base64.b64decode(long_packed[0])))
        self.assertEqual(long_combined.size, (16, 32))

    def test_resuming_run_rejects_different_guide(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            guide = root / 'guide.txt'
            guide.write_text('Original guide')
            run([], [], root / 'run', root / 'cache', guide_path=guide)
            original = (root / 'run' / 'config.json').read_text()
            guide.write_text('Changed guide')
            with self.assertRaisesRegex(ValueError, 'guide_sha256'):
                run([], [], root / 'run', root / 'cache', guide_path=guide)
            self.assertEqual((root / 'run' / 'config.json').read_text(), original)

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
