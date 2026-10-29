import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import export_viewer
from shotwell_bench.scoring import JUDGE_RUBRIC_VERSION, judgment_key


class ViewerExportTests(unittest.TestCase):
    def test_export_joins_description_judgment_and_boundary_result(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reference_path = root / 'episodes.json'
            guide_path = root / 'guide.md'
            run_dir = root / 'run'
            website_dir = root / 'website'
            (run_dir / 'judgments').mkdir(parents=True)
            episode_id = 'wgo-bench-homer_1'
            segment = {
                'start_boundary_id': 'start', 'end_boundary_id': 'end',
                'start': 0, 'end': 2, 'required': ['Pick up the cup.'], 'optional': [],
            }
            references = {
                'source_job': 690, 'source_dataset': 'macrodata/WGO-Bench',
                'license': 'CC BY-NC-SA 4.0', 'episodes': [{
                    'id': episode_id, 'source_id': 'homer_1',
                    'video_url': 'https://example.com/video.mp4', 'duration': 2,
                    'description_ready': True,
                    'boundaries': [
                        {'id': 'start', 'time': 0, 'minimum': 0, 'maximum': 0,
                         'type': 'major'},
                        {'id': 'end', 'time': 2, 'minimum': 2, 'maximum': 2,
                         'type': 'major'},
                    ],
                    'segments': [segment],
                }],
            }
            reference_path.write_text(json.dumps(references))
            guide_path.write_text('Test guide')
            config = {
                'reference_sha256': hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                'guide_sha256': hashlib.sha256(guide_path.read_bytes()).hexdigest(),
                'episode_ids': [episode_id],
                'fps': 8,
            }
            (run_dir / 'config.json').write_text(json.dumps(config))
            scores = {
                'episode_ids': [episode_id], 'models': {'astra': {
                    'model_id': 'test-astra', 'overall_f1': .5,
                    'refused_tasks': 0, 'description': {'f1': .5},
                    'segmentation': {'f1': 1, 'by_episode': {episode_id: {
                        'predicted_boundaries': [{'time': 1, 'description': 'Extra split'}],
                    }}},
                }},
            }
            (run_dir / 'scores.json').write_text(json.dumps(scores))
            task_id = f'd:{episode_id}:0'
            description = 'Pick up the cup.'
            (run_dir / 'astra.jsonl').write_text(json.dumps({
                'task_id': task_id, 'status': 'done',
                'output': {'description': description},
            }) + '\n')
            (run_dir / 'judgments' / 'astra.jsonl').write_text(json.dumps({
                'key': judgment_key(task_id, segment, description), 'status': 'done',
                'judgment': {'matched_required': [0], 'unsupported': [],
                             'format_violation': ''},
            }) + '\n')

            with patch.object(export_viewer, 'REFERENCE_PATH', reference_path), \
                    patch.object(export_viewer, 'GUIDE_PATH', guide_path):
                export_viewer.export(run_dir, website_dir, None)

            prefix = 'window.SHOTWELL_WGO_RESULTS = '
            results_text = (website_dir / 'results.js').read_text()
            self.assertTrue(results_text.startswith(prefix))
            result = json.loads(results_text[len(prefix):].rstrip(';\n'))
            model = result['models']['astra']['episodes'][episode_id]
            self.assertTrue(model['descriptions'][0]['correct'])
            self.assertEqual(model['descriptions'][0]['matchedRequired'], [0])
            self.assertEqual(model['boundaries'][0]['status'], 'extra')
            self.assertEqual(model['boundarySummary']['extra'], 1)
            self.assertIn('homer_1', (website_dir / 'data.js').read_text())
            chart = json.loads((website_dir.parent / 'scores.json').read_text())
            self.assertEqual(chart['fps'], 8)
            self.assertEqual(chart['judge_rubric_version'], JUDGE_RUBRIC_VERSION)
            self.assertEqual(chart['models']['astra']['description']['f1'], .5)

            data_path = website_dir / 'data.js'
            data = json.loads(data_path.read_text().removeprefix(
                'window.SHOTWELL_WGO_REFERENCE = ').rstrip(';\n'))
            data['episodes'][0]['task'] = 'Wash the dishes'
            data_path.write_text(export_viewer.js_assignment('SHOTWELL_WGO_REFERENCE', data))
            with patch.object(export_viewer, 'REFERENCE_PATH', reference_path):
                export_viewer.export(run_dir, website_dir, None, guide_path)
            self.assertIn('Wash the dishes', data_path.read_text())


if __name__ == '__main__':
    unittest.main()
