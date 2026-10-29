"""Publish the selected reference episodes and judged model outputs for the static viewer."""

import argparse
import hashlib
import json
from pathlib import Path

from shotwell_bench.boundary_score import score_boundaries
from shotwell_bench.runner import GUIDE_PATH, REFERENCE_PATH
from shotwell_bench.scoring import JUDGE_RUBRIC_VERSION, judgment_key


MODEL_NAMES = {
    'astra': 'GPT-6 Astra',
    'sol': 'GPT-6.1 Sol',
    'argon': 'Gemini 4 Argon',
    'opus': 'Claude Opus 5.5',
    'luna': 'GPT-6 Luna',
    'sonnet': 'Claude Sonnet 5.5',
    'qwen': 'Qwen3-VL 235B Instruct',
    'qwen35': 'Qwen3.5 397B',
    'flora': 'NVIDIA FLORA',
}


def latest_done(path, key='task_id'):
    rows = {}
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get('status') == 'done':
            rows[row[key]] = row
    return rows


def js_assignment(name, value):
    return f'window.{name} = {json.dumps(value, ensure_ascii=False, separators=(",", ":"))};\n'


def existing_task_labels(path):
    if not path.exists():
        return {}
    prefix = 'window.SHOTWELL_WGO_REFERENCE = '
    source = path.read_text()
    if not source.startswith(prefix):
        raise ValueError(f'Unexpected reference data format in {path}')
    data = json.loads(source[len(prefix):].strip().removesuffix(';'))
    return {episode['id']: episode.get('task', '') for episode in data['episodes']}


def export(run_dir, website_dir, task_labels, guide_path=None):
    guide_path = Path(guide_path) if guide_path else GUIDE_PATH
    config = json.loads((run_dir / 'config.json').read_text())
    for name, path in [('reference_sha256', REFERENCE_PATH), ('guide_sha256', guide_path)]:
        if config[name] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f'{path.name} differs from the completed run')
    scores = json.loads((run_dir / 'scores.json').read_text())
    if scores['episode_ids'] != config['episode_ids']:
        raise ValueError('Scored episode selection differs from the run')
    references = json.loads(REFERENCE_PATH.read_text())
    by_id = {episode['id']: episode for episode in references['episodes']}
    selected = [by_id[episode_id] for episode_id in scores['episode_ids']]
    if any(not episode['description_ready'] for episode in selected):
        raise ValueError('The viewer requires complete reference descriptions')
    labels = (json.loads(task_labels.read_text()) if task_labels else
              existing_task_labels(website_dir / 'data.js'))

    public_episodes = []
    for episode in selected:
        public_episodes.append({
            'id': episode['id'], 'sourceId': episode['source_id'],
            'task': labels.get(episode['id'], ''),
            'duration': episode['duration'], 'videoUrl': episode['video_url'],
            'boundaries': [{key: boundary[key] for key in
                            ('id', 'time', 'minimum', 'maximum', 'type')}
                           for boundary in episode['boundaries']],
            'segments': [{
                'startBoundaryId': segment['start_boundary_id'],
                'endBoundaryId': segment['end_boundary_id'],
                'start': segment['start'], 'end': segment['end'],
                'required': segment['required'], 'optional': segment['optional'],
            } for segment in episode['segments']],
        })

    public_models = {}
    for model, score in scores['models'].items():
        outputs = latest_done(run_dir / f'{model}.jsonl')
        judgments = latest_done(run_dir / 'judgments' / f'{model}.jsonl', 'key')
        model_episodes = {}
        for episode in selected:
            episode_id = episode['id']
            predicted = score['segmentation']['by_episode'][episode_id]['predicted_boundaries']
            reference_boundaries = [{
                'time': boundary['time'], 'minimum': boundary['minimum'],
                'maximum': boundary['maximum'], 'boundary_type': boundary['type'],
            } for boundary in episode['boundaries']
                if .001 < boundary['time'] < episode['duration'] - .001]
            boundary_score = score_boundaries(reference_boundaries, predicted)
            descriptions = []
            for index, segment in enumerate(episode['segments']):
                task_id = f'd:{episode_id}:{index}'
                description = outputs[task_id]['output']['description']
                judgment = judgments[judgment_key(task_id, segment, description)]['judgment']
                matched = sorted(judgment['matched_required'])
                unsupported = judgment['unsupported']
                format_violation = judgment['format_violation']
                descriptions.append({
                    'text': description, 'matchedRequired': matched,
                    'unsupported': unsupported, 'formatViolation': format_violation,
                    'correct': len(matched) == len(segment['required'])
                    and not unsupported and not format_violation,
                })
            summary = boundary_score['summary']
            model_episodes[episode_id] = {
                'descriptions': descriptions,
                'boundaries': [{
                    'time': prediction['time'],
                    'description': prediction.get('description', ''),
                    'status': prediction['status'],
                } for prediction in boundary_score['predictions']],
                'boundarySummary': {
                    'majorHits': summary['major_hits'],
                    'majorTotal': summary['major_total'],
                    'optionalHits': summary['minor_hits'],
                    'extra': summary['extra_predictions'],
                },
            }
        public_models[model] = {
            'name': MODEL_NAMES.get(model, score['model_id']),
            'modelId': score['model_id'],
            'overallF1': score['overall_f1'],
            'episodes': model_episodes,
        }

    website_dir.mkdir(parents=True, exist_ok=True)
    (website_dir / 'data.js').write_text(js_assignment('SHOTWELL_WGO_REFERENCE', {
        'sourceJobId': references['source_job'],
        'sourceDataset': references['source_dataset'],
        'license': references['license'], 'episodes': public_episodes,
    }))
    (website_dir / 'results.js').write_text(js_assignment('SHOTWELL_WGO_RESULTS', {
        'run': run_dir.name, 'judgeRubricVersion': JUDGE_RUBRIC_VERSION,
        'models': public_models,
    }))
    (website_dir.parent / 'scores.json').write_text(json.dumps({
        'benchmark': 'Shotwell-Bench',
        'episode_ids': scores['episode_ids'],
        'fps': config['fps'],
        'judge_rubric_version': JUDGE_RUBRIC_VERSION,
        'models': {
            model: {
                'model_id': score['model_id'],
                'segmentation': {'f1': score['segmentation']['f1']},
                'description': {'f1': score['description']['f1']},
                'overall_f1': score['overall_f1'],
                'refused_tasks': score['refused_tasks'],
            } for model, score in scores['models'].items()
        },
    }, indent=2) + '\n')
    print(f'Published {len(selected)} episodes and {len(public_models)} models to {website_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--website-dir', type=Path, required=True)
    parser.add_argument('--task-labels', type=Path)
    parser.add_argument('--guide', type=Path)
    args = parser.parse_args()
    export(args.run_dir, args.website_dir, args.task_labels, args.guide)


if __name__ == '__main__':
    main()
