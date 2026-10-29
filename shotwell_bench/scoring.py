"""One-to-one boundary matching and reference-constrained semantic grading."""

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .alternatives import expand_alternatives
from .boundary_score import score_boundaries
from .providers import JUDGE_MODEL, judge
from .runner import GUIDE_PATH, REFERENCE_PATH, tasks_for_episode

JUDGE_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['matched_required', 'unsupported', 'format_violation'],
    'properties': {
        'matched_required': {'type': 'array', 'items': {'type': 'integer'}},
        'unsupported': {'type': 'array', 'items': {'type': 'string'}},
        'format_violation': {'type': 'string'},
    },
}
JUDGE_RUBRIC_VERSION = 'core-action-v2'


def judgment_key(task_id, segment, description):
    fingerprint = hashlib.sha256(json.dumps(
        [JUDGE_RUBRIC_VERSION, segment['required'], segment['optional'], description],
        ensure_ascii=False).encode()).hexdigest()
    return f'{task_id}:{fingerprint}'


def _latest_rows(path, key_field='task_id'):
    if not path.exists():
        raise FileNotFoundError(path)
    rows = {}
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get('status') == 'done':
            rows[row[key_field]] = row
    return rows


def _f1(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1,
            'tp': tp, 'fp': fp, 'fn': fn}


def _segmentation(episodes, rows):
    major_total = major_hits = matched_predictions = prediction_total = 0
    optional_hits = 0
    by_episode = {}
    for episode in episodes:
        predictions = []
        for task in tasks_for_episode(episode, 'segmentation'):
            row = rows[task.id]
            segments = row['output'].get('segments')
            if not isinstance(segments, list):
                raise ValueError(f'Invalid segments for {task.id}')
            for segment in segments:
                t = float(segment['end'])
                if task.core_start < t <= task.core_end and 0 < t < episode['duration'] - .001:
                    predictions.append({'time': t, 'description': segment.get('description', '')})
        predictions.sort(key=lambda p: p['time'])
        deduped = []
        for prediction in predictions:
            if not deduped or prediction['time'] - deduped[-1]['time'] > .08:
                deduped.append(prediction)
        refs = [{'time': b['time'], 'minimum': b['minimum'],
                 'maximum': b['maximum'], 'boundary_type': b['type']}
                for b in episode['boundaries']
                if .001 < b['time'] < episode['duration'] - .001]
        scored = score_boundaries(refs, deduped)
        summary = scored['summary']
        major_total += summary['major_total']
        major_hits += summary['major_hits']
        optional_hits += summary['minor_hits']
        matched_predictions += summary['matched_predictions']
        prediction_total += summary['prediction_total']
        by_episode[episode['id']] = {**summary, 'predicted_boundaries': deduped}
    return {**_f1(matched_predictions, prediction_total - matched_predictions,
                  major_total - major_hits),
            'major_total': major_total, 'major_hits': major_hits,
            'optional_hits': optional_hits, 'prediction_total': prediction_total,
            'by_episode': by_episode}


def _judge_prompt(required, optional, description):
    reference = json.dumps({'required': [
        {'original': text, 'acceptable': expand_alternatives(text)} for text in required],
        'optional': [
        {'original': text, 'acceptable': expand_alternatives(text)} for text in optional],
                            'candidate': description}, ensure_ascii=False)
    return (
        'You are a blinded semantic judge for egocentric action annotations. '
        'Evaluate the candidate only against the reference facts below; do not infer '
        'events from an unseen video. Every distinct required action must be expressed. '
        'Optional actions are allowed but need not be expressed. Match a required action '
        'when its core verb (or equivalent), manipulated object, and acting hand(s) are '
        'correct. Allow a slight generalization that omits a nonessential modifier, '
        'tool, material, or destination, provided the same action is still clear. '
        'Allow compatible extra specificity about that same action, such as water for '
        'liquid, jug for container, or a more precise object subtype. Do not call a '
        'compatible refinement unsupported merely because the reference is less specific. '
        'Synonyms and grammatical differences are fine. Bracketed alternatives have '
        'already been expanded into the acceptable variants: [Rub / rotate] '
        'allows either verb, a single bracketed phrase may be omitted, and '
        'nested alternatives are also disjunctions. Match ONE acceptable '
        'variant to credit the original required detail. For IDLE, only an '
        'IDLE-equivalent candidate matches. Return zero-based indices of fully '
        'matched required details. List in "unsupported" only a clearly contradictory '
        'verb, object, hand, tool, or destination; a separate action absent from both '
        'required and optional details; or a distinct outcome, intention, or causal '
        'claim with no reference support. Duplicate actions also count as unsupported. '
        'A compatible elaboration of a referenced action is not a separate action. '
        'Do not penalize merely uncertain harmless specificity or a slight '
        'generalization. Set '
        '"format_violation" to a short reason if the candidate breaks the '
        'guide\'s imperative/compound-sentence form, otherwise an empty string. '
        'Do not credit a required action with a wrong core verb, object, or hand. '
        'Return only JSON.\n\n'
        + reference)


def _description(episodes, model_rows, judgment_path, workers):
    existing = _latest_rows(judgment_path, 'key') if judgment_path.exists() else {}
    pending = []
    all_tasks = []
    for episode in episodes:
        if not episode['description_ready']:
            raise ValueError(f'{episode["id"]} lacks complete descriptions')
        for task in tasks_for_episode(episode, 'description'):
            segment = episode['segments'][task.segment_index]
            description = model_rows[task.id]['output']['description']
            key = judgment_key(task.id, segment, description)
            all_tasks.append((task, segment, description, key))
            if key not in existing:
                pending.append((task, segment, description, key))
    lock = threading.Lock()
    judgment_path.parent.mkdir(parents=True, exist_ok=True)

    def work(task, segment, description, key):
        output = judge(_judge_prompt(segment['required'], segment['optional'], description),
                       JUDGE_SCHEMA)
        matched = output.get('matched_required')
        unsupported = output.get('unsupported')
        if (not isinstance(matched, list) or not isinstance(unsupported, list)
                or any(type(i) is not int or i < 0 or i >= len(segment['required'])
                       for i in matched) or len(set(matched)) != len(matched)
                or any(not isinstance(s, str) for s in unsupported)):
            raise ValueError(f'Invalid judge output for {task.id}: {output}')
        row = {'task_id': task.id, 'key': key, 'status': 'done',
               'candidate': description, 'judgment': output}
        with lock:
            with judgment_path.open('a') as output_file:
                output_file.write(json.dumps(row, ensure_ascii=False) + '\n')
        return row

    if pending:
        print(f'Judging {len(pending)} descriptions with {workers} workers', flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(work, *task) for task in pending]
            for index, future in enumerate(as_completed(futures), 1):
                future.result()
                if index % 20 == 0 or index == len(pending):
                    print(f'Judged {index}/{len(pending)}', flush=True)
        existing = _latest_rows(judgment_path, 'key')
    tp = fp = fn = valid_segments = 0
    by_episode = {}
    for task, segment, description, key in all_tasks:
        verdict = existing[key]['judgment']
        matched = len(verdict['matched_required'])
        missing = len(segment['required']) - matched
        extra = len(verdict['unsupported']) + bool(verdict['format_violation'])
        tp += matched
        fn += missing
        fp += extra
        valid_segments += not (missing or extra)
        item = by_episode.setdefault(task.episode_id,
                                     {'required': 0, 'matched': 0, 'unsupported': 0,
                                      'valid_segments': 0, 'segments': 0})
        item['required'] += len(segment['required'])
        item['matched'] += matched
        item['unsupported'] += extra
        item['valid_segments'] += not (missing or extra)
        item['segments'] += 1
    return {**_f1(tp, fp, fn), 'required_details': tp + fn,
            'valid_segments': valid_segments, 'segment_total': len(all_tasks),
            'by_episode': by_episode, 'judge_model': JUDGE_MODEL,
            'judge_rubric_version': JUDGE_RUBRIC_VERSION}


def score(models, episodes, run_dir, workers=24):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / 'config.json').read_text())
    if config['reference_sha256'] != hashlib.sha256(REFERENCE_PATH.read_bytes()).hexdigest():
        raise ValueError('Reference snapshot changed since generation')
    if config['guide_sha256'] != hashlib.sha256(GUIDE_PATH.read_bytes()).hexdigest():
        raise ValueError('Guide changed since generation')
    results = {'benchmark': 'Shotwell-WGO-Bench', 'fps': 8,
               'episode_ids': [e['id'] for e in episodes], 'models': {}}
    for model in models:
        rows = _latest_rows(run_dir / f'{model}.jsonl')
        required_ids = {task.id for episode in episodes for task in tasks_for_episode(episode)}
        missing = required_ids - rows.keys()
        if missing:
            raise ValueError(f'{model} has {len(missing)} unfinished tasks; rerun generation')
        segmentation = _segmentation(episodes, rows)
        description = _description(episodes, rows, run_dir / 'judgments' / f'{model}.jsonl',
                                   workers)
        results['models'][model] = {
            'model_id': rows[next(iter(required_ids))]['model_id'],
            'segmentation': segmentation, 'description': description,
            'refused_tasks': sum(bool(rows[task_id].get('refusal_reason'))
                                 for task_id in required_ids),
            'overall_f1': segmentation['f1'] * description['f1']}
        print(f'{model}: seg {segmentation["f1"]:.3f}, '
              f'desc {description["f1"]:.3f}, '
              f'overall {results["models"][model]["overall_f1"]:.3f}', flush=True)
        (run_dir / 'scores.json').write_text(json.dumps(results, indent=2) + '\n')
    return results
