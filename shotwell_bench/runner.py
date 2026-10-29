"""Generate segmentation and fixed-boundary descriptions at 8 fps."""

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .providers import MODELS, generate
from .video import FPS, FRAME_WIDTH, contact_sheets, video_path

REFERENCE_PATH = Path(__file__).resolve().parents[1] / 'references' / 'episodes.json'
GUIDE_PATH = Path(__file__).resolve().parents[1] / 'guide.md'
SEG_SCHEMA = {
    'type': 'object', 'additionalProperties': False, 'required': ['segments'],
    'properties': {'segments': {'type': 'array', 'items': {
        'type': 'object', 'additionalProperties': False,
        'required': ['end', 'description'],
        'properties': {'end': {'type': 'number'}, 'description': {'type': 'string'}}}}}}
DESC_SCHEMA = {'type': 'object', 'additionalProperties': False,
               'required': ['description'],
               'properties': {'description': {'type': 'string'}}}


@dataclass(frozen=True)
class Task:
    id: str
    kind: str
    episode_id: str
    start: float
    end: float
    core_start: float
    core_end: float
    segment_index: int | None = None


def references():
    return json.loads(REFERENCE_PATH.read_text())


def guide():
    return GUIDE_PATH.read_text(encoding='utf-8-sig')


def select_episodes(data, subset):
    if subset == 'first5':
        ids = {f'wgo-bench-homer_{n}' for n in range(1, 6)}
        return [e for e in data['episodes'] if e['id'] in ids]
    if subset == 'completed':
        return [e for e in data['episodes'] if e['description_ready']]
    if subset == 'all':
        return data['episodes']
    ids = {part.strip() for part in subset.split(',')}
    episodes = [e for e in data['episodes'] if e['id'] in ids or e['source_id'] in ids]
    if len(episodes) != len(ids):
        raise ValueError(f'Unrecognized episode selector: {subset}')
    return episodes


def tasks_for_episode(episode, stage='both'):
    tasks = []
    if stage in ('both', 'segmentation'):
        index = 0
        while index * 10 < episode['duration'] - 1e-6:
            core_start = index * 10.0
            core_end = min(episode['duration'], (index + 1) * 10.0)
            start = max(0.0, core_start - 1.0)
            end = min(episode['duration'], core_end + 1.0)
            tasks.append(Task(f's:{episode["id"]}:{index}', 'segmentation', episode['id'],
                              start, end, core_start, core_end))
            index += 1
    if stage in ('both', 'description') and episode['description_ready']:
        for index, segment in enumerate(episode['segments']):
            tasks.append(Task(f'd:{episode["id"]}:{index}', 'description', episode['id'],
                              segment['start'], segment['end'],
                              segment['start'], segment['end'], index))
    return tasks


def _prompt(task, text):
    if task.kind == 'segmentation':
        return (text + '\n\nSEGMENTATION TASK: These timestamped contact sheets sample a '
                f'continuous video at {FPS} frames per second from {task.start:.3f}s to '
                f'{task.end:.3f}s. Identify action segments under the guide. Return the '
                'end timestamp and a short guide-compliant description for every action '
                'that ends within this view. Timestamps must be absolute seconds in the '
                f'episode. The scoring core is {task.core_start:.3f}s to '
                f'{task.core_end:.3f}s; the extra context around it is only to resolve '
                'actions crossing a core edge. Do not invent a boundary at a window/core '
                'edge unless an action visibly ends there. Include IDLE only when the '
                'guide calls for it. Return JSON {"segments":[{"end":number,'
                '"description":string},...]}, ordered by end time. Do not return prose.')
    return (text + '\n\nDESCRIPTION TASK: The exact golden segmentation interval is '
            f'{task.start:.3f}s to {task.end:.3f}s. These contact sheets sample its frames '
            f'at {FPS} fps. Describe only what happens within that fixed interval, using '
            'the guide\'s imperative format and hand/tool details. For overlapping actions, '
            'use a compound sentence; for no action, return IDLE. Do not change the '
            'segment boundaries. Return only JSON {"description":"..."}.')


def _done_ids(path, retry_refusals=False):
    if not path.exists():
        return set()
    done = {}
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        output = row.get('output')
        expected = 'segments' if row.get('kind') == 'segmentation' else 'description'
        valid = (isinstance(output, dict) and expected in output and
                 isinstance(output[expected], list if expected == 'segments' else str))
        if row.get('status') == 'done' and valid:
            done[row['task_id']] = not (retry_refusals and row.get('refusal_reason'))
    return {task_id for task_id, complete in done.items() if complete}


def run(models, episodes, out_dir, cache, workers=32, stage='both', limit_tasks=None,
        retry_refusals=False, guide_path=None):
    out_dir = Path(out_dir)
    cache = Path(cache)
    guide_path = Path(guide_path) if guide_path else GUIDE_PATH
    out_dir.mkdir(parents=True, exist_ok=True)
    guide_text = guide_path.read_text(encoding='utf-8-sig')
    episode_by_id = {e['id']: e for e in episodes}
    tasks = [task for e in episodes for task in tasks_for_episode(e, stage)]
    if limit_tasks is not None:
        tasks = tasks[:limit_tasks]
    metadata = {'fps': FPS, 'frame_width': FRAME_WIDTH, 'contact_sheet': '4x2',
                'segmentation_core_seconds': 10, 'context_seconds': 1,
                'guide_sha256': hashlib.sha256(guide_path.read_bytes()).hexdigest(),
                'reference_sha256': hashlib.sha256(REFERENCE_PATH.read_bytes()).hexdigest(),
                'models': {name: MODELS[name][1] for name in models},
                'episode_ids': list(episode_by_id)}
    config_path = out_dir / 'config.json'
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        for key in ('fps', 'frame_width', 'contact_sheet', 'segmentation_core_seconds',
                    'context_seconds', 'guide_sha256', 'reference_sha256', 'episode_ids'):
            if existing[key] != metadata[key]:
                raise ValueError(f'Run config mismatch for {key}; use a new run directory')
        for name, model_id in metadata['models'].items():
            if name in existing['models'] and existing['models'][name] != model_id:
                raise ValueError(f'Run config mismatch for model {name}')
        metadata['models'] = {**existing['models'], **metadata['models']}
        metadata = {**existing, **metadata}
    config_path.write_text(json.dumps(metadata, indent=2) + '\n')

    done_by_model = {model: _done_ids(out_dir / f'{model}.jsonl', retry_refusals)
                     for model in models}
    pending = [(model, task, out_dir / f'{model}.jsonl')
               for task in tasks for model in models
               if task.id not in done_by_model[model]]
    if not pending:
        print('All requested model tasks are already complete.', flush=True)
        return

    # Compute each remaining visual input once while interleaving provider calls.
    needed_ids = {task.id for _, task, _ in pending}
    needed = [(task, episode_by_id[task.episode_id])
              for task in tasks if task.id in needed_ids]
    with ThreadPoolExecutor(max_workers=min(workers, 12)) as pool:
        futures = [pool.submit(contact_sheets, ep, task.start, task.end, cache)
                   for task, ep in needed]
        for future in as_completed(futures):
            future.result()
    print(f'Prepared 8 fps visual inputs for {len(tasks)} tasks', flush=True)

    print(f'Running {len(pending)} model calls with {workers} workers', flush=True)
    output_lock = threading.Lock()
    completed = 0
    errors = 0

    def work(model, task, path):
        ep = episode_by_id[task.episode_id]
        images = contact_sheets(ep, task.start, task.end, cache)
        schema = SEG_SCHEMA if task.kind == 'segmentation' else DESC_SCHEMA
        try:
            max_tokens = (1000 if task.kind == 'segmentation' else 300) if model == 'qwen35' else (
                5000 if task.kind == 'segmentation' else 2500)
            result = generate(model, _prompt(task, guide_text), images, schema,
                              max_tokens=max_tokens)
            row = {'task_id': task.id, 'kind': task.kind, 'episode_id': task.episode_id,
                   'model': model, 'model_id': result['model_id'], 'status': 'done',
                   'start': task.start, 'end': task.end,
                   'core_start': task.core_start, 'core_end': task.core_end,
                   'segment_index': task.segment_index,
                   'output': result['output'], 'usage': result['usage']}
            if result.get('refusal_reason'):
                row['refusal_reason'] = result['refusal_reason']
        except Exception as exc:
            row = {'task_id': task.id, 'kind': task.kind, 'episode_id': task.episode_id,
                   'model': model, 'status': 'error', 'error': str(exc)[:1000]}
        with output_lock:
            with path.open('a') as output:
                output.write(json.dumps(row, ensure_ascii=False) + '\n')
        return row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, *item) for item in pending]
        for future in as_completed(futures):
            row = future.result()
            completed += 1
            errors += row['status'] != 'done'
            if completed % 20 == 0 or row['status'] != 'done' or completed == len(pending):
                print(f'{completed}/{len(pending)} complete, {errors} errors; '
                      f'latest {row["model"]} {row["task_id"]}: {row["status"]}', flush=True)
    if errors:
        raise RuntimeError(f'{errors} model calls failed; rerun to retry them')
