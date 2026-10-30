"""Run the public WGO-Bench on the model versions in Shotwell-Bench.

The generated scores are independent replications, not NVIDIA's published runs.
Source videos stay in the local cache and are never written to this repository.
"""

import argparse
import base64
import io
import json
import math
import subprocess
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image, ImageDraw

from shotwell_bench.cli import load_env
from shotwell_bench import providers
from shotwell_bench.providers import generate

DATASET = 'https://huggingface.co/datasets/macrodata/WGO-Bench'
SEGMENT_PROMPT = """Reconstruct the sequence of manipulation events in this video from timestamped contact sheets.

Return only JSON with this shape: {{"segments":[{{"start_sec":0.0,"end_sec":1.0,"subtask":"short action description"}}]}}

Rules:
- Segment completed manipulation events, not every visible movement.
- Good boundaries happen when a held object changes, an object is placed or released, a tool starts or stops changing a surface, a container/door/lid opens or closes, or contents move between containers.
- Do not split approach, grasp adjustment, small repositioning, and retreat unless the world state changes.
- Do not merge separate pick/place/open/close/pour/wipe events when they complete different states.
- Most segments should be 2-10 seconds. Shorter segments are fine for quick pick, place, open, close, or release events.
- Use the visible timestamps for start_sec and end_sec.
- Prioritize temporal boundaries over label wording.

Episode instruction: {instruction}"""
CAPTION_PROMPT = """Annotate one fixed segment from a longer video.
Return only JSON with a short descriptive subtask label in the description field.
The three images are the previous, current, and next predicted segments, respectively. A missing neighbor is blank. Each image is timestamped with absolute video time.

Episode instruction: {instruction}
Target segment: {index} of {count}
Target time: {start:.2f}s to {end:.2f}s
Original predicted label for this exact segment: {seed}

Label only the current segment. Use neighboring images only to disambiguate what changed during the current segment. Treat the original predicted label as a strong prior, not ground truth. Verify it and make only necessary corrections: keep the main action and object when they are right, but replace a wrong action, object, destination, or state change. Name the completed visible change and its main object in one concise phrase. Do not split or merge this segment, describe neighboring events, or mention uncertainty, timestamps, or invisible intent."""
SEGMENT_SCHEMA = {'type': 'object', 'properties': {'segments': {'type': 'array', 'items': {
    'type': 'object', 'properties': {'start_sec': {'type': 'number'},
                                  'end_sec': {'type': 'number'},
                                  'subtask': {'type': 'string'}},
    'required': ['start_sec', 'end_sec', 'subtask'], 'additionalProperties': False}}},
    'required': ['segments'], 'additionalProperties': False}
CAPTION_SCHEMA = {'type': 'object', 'properties': {'description': {'type': 'string'}},
                  'required': ['description'], 'additionalProperties': False}
_write_lock = threading.Lock()
_frame_locks = {}
_frame_locks_guard = threading.Lock()


def _rows(parquet):
    file = pq.ParquetFile(parquet)
    for index in range(file.num_row_groups):
        row = file.read_row_group(index, columns=['id', 'instruction', 'segments', 'metadata'])
        yield index, row.to_pylist()[0]


def _video(parquet, index, episode_id, cache):
    path = cache / 'videos' / f'{episode_id}.mp4'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        row = pq.ParquetFile(parquet).read_row_group(index, columns=['video']).to_pylist()[0]
        temporary = path.with_suffix('.download')
        temporary.write_bytes(row['video'])
        temporary.replace(path)
    return path


def _video_duration(video):
    return float(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', str(video),
    ], text=True, timeout=30).strip())


def _sheet(frames, fps, start_time=0, timestamps=None):
    result = []
    for first in range(0, len(frames), 20):
        chunk = frames[first:first + 20]
        canvas = Image.new('RGB', (1120, 224 * math.ceil(len(chunk) / 5)), '#111111')
        draw = ImageDraw.Draw(canvas)
        for offset, frame in enumerate(chunk):
            x, y = (offset % 5) * 224, (offset // 5) * 224
            with Image.open(frame) as image:
                canvas.paste(image.convert('RGB'), (x, y))
            draw.rectangle((x + 2, y + 2, x + 92, y + 20), fill='#111111')
            timestamp = (timestamps[first + offset] if timestamps is not None
                         else start_time + (first + offset) / fps)
            draw.text((x + 5, y + 5), f'{timestamp:.2f}s', fill='white')
        buffer = io.BytesIO()
        canvas.save(buffer, format='JPEG', quality=90)
        result.append(base64.b64encode(buffer.getvalue()).decode())
    return result


def _frames(video, folder, fps, start=0, duration=None):
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / '.complete'
    with _frame_locks_guard:
        lock = _frame_locks.setdefault(str(folder), threading.Lock())
    with lock:
        if not marker.exists():
            command = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
                       '-threads', '1', '-ss', str(start), '-i', str(video)]
            if duration is not None:
                command += ['-t', str(max(duration, .25))]
            command += ['-vf', f'fps={fps},scale=224:224:force_original_aspect_ratio=decrease,pad=224:224:(ow-iw)/2:(oh-ih)/2',
                        '-q:v', '2', str(folder / '%05d.jpg')]
            subprocess.run(command, check=True, timeout=900)
            if not list(folder.glob('*.jpg')):
                raise ValueError(f'No video frames: {video} {start}-{duration}')
            marker.touch()
        return sorted(folder.glob('*.jpg'))


def _segments(episode, parquet, cache):
    video = _video(parquet, episode['index'], episode['id'], cache)
    frames = _frames(video, cache / 'wgo-frames' / episode['id'] / '2fps-224', 2)
    if len(frames) > 1024:
        frames = [frames[round(i * (len(frames) - 1) / 1023)] for i in range(1024)]
    return _sheet(frames, 2)


def _caption_images(episode, segments, index, parquet, cache):
    video = _video(parquet, episode['index'], episode['id'], cache)
    all_frames = _frames(video, cache / 'wgo-frames' / episode['id'] / '4fps-224', 4)
    result = []
    for neighbor in (index - 1, index, index + 1):
        if neighbor < 0 or neighbor >= len(segments):
            result.append(_sheet([], 4))
            continue
        segment = segments[neighbor]
        left = max(0, math.ceil(segment['start_sec'] * 4))
        right = min(len(all_frames), math.ceil(segment['end_sec'] * 4))
        indices = list(range(left, right))
        if len(indices) > 5:
            indices = [indices[round(i * (len(indices) - 1) / 4)] for i in range(5)]
        result.append(_sheet([all_frames[frame] for frame in indices], 4,
                             timestamps=[frame / 4 for frame in indices]) if indices else [])
    return result


def _load(path, key):
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        rows[row[key]] = row
    return rows


def _append(path, row):
    with _write_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as output:
            output.write(json.dumps(row, ensure_ascii=False) + '\n')


def _run_segmentation(model, episode, parquet, cache, output):
    prompt = SEGMENT_PROMPT.format(instruction=episode['instruction'])
    result = generate(model, prompt, _segments(episode, parquet, cache),
                      SEGMENT_SCHEMA, max_tokens=16000)
    raw = result['output']['segments']
    duration = episode['duration']
    segments = []
    for item in raw:
        try:
            start, end = float(item['start_sec']), float(item['end_sec'])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(start) and math.isfinite(end) and end > start:
            segments.append({'start_sec': max(0, min(start, duration)),
                             'end_sec': max(0, min(end, duration)),
                             'draft': item.get('subtask', '')})
    segments = [item for item in segments if item['end_sec'] > item['start_sec']]
    segments.sort(key=lambda item: item['start_sec'])
    _append(output, {'episode_id': episode['id'], 'model': model,
                     'model_id': result['model_id'], 'segments': segments,
                     'refusal_reason': result.get('refusal_reason'),
                     'usage': result['usage']})
    return model, episode['id'], len(segments)


def _run_caption(model, episode, index, segments, parquet, cache, output):
    segment = segments[index]
    blocks = _caption_images(episode, segments, index, parquet, cache)
    prompt = CAPTION_PROMPT.format(instruction=episode['instruction'],
                                   index=index + 1, count=len(segments),
                                   start=segment['start_sec'], end=segment['end_sec'],
                                   seed=segment['draft'])
    images = []
    for sheets in blocks:
        if sheets:
            images.extend(sheets)
        else:
            blank = Image.new('RGB', (1120, 224), '#111111')
            buffer = io.BytesIO()
            blank.save(buffer, format='JPEG', quality=90)
            images.append(base64.b64encode(buffer.getvalue()).decode())
    result = generate(model, prompt, images, CAPTION_SCHEMA, max_tokens=4000)
    _append(output, {'key': f'{episode["id"]}:{index}', 'model': model,
                     'description': result['output']['description'],
                     'recipe': 'wgo-neighbor-seeded-v1',
                     'refusal_reason': result.get('refusal_reason'),
                     'usage': result['usage']})
    return model, episode['id'], index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['prepare', 'segment', 'caption'], required=True)
    parser.add_argument('--models', default='sol,luna,argon,opus,sonnet,kimi31pro')
    parser.add_argument('--parquet', type=Path,
                        default=Path.home() / '.cache/shotwell-bench/wgo-annotations.parquet')
    parser.add_argument('--cache', type=Path,
                        default=Path.home() / '.cache/shotwell-bench')
    parser.add_argument('--run-dir', type=Path, default=Path('benchmark-results/wgo-ranking-2026-10-30-v3'))
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--env', type=Path)
    args = parser.parse_args()
    load_env(args.env)
    providers._provider_limits['openai'] = threading.BoundedSemaphore(4)
    episodes = []
    for index, row in _rows(args.parquet):
        episodes.append({'id': row['id'], 'index': index,
                         'instruction': row['instruction'],
                         'duration': None if args.stage == 'prepare' else
                         _video_duration(_video(args.parquet, index, row['id'], args.cache)),
                         'subset': row['id'].split('_')[0],
                         'reference': row['segments']})
    args.run_dir.mkdir(parents=True, exist_ok=True)
    models = args.models.split(',')
    manifest = {
        'dataset': DATASET,
        'episodes': len(episodes),
        'prediction_duration_source': 'ffprobe_video_duration',
        'segmentation_prompt': SEGMENT_PROMPT,
        'caption_prompt': CAPTION_PROMPT,
        'segmentation_input': {'fps': 2, 'tile': [224, 224], 'sheet': [5, 4],
                               'jpeg_quality': 90, 'frame_cap': 1024},
        'caption_input': {'frame_grid_fps': 4, 'uniform_frames_per_predicted_segment': 5,
                          'context': 'previous,current,next_predicted_segments',
                          'seeded_relabeling': True, 'recipe': 'wgo-neighbor-seeded-v1'},
    }
    manifest_path = args.run_dir / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('WGO run manifest differs from the current recipe')
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    jobs = []
    if args.stage == 'prepare':
        jobs = [(episode['id'], _video, (args.parquet, episode['index'], episode['id'], args.cache))
                for episode in episodes]
    elif args.stage == 'segment':
        done = {model: _load(args.run_dir / f'{model}-segments.jsonl', 'episode_id')
                for model in models}
        for episode in episodes:
            for model in models:
                if episode['id'] not in done[model]:
                    output = args.run_dir / f'{model}-segments.jsonl'
                    jobs.append((f'{model}:{episode["id"]}', _run_segmentation,
                                 (model, episode, args.parquet, args.cache, output)))
    else:
        segments_by_model = {model: _load(args.run_dir / f'{model}-segments.jsonl', 'episode_id')
                             for model in models}
        done_by_model = {model: _load(args.run_dir / f'{model}-captions.jsonl', 'key')
                         for model in models}
        for episode in episodes:
            for model in models:
                segments = segments_by_model[model]
                if episode['id'] not in segments:
                    continue
                output = args.run_dir / f'{model}-captions.jsonl'
                done = done_by_model[model]
                for index, segment in enumerate(segments[episode['id']]['segments']):
                    key = f'{episode["id"]}:{index}'
                    if key not in done:
                        jobs.append((f'{model}:{key}', _run_caption,
                                     (model, episode, index, segments[episode['id']]['segments'],
                                      args.parquet, args.cache, output)))
    print(f'{args.stage}: {len(jobs)} tasks; {len(episodes)} episodes; models={models}', flush=True)
    errors = Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fn, *arguments): key for key, fn, arguments in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as exc:
                errors[type(exc).__name__] += 1
                print(f'FAILED {futures[future]}: {exc}', flush=True)
            if index % 20 == 0 or index == len(futures):
                print(f'{args.stage}: {index}/{len(futures)}, errors={sum(errors.values())}', flush=True)
    if errors:
        raise RuntimeError(f'{dict(errors)}; rerun the same command to resume')


if __name__ == '__main__':
    main()
