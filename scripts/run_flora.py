"""Run NVIDIA FLORA on the benchmark episodes using Modal H100 workers.

FLORA's temporal model samples internally at 4 fps. Inputs are normalized to
8 fps, but this is not the same visual exposure as the API model runs.
"""

import hashlib
import json
import subprocess
from pathlib import Path

import modal

REVISION = '78a0e4ac75d33509f077fdf985ebb51153920e7c'
MODEL_ID = f'nvidia/Flora-TempSeg+Flora-Captioner@{REVISION}'
INSTRUCTION = (
    'Follow Shotwell-WGO annotation rules: segment complete or attempted hand actions, '
    'navigation, and IDLE. Each distinct action merits a segment; group overlapping '
    'actions, and group short repetitive motions using the same object and hand. '
    'Pick up only when an item is fully lifted from its supporting surface; moving '
    'an item along a surface is one Move action, not Pick up plus Place. Ignore '
    'contact without effect, brief pats and clothing alignment, passive rinsing, '
    'and head-camera adjustment. IDLE requires at least 1.5 seconds, navigation '
    'at least 1 second. Caption in imperative form: verb, object, acting hand(s), '
    'and tool when used. For overlapping actions, join full clauses with and. '
    'For a placed item, give the destination; for an item slid on a surface, '
    'give direction. Do not invent intentions or unseen details.'
)

app = modal.App('shotwell-wgo-flora')
weights = modal.Volume.from_name('flora-eval')
image = (
    modal.Image.debian_slim(python_version='3.12')
    .apt_install('ffmpeg')
    .run_commands('pip install torch==2.11.0 torchvision==0.26.0 '
                  '--index-url https://download.pytorch.org/whl/cu128')
    .pip_install('huggingface_hub==1.33.0', 'transformers==5.8.1',
                 'peft==0.18.1', 'accelerate==1.12.0', 'safetensors==0.7.0',
                 'av==16.0.1', 'numpy<3', 'Pillow>=11,<13', 'requests')
    .run_commands(
        'python -c \'from huggingface_hub import snapshot_download; '
        f'snapshot_download("nvidia/Flora-TempSeg", revision="{REVISION}", '
        'allow_patterns=["infer.py", "batch_utils/**"], local_dir="/opt/flora")\'',
        'test -f /opt/flora/batch_utils/api.py',
    )
    .env({'FLORA_MODEL_CACHE': '/flora/models', 'HF_HOME': '/flora/hf'})
)


def clip_plan(episode):
    duration = float(episode['duration'])
    if duration <= 120:
        clips = [(0.0, duration)]
        if duration > 45:
            split_candidates = [float(segment['end']) for segment in episode['segments']
                                if duration * .3 <= float(segment['end']) <= duration * .7]
            if split_candidates:
                split = min(split_candidates, key=lambda time: abs(time - duration / 2))
                clips = [(0.0, split), (split, duration)]
    elif duration <= 200:
        clips = [(0.0, 100.0), (80.0, duration)]
    else:
        raise ValueError(f'FLORA clip planner does not support {duration:.1f}s')
    assignments = {index: [] for index in range(len(clips))}
    for index, segment in enumerate(episode['segments']):
        start, end = float(segment['start']), float(segment['end'])
        containing = [i for i, (lo, hi) in enumerate(clips)
                      if lo <= start + 1e-4 and end <= hi + 1e-4]
        if containing:
            chosen = min(containing, key=lambda i: abs((start + end) / 2 - sum(clips[i]) / 2))
        else:
            if end - start > 119.5:
                raise ValueError(f'Golden segment {index} exceeds FLORA limit')
            lo = max(0.0, min(start - 2.0, duration - 119.5))
            hi = min(duration, lo + 119.5)
            clips.append((lo, hi))
            chosen = len(clips) - 1
            assignments[chosen] = []
        assignments[chosen].append((index, start - clips[chosen][0], end - clips[chosen][0]))
    return clips, assignments


class LoadedModel:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def load(self, _kind):
        return self.model, self.processor


@app.function(image=image, gpu='H100', volumes={'/flora': weights},
              timeout=5 * 60 * 60, max_containers=5)
def infer_episode(episode):
    import sys
    import tempfile
    import traceback

    import requests

    sys.path.insert(0, '/opt/flora')
    from batch_utils.api import DataEnginePipeline
    from batch_utils import pipeline
    from batch_utils.config import MODELS as FLORA_MODELS
    from batch_utils.models import Session

    try:
        clips, assignments = clip_plan(episode)
        with tempfile.TemporaryDirectory(prefix='wgo-flora-') as temporary:
            root = Path(temporary)
            source = root / 'source.mp4'
            with requests.get(episode['video_url'], stream=True, timeout=120) as response:
                response.raise_for_status()
                with source.open('wb') as output:
                    for chunk in response.iter_content(1024 * 1024):
                        output.write(chunk)
            videos_by_clip = []
            for index, (start, end) in enumerate(clips):
                target = root / f'clip_{index}.mp4'
                subprocess.run([
                    'ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
                    '-ss', str(start), '-i', str(source), '-t', str(end - start),
                    '-vf', 'fps=8', '-an', '-c:v', 'libx264', '-crf', '19',
                    '-preset', 'fast', str(target),
                ], check=True, timeout=900)
                videos, _ = DataEnginePipeline._inputs(
                    target, 'human', None, None, False, INSTRUCTION)
                videos_by_clip.append(videos)

            session = Session()
            predicted_by_clip = {}
            try:
                model, processor = session.load('tempseg')
                segmenter = LoadedModel(model, processor)
                for index in range(min(len(clips), 2)):
                    segments, _ = pipeline.segment(
                        videos_by_clip[index], 'human', segmenter,
                        INSTRUCTION, lambda *args, **kwargs: None)
                    predicted_by_clip[index] = segments
                model, processor = session.load('captioner')
                captioner = LoadedModel(model, processor)
                predictions = []
                descriptions = {}
                split_point = ((clips[0][1] + clips[1][0]) / 2
                               if len(clips) > 1 else None)
                for index, videos in enumerate(videos_by_clip):
                    native = predicted_by_clip.get(index, [])
                    golden = assignments[index]
                    intervals = native + [
                        {'start_s': start, 'end_s': end}
                        for _, start, end in golden
                    ]
                    if not intervals:
                        continue
                    captioned, _ = pipeline.caption_single(
                        videos, intervals, 'Human', captioner,
                        INSTRUCTION, lambda *args, **kwargs: None)
                    if len(captioned) != len(intervals):
                        raise RuntimeError('FLORA caption count differs from interval count')
                    for segment, caption in zip(native, captioned[:len(native)]):
                        absolute = clips[index][0] + float(segment['end_s'])
                        if not 0 < absolute < episode['duration'] - .001:
                            continue
                        if split_point is not None and (index == 0 and absolute > split_point
                                                        or index == 1 and absolute <= split_point):
                            continue
                        predictions.append({
                            'time': absolute,
                            'description': caption['caption'].strip()
                            if caption['status'] == 'complete' else '',
                        })
                    for (segment_index, _, _), caption in zip(golden, captioned[len(native):]):
                        descriptions[segment_index] = (
                            caption['caption'].strip() if caption['status'] == 'complete' else ''
                        )
                if len(descriptions) != len(episode['segments']):
                    raise RuntimeError('FLORA did not process every golden interval')
            finally:
                session.close()
        return {'episode_id': episode['id'], 'predictions': predictions,
                'descriptions': descriptions, 'clips': clips,
                'model_revisions': FLORA_MODELS}
    except Exception as exc:
        return {'episode_id': episode['id'], 'error': repr(exc),
                'traceback': traceback.format_exc()[-4000:]}


def append_rows(path, episode, result):
    from shotwell_bench.runner import tasks_for_episode

    predictions = result['predictions']
    with path.open('a') as output:
        for task in tasks_for_episode(episode):
            if task.kind == 'segmentation':
                segments = [
                    {'end': prediction['time'], 'description': prediction['description']}
                    for prediction in predictions
                    if task.core_start < prediction['time'] <= task.core_end
                ]
                value = {'segments': segments}
            else:
                value = {'description': result['descriptions'][task.segment_index]}
            row = {
                'task_id': task.id, 'kind': task.kind, 'episode_id': episode['id'],
                'model': 'flora', 'model_id': MODEL_ID, 'status': 'done',
                'start': task.start, 'end': task.end,
                'core_start': task.core_start, 'core_end': task.core_end,
                'segment_index': task.segment_index, 'output': value, 'usage': {},
                'model_revisions': result.get('model_revisions', {}),
            }
            output.write(json.dumps(row, ensure_ascii=False) + '\n')


@app.local_entrypoint()
def main(run_dir: str, guide: str, subset: str = 'completed'):
    from shotwell_bench.runner import REFERENCE_PATH, tasks_for_episode, references, select_episodes

    run_path = Path(run_dir)
    guide_path = Path(guide)
    episodes = select_episodes(references(), subset)
    config_path = run_path / 'config.json'
    config = json.loads(config_path.read_text())
    assert config['episode_ids'] == [episode['id'] for episode in episodes]
    assert config['reference_sha256'] == hashlib.sha256(REFERENCE_PATH.read_bytes()).hexdigest()
    assert config['guide_sha256'] == hashlib.sha256(guide_path.read_bytes()).hexdigest()
    assert len(INSTRUCTION) <= 2000
    if config['models'].get('flora') not in (None, MODEL_ID):
        raise ValueError('FLORA checkpoint differs from the existing run')
    config['models']['flora'] = MODEL_ID
    config.setdefault('model_inputs', {})['flora'] = {
        'source_fps': 8, 'temporal_model_fps': 4,
        'pipeline': 'NVIDIA native TempSeg + Captioner',
        'instruction': INSTRUCTION,
    }
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    output = run_path / 'flora.jsonl'
    completed = set()
    if output.exists():
        for line in output.read_text().splitlines():
            row = json.loads(line)
            if row.get('status') == 'done':
                completed.add(row['task_id'])
    pending = [episode for episode in episodes
               if any(task.id not in completed for task in tasks_for_episode(episode))]
    print(f'FLORA episodes pending: {len(pending)}', flush=True)
    by_id = {episode['id']: episode for episode in pending}
    failures = []
    for result in infer_episode.map(pending, order_outputs=False):
        episode_id = result['episode_id']
        if 'error' in result:
            failures.append(result)
            print(f'FLORA failed {episode_id}: {result["error"]}\n'
                  f'{result.get("traceback", "")}', flush=True)
            continue
        append_rows(output, by_id[episode_id], result)
        print(f'FLORA saved {episode_id}: {len(result["predictions"])} boundaries, '
              f'{len(result["descriptions"])} descriptions', flush=True)
    if failures:
        raise RuntimeError(f'{len(failures)} FLORA episodes failed: {failures}')
