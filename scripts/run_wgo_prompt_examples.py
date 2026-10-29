"""Run Astra with WGO-Bench's published segmentation prompt on blog examples."""

import argparse
import base64
import io
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageDraw

from shotwell_bench.cli import load_env
from shotwell_bench.providers import generate
from shotwell_bench.runner import references
from shotwell_bench.video import video_path

FPS = 2
TILE_WIDTH = 224
TILES_PER_SHEET = 20
COLUMNS = 5
TASKS = {
    'homer_1': 'use pitcher to pour water into tall glass and wine glass',
    'homer_2': 'wash the dishes in the sink',
    'homer_3': 'use a spoon to stir contents in a bowl',
    'homer_4': 'Clean tables, desks, or shelves with a cloth',
    'homer_5': 'use a cloth to dust white shelf',
    'homer_9': 'wash the two glasses in the sink',
    'homer_33': 'dice an onion with a nice',
}
SOURCE = 'https://macrodata.co/blog/annotating-robot-video-subtasks'
PROMPT = """Reconstruct the sequence of manipulation events in this robot video from the timestamped contact sheets.

Return only JSON with this shape:
{{"segments":[{{"start_sec":0.0,"end_sec":1.0,"subtask":"short action description"}}]}}

Rules:
- Segment only completed robot manipulation events, not every visible movement.
- Good boundaries happen when a held object changes, an object is placed or released, a tool starts/stops changing a surface, a container/door/lid opens or closes, or contents move between containers.
- Do not split approach, grasp adjustment, small repositioning, and retreat unless the world state changes.
- Do not merge separate pick/place/open/close/pour/wipe events when they complete different states.
- Most segments should be 2-10 seconds. Shorter segments are okay only for fast pick, place, open, close, or release events.
- Use the visible timestamps for start_sec and end_sec.
- Ignore label wording quality; prioritize temporally correct boundaries.

Episode instruction: {task}"""
SCHEMA = {
    'type': 'object', 'properties': {
        'segments': {'type': 'array', 'items': {
            'type': 'object', 'properties': {
                'start_sec': {'type': 'number'},
                'end_sec': {'type': 'number'},
                'subtask': {'type': 'string'},
            },
            'required': ['start_sec', 'end_sec', 'subtask'],
            'additionalProperties': False,
        }},
    },
    'required': ['segments'], 'additionalProperties': False,
}


def contact_sheets(episode, cache):
    with tempfile.TemporaryDirectory(prefix='wgo-example-frames-') as temporary:
        output = Path(temporary) / '%05d.jpg'
        subprocess.run([
            'ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
            '-i', str(video_path(episode, cache)),
            '-vf', f'fps={FPS},scale={TILE_WIDTH}:-2', '-q:v', '5', str(output),
        ], check=True, timeout=900)
        files = sorted(Path(temporary).glob('*.jpg'))
        if not files:
            raise ValueError(f'No sampled frames for {episode["id"]}')
        images = []
        for start in range(0, len(files), TILES_PER_SHEET):
            group = files[start:start + TILES_PER_SHEET]
            with Image.open(group[0]) as first:
                height = first.height
            sheet = Image.new('RGB', (COLUMNS * TILE_WIDTH, 4 * height), '#111111')
            draw = ImageDraw.Draw(sheet)
            for offset, path in enumerate(group):
                x = (offset % COLUMNS) * TILE_WIDTH
                y = (offset // COLUMNS) * height
                with Image.open(path) as frame:
                    sheet.paste(frame, (x, y))
                timestamp = (start + offset) / FPS
                draw.rectangle((x + 3, y + 3, x + 59, y + 18), fill='#101010')
                draw.text((x + 5, y + 4), f'{timestamp:.1f}s', fill='white')
            encoded = io.BytesIO()
            sheet.save(encoded, format='JPEG', quality=82)
            images.append(base64.b64encode(encoded.getvalue()).decode())
    return images


def run_episode(episode, cache):
    task = TASKS[episode['source_id']]
    images = contact_sheets(episode, cache)
    result = generate('astra', PROMPT.format(task=task), images, SCHEMA,
                      max_tokens=16000)
    segments = result['output']['segments']
    if not all(0 <= segment['start_sec'] < segment['end_sec'] <= episode['duration'] + .5
               for segment in segments):
        raise ValueError(f'Invalid segment times for {episode["id"]}')
    return {
        'episode_id': episode['id'], 'source_id': episode['source_id'],
        'model_id': result['model_id'], 'prompt': PROMPT.format(task=task),
        'prompt_source': SOURCE, 'fps': FPS, 'tile_width': TILE_WIDTH,
        'tiles_per_sheet': TILES_PER_SHEET, 'sheet_columns': COLUMNS,
        'reasoning_effort': 'medium', 'segments': segments, 'usage': result['usage'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=Path)
    parser.add_argument('--cache', type=Path,
                        default=Path.home() / '.cache/shotwell-bench')
    parser.add_argument('--out', type=Path,
                        default=Path('benchmark-results/wgo-prompt-examples/astra-official.jsonl'))
    args = parser.parse_args()
    load_env(args.env)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        done = {json.loads(line)['episode_id'] for line in args.out.read_text().splitlines()}
    episodes = [episode for episode in references()['episodes']
                if episode['source_id'] in TASKS and episode['id'] not in done]
    failures = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        futures = [pool.submit(run_episode, episode, args.cache) for episode in episodes]
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                failures.append(str(exc))
                print(f'WGO-prompt episode failed: {exc}', flush=True)
                continue
            with args.out.open('a') as output:
                output.write(json.dumps(result, ensure_ascii=False) + '\n')
            print(f'Saved {result["source_id"]}: {len(result["segments"])} segments', flush=True)
    if failures:
        raise RuntimeError(f'{len(failures)} WGO-prompt episodes failed; rerun to resume')


if __name__ == '__main__':
    main()
