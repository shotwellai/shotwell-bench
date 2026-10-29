"""Public-video cache and identical 8 fps visual inputs for every model."""

import base64
import hashlib
import io
import subprocess
import tempfile
import threading
from pathlib import Path

import requests
from PIL import Image, ImageDraw

FPS = 8
FRAME_WIDTH = 512
TILES_PER_SHEET = 8
_locks = {}
_locks_guard = threading.Lock()


def _lock(key):
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


def video_path(episode, cache):
    path = Path(cache) / 'videos' / (episode['id'] + '.mp4')
    with _lock(str(path)):
        if path.exists() and path.stat().st_size:
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.download')
        try:
            with requests.get(episode['video_url'], stream=True, timeout=120) as response:
                response.raise_for_status()
                with tmp.open('wb') as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        output.write(chunk)
            observed = float(subprocess.check_output(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', str(tmp)],
                text=True, timeout=30).strip())
            if abs(observed - episode['duration']) > .75:
                raise ValueError(f'Video duration mismatch for {episode["id"]}: {observed}')
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)
    return path


def _frames(video, start, end):
    with tempfile.TemporaryDirectory(prefix='shotwell-wgo-frames-') as folder:
        output = Path(folder) / '%05d.jpg'
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                        '-threads', '1', '-ss', str(start), '-i', str(video),
                        '-t', str(max(.1, end - start)),
                        '-vf', f'fps={FPS},scale={FRAME_WIDTH}:-2',
                        '-q:v', '5', str(output)], check=False, timeout=180)
        files = sorted(Path(folder).glob('*.jpg'))
        first_time = start
        if not files:
            first_time = max(0, start - .25)
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                            '-threads', '1', '-ss', str(first_time), '-i', str(video),
                            '-frames:v', '1', str(output)], check=True, timeout=60)
            files = sorted(Path(folder).glob('*.jpg'))
        if not files:
            raise ValueError(f'No frames at {start:.3f}-{end:.3f}')
        return [(first_time + i / FPS, Image.open(path).convert('RGB').copy())
                for i, path in enumerate(files)]


def contact_sheets(episode, start, end, cache):
    """Every frame is sampled at 8 fps; 4x2 sheets reduce API image count."""
    key = hashlib.sha256(f'{episode["id"]}:{start:.6f}:{end:.6f}:{FPS}:{FRAME_WIDTH}'.encode()).hexdigest()[:20]
    folder = Path(cache) / 'sheets' / episode['id'] / key
    marker = folder / '.complete'
    with _lock(str(folder)):
        if not marker.exists():
            folder.mkdir(parents=True, exist_ok=True)
            frames = _frames(video_path(episode, cache), start, end)
            for index in range(0, len(frames), TILES_PER_SHEET):
                chunk = frames[index:index + TILES_PER_SHEET]
                tile_height = chunk[0][1].height
                sheet = Image.new('RGB', (FRAME_WIDTH * 4, tile_height * 2), '#111111')
                draw = ImageDraw.Draw(sheet)
                for offset, (timestamp, frame) in enumerate(chunk):
                    x = (offset % 4) * FRAME_WIDTH
                    y = (offset // 4) * tile_height
                    sheet.paste(frame, (x, y))
                    draw.rectangle((x + 3, y + 3, x + 75, y + 19), fill='#101010')
                    draw.text((x + 6, y + 5), f'{timestamp:.2f}s', fill='white')
                sheet.save(folder / f'{index // TILES_PER_SHEET:04d}.jpg',
                           format='JPEG', quality=78, optimize=True)
            marker.touch()
        paths = sorted(folder.glob('*.jpg'))
    return [base64.b64encode(path.read_bytes()).decode() for path in paths]
