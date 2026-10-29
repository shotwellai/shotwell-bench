"""Maintainer-only export from Job 690; benchmark runs never need database access."""

import argparse
import json
import os
import shlex
from datetime import datetime, timezone
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

IDS = (1, 2, 3, 4, 5, 9, 33, 37, 40, 41, 48)
VIDEO_BASE = ('https://huggingface.co/datasets/Nano1337/wgo-bench-videos/resolve/'
              '3f3d01542aad6ed66332001f077ff1b208700ae7/videos/')


def load_env(path):
    if not path:
        return
    for line in path.read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            key, value = line.split('=', 1)
            if key.strip().isidentifier():
                parts = shlex.split(value, comments=True)
                if parts:
                    os.environ.setdefault(key.strip(), parts[0])


def export(conn):
    ids = [f'wgo-bench-homer_{n}' for n in IDS]
    rows = conn.execute('''SELECT a.episode_id,a.annotations,e.video_duration
        FROM annotations a JOIN episodes e ON e.episode_id=a.episode_id
        WHERE a.job_id=690 AND a.username='consolidated' AND a.superseded_at IS NULL
          AND a.episode_id=ANY(%s)''', (ids,)).fetchall()
    if len(rows) != len(ids):
        raise ValueError(f'Expected {len(ids)} reference timelines, found {len(rows)}')
    detail_rows = conn.execute('''SELECT episode_id,start_boundary_id,end_boundary_id,
        start_time,end_time,items FROM job690_segment_details
        WHERE episode_id=ANY(%s)''', (ids,)).fetchall()
    by_key = {(r['episode_id'], r['start_boundary_id'], r['end_boundary_id']): r
              for r in detail_rows}
    episodes = []
    for row in sorted(rows, key=lambda r: int(r['episode_id'].split('_')[-1])):
        eid = row['episode_id']
        marks = sorted(row['annotations'], key=lambda mark: float(mark['timestamp']))
        boundaries = [{
            'id': mark['reference_id'],
            'time': float(mark['timestamp']),
            'minimum': float(mark.get('minimum', mark['timestamp'])),
            'maximum': float(mark.get('maximum', mark['timestamp'])),
            'type': mark.get('boundary_type', 'major'),
            'golden': (mark.get('boundary_type', 'major') == 'major'
                       or mark.get('golden') is True),
        } for mark in marks]
        if not boundaries or abs(boundaries[0]['time']) > .001:
            raise ValueError(f'{eid} is missing its start boundary')
        golden = [b for b in boundaries if b['golden']]
        segments = []
        for start, end in zip(golden, golden[1:]):
            detail = by_key.get((eid, start['id'], end['id']))
            if detail is None:
                raise ValueError(f'Missing detail row: {eid} {start["id"]}-{end["id"]}')
            decisions = {name: [item['text'] for item in detail['items']
                                if item.get('decision') == name]
                         for name in ('required', 'optional')}
            segments.append({'start_boundary_id': start['id'], 'end_boundary_id': end['id'],
                             'start': start['time'], 'end': end['time'], **decisions})
        complete = bool(segments) and all(s['required'] for s in segments)
        if int(eid.split('_')[-1]) <= 5 and not complete:
            raise ValueError(f'{eid} is not description-complete')
        episodes.append({'id': eid, 'source_id': eid.removeprefix('wgo-bench-'),
                         'video_url': VIDEO_BASE + eid.removeprefix('wgo-bench-') + '.mp4',
                         'duration': float(row['video_duration']),
                         'description_ready': complete,
                         'boundaries': boundaries, 'segments': segments})
    return {'schema_version': 1, 'source_job': 690,
            'source_dataset': 'macrodata/WGO-Bench',
            'video_mirror': 'Nano1337/wgo-bench-videos',
            'license': 'CC BY-NC-SA 4.0',
            'exported_at': datetime.now(timezone.utc).isoformat(), 'episodes': episodes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=Path)
    parser.add_argument('--output', type=Path, default=Path('references/episodes.json'))
    args = parser.parse_args()
    load_env(args.env)
    with psycopg.connect(os.environ['DATABASE_URL'], row_factory=dict_row) as conn:
        data = export(conn)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    print(f'Exported {len(data["episodes"])} episodes to {args.output}')


if __name__ == '__main__':
    main()
