"""Command-line interface for generation and scoring."""

import argparse
import os
import shlex
from pathlib import Path

from .providers import MODELS
from .runner import references, run, select_episodes
from .scoring import score


def load_env(path):
    if path is None:
        return
    for line in Path(path).read_text().splitlines():
        if '=' not in line or line.lstrip().startswith('#'):
            continue
        key, value = line.split('=', 1)
        if key.strip().isidentifier():
            parsed = shlex.split(value, comments=True)
            if parsed:
                os.environ.setdefault(key.strip(), parsed[0])


def main():
    parser = argparse.ArgumentParser(prog='shotwell-wgo-bench')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('run', 'score'):
        command = commands.add_parser(name)
        command.add_argument('--model', choices=[*MODELS, 'all'], default='all')
        command.add_argument('--subset', default='completed',
                             help='completed, first5, all, or comma-separated episode IDs')
        command.add_argument('--run-dir', type=Path, default=Path('runs/first5-8fps'))
        command.add_argument('--workers', type=int, default=32 if name == 'run' else 24)
        command.add_argument('--env', type=Path)
    run_command = commands.choices['run']
    run_command.add_argument('--cache', type=Path,
                             default=Path.home() / '.cache/shotwell-wgo-bench')
    run_command.add_argument('--stage', choices=['both', 'segmentation', 'description'],
                             default='both')
    run_command.add_argument('--limit-tasks', type=int)
    run_command.add_argument('--retry-refusals', action='store_true',
                             help='Rerun policy-refused tasks with the same cached frames')
    args = parser.parse_args()
    load_env(args.env)
    models = list(MODELS) if args.model == 'all' else [args.model]
    episodes = select_episodes(references(), args.subset)
    if not episodes:
        parser.error('No episodes selected')
    if args.command == 'run':
        run(models, episodes, args.run_dir, args.cache,
            workers=args.workers, stage=args.stage, limit_tasks=args.limit_tasks,
            retry_refusals=args.retry_refusals)
    else:
        score(models, episodes, args.run_dir, workers=args.workers)


if __name__ == '__main__':
    main()
