"""Score WGO comparison outputs with the public joint semantic F1 protocol."""

import argparse
import json
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
import requests

from shotwell_bench.cli import load_env
from shotwell_bench.providers import _vertex_token

JUDGE = 'gemini-3.5-flash'
_write_lock = threading.Lock()


def load_rows(path, key):
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        result[row[key]] = row
    return result


def references(parquet):
    file = pq.ParquetFile(parquet)
    for index in range(file.num_row_groups):
        row = file.read_row_group(index, columns=['id', 'instruction', 'segments']).to_pylist()[0]
        yield row


def subset_of(episode_id):
    prefix = episode_id.split('_', 1)[0]
    return {'homer': 'HomER', 'robointer': 'DROID', 'galaxea': 'Galaxea'}[prefix]


def tiou(a, b):
    overlap = max(0, min(a['end_sec'], b['end_sec']) - max(a['start_sec'], b['start_sec']))
    union = max(a['end_sec'], b['end_sec']) - min(a['start_sec'], b['start_sec'])
    return overlap / union if union else 0


def snapped(predictions, gold):
    result = [dict(segment) for segment in predictions]
    if result:
        result[0]['start_sec'] = gold[0]['start_sec']
        result[-1]['end_sec'] = gold[-1]['end_sec']
    return result


def eligible_edges(predictions, gold):
    return [(i, j) for i, prediction in enumerate(predictions)
            for j, reference in enumerate(gold) if tiou(prediction, reference) >= .75]


def maximum_matches(edges, predictions_count, gold_count):
    neighbors = [[] for _ in range(predictions_count)]
    for i, j in edges:
        neighbors[i].append(j)
    gold_match = [-1] * gold_count

    def visit(i, seen):
        for j in neighbors[i]:
            if j in seen:
                continue
            seen.add(j)
            if gold_match[j] == -1 or visit(gold_match[j], seen):
                gold_match[j] = i
                return True
        return False

    for i in range(predictions_count):
        visit(i, set())
    return sum(index != -1 for index in gold_match)


def judge_caption(gold, predicted, instruction):
    prompt = f"""You are judging whether a predicted subtask label matches a gold subtask label.

Gold label: {gold}
Predicted label: {predicted}
Episode instruction: {instruction}

Accept if it describes the same manipulation event or world-state change, with the correct main action and manipulated object. Source, destination, direction, or spatial relation must be correct when central to the event. Synonyms and slight loss of detail are acceptable if the label remains useful.
Reject a wrong action, object, destination or direction, a different event, a label too vague to identify the subtask, or a hallucinated important object or action. Ignore grammar and timing. Return JSON with one boolean field named match."""
    payload = {'contents': [{'role': 'user', 'parts': [{'text': prompt}]}],
               'generationConfig': {'temperature': 0, 'maxOutputTokens': 8192,
                                    'responseMimeType': 'application/json',
                                    'responseSchema': {'type': 'OBJECT', 'properties': {
                                        'match': {'type': 'BOOLEAN'}}, 'required': ['match']}}}
    project = os.environ.get('GOOGLE_CLOUD_PROJECT', 'robot-data-qa')
    url = (f'https://aiplatform.googleapis.com/v1/projects/{project}/locations/global/'
           f'publishers/google/models/{JUDGE}:generateContent')
    for attempt in range(5):
        try:
            response = requests.post(url, json=payload,
                                     headers={'Authorization': 'Bearer ' + _vertex_token()},
                                     timeout=180)
            response.raise_for_status()
            body = response.json()
            answer = ''.join(part.get('text', '') for candidate in body['candidates']
                             for part in candidate['content']['parts'])
            result = json.loads(answer)
            if type(result.get('match')) is not bool:
                raise ValueError(f'Invalid judgment: {answer[:120]}')
            return result['match']
        except (requests.RequestException, KeyError, ValueError) as exc:
            if (isinstance(exc, requests.HTTPError) and exc.response is not None
                    and exc.response.status_code in (400, 401, 402, 403, 404)):
                raise RuntimeError(f'Judge request rejected: {exc.response.text[:300]}') from exc
            if attempt == 4:
                raise RuntimeError(f'Judge failed: {exc}') from exc
            time.sleep(min(45, 2 ** attempt * 3))


def score_model(model, episodes, run_dir, workers):
    segments = load_rows(run_dir / f'{model}-segments.jsonl', 'episode_id')
    captions = load_rows(run_dir / f'{model}-captions.jsonl', 'key')
    if len(segments) != len(episodes):
        raise ValueError(f'{model}: only {len(segments)}/{len(episodes)} segmentation outputs')
    expected_captions = sum(len(row['segments']) for row in segments.values())
    if len(captions) != expected_captions:
        raise ValueError(f'{model}: only {len(captions)}/{expected_captions} caption outputs')
    if any(row.get('recipe') != 'wgo-neighbor-seeded-v1' for row in captions.values()):
        raise ValueError(f'{model}: mixed caption input recipes')
    judgments_path = run_dir / f'{model}-judgments.jsonl'
    judgments = load_rows(judgments_path, 'key')
    tasks = []
    episode_edges = {}
    for episode in episodes:
        episode_id = episode['id']
        gold = episode['segments']
        predicted = snapped(segments[episode_id]['segments'], gold)
        edges = eligible_edges(predicted, gold)
        episode_edges[episode_id] = edges
        for i, j in edges:
            key = f'{episode_id}:{i}:{j}'
            if key in judgments:
                continue
            caption = captions[f'{episode_id}:{i}']['description']
            if not caption.strip():
                row = {'key': key, 'match': False}
                with _write_lock:
                    with judgments_path.open('a') as output:
                        output.write(json.dumps(row) + '\n')
                judgments[key] = row
            else:
                tasks.append((key, gold[j]['subtask'], caption, episode['instruction']))

    def work(task):
        key, reference, predicted, instruction = task
        verdict = judge_caption(reference, predicted, instruction)
        row = {'key': key, 'match': verdict}
        with _write_lock:
            with judgments_path.open('a') as output:
                output.write(json.dumps(row) + '\n')
        return row

    print(f'{model}: judging {len(tasks)} temporal matches', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            if index % 25 == 0 or index == len(futures):
                print(f'{model}: judged {index}/{len(futures)}', flush=True)
    judgments = load_rows(judgments_path, 'key')
    totals = defaultdict(lambda: {'predicted': 0, 'reference': 0, 'matched': 0,
                                  'segmentation_matched': 0})
    for episode in episodes:
        episode_id = episode['id']
        subset = subset_of(episode_id)
        gold = episode['segments']
        predicted = segments[episode_id]['segments']
        edges = episode_edges[episode_id]
        semantic = [(i, j) for i, j in edges if judgments[f'{episode_id}:{i}:{j}']['match']]
        item = totals[subset]
        item['predicted'] += len(predicted)
        item['reference'] += len(gold)
        item['matched'] += maximum_matches(semantic, len(predicted), len(gold))
        item['segmentation_matched'] += maximum_matches(edges, len(predicted), len(gold))
    for item in totals.values():
        denominator = item['predicted'] + item['reference']
        item['joint_f1'] = 2 * item['matched'] / denominator if denominator else 0
        item['segmentation_f1'] = 2 * item['segmentation_matched'] / denominator if denominator else 0
    overall = sum(totals[subset]['joint_f1'] for subset in ('HomER', 'DROID', 'Galaxea')) / 3
    return {'model': model, 'joint_f1': overall, 'subsets': dict(totals),
            'refused_segmentation_calls': sum(bool(row.get('refusal_reason'))
                                              for row in segments.values()),
            'refused_caption_calls': sum(bool(row.get('refusal_reason'))
                                         for row in captions.values()),
            'source': 'independent_replication', 'judge': JUDGE,
            'dataset': 'macrodata/WGO-Bench', 'episodes': len(episodes)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', default='sol,luna,argon,opus,sonnet,kimi31pro')
    parser.add_argument('--parquet', type=Path,
                        default=Path.home() / '.cache/shotwell-bench/wgo-annotations.parquet')
    parser.add_argument('--run-dir', type=Path, default=Path('benchmark-results/wgo-ranking-2026-10-30-v3'))
    parser.add_argument('--workers', type=int, default=8,
                        help='Concurrent judgments per model')
    parser.add_argument('--model-workers', type=int, default=6,
                        help='Models scored concurrently')
    parser.add_argument('--env', type=Path)
    args = parser.parse_args()
    load_env(args.env)
    episodes = list(references(args.parquet))
    scores = {}
    with ThreadPoolExecutor(max_workers=args.model_workers) as pool:
        futures = {pool.submit(score_model, model, episodes, args.run_dir, args.workers): model
                   for model in args.models.split(',')}
        for future in as_completed(futures):
            model = futures[future]
            scores[model] = future.result()
            print(f'{model}: joint semantic F1={scores[model]["joint_f1"]:.3f}', flush=True)
            (args.run_dir / 'scores.json').write_text(json.dumps(scores, indent=2) + '\n')


if __name__ == '__main__':
    main()
