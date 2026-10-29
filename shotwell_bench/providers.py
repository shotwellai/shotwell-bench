"""Small HTTP adapters for the five candidate models and the text-only judge."""

import json
import os
import subprocess
import threading
import time

import requests

MODELS = {
    'astra': ('openai', 'gpt-6-astra'),
    'sol': ('openai', 'gpt-6.1-sol'),
    'luna': ('openai', 'gpt-6-luna'),
    'argon': ('vertex', 'gemini-4-argon'),
    'opus': ('openrouter', 'anthropic/claude-opus-5.5'),
}
JUDGE_MODEL = 'gpt-6-astra'
_token_lock = threading.Lock()
_token = {'value': None, 'until': 0.0}
_provider_limits = {'openai': threading.BoundedSemaphore(16),
                    'vertex': threading.BoundedSemaphore(8),
                    'openrouter': threading.BoundedSemaphore(8)}


def _vertex_token():
    with _token_lock:
        if _token['value'] and time.monotonic() < _token['until']:
            return _token['value']
        try:
            r = requests.get(
                'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token',
                headers={'Metadata-Flavor': 'Google'}, timeout=2)
            r.raise_for_status()
            data = r.json()
            value, lifetime = data['access_token'], int(data.get('expires_in', 3300))
        except requests.RequestException:
            value = subprocess.check_output(['gcloud', 'auth', 'print-access-token'],
                                            text=True, timeout=30).strip()
            lifetime = 3300
        _token.update(value=value, until=time.monotonic() + max(60, lifetime - 120))
        return value


def _json_text(text):
    if isinstance(text, list):
        text = ''.join(part.get('text', '') for part in text if isinstance(part, dict))
    text = text.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
    return json.loads(text)


def _openai(model, prompt, images, schema, max_tokens):
    content = []
    for image in images:
        content.append({'type': 'input_image',
                        'image_url': 'data:image/jpeg;base64,' + image, 'detail': 'high'})
    if not content:
        content = [{'type': 'input_text', 'text': 'Evaluate the supplied text.'}]
    payload = {'model': model, 'instructions': prompt,
               'input': [{'role': 'user', 'content': content}],
               'reasoning': {'effort': 'medium'},
               'text': {'format': {'type': 'json_schema', 'name': 'bench_output',
                                   'schema': schema, 'strict': True}},
               'max_output_tokens': max_tokens}
    r = requests.post('https://api.openai.com/v1/responses', json=payload,
                      headers={'Authorization': 'Bearer ' + os.environ['OPENAI_KEY']},
                      timeout=900)
    r.raise_for_status()
    body = r.json()
    raw = body.get('output_text') or ''.join(
        part.get('text', '') for output in body.get('output', [])
        for part in output.get('content', []) if part.get('type') == 'output_text')
    return _json_text(raw), body.get('usage', {})


def _vertex(model, prompt, images, schema, max_tokens):
    parts = [{'text': prompt}]
    parts.extend({'inline_data': {'mime_type': 'image/jpeg', 'data': image}}
                 for image in images)
    def vertex_schema(node):
        if isinstance(node, dict):
            return {key: vertex_schema(value) for key, value in node.items()
                    if key != 'additionalProperties'}
        if isinstance(node, list):
            return [vertex_schema(value) for value in node]
        return node
    project = os.environ.get('GOOGLE_CLOUD_PROJECT', 'robot-data-qa')
    url = (f'https://aiplatform.googleapis.com/v1/projects/{project}/locations/global/'
           f'publishers/google/models/{model}:generateContent')
    r = requests.post(url, json={'contents': [{'role': 'user', 'parts': parts}],
                                 'generationConfig': {'responseMimeType': 'application/json',
                                                      'responseSchema': vertex_schema(schema),
                                                      'maxOutputTokens': max_tokens}},
                      headers={'Authorization': 'Bearer ' + _vertex_token()}, timeout=900)
    r.raise_for_status()
    body = r.json()
    raw = ''.join(part.get('text', '') for candidate in body.get('candidates', [])
                  for part in candidate.get('content', {}).get('parts', []))
    return _json_text(raw), body.get('usageMetadata', {})


def _openrouter(model, prompt, images, schema, max_tokens):
    content = [{'type': 'text', 'text': prompt + '\nReturn only valid JSON.'}]
    content.extend({'type': 'image_url',
                    'image_url': {'url': 'data:image/jpeg;base64,' + image}}
                   for image in images)
    r = requests.post('https://openrouter.ai/api/v1/chat/completions',
                      json={'model': model, 'messages': [{'role': 'user', 'content': content}],
                            'temperature': 0, 'max_tokens': max_tokens},
                      headers={'Authorization': 'Bearer ' + os.environ['OPENROUTER_API_KEY'],
                               'HTTP-Referer': 'https://shotwell.ai/',
                               'X-Title': 'Shotwell-WGO-Bench'}, timeout=900)
    r.raise_for_status()
    body = r.json()
    return _json_text(body['choices'][0]['message']['content']), body.get('usage', {})


def generate(model_name, prompt, images, schema, max_tokens=4000):
    provider, model = MODELS[model_name]
    required_key = {'openai': 'OPENAI_KEY', 'openrouter': 'OPENROUTER_API_KEY'}.get(provider)
    if required_key and not os.environ.get(required_key):
        raise ValueError(f'{required_key} is not configured')
    caller = {'openai': _openai, 'vertex': _vertex, 'openrouter': _openrouter}[provider]
    last = None
    with _provider_limits[provider]:
        for attempt in range(5):
            try:
                result, usage = caller(model, prompt, images, schema, max_tokens)
                return {'output': result, 'usage': usage, 'model_id': model}
            except (requests.RequestException, ValueError, KeyError,
                    json.JSONDecodeError, subprocess.SubprocessError) as exc:
                last = exc
                if attempt < 4:
                    time.sleep(min(90, 3 * 2 ** attempt))
    raise RuntimeError(f'{model_name} failed after five attempts: {last}')


def judge(prompt, schema):
    if not os.environ.get('OPENAI_KEY'):
        raise ValueError('OPENAI_KEY is not configured for the judge')
    with _provider_limits['openai']:
        for attempt in range(5):
            try:
                return _openai(JUDGE_MODEL, prompt, [], schema, 1800)[0]
            except (requests.RequestException, ValueError, KeyError,
                    json.JSONDecodeError):
                if attempt == 4:
                    raise
                time.sleep(min(90, 3 * 2 ** attempt))
