# Shotwell-WGO-Bench

Shotwell-WGO-Bench measures how well a vision-language model (1) segments an egocentric video into actions and (2) describes actions inside fixed reference intervals. It builds on [WGO-Bench](https://huggingface.co/datasets/macrodata/WGO-Bench) with a detailed [annotation guide](https://docs.google.com/document/d/1DT8Nu9IEP5oZ7uSj0GgQoOuZyltjdQMlGYVQgc76qVI/edit) and references that accept boundary ranges and optional splits. See the [article](https://shotwell.ai/blog/shotwell-wgo-bench/) and [reference browser](https://shotwell.ai/blog/shotwell-wgo-bench/episodes/).

The repo contains **no videos**. `references/episodes.json` links each episode to a pinned, publicly hosted MP4 in [Nano1337/wgo-bench-videos](https://huggingface.co/datasets/Nano1337/wgo-bench-videos). The source dataset is [macrodata/WGO-Bench](https://huggingface.co/datasets/macrodata/WGO-Bench); its listed license is CC BY-NC-SA 4.0. The runner caches downloaded MP4s outside the repo. `guide.txt` is a snapshot of the linked guide used by this run.

## Reference status

The snapshot contains the 11 visible Job 690 episodes: `homer_1`, `homer_2`, `homer_3`, `homer_4`, `homer_5`, `homer_9`, `homer_33`, `homer_37`, `homer_40`, `homer_41`, and `homer_48`. All have boundary references. **Only the first five currently have a required detail in every golden segment** (104 fixed segments, 145 required details). The `description_ready` flag is authoritative; an episode without it can be segmented but cannot receive an overall score. The reference is a dated export, not a live database query.

## Five-episode pilot

The first five episodes were run at 8 fps with the guide in this repo. Scores below are micro F1 for segmentation and descriptions; overall F1 is their product. [Full scores, raw model outputs, judge decisions, and run config](benchmark-results/first5-8fps/) are included for audit.

| Model | Segmentation F1 | Description F1 | Overall F1 | Refused windows |
| --- | ---: | ---: | ---: | ---: |
| GPT-6 Astra | 0.702 | 0.629 | **0.442** | 0 |
| GPT-6.1 Sol | 0.708 | 0.598 | 0.423 | 0 |
| Gemini 4 Argon | 0.584 | 0.609 | 0.355 | 2 |
| Claude Opus 5.5 | 0.542 | 0.495 | 0.268 | 0 |
| GPT-6 Luna | 0.491 | 0.362 | 0.178 | 0 |

Argon refused two image windows under its provider's content policy, including on an identical-input retry. They count as missing annotations; the score should be read with that caveat. This is a small pilot, not a full-benchmark estimate. The description judge is a model and merits human audit.

## Install

Python 3.11+, `ffmpeg`, and `ffprobe` are required. Install with `python3 -m venv .venv && .venv/bin/pip install -e .` and activate the environment. Copy `.env.example` to `.env` and fill in your own credentials, or set environment variables directly:

- `OPENAI_KEY`: GPT-6 Astra, GPT-6.1 Sol, GPT-6 Luna, and the text-only judge.
- `OPENROUTER_API_KEY`: Claude Opus 5.5.
- Google Cloud ADC or `gcloud auth login`: Gemini 4 Argon via Vertex AI. `GOOGLE_CLOUD_PROJECT` defaults to `robot-data-qa`; set it to a project with model access.

Never commit `.env` or run outputs with credentials. The first run downloads only the selected public MP4s to `~/.cache/shotwell-wgo-bench` (override with `--cache`).

## Run

All five models, first five completed episodes, with up to 32 concurrent calls:

```bash
python3 -m shotwell_bench run --model all --subset first5 \
  --workers 32 --env .env --run-dir runs/first5-8fps
python3 -m shotwell_bench score --model all --subset first5 \
  --workers 24 --env .env --run-dir runs/first5-8fps
```

Run one model with `scripts/run_astra.sh`, `run_sol.sh`, `run_argon.sh`, `run_opus.sh`, or `run_luna.sh`, passing the same options. `scripts/run_all.sh` runs all five. `--subset completed` (default) selects every description-ready episode; `--subset all` generates segmentation for all 11 and descriptions for ready episodes, but `score` requires a description-complete selection. A comma-separated list such as `--subset homer_1,homer_3` is also accepted. `--stage segmentation` or `--stage description` isolates a pass. Rerunning resumes from successful JSONL checkpoints and retries failed tasks.

Results appear in `<run-dir>/scores.json`; per-model raw outputs and blinded judge decisions remain in JSONL files alongside it. `config.json` records exact model IDs, guide/reference hashes, frame settings, and episode IDs. Keep it with the run for reproducibility. The candidate models are `gpt-6-astra`, `gpt-6.1-sol`, `gpt-6-luna`, `google/gemini-4-argon`, and `anthropic/claude-opus-5.5` (Opus via OpenRouter).

`--retry-refusals` reruns only policy-refused tasks with the same cached 8 fps frames, retaining the earlier refusal rows for audit. Repeated policy blocks remain recorded as refusals.

## Visual input and prompts

**Every candidate model receives frames sampled at 8 fps** and resized to 512 pixels wide. Eight consecutive timestamped frames are arranged in a 4-by-2 contact sheet; this reduces API image count without dropping sampled frames. The same cached sheets and full guide are passed to every model. The segmentation pass uses 10-second nonoverlapping scoring cores with one second of context on either side. It predicts action ends and captions without seeing reference boundaries. Only predicted ends inside the core are kept; the structural video start/end are never scored. The description pass gives the model a fixed golden interval and its frames, but no reference detail text. Each fixed interval returns one imperative description or `IDLE`.

## Scoring

**Segmentation.** Predictions are matched one-to-one to reviewed boundary windows (inclusive minimum/maximum) using the same minimum-cost assignment as Job 690's boundary scorer. Major boundaries contribute to recall; minor boundaries are optional valid splits and contribute to precision when matched. Unmatched predictions are false positives; missed major boundaries are false negatives. Start/end markers are structural, not scored. Predictions within 0.08 s of each other are deduplicated. Segmentation F1 is the micro F1 across selected episodes.

**Descriptions.** A blinded, text-only GPT-6 Astra judge sees a model's output plus that interval's required and optional reference details. It checks semantic equivalence of the action, object, hand, tool, and destination; bracketed alternatives are acceptable. Each fully matched required detail is a true positive, each missing required detail is a false negative, and each unsupported action/factual claim or format violation is a false positive. Optional details may be mentioned but need not be. The judge has no video access and is instructed not to infer events outside the reference. Description F1 is the micro F1 over required details and unsupported claims. **Overall F1 = segmentation F1 × description F1.** The result JSON includes counts and per-episode diagnostics, not only rounded scores.

If a provider refuses a visual input under its content policy, the run records the refusal and scores that window as producing no annotation. It does not alter the 8 fps input to evade the block. The aggregate results report refused-task counts so model scores can be interpreted with that caveat.

This judge is itself a model and uses Astra, one of the candidates. Its decisions should be audited before making strong comparative claims. The reference can also be incomplete or ambiguous. The current five-episode evaluation is a pilot, not a population estimate; it should not be presented as a result across all of WGO-Bench. Contact sheets preserve 8 fps sampling but are not the same input format as a native video API.

## Maintainer reference export

The runner never connects to Shotwell's database. Maintainers can refresh the public JSON with `pip install -e '.[export]'` and `python scripts/export_references.py --env /path/to/private.env`. The export includes only visible episode IDs, timestamps, tolerances, golden/optional boundary status, and human decisions marked `required` or `optional`; it excludes model suggestions, review identities, and credentials. Review the diff before publishing an updated reference snapshot.

Run `python -m unittest discover -s tests -v` for the reference and scoring checks.
