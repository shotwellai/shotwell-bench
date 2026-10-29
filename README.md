# Shotwell-WGO-Bench

Shotwell-WGO-Bench measures how well a vision-language model (1) segments an egocentric video into actions and (2) describes actions inside fixed reference intervals. It builds on [WGO-Bench](https://huggingface.co/datasets/macrodata/WGO-Bench) with a detailed [annotation guide](https://docs.google.com/document/d/1DT8Nu9IEP5oZ7uSj0GgQoOuZyltjdQMlGYVQgc76qVI/edit) and references that accept boundary ranges and optional splits. See the [article](https://shotwell.ai/blog/shotwell-wgo-bench/) and [reference browser](https://shotwell.ai/blog/shotwell-wgo-bench/episodes/).

The repo contains **no videos**. `references/episodes.json` links each episode to a pinned, publicly hosted MP4 in [Nano1337/wgo-bench-videos](https://huggingface.co/datasets/Nano1337/wgo-bench-videos). The source dataset is [macrodata/WGO-Bench](https://huggingface.co/datasets/macrodata/WGO-Bench); its listed license is CC BY-NC-SA 4.0. The runner caches downloaded MP4s outside the repo. The current [Markdown guide](guide.md) matches the linked Google Doc in prose; the exact text guide used for the completed run is archived with its results.

## Reference status

The snapshot contains the 11 selected Job 690 episodes: `homer_1`, `homer_2`, `homer_3`, `homer_4`, `homer_5`, `homer_9`, `homer_33`, `homer_37`, `homer_40`, `homer_41`, and `homer_48`. All 11 now have a required detail in every golden segment (221 fixed segments, 325 required details). The `description_ready` flag is authoritative; an episode without it can be segmented but cannot receive an overall score. The reference is a dated export, not a live database query.

## Eleven-episode run

All 11 completed episodes were evaluated at 8 fps with the updated guide and description judge. The seven-model extension reused the original five model outputs and added Sonnet and Qwen on the same guide and reference snapshot. The exact guide text, run config, model outputs, judgments, and scores are in [the archived results](benchmark-results/all11-8fps-2026-10-29-plus2/). The Markdown guide adds formatting only, but changes its byte hash, so use a new run directory for future runs.

| Model | Segmentation F1 | Description F1 | Overall F1 |
| --- | ---: | ---: | ---: |
| GPT-6 Astra | 0.773 | 0.709 | **0.548** |
| GPT-6.1 Sol | 0.756 | 0.676 | 0.511 |
| Gemini 4 Argon | 0.694 | 0.672 | 0.466 |
| Claude Opus 5.5 | 0.548 | 0.590 | 0.323 |
| Claude Sonnet 5.5 | 0.520 | 0.489 | 0.254 |
| GPT-6 Luna | 0.474 | 0.432 | 0.204 |
| Qwen3-VL 235B Instruct | 0.339 | 0.319 | 0.108 |

## Five-episode pilot

The first five episodes were run at 8 fps with the [earlier guide revision](https://github.com/shotwellai/shotwell-wgo-bench/blob/ed6788bcf9e1e437556748c8579bec6dec45476f/guide.txt) and the earlier, stricter description judge. The current [guide](guide.md) adds a rule treating head-camera adjustment as IDLE, and the current judge accepts compatible specificity and slight generalizations. The first five golden references are unchanged, but the scores below have not been rerun under either revision and should not be compared directly with the new run. Scores are micro F1 for segmentation and descriptions; overall F1 is their product. [Full scores, raw model outputs, judge decisions, and run config](benchmark-results/first5-8fps/) are included for audit.

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
- `ANTHROPIC_KEY`: Claude Opus 5.5 and Sonnet 5.5 via Anthropic's API.
- `OPENROUTER_API_KEY`: Qwen3-VL-235B-A22B-Instruct and Qwen3.5-397B-A17B via OpenRouter.
- Google Cloud ADC or `gcloud auth login`: Gemini 4 Argon via Vertex AI. `GOOGLE_CLOUD_PROJECT` defaults to `robot-data-qa`; set it to a project with model access.

Never commit `.env` or run outputs with credentials. The first run downloads only the selected public MP4s to `~/.cache/shotwell-wgo-bench` (override with `--cache`).

## Run

All eight API models, first five completed episodes, with up to 32 concurrent calls:

```bash
python3 -m shotwell_bench run --model all --subset first5 \
  --workers 32 --env .env --run-dir runs/first5-8fps
python3 -m shotwell_bench score --model all --subset first5 \
  --workers 24 --env .env --run-dir runs/first5-8fps
```

Run one API model with the corresponding script in `scripts/`, including `run_sonnet.sh`, `run_qwen.sh`, and `run_qwen35.sh`. `scripts/run_all.sh` runs all eight API models; FLORA has a separate GPU runner below. `--subset completed` (default) selects every description-ready episode; with the current reference export, all 11 are ready, so it selects the same episodes as `--subset all`. A comma-separated list such as `--subset homer_1,homer_3` is also accepted. `--stage segmentation` or `--stage description` isolates a pass. Rerunning resumes from successful JSONL checkpoints and retries failed tasks.

`scripts/run_qwen35.sh` adds the open-weight Qwen3.5-397B-A17B model through OpenRouter. This adapter disables its default thinking mode so the provider returns a JSON answer within the per-task output limit; the model name and this setting should both be reported with its score. The older Qwen3-VL run remains available for comparison.

NVIDIA FLORA runs through its pinned native TempSeg and Captioner pipeline on Modal H100 workers. Install the optional `modal` package, configure a Modal profile with access to the `flora-eval` volume, then run:

```bash
modal run scripts/run_flora.py --run-dir runs/your-run \
  --guide path/to/the-run-guide.txt --subset completed
python3 -m shotwell_bench score --model flora --run-dir runs/your-run \
  --guide path/to/the-run-guide.txt --subset completed --env .env
```

The FLORA runner converts source video to 8 fps, but its released temporal model samples frames internally at **4 fps**. It accepts at most 2,000 instruction characters, so the adapter passes a concise guide summary rather than the full benchmark guide. These differences are recorded in `config.json` and must be disclosed beside any comparison with the API models' 8-fps contact-sheet results. FLORA's weights and inference code use NVIDIA's noncommercial research license; see the [official release](https://github.com/NVIDIA/Flora).

Results appear in `<run-dir>/scores.json`; per-model raw outputs and blinded judge decisions remain in JSONL files alongside it. `config.json` records exact model IDs, guide/reference hashes, frame settings, and episode IDs. Keep it with the run for reproducibility. The API candidates are `gpt-6-astra`, `gpt-6.1-sol`, `gpt-6-luna`, `gemini-4-argon`, `claude-opus-5-5`, `claude-sonnet-5-5`, `qwen/qwen3-vl-235b-a22b-instruct`, and `qwen/qwen3.5-397b-a17b`; NVIDIA FLORA uses the separate GPU adapter above. The archived five-episode pilot used Opus through OpenRouter; the current run uses Anthropic directly. Qwen3-VL's Thinking variant was unavailable through the configured providers, so the older run uses the available Instruct variant.

`--retry-refusals` reruns only policy-refused tasks with the same cached 8 fps frames, retaining the earlier refusal rows for audit. Repeated policy blocks remain recorded as refusals. Use a new run directory when changing the guide. To extend the archived eleven-episode run with another model, pass `--guide benchmark-results/all11-8fps-2026-10-29-plus2/guide.txt` so the exact guide hash matches; the root Markdown guide has the same prose but different bytes.

## Visual input and prompts

**Every API candidate receives frames sampled at 8 fps** and resized to 512 pixels wide. Eight consecutive timestamped frames are arranged in a 4-by-2 contact sheet; this reduces API image count without dropping sampled frames. The same cached sheets and full guide are passed to every API model. For unusually long intervals, the Qwen adapter stacks adjacent sheets vertically to stay within its provider's 32-image limit; every sampled frame remains present. The segmentation pass uses 10-second nonoverlapping scoring cores with one second of context on either side. It predicts action ends and captions without seeing reference boundaries. Only predicted ends inside the core are kept; the structural video start/end are never scored. The description pass gives the model a fixed golden interval and its frames, but no reference detail text. Each fixed interval returns one imperative description or `IDLE`. FLORA uses its native pipeline and the input differences documented above.

## WGO-prompt comparison examples

The article's "Astra on WGO" rows are a separate segmentation run, not copies of
the Shotwell-guided output. Reproduce them with:

```bash
python3 -m scripts.run_wgo_prompt_examples --env .env
```

This uses WGO-Bench's published completed-events prompt, 2-fps contact sheets
with 224-pixel tiles, and the source dataset's episode instructions. Its raw
segments and prompt are in
`benchmark-results/wgo-prompt-examples/astra-official.jsonl`. The article's
"Astra on Shotwell-WGO" rows instead come from the 8-fps guided run. The
comparison markers report segmentation only and apply each benchmark's own
boundary-matching rule, not the description judge.

## Scoring

**Segmentation.** Predictions are matched one-to-one to reviewed boundary windows (inclusive minimum/maximum) using the same minimum-cost assignment as Job 690's boundary scorer. Major boundaries contribute to recall; minor boundaries are optional valid splits and contribute to precision when matched. Unmatched predictions are false positives; missed major boundaries are false negatives. Start/end markers are structural, not scored. Predictions within 0.08 s of each other are deduplicated. Segmentation F1 is the micro F1 across selected episodes.

**Descriptions.** A blinded, text-only GPT-6 Astra judge sees a model's output plus that interval's required and optional reference details. Each distinct required action must be present with the correct core verb, object, and hand; bracketed alternatives are acceptable. The judge allows compatible extra specificity (for example, “water” for “liquid”) and slight generalizations that leave the core action clear. It still rejects contradictions, separate actions absent from the reference, and unsupported outcomes or intentions. Each matched required detail is a true positive, each missing required detail is a false negative, and each unsupported action/factual claim or format violation is a false positive. Optional details may be mentioned but need not be. The judge has no video access. Description F1 is the micro F1 over required details and unsupported claims. **Overall F1 = segmentation F1 × description F1.** The result JSON includes counts and per-episode diagnostics, not only rounded scores. Judgment checkpoints include a rubric version in their key, so changes to this policy trigger fresh judgments.

If a provider refuses a visual input under its content policy, the run records the refusal and scores that window as producing no annotation. It does not alter the 8 fps input to evade the block. The aggregate results report refused-task counts so model scores can be interpreted with that caveat.

This judge is itself a model and uses Astra, one of the candidates. Its decisions should be audited before making strong comparative claims. The reference can also be incomplete or ambiguous. These 11 selected episodes are not a population estimate for all of WGO-Bench. Contact sheets preserve 8 fps sampling but are not the same input format as a native video API.

## Maintainer reference export

The runner never connects to Shotwell's database. Maintainers can refresh the public JSON with `pip install -e '.[export]'` and `python scripts/export_references.py --env /path/to/private.env`. The export includes only visible episode IDs, timestamps, tolerances, golden/optional boundary status, and human decisions marked `required` or `optional`; it excludes model suggestions, review identities, and credentials. Review the diff before publishing an updated reference snapshot.

Run `python -m unittest discover -s tests -v` for the reference and scoring checks.
