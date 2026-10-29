#!/usr/bin/env bash
set -euo pipefail
python3 -m shotwell_bench run --model sol "$@"
