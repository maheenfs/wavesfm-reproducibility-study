#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SESSION_NAME="third_try_lwm_beam_challenge_$(date +%Y%m%d_%H%M%S)-$(hostname -s)"

python3 phase2_vivor4/scripts/prepare_clean_launch_root.py \
  --tasks lwm-beam-challenge \
  --modes lp ft2 lora \
  --seeds 0 1 2 \
  --session-name "$SESSION_NAME" \
  --device cuda \
  --num-workers 4 \
  --save-every 5
