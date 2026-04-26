#!/usr/bin/env bash
set -euo pipefail

cd "/home/maheenfs/wavesfm_vivor4_m2"

/home/maheenfs/wavesfm_vivor4_m2/.venv/bin/python \
  phase2_vivor4/scripts/run_tracker_supervised.py \
  --session-root \
  /local/data0/maheenfs/wavesfm_vivor4_m2/phase2_vivor4/runs/modulation_subset_50pct_20260424_ngwn06 \
  --label \
  modulation_subset_50pct_20260424_ngwn06 \
  --restart-delay-seconds \
  20 \
  --poll-seconds \
  15 \
  --stale-seconds \
  900 \
  --startup-grace-seconds \
  180 \
  --graceful-stop-seconds \
  20 \
  --max-restarts \
  50 \
  -- \
  --session-root \
  /local/data0/maheenfs/wavesfm_vivor4_m2/phase2_vivor4/runs/modulation_subset_50pct_20260424_ngwn06 \
  --tasks \
  rml \
  radcom \
  --modes \
  lp \
  ft2 \
  lora \
  --seeds \
  0 \
  1 \
  2 \
  --num-workers \
  4 \
  --save-every \
  5 \
  --train-subset-fraction \
  0.5
