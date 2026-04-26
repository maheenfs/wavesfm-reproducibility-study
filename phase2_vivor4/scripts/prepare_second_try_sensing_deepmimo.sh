#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HOST_SHORT="$(hostname -s 2>/dev/null || echo host)"
STAMP="$(date +%Y%m%d_%H%M%S)"
SESSION_NAME="second_try_sensing_deepmimo_${STAMP}-${HOST_SHORT}"

cd "$ROOT"

python3 phase2_vivor4/scripts/prepare_clean_launch_root.py \
  --device cuda \
  --num-workers 4 \
  --tasks sensing deepmimo-los deepmimo-beam \
  --session-name "$SESSION_NAME"

echo
echo "Prepared second-try session: $SESSION_NAME"
echo "Existing first-pass sessions remain in phase2_vivor4/runs/; this helper only adds a new named session."
echo "Next:"
echo "  source .venv/bin/activate"
echo "  export WAVESFM_FORCE_DEVICE=cuda"
echo "  python3 phase2_vivor4/scripts/preflight_check.py --session-root phase2_vivor4/runs/$SESSION_NAME --tasks sensing deepmimo-los deepmimo-beam --modes lp ft2 lora --seeds 0 1 2 --radcom-cache datasets_h5/radcom.h5"
echo "  python3 phase2_vivor4/scripts/wait_for_radcom_and_run_next.py --session-root phase2_vivor4/runs/$SESSION_NAME --tasks sensing deepmimo-los deepmimo-beam --modes lp ft2 lora --seeds 0 1 2 --radcom-cache datasets_h5/radcom.h5 --num-workers 4 --save-every 5"
