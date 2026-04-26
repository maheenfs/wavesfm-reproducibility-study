#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_modulation_subset_fresh.sh <subset-label> <subset-fraction> <task> [task ...] [-- extra tracker args]

Examples:
  run_modulation_subset_fresh.sh 1pct 0.01 rml
  run_modulation_subset_fresh.sh 5pct 0.05 radcom
  run_modulation_subset_fresh.sh 10pct 0.10 rml radcom
  run_modulation_subset_fresh.sh 5pct 0.05 rml -- --modes lora --seeds 0

Behavior:
  - creates a fresh timestamped session root
  - defaults to modes: lp ft2 lora
  - defaults to seeds: 0 1 2
  - defaults to num-workers=4 and save-every=5
  - appends any extra tracker args after '--'
EOF
}

if [[ $# -lt 3 ]]; then
  usage
  exit 1
fi

subset_label="$1"
subset_fraction="$2"
shift 2

tasks=()
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--" ]]; then
    shift
    break
  fi
  tasks+=("$1")
  shift
done

extra_args=("$@")

if [[ ${#tasks[@]} -eq 0 ]]; then
  usage
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../../.." && pwd)"
project_name="$(basename "$repo_root")"
user_name="$(whoami)"
host_name="$(hostname -s | tr ' ' '-')"

preferred_runs_root="/local/data0/${user_name}/${project_name}/phase2_vivor4/runs"
if [[ -d "$preferred_runs_root" ]]; then
  runs_root="$preferred_runs_root"
else
  runs_root="$repo_root/phase2_vivor4/runs"
fi

mkdir -p "$runs_root"

python_bin="$repo_root/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  python_bin="python3"
fi

task_slug="$(printf '%s_' "${tasks[@]}")"
task_slug="${task_slug%_}"
stamp="$(date -u +%Y%m%d_%H%M%S)"
session_label="modulation_subset_${task_slug}_${subset_label}_${stamp}_${host_name}"
session_root="$runs_root/$session_label"

tracker_args=(
  --session-root "$session_root"
  --tasks "${tasks[@]}"
  --modes lp ft2 lora
  --seeds 0 1 2
  --num-workers 4
  --save-every 5
  --train-subset-fraction "$subset_fraction"
)

if [[ ${#extra_args[@]} -gt 0 ]]; then
  tracker_args+=("${extra_args[@]}")
fi

cmd=(
  "$python_bin"
  "$repo_root/phase2_vivor4/scripts/run_tracker_supervised.py"
  --session-root "$session_root"
  --label "$session_label"
  --restart-delay-seconds 20
  --poll-seconds 15
  --stale-seconds 900
  --startup-grace-seconds 180
  --graceful-stop-seconds 20
  --max-restarts 50
  --
  "${tracker_args[@]}"
)

echo "[info] repo_root=$repo_root"
echo "[info] runs_root=$runs_root"
echo "[info] session_root=$session_root"
echo "[info] tasks=${tasks[*]}"
echo "[info] subset_label=$subset_label"
echo "[info] subset_fraction=$subset_fraction"
echo "[info] command=${cmd[*]}"

cd "$repo_root"
"${cmd[@]}"
