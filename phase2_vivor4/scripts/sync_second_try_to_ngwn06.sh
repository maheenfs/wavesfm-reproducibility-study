#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REMOTE="<lab-host>"
REMOTE_ROOT="/home/<user>/wavesfm_vivor4_m2"

DOCS=(
  "phase2_vivor4/README.md"
  "notes/01_project_overview.md"
  "notes/02_environment_setup.md"
  "notes/03_code_changes.md"
  "notes/04_reproduction_log.md"
  "notes/05_modulation_subset_experiments.md"
)

CODE=(
  "wavesfm/data.py"
  "wavesfm/dataset_classes/deepmimo.py"
  "wavesfm/preprocessing/preprocess_csi_sensing.py"
  "wavesfm/preprocessing/preprocess_deepmimo.py"
  "wavesfm/preprocessing/preprocess_lwm_beam_challenge.py"
  "wavesfm/run_finetune_all.py"
  "phase2_vivor4/scripts/benchmark_config.py"
  "phase2_vivor4/scripts/compare_with_official.py"
  "phase2_vivor4/scripts/export_official_results.py"
  "phase2_vivor4/scripts/run_all_tasks.py"
  "phase2_vivor4/scripts/wait_for_radcom_and_run_next.py"
  "phase2_vivor4/scripts/patch_live_dashboard_scroll.py"
  "phase2_vivor4/scripts/prepare_clean_launch_root.py"
  "phase2_vivor4/scripts/prepare_second_try_sensing_deepmimo.sh"
  "phase2_vivor4/scripts/prepare_third_try_lwm_beam_challenge.sh"
  "phase2_vivor4/scripts/storage_offload.py"
  "phase2_vivor4/scripts/dashboard_tunnel_watch.sh"
  "phase2_vivor4/scripts/sync_second_try_to_ngwn06.sh"
)

CACHES=(
  "datasets_h5/has.h5"
  "datasets_h5/deepmimo.h5"
  "datasets_h5/lwm-beam-challenge.h5"
)

RAW_DEEPMIMO_BEAM_TRAIN=(
  "datasets_raw/deepmimo/lwm_beam_labels/beam_prediction_challenge/train/bp_data_train.p"
  "datasets_raw/deepmimo/lwm_beam_labels/beam_prediction_challenge/train/bp_label_train.p"
  "datasets_raw/deepmimo/lwm_beam_labels/beam_prediction_challenge/train/bp_label_train.p.official"
)

RAW_DEEPMIMO_BEAM_TEST=(
  "datasets_raw/deepmimo/lwm_beam_labels/beam_prediction_challenge/test/bp_data_test.p"
)

cd "$ROOT"

echo "[1/7] Sync docs"
/opt/homebrew/bin/rsync -avhR "${DOCS[@]}" "${REMOTE}:${REMOTE_ROOT}/"

echo "[2/7] Sync code and helper scripts"
/opt/homebrew/bin/rsync -avhR "${CODE[@]}" "${REMOTE}:${REMOTE_ROOT}/"

echo "[3/7] Reconcile remote storage offload layout"
ssh "$REMOTE" "cd ${REMOTE_ROOT} && python3 phase2_vivor4/scripts/storage_offload.py apply --quiet"

echo "[4/7] Verify remote storage protocol"
ssh "$REMOTE" "cd ${REMOTE_ROOT} && python3 - <<'PY'
from pathlib import Path
import sys

root = Path('.').resolve()
offload_targets = [
    Path('datasets_raw'),
    Path('datasets_h5'),
    Path('_transfer_quarantine'),
    Path('phase2_vivor4/automation_logs'),
    Path('phase2_vivor4/comparisons'),
    Path('phase2_vivor4/local_results'),
    Path('phase2_vivor4/plots'),
    Path('phase2_vivor4/runs'),
]
home_targets = [
    Path('notes'),
    Path('phase2_vivor4/README.md'),
    Path('phase2_vivor4/scripts'),
]

errors = []
for rel in offload_targets:
    path = root / rel
    if not path.exists():
        errors.append(f'missing offload target: {rel}')
    elif not path.is_symlink():
        errors.append(f'offload target is not a symlink: {rel}')
    else:
        print(f'offload {rel} -> {path.resolve()}')

for rel in home_targets:
    path = root / rel
    if not path.exists():
        print(f'home {rel} missing before sync; will be created if included in sync set')
    elif path.is_symlink():
        errors.append(f'canonical home path unexpectedly symlinked: {rel}')
    else:
        print(f'home {rel}')

if errors:
    print('\\n'.join(errors), file=sys.stderr)
    sys.exit(1)
PY"

echo "[5/7] Sync official DeepMIMO beam challenge raw artifacts into offloaded datasets_raw"
ssh "$REMOTE" "mkdir -p ${REMOTE_ROOT}/datasets_raw/deepmimo/lwm_beam_labels/beam_prediction_challenge/train ${REMOTE_ROOT}/datasets_raw/deepmimo/lwm_beam_labels/beam_prediction_challenge/test"
/opt/homebrew/bin/rsync -avh "${RAW_DEEPMIMO_BEAM_TRAIN[@]}" "${REMOTE}:${REMOTE_ROOT}/datasets_raw/deepmimo/lwm_beam_labels/beam_prediction_challenge/train/"
/opt/homebrew/bin/rsync -avh "${RAW_DEEPMIMO_BEAM_TEST[@]}" "${REMOTE}:${REMOTE_ROOT}/datasets_raw/deepmimo/lwm_beam_labels/beam_prediction_challenge/test/"

echo "[6/7] Sync rebuilt caches into offloaded datasets_h5"
/opt/homebrew/bin/rsync -avh "${CACHES[@]}" "${REMOTE}:${REMOTE_ROOT}/datasets_h5/"

echo "[7/7] Verify remote cache metadata, raw artifacts, and live paths"
ssh "$REMOTE" "cd ${REMOTE_ROOT} && ./.venv/bin/python - <<'PY'
import pickle
import h5py
from pathlib import Path

for rel in [
    Path('phase2_vivor4/README.md'),
    Path('phase2_vivor4/scripts/wait_for_radcom_and_run_next.py'),
    Path('phase2_vivor4/scripts/patch_live_dashboard_scroll.py'),
    Path('phase2_vivor4/scripts/prepare_clean_launch_root.py'),
    Path('phase2_vivor4/scripts/prepare_second_try_sensing_deepmimo.sh'),
    Path('phase2_vivor4/scripts/prepare_third_try_lwm_beam_challenge.sh'),
    Path('phase2_vivor4/scripts/storage_offload.py'),
    Path('phase2_vivor4/scripts/sync_second_try_to_ngwn06.sh'),
]:
    print(rel, 'exists=', rel.exists())

print('datasets_raw ->', Path('datasets_raw').resolve())
print('datasets_h5 ->', Path('datasets_h5').resolve())
print('phase2_vivor4/runs ->', Path('phase2_vivor4/runs').resolve())
for path in [
    'datasets_raw/deepmimo/lwm_beam_labels/beam_prediction_challenge/train/bp_data_train.p',
    'datasets_raw/deepmimo/lwm_beam_labels/beam_prediction_challenge/train/bp_label_train.p',
    'datasets_raw/deepmimo/lwm_beam_labels/beam_prediction_challenge/test/bp_data_test.p',
]:
    p = Path(path)
    print(path, 'size=', p.stat().st_size if p.exists() else 'missing')
    if p.exists():
        with p.open('rb') as f:
            obj = pickle.load(f)
        print(' shape=', getattr(obj, 'shape', None), 'dtype=', getattr(obj, 'dtype', None))
for path in ['datasets_h5/has.h5', 'datasets_h5/deepmimo.h5']:
    print(path)
    with h5py.File(path, 'r') as h5:
        print(' version', h5.attrs.get('version'))
        print(' input_source', h5.attrs.get('input_source'))
        if 'source_split' in h5:
            print(' source_split', True, 'source_splits', h5.attrs.get('source_splits'))
        for b in [16, 32, 64]:
            eff = h5.attrs.get(f'effective_n_beams_{b}', None)
            miss = h5.attrs.get(f'missing_beams_{b}', None)
            if eff is not None or miss is not None:
                print(' ', b, eff, miss)
path = 'datasets_h5/lwm-beam-challenge.h5'
print(path)
with h5py.File(path, 'r') as h5:
    print(' version', h5.attrs.get('version'))
    print(' input_source', h5.attrs.get('input_source'))
    print(' sample', h5['sample'].shape, h5['sample'].dtype)
    print(' label', h5['label'].shape, h5['label'].dtype)
    if 'public_test_sample' in h5:
        print(' public_test_sample', h5['public_test_sample'].shape, h5['public_test_sample'].dtype)
PY"

echo "Sync complete."
