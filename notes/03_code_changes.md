# Code Changes

This note documents every modification made to the original WavesFM codebase
(`wavesfm/`) and describes the local benchmark harness added alongside it
(`phase2_vivor4/`). Changes are categorized by their impact: bug fixes required
to run the code, portability adaptations for our hardware, and benchmark protocol
changes that may affect results.

---

## Changes Inside `wavesfm/` (Original Codebase)

### Bug Fixes (required for execution)

**1. `run_finetune_all.py` -- broken import**
- Removed `from run_finetune_all_jepa import USE_CONDITIONAL_LN`, which referenced
  a module that does not exist in the release. The script crashed at import time
  without this fix.

**2. `preprocessing/preprocess_uwb_loc.py` -- non-existent CLI argument**
- Removed the line `root = Path(args.root).expanduser()`. The `--root` argument
  does not exist in the script's argument parser, causing UWB indoor preprocessing
  to crash.

### Portability Adaptations (no effect on benchmark results)

**3. `main_finetune.py` -- device selection**
- Changed default device from hard-coded `cuda` to automatic selection:
  `cuda` if available, then `mps` (Mac GPU), then `cpu`.
- AMP `GradScaler` is now enabled only on CUDA (it is not supported on MPS).
- Pin memory is CUDA-only; persistent workers and prefetching enabled when
  `num_workers > 0`.

**4. `main_finetune.py` and `utils.py` -- checkpoint and restart hardening**
- Checkpoint writes are now atomic (write to temp file, then rename) to prevent
  corruption if a run is interrupted mid-save.
- Checkpoints now store best-key metadata, last epoch's train/val stats, and
  full RNG state (PyTorch, NumPy, Python random).
- Resume restores RNG state and the DataLoader shuffle generator so restarted
  runs track the pre-crash run more closely.
- `set_seed` now also seeds NumPy when available.
- JSONL logger uses a single fsynced append per epoch (prevents partial writes
  after crashes).

**5. `requirements.txt` -- added matplotlib**
- Added `matplotlib==3.10.7` for local plot generation. Not a benchmark change.

### Benchmark Protocol Changes (may affect results)

**6. `main_finetune.py` and `run_finetune_all.py` -- DeepMIMO beam default**
- Changed the default `--deepmimo-n-beams` from 16 to 64 to match the public
  WavesFM DeepMIMO setup. This changes which beam labels are generated when the
  argument is not explicitly passed.

**7. `preprocessing/preprocess_deepmimo.py` -- beam codebook fix**
- Fixed the beam sweep from an inclusive `[-90, +90]` interval to a half-open
  `[-90, 90)` interval. The original code duplicated the endpoint beam and left
  the highest beam class (class 63) unreachable. This is the root cause of the
  first-pass 63-class cache problem.
- Added `--data-pickle` as an alternate input path for pre-generated DeepMIMO
  scenario data, with pickle structure validation.
- Added cache attributes recording input source (folder vs pickle), missing beam
  classes, and effective beam-count coverage per codebook size.
- Class weights now always match the selected codebook size (e.g., 64 weights for
  64 beams) rather than being sized to the observed max label.

**8. `data.py` -- beam head sizing and new task**
- DeepMIMO beam task output size now follows the selected codebook (64) rather
  than the observed effective label count. This ensures the model trains a proper
  64-way classifier even if not all classes appear in a particular cache.
- Added a separate `lwm-beam-challenge` task ID for the third-run hypothesis
  experiment, keeping it distinct from the official `deepmimo-beam` task.

**9. `dataset_classes/deepmimo.py` -- cache validation**
- Pickle-sourced caches with malformed beam coverage now fail fast instead of
  silently training on incomplete labels.
- Official-folder caches with unresolved coverage gaps emit explicit warnings.

**10. `preprocessing/preprocess_csi_sensing.py` -- provenance tracking**
- When the raw data contains `train_amp` and `test_amp` directories, the cache
  now preserves source-relative filenames and records a `source_split` dataset.
  This makes it auditable which samples came from which original split.
- Cache metadata version bumped from v1 to v2. No labels or tensors are changed.

**11. `preprocessing/preprocess_radcom.py` -- single-pass refactor**
- Removed the separate first pass that re-read the source file just to compute
  mean/std. The script now reads raw samples once, accumulates statistics during
  that write pass, and normalizes in place. Same output schema, less I/O.

**12. `preprocessing/preprocess_lwm_beam_challenge.py` -- new preprocessor**
- New local script that converts official LWM beam-challenge pickle files
  (`bp_data_train.p`, `bp_label_train.p`) into `datasets_h5/lwm-beam-challenge.h5`.
  Used only for the third-run hypothesis experiment.

**13. `run_finetune_all.py` -- sensing stratified split**
- Added `sensing` to the stratified-split task set for the second-try rerun.

### Experimental-Only Changes (not part of the reproduction benchmark)

**14. `main_finetune.py` and `data.py` -- train-subset controls for follow-up studies**
- Added `--train-subset-fraction` and `--train-subset-size` to support
  reduced-data fine-tuning experiments.
- The subset is applied **after** the normal train/validation split so the
  validation set remains unchanged.
- For classification tasks, the reduced training subset is sampled
  stratified-by-class to avoid dropping classes accidentally at very small
  fractions.
- These flags were added specifically for post-reproduction modulation-task
  experiments (`rml`, `radcom`) requested after the main benchmark had already
  been run and analyzed.

**15. `phase2_vivor4/scripts/benchmark_config.py`, `run_all_tasks.py`, and `wait_for_radcom_and_run_next.py` -- session-scoped subset experiment wiring**
- Added pass-through support so the new train-subset flags can be launched from
  the batch runner and the managed session tracker.
- This allows small-subset experiments to be stored in their own dedicated
  session roots under `phase2_vivor4/runs/<session>/` instead of contaminating
  the main reproduction outputs.
- These changes are operational support for follow-up experiments, not part of
  the original reproduction protocol.

**16. `phase2_vivor4/scripts/wait_for_radcom_and_run_next.py`, `plot_local_detailed_eval.py`, `summarize_local_results.py`, and `compare_with_official.py` -- experiment-aware dashboard and manifests**
- The session dashboard now shows whether a run is a full-data session or a
  train-subset study, which subset control is being used, and the exact
  session-scoped output roots for runs, plots, summaries, and comparisons.
- Detailed-eval plot headers and the plot manifest now record the reduced-data
  setup (subset label, train/eval split sizes, and sampling policy) so subset
  runs are visually distinct from the original full-data reproduction plots.
- Summary and comparison stages now write explicit manifest files containing
  experiment metadata, keeping these follow-up study artifacts self-contained
  under their session root.

---

## Local Benchmark Harness (`phase2_vivor4/`)

Everything under `phase2_vivor4/` is local tooling built around the original
codebase. It is **not** part of the original WavesFM release and should not be
reported as modifications to the upstream method. Its purpose is operational:

### Scripts (`phase2_vivor4/scripts/`)

| Script | Purpose |
|--------|---------|
| `benchmark_config.py` | Task specs: batch sizes, epochs, label smoothing, per-task flags |
| `run_all_tasks.py` | Queue runner: iterates tasks x modes x seeds |
| `preprocess_all_tasks.py` | Batch preprocessing through one config layer |
| `export_official_results.py` | Exports official website numbers to local JSON |
| `compare_with_official.py` | Generates local-vs-official comparison JSONs |
| `summarize_local_results.py` | Aggregates per-seed results into per-task summaries |
| `plot_local_detailed_eval.py` | Generates confusion matrices and error density plots |
| `prepare_clean_launch_root.py` | Creates clean session directories for new runs |
| `prepare_second_try_*.sh` | Session-specific launch scripts |
| `prepare_modulation_subset_study.py` | Scaffolds the rml/radcom reduced-data study under `phase2_vivor4/experiments/` |
| `wait_for_radcom_and_run_next.py` | Serial queue with GPU safety guards |
| `storage_offload.py` | Manages large-file offloading to lab storage |
| `sync_second_try_to_ngwn06.sh` | Mac-to-lab sync for specific sessions |
| `run_tracker_supervised.py` | Supervised tracker with thermal/pressure guards |

### Official Results (`phase2_vivor4/official_results/`)

Baseline numbers from the WavesFM detailed evaluation page, stored as JSON.
Snapshot date: 2026-01-23. These are the comparison targets for all local runs.

### Run Sessions (`phase2_vivor4/runs/`)

Each session is self-contained with its own results, comparisons, plots, and
manifest. See `04_reproduction_log.md` for session details and verdicts.

### Follow-up study scaffold (`phase2_vivor4/experiments/`)

`prepare_modulation_subset_study.py` writes a study root under
`phase2_vivor4/experiments/modulation_subset_study_<date>/` containing:

- `commands/launch_*.sh` -- one launcher per subset level (1, 3, 5, 10, 50%)
- `tables/table_a..table_f.csv` -- prefilled descriptor tables (datasets,
  modes, run config, results, comparison, runtime)
- `results/full_data_baselines.csv` and `planned_sessions.csv`
- `study_manifest.json` -- machine-readable index

This is operational scaffolding only. All actual run outputs land under
`phase2_vivor4/runs/modulation_subset_<N>pct_*/`.

---

## Report and Presentation Tooling (`report/`)

| Script | Purpose |
|--------|---------|
| `create_architecture_diagram.py` | Renders 5 architecture PNGs (pretraining, transformer block, fine-tuning flow, three modes, LoRA deep dive) |
| `create_architecture_pptx.py` | Builds the editable architecture PPTX from native shapes |
| `create_presentation.py` | Builds the main 22-slide reproduction PPTX |
| `create_subset_presentation.py` | Builds the dedicated modulation-subset PPTX (datasets, preprocessing, gallery, results, runtime) |
| `create_task_visualizations.py` | Generates confusion matrices and error-density plots used in the report |
| `extract_presentation_evidence.py` | Pulls real data points (sample tensors, label distributions) for the slide gallery |

Generated assets live under `report/figures/`. Subset-study assets live under
`report/figures/datasets/` (per-modulation IQ plots, constellations, amplitude
envelopes, before/after normalisation, task input/output diagrams).

The PPTX files in `report/` are ALWAYS regenerated from the Python scripts
above. Manual edits in PowerPoint will be overwritten on next regeneration.

---

## Summary for Reporting

When writing about this work, the following distinctions matter:

- **Bug fixes** (broken import, UWB CLI arg): required for execution, not
  scientific choices. Disclose as preprocessing/runtime fixes.
- **Portability changes** (device selection, AMP, checkpoint hardening): do not
  affect benchmark definitions. Disclose as execution adaptations.
- **Beam codebook fix** (16→64 default, half-open sweep): directly affects
  DeepMIMO results. Must be disclosed prominently.
- **Sensing provenance tracking**: does not change labels or tensors, but changes
  cache schema. Disclose if discussing split protocols.
- **Harness scripts**: are local operational tooling, not method modifications.
  Mention only as part of the reproduction infrastructure description.
- **Train-subset flags**: are experimental follow-up tooling added after the
  reproduction study. Do not mix their outputs into the main reproduction
  verdict tables or session summaries. Document them separately as a reduced-data
  transfer study.
