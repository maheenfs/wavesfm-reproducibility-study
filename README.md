# WavesFM Reproducibility Study

A full reproducibility study of **WavesFM**, a multimodal wireless foundation
model for sensing, communication, and localization. This repository contains
the adapted framework code, the local benchmark harness, all run logs, the
final report, and a reduced-data follow-up study on the modulation tasks.

- **Paper:** [Multimodal Wireless Foundation Models](https://arxiv.org/abs/2511.15162) (Aboulfotouh & Abou-Zeid, 2026)
- **Official site:** https://wavesfm.waveslab.ai/
- **Local paper copy:** `Multi_Model.pdf`

---

## Project Structure

```
wavesfm_vivor4_m2/
├── wavesfm/                      # WavesFM framework code (adapted from official release)
│   ├── main_finetune.py          # Training entry point
│   ├── models_vit.py             # ViT model architecture
│   ├── engine.py                 # Training / evaluation loop
│   ├── data.py                   # Task registry, splits, train-subset support
│   ├── lora.py                   # LoRA adapter implementation
│   ├── dataset_classes/          # Per-task dataset wrappers
│   └── preprocessing/            # Raw → HDF5 preprocessing scripts
│
├── checkpoints/                  # Pretrained weights (wavesfm-v1p0.pth)
├── datasets_raw/                 # Raw datasets (per-task subdirectories)
├── datasets_h5/                  # Preprocessed HDF5 caches
│
├── phase2_vivor4/                # Local benchmark harness and results
│   ├── scripts/                  # Orchestration, comparison, plotting scripts
│   ├── official_results/         # Published WavesFM baseline numbers (JSON)
│   ├── runs/                     # All experimental sessions (see below)
│   └── experiments/              # Follow-up study scaffolds + tables
│
├── report/                       # Final deliverables
│   ├── wavesfm_reproduction_report.tex   # LaTeX source
│   ├── wavesfm_reproduction_report.pdf   # Compiled report (34 pages)
│   ├── wavesfm_reproduction_presentation.pptx  # Reproduction presentation (polished)
│   ├── wavesfm_subset_study.pptx         # Subset-study presentation
│   ├── wavesfm_presentation.pptx         # Auto-generated reproduction PPTX
│   ├── create_presentation.py            # Generates wavesfm_presentation.pptx
│   ├── create_subset_presentation.py     # Generates wavesfm_subset_study.pptx
│   ├── create_architecture_pptx.py       # Generates the architecture PPTX
│   ├── create_architecture_diagram.py    # Generates architecture PNGs
│   ├── create_task_visualizations.py     # Generates confusion-matrix and density plots
│   └── figures/                          # All plots + dataset visualisations
│
├── notes/                        # Documentation
│   ├── 01_project_overview.md            # What WavesFM is, tasks, metrics
│   ├── 02_environment_setup.md           # Mac + lab setup, SSH, sync workflow
│   ├── 03_code_changes.md                # Every modification to the upstream code
│   ├── 04_reproduction_log.md            # Full reproduction journey + verdicts
│   └── 05_modulation_subset_experiments.md  # Reduced-data follow-up study
│
├── README.md                     # This file
└── Multi_Model.pdf               # Original paper (local copy)
```

---

## Two Studies, Cleanly Separated

This repository contains two **independent** studies. They share the same
codebase and the same pretrained checkpoint but answer different questions and
must not be mixed in any results table.

| | Reproduction Study | Modulation Subset Study |
|---|---|---|
| **Question** | Do the published WavesFM benchmark numbers reproduce from public artifacts? | How label-efficient is WavesFM on the two modulation tasks? |
| **Tasks** | All 11 (10 attempted, 1 blocked) | rml + radcom only |
| **Modes** | LP, FT2, LoRA | LP, FT2, LoRA |
| **Subset levels** | 100% (full data) | 1, 3, 5, 10, 50% |
| **Seeds per cell** | 3 | 3 |
| **Total runs** | 90 + 27 + 9 = 126 | 90 |
| **Sessions** | 3 reproduction sessions under `phase2_vivor4/runs/` | 5 subset sessions under `phase2_vivor4/runs/modulation_subset_*` |
| **Notes file** | `notes/04_reproduction_log.md` | `notes/05_modulation_subset_experiments.md` |
| **Presentation** | `report/wavesfm_reproduction_presentation.pptx` | `report/wavesfm_subset_study.pptx` |

The subset study was added AFTER the reproduction work was complete. Its
results never enter the reproduction verdict table and live in dedicated
session roots.

---

# Reproduction Study

## Experimental Sessions

| Session folder under `phase2_vivor4/runs/` | Scope | Purpose |
|---|---|---|
| `20260403_004122-ngwn06` | 10 tasks × 3 modes × 3 seeds = 90 runs | First full benchmark sweep |
| `second_try_sensing_deepmimo_20260415_005633-ngwn06` | sensing + deepmimo-{los, beam} × 3 × 3 = 27 runs | Rerun after the DeepMIMO beam codebook fix and the sensing source-split rebuild |
| `third_try_lwm_beam_challenge_20260415_140720-ngwn06` | lwm-beam-challenge × 3 × 3 = 9 runs | Hypothesis test that ruled out the public LWM beam-challenge data as the source of the official beam numbers |

Each session contains `local_results/`, `comparisons/`, `plots/`,
`session_manifest.json`, and a `supervisor.log`.

## Reproduction Results

Format: `local (official)`. Classification metric = PCA %; regression = mean
distance error (m).

| Task | LP | FT2 | LoRA | Verdict |
|------|----:|-----:|------:|---------|
| rfp | 98.98 (98.97) | 99.70 (99.70) | 99.84 (99.82) | Reproduced |
| rml | 50.39 (50.39) | 55.14 (55.16) | 56.42 (56.49) | Reproduced |
| radcom | 90.12 (90.10) | 94.61 (94.53) | 94.07 (93.78) | Reproduced |
| interf | 71.17 (71.50) | 78.83 (78.94) | 79.67 (78.94) | Reproduced |
| deepmimo-los | 95.12 (95.02) | 95.36 (95.46) | 95.16 (95.24) | Reproduced |
| pos | 3.14 m (3.17 m) | 1.26 m (1.24 m) | 1.51 m (1.45 m) | Reproduced |
| uwb-indoor | 1.44 m (1.43 m) | 0.82 m (0.83 m) | 0.65 m (0.65 m) | Reproduced |
| uwb-industrial | 2.71 m (2.71 m) | 0.88 m (0.88 m) | 0.74 m (0.74 m) | Reproduced |
| sensing | 93.89 (94.42) | 98.47 (99.31) | 99.31 (98.47) | Close (split caveat) |
| deepmimo-beam | 56.37 (67.67) | 69.44 (78.38) | 67.84 (77.38) | **Not reproduced** |
| rfs | — (42.85) | — (83.41) | — (84.49) | **Blocked** (public data format mismatch) |

Eight of ten attempted tasks reproduced the published numbers within tight
margins (≤ 0.73 pp / 0.06 m). DeepMIMO beam prediction has a 9–11 pp gap that
three diagnostic runs could not close — the most likely cause is that the
official cache or split indices are not part of the public release.

## Code Changes Made for Reproduction

All modifications are documented in `notes/03_code_changes.md`. Quick map:

| Change | File | Why |
|---|---|---|
| Removed broken `run_finetune_all_jepa` import | `wavesfm/run_finetune_all.py` | Module does not exist in the public release |
| Removed dead `args.root` access | `wavesfm/preprocessing/preprocess_uwb_loc.py` | CLI flag never declared |
| `cuda → mps → cpu` device autoselect, AMP gated to CUDA | `wavesfm/main_finetune.py` | Mac portability for development |
| Atomic checkpoint writes + RNG state save/restore + fsynced JSONL log | `wavesfm/main_finetune.py`, `wavesfm/utils.py` | Survive crashes and preempted SLURM jobs |
| DeepMIMO beam default 16 → 64 | `wavesfm/main_finetune.py`, `wavesfm/run_finetune_all.py` | Match published 64-beam codebook |
| Beam sweep `[-90, 90]` (closed) → `[-90, 90)` (half-open) | `wavesfm/preprocessing/preprocess_deepmimo.py` | Endpoint duplication left class 63 unreachable |
| `--data-pickle` alt input + cache attribute provenance | `wavesfm/preprocessing/preprocess_deepmimo.py` | Track which DeepMIMO source built each cache |
| Beam head sized from selected codebook (not observed labels) | `wavesfm/data.py` | Train a proper 64-way classifier even on partial caches |
| New `lwm-beam-challenge` task id | `wavesfm/data.py` | Keep the third-run hypothesis isolated |
| Cache validation on pickle-sourced DeepMIMO | `wavesfm/dataset_classes/deepmimo.py` | Fail fast on malformed beam coverage |
| Sensing cache provenance + version bump v1 → v2 | `wavesfm/preprocessing/preprocess_csi_sensing.py` | Auditable train/test source split |
| Single-pass refactor (read+stat in one pass) | `wavesfm/preprocessing/preprocess_radcom.py` | Halves I/O |
| New preprocessor for LWM beam-challenge | `wavesfm/preprocessing/preprocess_lwm_beam_challenge.py` | Third-run hypothesis cache |
| `sensing` added to stratified-split set | `wavesfm/run_finetune_all.py` | Second-pass rerun |
| `matplotlib==3.10.7` added to requirements | `wavesfm/requirements.txt` | Local plot generation |

## How to Reproduce — Step by Step

```bash
# 0. Hardware: NVIDIA GPU with ≥16 GB VRAM (RTX 5090 used here).
#    Mac (MPS) works for development but not for the full sweep.

# 1. Clone the repo and create the Python environment
conda create -n wavesfm python=3.10 -y
conda activate wavesfm
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r wavesfm/requirements.txt

# 2. Place the pretrained checkpoint under checkpoints/
#    Source: WavesFM release page → wavesfm-v1p0.pth (~30 MB)

# 3. Download the raw datasets into datasets_raw/
#    Sources are listed in notes/04_reproduction_log.md (Phase 1 table).

# 4. Build the HDF5 caches one task at a time
python wavesfm/preprocessing/preprocess_rml.py        --data-file datasets_raw/rml22/RML22.01A --version 2022 --output datasets_h5/rml22.h5
python wavesfm/preprocessing/preprocess_radcom.py     --input datasets_raw/radcom/RadComOta2.45GHz.hdf5 --output datasets_h5/radcom.h5
python wavesfm/preprocessing/preprocess_csi_sensing.py --data-root datasets_raw/has/NTU-Fi_HAR --output datasets_h5/has.h5
python wavesfm/preprocessing/preprocess_deepmimo.py   --data-root datasets_raw/deepmimo/lwm_snapshot --output datasets_h5/deepmimo.h5
# (other preprocessors work the same way — see wavesfm/preprocessing/)

# 5. Single-task fine-tuning example (rml, LoRA, seed 0)
python wavesfm/main_finetune.py \
    --task rml \
    --train-data datasets_h5/rml22.h5 \
    --finetune checkpoints/wavesfm-v1p0.pth \
    --epochs 50 --batch-size 2048 --num-workers 4 \
    --model vit_multi_small --use-conditional-ln \
    --frozen-blocks 6 \
    --output-dir runs/example_rml_lora_s0 \
    --seed 0

# 6. Full reproduction sweep (all tasks × 3 modes × 3 seeds, in parallel-by-design)
python phase2_vivor4/scripts/run_all_tasks.py
# or, with the supervised tracker that resumes after thermal pauses:
python phase2_vivor4/scripts/wait_for_radcom_and_run_next.py \
    --session-root phase2_vivor4/runs/<your_session_label> \
    --tasks rfp rml radcom interf sensing deepmimo-los deepmimo-beam pos uwb-indoor uwb-industrial \
    --modes lp ft2 lora --seeds 0 1 2

# 7. Aggregate results and produce the comparison artefacts
python phase2_vivor4/scripts/summarize_local_results.py --session-root <session>
python phase2_vivor4/scripts/compare_with_official.py   --session-root <session>
python phase2_vivor4/scripts/plot_local_detailed_eval.py --session-root <session>
```

The session folder ends up with `local_results/`, `comparisons/`, `plots/`,
and `session_manifest.json`. Compare against the published numbers in
`phase2_vivor4/official_results/official_results_all.json`.

## Reproduction Deliverables

- **Report:** `report/wavesfm_reproduction_report.pdf` (34 pages)
- **Presentation:** `report/wavesfm_reproduction_presentation.pptx`
- **Auto-generated reference PPTX:** `report/wavesfm_presentation.pptx`
  (regenerate via `python report/create_presentation.py`)
- **Notes:** `notes/01_project_overview.md`, `notes/02_environment_setup.md`,
  `notes/03_code_changes.md`, `notes/04_reproduction_log.md`

---

# Modulation Subset Study (Follow-Up)

A reduced-data label-efficiency study on `rml` and `radcom` only. Asks: what
fraction of the full-data PCA does WavesFM retain when fine-tuning sees only
1, 3, 5, 10, or 50 percent of the labelled training split?

## Subset Sessions

All under `phase2_vivor4/runs/`:

- `modulation_subset_1pct_20260424_ngwn06`
- `modulation_subset_3pct_20260424_ngwn06`
- `modulation_subset_5pct_20260424_ngwn06`
- `modulation_subset_10pct_20260424_ngwn06`
- `modulation_subset_50pct_20260424_ngwn06`

Each: 2 tasks × 3 modes × 3 seeds = 18 runs. Total = 90 runs ≈ 17 GPU-hours
on RTX 5090.

## Subset Results — PCA Mean Across 3 Seeds

**RML22**

| Subset | LP | FT2 | LoRA |
|---|---:|---:|---:|
| 1% | 30.75 | 33.42 | 34.52 |
| 3% | 36.26 | 36.24 | 38.07 |
| 5% | 38.07 | 38.26 | 40.01 |
| 10% | 40.46 | 41.64 | 43.85 |
| 50% | 47.36 | 51.41 | 53.71 |
| **100% baseline** | **50.39** | **55.14** | **56.42** |

**RadCom OTA**

| Subset | LP | FT2 | LoRA |
|---|---:|---:|---:|
| 1% | 80.17 | 82.22 | 81.91 |
| 3% | 83.36 | 83.84 | 84.03 |
| 5% | 84.42 | 85.68 | 85.37 |
| 10% | 85.65 | 88.51 | 88.70 |
| 50% | 89.21 | 93.79 | 93.69 |
| **100% baseline** | **90.12** | **94.61** | **94.07** |

Headline finding: at 10% data, RadCom retains 93–95% of full-data PCA across
all three modes. RML22 needs more data — at 50% it still leaves 3–4 PCA on
the table.

## Code Changes Made for the Subset Study

| Change | File | Why |
|---|---|---|
| `--train-subset-fraction`, `--train-subset-size` flags | `wavesfm/main_finetune.py` | Reduce labelled training pool |
| Stratified subset sampling (post-split, class-balanced) | `wavesfm/data.py` | Preserve class distribution at every subset level |
| Pass-through of subset flags through batch runner | `phase2_vivor4/scripts/run_all_tasks.py` | Drive the study from the queue |
| Pass-through through the supervised tracker | `phase2_vivor4/scripts/wait_for_radcom_and_run_next.py` | Each subset gets its own session root |
| Experiment-aware dashboard, plot, summary, comparison manifests | `phase2_vivor4/scripts/{plot_local_detailed_eval,summarize_local_results,compare_with_official}.py` | Subset runs render visually distinct from the full-data ones |
| Study scaffold script | `phase2_vivor4/scripts/prepare_modulation_subset_study.py` | Creates `phase2_vivor4/experiments/modulation_subset_study_<date>/` with prefilled tables and per-subset launchers |
| Subset-study presentation generator | `report/create_subset_presentation.py` | Builds `report/wavesfm_subset_study.pptx` from raw run logs |
| Per-class IQ visualisations (separate PNG per class/pair) | `report/figures/datasets/` | Used by the subset PPTX |

## How to Run the Subset Study

```bash
# Prepare the study scaffold (writes phase2_vivor4/experiments/modulation_subset_study_<date>/)
python phase2_vivor4/scripts/prepare_modulation_subset_study.py

# Run one subset level (both rml and radcom, 3 modes, 3 seeds, 18 runs):
bash phase2_vivor4/experiments/modulation_subset_study_<date>/commands/launch_5pct.sh

# Or use the queue runner directly:
python phase2_vivor4/scripts/wait_for_radcom_and_run_next.py \
    --session-root phase2_vivor4/runs/modulation_subset_5pct_<date>_<host> \
    --tasks rml radcom \
    --modes lp ft2 lora \
    --seeds 0 1 2 \
    --train-subset-fraction 0.05
```

A full copy-paste command reference (Mac-side, lab-side, individual rml/radcom,
combined runs) lives at:

```
phase2_vivor4/experiments/modulation_subset_study_20260424/commands/COMMAND_REFERENCE.md
```

## Subset Study Deliverables

- **Presentation:** `report/wavesfm_subset_study.pptx` (60+ slides)
- **Notes:** `notes/05_modulation_subset_experiments.md`
- **Tables:** `phase2_vivor4/experiments/modulation_subset_study_20260424/tables/`
  (`table_a` … `table_f`: dataset, mode, run-config, results, comparison-vs-full-data, runtime)

---

## Hardware

- **Development:** MacBook Pro (code editing, preprocessing, dashboard, sync orchestration)
- **Training:** University lab server `ngwn06` — NVIDIA RTX 5090, 24 cores, 62 GB RAM, PyTorch 2.9.1+cu128

## Software Versions

| Package | Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.9.1 |
| torchvision | 0.24.1 |
| CUDA | 12.4 |
| timm | 1.0.11 |
| h5py | 3.15.1 |
| matplotlib | 3.10.7 |
| python-pptx | latest |
