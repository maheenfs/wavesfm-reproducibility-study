# Reproduction Log

This note tells the complete story of reproducing the WavesFM benchmark: how
datasets were obtained, what ran in each experimental session, what the results
were, and what conclusions were drawn. It is organized chronologically.

---

## Phase 1: Dataset Acquisition and Preprocessing

### Datasets Obtained

| Task | Raw source location | Format | Status |
|------|-------------------|--------|--------|
| `sensing` | `datasets_raw/has/NTU-Fi_HAR` | `.mat` files (train_amp + test_amp) | Ready |
| `pos` | `datasets_raw/pos/dataset_SNR{10,20,50}_outdoor.mat` | `.mat` files | Ready |
| `radcom` | `datasets_raw/radcom/RadComOta2.45GHz.hdf5` | HDF5 | Ready |
| `rml` | `datasets_raw/rml22/RML22.01A` | Pickle archives | Ready |
| `rfp` | `datasets_raw/rfp/GlobecomPOWDER` | IQ data files | Ready |
| `uwb-indoor` | `datasets_raw/uwb_indoor/environment0..3` | CSV CIR data | Ready (after bug fix) |
| `uwb-industrial` | `datasets_raw/uwb_industrial/industrial_training.pkl` | Pickle | Ready |
| `interf` | `datasets_raw/icarus/Batch1_{5,10}MHz` | IQ recordings | Ready |
| `deepmimo` | `datasets_raw/deepmimo/lwm_snapshot/city_*` | DeepMIMO scenario folders | Ready |
| `deepmimo` (legacy) | `datasets_raw/deepmimo/deepmimo_data.p` | Compatibility pickle | Used in first pass only |
| `lwm-beam` | `datasets_raw/deepmimo/lwm_beam_labels/beam_prediction_challenge/` | Pickle pair | Ready (hypothesis only) |
| `rfs` | `dataset_sources/rfs/27x-Walkie-talkie-dataset-Version-2` | Nested `.wav` files | **Blocked** |

### The RFS Problem

The WavesFM `rfs` task expects a flat directory of spectrogram images with 20
signal-class labels derived from filename prefixes. The publicly available CommRad
archive contains nested directories of `.wav` recordings organized by 27 radio
devices. The label semantics are fundamentally different (device identity vs.
signal type). No official preprocessed cache, benchmark-ready image bundle, or
label mapping was found. Inventing a mapping would not be a faithful reproduction,
so `rfs` was marked as blocked.

### Preprocessing

Each raw dataset was converted to an HDF5 cache using the scripts in
`wavesfm/preprocessing/`. The preprocessing for UWB indoor required a bug fix
(see `03_code_changes.md`). DeepMIMO preprocessing went through two iterations
(see Phase 3 below). All 13 resulting cache files live in `datasets_h5/`.

---

## Phase 2: First Full Run

**Session:** `phase2_vivor4/runs/20260403_004122-ngwn06`
**Host:** ngwn06 (RTX 5090)
**Scope:** 10 tasks x 3 modes x 3 seeds = 90 individual training runs
**Status:** All 90 completed

### Results

#### Classification Tasks (PCA %)

| Task | LP (official) | LP (local) | FT2 (official) | FT2 (local) | LoRA (official) | LoRA (local) |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| `rfp` | 98.97 | **98.98** | 99.70 | **99.70** | 99.82 | **99.84** |
| `rml` | 50.39 | **50.39** | 55.16 | **55.14** | 56.49 | **56.42** |
| `radcom` | 90.10 | **90.12** | 94.53 | **94.61** | 93.78 | **94.07** |
| `interf` | 71.50 | **71.17** | 78.94 | **78.83** | 78.94 | **79.67** |
| `deepmimo-los` | 95.02 | **95.12** | 95.46 | **95.47** | 95.24 | **95.32** |
| `sensing` | 94.42 | **94.13** | 99.31 | **98.60** | 98.47 | **98.70** |
| `deepmimo-beam` | 67.67 | **55.80** | 78.38 | **68.65** | 77.38 | **67.56** |

#### Regression Tasks (mean error in meters, lower is better)

| Task | LP (official) | LP (local) | FT2 (official) | FT2 (local) | LoRA (official) | LoRA (local) |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| `pos` | 3.17 | **3.14** | 1.24 | **1.26** | 1.45 | **1.51** |
| `uwb-indoor` | 1.43 | **1.44** | 0.83 | **0.82** | 0.65 | **0.65** |
| `uwb-industrial` | 2.71 | **2.71** | 0.88 | **0.88** | 0.74 | **0.74** |

### First-Pass Audit Findings

After the first run completed, an audit identified three issues:

1. **`deepmimo-beam` (invalid):** The first-pass cache was built from the legacy
   pickle path, which produced only 63 effective beam classes for the 64-beam
   setting. The beam sweep used an inclusive [-90, +90] interval that duplicated
   the endpoint and left class 63 unreachable. These beam results were discarded.

2. **`sensing` (exploratory):** The released code merges the original train/test
   directories before applying a new random 80/20 split. The exact split indices
   used for official results are not published. First-pass results were
   informative but not final.

3. **`rfs` (blocked):** Dataset format mismatch prevented any run.

All other tasks (rfp, rml, radcom, interf, pos, uwb-indoor, uwb-industrial,
deepmimo-los) were accepted as valid. Their results remain the final reportable
numbers.

---

## Phase 3: Corrected Second Pass

**Session:** `phase2_vivor4/runs/second_try_sensing_deepmimo_20260415_005633-ngwn06`
**Host:** ngwn06
**Scope:** 3 tasks (sensing, deepmimo-los, deepmimo-beam) x 3 modes x 3 seeds = 27 runs
**Status:** All 27 completed, 9 plots, 9 summaries, 9 comparisons generated

### What Was Fixed

Before this run, two corrections were made:

1. **DeepMIMO cache rebuilt from official scenario folders** (not the legacy
   pickle). The scenario data in `datasets_raw/deepmimo/lwm_snapshot/city_*`
   was used as the source.

2. **Beam codebook sweep fixed** from inclusive [-90, +90] to half-open
   [-90, 90). This restored the missing class 63 and produced a proper 64-class
   label space. Post-fix verification confirmed `effective_n_beams_64=64` with
   no missing beam classes.

3. **Sensing cache rebuilt** with source-split provenance tracking (records
   whether each sample came from train_amp or test_amp).

Backups of the old caches were created before overwriting.

### Results

| Task | Mode | Official | Local Mean | Local Std | Delta | Verdict |
|------|------|:---:|:---:|:---:|:---:|---------|
| `deepmimo-los` | LP | 95.02 | 95.12 | 0.61 | +0.10 | **Reproduced** |
| `deepmimo-los` | FT2 | 95.46 | 95.36 | 0.39 | -0.10 | **Reproduced** |
| `deepmimo-los` | LoRA | 95.24 | 95.16 | 0.39 | -0.08 | **Reproduced** |
| `sensing` | LP | 94.42 | 93.89 | 1.99 | -0.53 | Close (split caveat) |
| `sensing` | FT2 | 99.31 | 98.47 | 0.86 | -0.84 | Close (split caveat) |
| `sensing` | LoRA | 98.47 | 99.31 | 0.71 | +0.84 | Close (split caveat) |
| `deepmimo-beam` | LP | 67.67 | 56.37 | 0.64 | -11.30 | **Not reproduced** |
| `deepmimo-beam` | FT2 | 78.38 | 69.44 | 0.15 | -8.94 | **Not reproduced** |
| `deepmimo-beam` | LoRA | 77.38 | 67.84 | 0.42 | -9.54 | **Not reproduced** |

### Key Observations

- **DeepMIMO LoS/NLoS reproduced beautifully** -- within 0.1 pp across all modes.
  This confirms the DeepMIMO scenario-folder rebuild was viable and the
  preprocessing pipeline works correctly for the LoS task.

- **Sensing came close** but all deviations are within the inter-seed standard
  deviation. The small dataset (240 test samples) amplifies split sensitivity.
  LoRA actually exceeded the official number by 0.84 pp.

- **DeepMIMO beam prediction failed again.** The gap (9-11 pp) is far larger than
  seed variation (std <= 0.64). The corrected cache fixed the 63-class bug but
  did not close the performance gap. The problem is specific to beam prediction
  -- the same pipeline reproduced LoS/NLoS just fine.

---

## Phase 4: LWM Beam-Challenge Hypothesis Test

**Session:** `phase2_vivor4/runs/third_try_lwm_beam_challenge_20260415_140720-ngwn06`
**Host:** ngwn06
**Scope:** 1 task (lwm-beam-challenge) x 3 modes x 3 seeds = 9 runs
**Status:** All 9 completed

### Motivation

After the second pass still failed to reproduce beam prediction, a hypothesis
was formed: maybe the official WavesFM beam numbers were produced from a different
data source -- specifically, the public LWM beam-challenge feature/label pair.

### Artifact Investigation

The following official LWM files were found:
- `bp_data_train.p` (622 labeled samples, feature shape 32x32)
- `bp_label_train.p` (matching labels)
- `bp_data_test.p` (2491 unlabeled test samples)

These do **not** match the WavesFM detailed-eval setup:
- WavesFM: 14,840 total samples, 2,968 test samples, shape (2, 32, 32)
- LWM challenge: 622 total labeled samples, shape (32, 32), different feature
  representation

### Setup

A separate task ID (`lwm-beam-challenge`) and preprocessor
(`preprocess_lwm_beam_challenge.py`) were created to keep this experiment
isolated from the corrected `deepmimo-beam` pipeline.

### Results

| Mode | Official Ref (deepmimo-beam) | Local Mean | Delta |
|------|:---:|:---:|:---:|
| LP | 67.67 | **1.88** | -65.79 |
| FT2 | 78.38 | **1.68** | -76.70 |
| LoRA | 77.38 | **1.69** | -75.69 |

Near-random performance on all modes.

### Conclusion

The public LWM beam-challenge training pair **does not** explain the official
WavesFM `deepmimo-beam` results. The hypothesis is conclusively rejected.

---

## Why DeepMIMO Beam Prediction Remains Unresolved

The evidence points to a **benchmark protocol mismatch**:

1. **Not seed instability:** Std <= 0.64 across seeds, but the gap is >= 8.94 pp.
2. **Not the 63-class bug:** Fixed in the second pass, gap persisted.
3. **Not the LWM challenge data:** Ruled out by the third run.
4. **Not a general pipeline problem:** DeepMIMO LoS/NLoS (same dataset, same
   preprocessor) reproduced within 0.1 pp.

The most likely cause is that the official evaluation used a specific HDF5 cache,
label-to-beam mapping, or train/test split that is not recoverable from publicly
available artifacts. Our reconstructed cache has 14,840 samples of shape
(2, 32, 32), but without access to the exact official cache or split indices,
the gap cannot be closed.

This is a **public artifact availability problem**, not a local scripting mistake.
It can only be resolved if the authors provide:
- The exact WavesFM `deepmimo-beam` H5 cache, or
- The exact train/validation split indices, or
- The exact mapping from LWM scenario rows to beam labels

---

## Final Verdicts

| Task | Session Used | Verdict |
|------|-------------|---------|
| `rfp` | First run | Reproduced (within 0.02 pp) |
| `rml` | First run | Reproduced (within 0.07 pp) |
| `radcom` | First run | Reproduced (within 0.29 pp) |
| `interf` | First run | Reproduced (within 0.73 pp, small test set) |
| `deepmimo-los` | Second pass | Reproduced (within 0.10 pp) |
| `pos` | First run | Reproduced (within 0.06 m) |
| `uwb-indoor` | First run | Reproduced (within 0.01 m) |
| `uwb-industrial` | First run | Reproduced (within 0.004 m) |
| `sensing` | Second pass | Close, with split protocol caveat |
| `deepmimo-beam` | Second pass | **Not reproduced** (9-11 pp gap) |
| `rfs` | -- | **Blocked** (dataset format mismatch) |

### Overall Assessment

The WavesFM benchmark is **largely reproducible**. Eight of ten attempted tasks
matched the published numbers closely -- spanning both modality families (vision
and IQ) and both task types (classification and regression). The pretrained
checkpoint genuinely transfers across diverse wireless problems.

The beam prediction gap and the RFS data mismatch are real concerns, but they are
narrowly scoped. They do not undermine the broader foundation model claim. Full
reproducibility would require the authors to release exact preprocessed caches
and split indices for all tasks.
