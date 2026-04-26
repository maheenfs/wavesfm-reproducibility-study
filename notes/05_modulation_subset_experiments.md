# Modulation Subset Experiments

This note defines a **post-reproduction follow-up study** for the modulation
tasks `rml` and `radcom`. Its purpose is to test how much of the reported
transfer performance remains when downstream fine-tuning is restricted to a
small labeled training subset instead of the full task dataset.

These experiments are **not part of the main WavesFM reproduction benchmark**.
They should be treated as a separate reduced-data transfer study.

---

## Why This Exists

The reproduction benchmark used the full downstream task caches (with the usual
validation split) for fine-tuning. A follow-up concern was raised: if the
pretrained backbone is genuinely useful, what happens when the downstream
fine-tuning stage is given only a small fraction of labeled data?

This note isolates that question for the two modulation-related tasks:

- `rml` -- RadioML 2022 automatic modulation recognition
- `radcom` -- RadCom OTA modulation/signal classification

---

## Separation From Reproduction

The outputs from these runs must remain separate from the main reproduction
artifacts.

Do **not**:
- write these runs into the top-level `phase2_vivor4/local_results/` tree
- mix them into `notes/04_reproduction_log.md`
- merge them into the root `README.md` reproduction verdict table
- treat them as replacements for the original full-data reproduction sessions

Do:
- create a fresh named session under `phase2_vivor4/runs/`
- keep summaries, comparisons, and plots inside that session only
- describe them as reduced-data or subset fine-tuning experiments

Recommended session-name prefix:
- `modulation_subset_`

Examples:
- `phase2_vivor4/runs/modulation_subset_1pct_20260424_ngwn06/`
- `phase2_vivor4/runs/modulation_subset_5pct_20260424_ngwn06/`
- `phase2_vivor4/runs/modulation_subset_10pct_20260424_ngwn06/`

Within each session, store outputs in the standard session layout:
- `runs/<session>/local_results/by_task/...`
- `runs/<session>/local_results/summaries/...`
- `runs/<session>/comparisons/...`
- `runs/<session>/plots/...`
- `runs/<session>/session_manifest.json`

---

## Code Changes Supporting These Experiments

The following changes were added specifically for this reduced-data study:

### `wavesfm/main_finetune.py`

- Added `--train-subset-fraction`
- Added `--train-subset-size`

These flags reduce the amount of training data used for a run.

### `wavesfm/data.py`

- The subset is applied **after** the normal train/validation split.
- The validation set remains full and unchanged.
- For classification tasks, the training subset is sampled **stratified by
  class** so small subsets do not accidentally drop classes.

### `phase2_vivor4/scripts/run_all_tasks.py`

- Added pass-through support for the subset flags.

### `phase2_vivor4/scripts/wait_for_radcom_and_run_next.py`

- Added pass-through support so subset experiments can still use the normal
  session-based tracker layout under `phase2_vivor4/runs/<session>/`.

### `phase2_vivor4/scripts/plot_local_detailed_eval.py`

- Plot headers and the plot manifest now preserve the experiment setup:
  subset label, train split size, train size used, evaluation split size, and
  sampling policy.

### `phase2_vivor4/scripts/summarize_local_results.py` and `compare_with_official.py`

- Summary and comparison outputs now preserve the experiment label and subset
  settings and write stage manifests inside the session tree.

### Dashboard behavior

- The live dashboard now shows the experiment type and the exact session-scoped
  folders for run outputs, plots, summaries, and comparisons.

These changes are documented as **experimental-only deviations** in
`notes/03_code_changes.md`.

---

## Ready-To-Run Scaffold

A local scaffold script now prepares the study before launch:

```bash
python3 phase2_vivor4/scripts/prepare_modulation_subset_study.py
```

It creates:

- a study root under `phase2_vivor4/experiments/`
- prefilled CSV tables for dataset, mode, configuration, results, comparison,
  and runtime tracking
- per-subset launch scripts
- planned session roots under `phase2_vivor4/runs/`
- baseline reference files and a machine-readable study manifest

This is operational prep for the follow-up study only. It does not change the
main reproduction outputs.

---

## Exact Experimental Question

For `rml` and `radcom`, how do LP, FT2, and LoRA perform when fine-tuning uses
only a small fraction of the downstream training split?

The baseline for comparison is the existing **full-data reproduction**:
- full train split
- same architecture
- same checkpoint
- same mode definitions
- same seeds
- same validation protocol

The only intended change is the amount of downstream training data.

---

## Recommended Experiment Matrix

Tasks:
- `rml`
- `radcom`

Modes:
- `lp`
- `ft2`
- `lora`

Seeds:
- `0`
- `1`
- `2`

Suggested subset levels:
- `1%`
- `3%`
- `5%`
- `10%`
- `50%`

This gives:
- `2 tasks x 3 modes x 3 seeds = 18 runs` per subset level

If all five subset levels are run:
- `90 total runs`

---

## Subset Sizes With Default `--val-split 0.2`

Because the subset is applied **after** the train/validation split, the train
split sizes are:

- `rml`: total `462,000` -> train split `369,600`
- `radcom`: total `567,000` -> train split `453,600`

Approximate subset counts:

| Fraction | `rml` train samples | `radcom` train samples |
|---|---:|---:|
| `1%`  | `3,696`  | `4,536`  |
| `3%`  | `11,088` | `13,608` |
| `5%`  | `18,480` | `22,680` |
| `10%` | `36,960` | `45,360` |
| `50%` | `184,800` | `226,800` |

Validation sizes remain unchanged for all subset experiments:

- `rml` validation split: `92,400`
- `radcom` validation split: `113,400`

---

## Recommended Tables

Do not collapse everything into one oversized table. Use a small set of
focused tables so the dataset description, the mode definition, the run
configuration, the measured results, and the full-data comparison are each easy
to read.

### Table A -- Dataset And Task Summary

Use this once for `rml` and `radcom`.

Recommended columns:

- `Task ID`
- `Task Name`
- `Dataset Name`
- `Modality`
- `Task Type`
- `Label / Output Meaning`
- `# Classes`
- `Input Shape`
- `Output Type`
- `Total Size`
- `Train Size`
- `Validation Size`
- `Primary Metric`
- `Secondary Metrics`
- `Loss Function`
- `Class Balance Notes`
- `Normalization / Preprocessing`

Important note:
- In the current code path this is a **validation split**, not a separate test
  set. If the table uses a `Test Size` column, it should be labeled
  `Validation Size` unless a separate held-out test cache is later introduced.

### Table B -- Fine-Tuning Mode Definition

Use this once for `LP`, `FT2`, and `LoRA`.

Recommended columns:

- `Mode`
- `What Trains`
- `What Stays Frozen`
- `Trainable Parameters`
- `Total Parameters`
- `% Trainable`
- `Frozen Blocks`
- `LoRA Rank`
- `LoRA Alpha`
- `Checkpoint Used`
- `Checkpoint Selection Metric`

### Table C -- Run Configuration Matrix

Use this for each task x subset level x mode.

Recommended columns:

- `Task`
- `Subset %`
- `Sampling Size`
- `Sampling Policy`
- `Train Size Used`
- `Validation Size`
- `Total Size`
- `Mode`
- `Seeds`
- `Batch Size`
- `Accum Steps`
- `Effective Batch Size`
- `Epochs`
- `Loss Function`
- `Optimizer`
- `Learning Rate`
- `Weight Decay`
- `Warmup`
- `Label Smoothing`
- `Class Weights`
- `Trainable Parameters`
- `System Resources`

For this study, the sampling-policy entry should explicitly say:
- `stratified subset after train/validation split`

### Table D -- Results By Task, Mode, And Subset

Use this for the measured results.

Recommended columns:

- `Task`
- `Mode`
- `Subset %`
- `Train Size Used`
- `Best Epoch`
- `Validation Loss`
- `Primary Metric Mean`
- `Primary Metric Std`
- `Acc1`
- `Acc3`
- `Macro-F1`
- `Training Time`
- `Total Run Time`
- `System Resources`
- `Task Time For This Sampling Level`

Task-specific metric additions:

- For `rml`, include:
  - `PCA`
  - `Acc1`
  - `Acc3`
  - `Macro-F1`

- For `radcom`, include:
  - `PCA`
  - `Acc1`
  - `Acc3`
  - `mod_acc`
  - `sig_acc`

### Table E -- Comparison Against Original Full-Data Results

This table is critical. It should check each subset result against the original
full-data reproduction baseline, not only against the official website number.

Recommended columns:

- `Task`
- `Mode`
- `Subset %`
- `Subset Train Size`
- `Subset Primary Metric Mean`
- `Full-Data Local Baseline`
- `Delta vs Full-Data Local`
- `Official Reference`
- `Delta vs Official`
- `Retention vs Full-Data Local (%)`
- `Retention vs Official (%)`
- `Validation Loss`
- `Training Time`
- `Relative Time vs Full-Data`

### Table F -- Runtime Summary By Sampling Level

Use this to summarize total experiment cost, not per-run performance.

Recommended columns:

- `Task`
- `Subset %`
- `# Runs`
- `Total Task Time`
- `Mean Time / Run`
- `Median Time / Run`
- `Total Training Time`
- `Plot + Summary + Comparison Time`
- `Total Session Time`
- `Peak GPU Memory`
- `Peak Host RAM`
- `GPU Type`
- `CPU / Worker Count`

---

## Original Full-Data Baselines For Comparison

These are the full-data baselines that the subset tables should compare against.

### Local full-data reproduction baseline

| Task | LP | FT2 | LoRA |
|---|---:|---:|---:|
| `rml` | `50.39` | `55.14` | `56.42` |
| `radcom` | `90.12` | `94.61` | `94.07` |

Source:
- `notes/04_reproduction_log.md`
- `README.md`

### Official reference baseline

| Task | LP | FT2 | LoRA |
|---|---:|---:|---:|
| `rml` | `50.39` | `55.16` | `56.49` |
| `radcom` | `90.10` | `94.53` | `93.78` |

Source:
- `phase2_vivor4/official_results/official_results_all.json`

### Comparison calculations

For each subset result, compute:

- `Delta vs Full-Data Local = subset_metric - full_data_local_metric`
- `Delta vs Official = subset_metric - official_metric`
- `Retention vs Full-Data Local (%) = 100 * subset_metric / full_data_local_metric`
- `Retention vs Official (%) = 100 * subset_metric / official_metric`

For time:

- `Relative Time vs Full-Data = subset_total_time / full_data_total_time`

If lower-is-better metrics are added later, invert the interpretation logic for
retention and deltas. For the current modulation tasks, the main metrics are
higher-is-better classification metrics.

---

## Recommended Launch Commands

### 1% session

```bash
python3 phase2_vivor4/scripts/wait_for_radcom_and_run_next.py \
  --session-root phase2_vivor4/runs/modulation_subset_1pct_20260424_ngwn06 \
  --tasks rml radcom \
  --modes lp ft2 lora \
  --seeds 0 1 2 \
  --train-subset-fraction 0.01
```

### 3% session

```bash
python3 phase2_vivor4/scripts/wait_for_radcom_and_run_next.py \
  --session-root phase2_vivor4/runs/modulation_subset_3pct_20260424_ngwn06 \
  --tasks rml radcom \
  --modes lp ft2 lora \
  --seeds 0 1 2 \
  --train-subset-fraction 0.03
```

### 5% session

```bash
python3 phase2_vivor4/scripts/wait_for_radcom_and_run_next.py \
  --session-root phase2_vivor4/runs/modulation_subset_5pct_20260424_ngwn06 \
  --tasks rml radcom \
  --modes lp ft2 lora \
  --seeds 0 1 2 \
  --train-subset-fraction 0.05
```

### 10% session

```bash
python3 phase2_vivor4/scripts/wait_for_radcom_and_run_next.py \
  --session-root phase2_vivor4/runs/modulation_subset_10pct_20260424_ngwn06 \
  --tasks rml radcom \
  --modes lp ft2 lora \
  --seeds 0 1 2 \
  --train-subset-fraction 0.10
```

### 50% session

```bash
python3 phase2_vivor4/scripts/wait_for_radcom_and_run_next.py \
  --session-root phase2_vivor4/runs/modulation_subset_50pct_20260424_ngwn06 \
  --tasks rml radcom \
  --modes lp ft2 lora \
  --seeds 0 1 2 \
  --train-subset-fraction 0.50
```

If an absolute sample count is preferred instead of a fraction, use
`--train-subset-size <n>` instead.

---

## Interpretation Rules

When analyzing these experiments:

- Compare them first against the corresponding **full-data local reproduction**
  for `rml` and `radcom`, not only against the official website numbers.
- Treat any performance drop as the cost of using less labeled downstream data.
- Read the subset curves together with training time. A subset run that retains
  most of the full-data metric at a small fraction of the time is evidence of
  practical label efficiency.
- If LP remains strong even at very small subsets, that strengthens the claim
  that the pretrained encoder learned genuinely transferable features.
- If FT2 or LoRA degrade much more slowly than expected, that suggests the
  pretrained backbone is still doing substantial work even under reduced-data
  adaptation.

---

## Reporting Rules

When writing about these runs:

- call them **modulation subset experiments** or **reduced-data fine-tuning
  experiments**
- state clearly that they were run **after** the main reproduction study
- state clearly that they are **not** part of the original reproduction verdicts
- keep their plots, summary tables, and comparisons in their own subsection or
  appendix

They should answer a new question:

> How label-efficient is WavesFM on downstream modulation tasks?

They should **not** be presented as corrections to the original reproduction.
