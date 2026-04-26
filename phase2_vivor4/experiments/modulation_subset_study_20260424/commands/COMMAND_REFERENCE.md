# Command Reference

Copy-paste reference for running every experiment in this repository.
Two studies are documented here, kept fully separate:

- **Reproduction study** — full-data benchmark across all 11 tasks
  (results live under `phase2_vivor4/runs/<reproduction_session>/`)
- **Modulation subset study** — reduced-data follow-up on `rml` and `radcom`
  only (results live under `phase2_vivor4/runs/modulation_subset_*/`)

All commands assume the current working directory is the project root
(`~/wavesfm_vivor4_m2` on the lab machine). SSH first when running from the
Mac:

```bash
ssh ngwn06
```

---

# Reproduction Study Commands

## Full reproduction sweep (all 10 tasks together)

The recommended path for a from-scratch reproduction is the supervised
tracker, which queues every (task, mode, seed) cell and resumes after thermal
or pressure pauses:

```bash
python phase2_vivor4/scripts/wait_for_radcom_and_run_next.py \
  --session-root phase2_vivor4/runs/<your_session_label> \
  --tasks rfp rml radcom interf sensing deepmimo-los deepmimo-beam pos uwb-indoor uwb-industrial \
  --modes lp ft2 lora \
  --seeds 0 1 2 \
  --num-workers 4 --save-every 5
```

A simpler alternative without the supervisor:

```bash
python phase2_vivor4/scripts/run_all_tasks.py
```

## Individual reproduction tasks

Each example fine-tunes one task with one mode and one seed. Replace
`--task`, `--train-data`, `--frozen-blocks`, and the output folder for other
combinations.

```bash
# rml — Modulation Classification (RML22)
python wavesfm/main_finetune.py \
  --task rml --train-data datasets_h5/rml22.h5 \
  --finetune checkpoints/wavesfm-v1p0.pth \
  --epochs 50 --batch-size 2048 --num-workers 4 \
  --model vit_multi_small --use-conditional-ln \
  --frozen-blocks 6 --seed 0 \
  --output-dir phase2_vivor4/runs/<session>/local_results/by_task/rml/ft2/s0

# radcom — RADCOM Signal Classification
python wavesfm/main_finetune.py \
  --task radcom --train-data datasets_h5/radcom.h5 \
  --finetune checkpoints/wavesfm-v1p0.pth \
  --epochs 50 --batch-size 2048 --num-workers 4 \
  --model vit_multi_small --use-conditional-ln \
  --frozen-blocks 6 --seed 0 \
  --output-dir phase2_vivor4/runs/<session>/local_results/by_task/radcom/ft2/s0

# rfp — RF Fingerprinting (POWDER)
python wavesfm/main_finetune.py \
  --task rfp --train-data datasets_h5/rfp.h5 \
  --finetune checkpoints/wavesfm-v1p0.pth \
  --epochs 10 --batch-size 256 --smoothing 0.1 \
  --model vit_multi_small --use-conditional-ln \
  --frozen-blocks 6 --seed 0 \
  --output-dir phase2_vivor4/runs/<session>/local_results/by_task/rfp/ft2/s0

# interf — Interference Classification (ICARUS)
python wavesfm/main_finetune.py \
  --task interf --train-data datasets_h5/interf.h5 \
  --finetune checkpoints/wavesfm-v1p0.pth \
  --epochs 35 --batch-size 256 --accum-steps 2 --smoothing 0.02 \
  --model vit_multi_small --use-conditional-ln --stratified-split --class-weights \
  --frozen-blocks 6 --seed 0 \
  --output-dir phase2_vivor4/runs/<session>/local_results/by_task/interf/ft2/s0

# sensing — Human Activity Sensing (HAS / EfficientFi CSI)
python wavesfm/main_finetune.py \
  --task sensing --train-data datasets_h5/has.h5 \
  --finetune checkpoints/wavesfm-v1p0.pth \
  --epochs 100 --batch-size 256 --smoothing 0.1 \
  --model vit_multi_small --use-conditional-ln --stratified-split \
  --frozen-blocks 6 --seed 0 \
  --output-dir phase2_vivor4/runs/<session>/local_results/by_task/sensing/ft2/s0

# deepmimo-los — LoS / NLoS Classification
python wavesfm/main_finetune.py \
  --task deepmimo-los --train-data datasets_h5/deepmimo.h5 \
  --finetune checkpoints/wavesfm-v1p0.pth \
  --epochs 100 --batch-size 256 \
  --model vit_multi_small --use-conditional-ln --stratified-split --class-weights \
  --vis-img-size 32 --frozen-blocks 6 --seed 0 \
  --output-dir phase2_vivor4/runs/<session>/local_results/by_task/deepmimo-los/ft2/s0

# deepmimo-beam — 64-way Beam Prediction
python wavesfm/main_finetune.py \
  --task deepmimo-beam --train-data datasets_h5/deepmimo.h5 \
  --finetune checkpoints/wavesfm-v1p0.pth \
  --epochs 100 --batch-size 256 --deepmimo-n-beams 64 \
  --model vit_multi_small --use-conditional-ln --stratified-split --class-weights \
  --vis-img-size 32 --frozen-blocks 6 --seed 0 \
  --output-dir phase2_vivor4/runs/<session>/local_results/by_task/deepmimo-beam/ft2/s0

# pos — 5G NR Positioning (regression)
python wavesfm/main_finetune.py \
  --task pos --train-data datasets_h5/pos.h5 \
  --finetune checkpoints/wavesfm-v1p0.pth \
  --epochs 100 --batch-size 256 \
  --model vit_multi_small --use-conditional-ln \
  --frozen-blocks 6 --seed 0 \
  --output-dir phase2_vivor4/runs/<session>/local_results/by_task/pos/ft2/s0

# uwb-indoor — UWB Indoor Positioning (regression)
python wavesfm/main_finetune.py \
  --task uwb-indoor --train-data datasets_h5/uwb-indoor.h5 \
  --finetune checkpoints/wavesfm-v1p0.pth \
  --epochs 50 --batch-size 256 \
  --model vit_multi_small --use-conditional-ln \
  --frozen-blocks 6 --seed 0 \
  --output-dir phase2_vivor4/runs/<session>/local_results/by_task/uwb-indoor/ft2/s0

# uwb-industrial — UWB Industrial Localization (regression)
python wavesfm/main_finetune.py \
  --task uwb-industrial --train-data datasets_h5/uwb-industrial.h5 \
  --finetune checkpoints/wavesfm-v1p0.pth \
  --epochs 50 --batch-size 512 \
  --model vit_multi_small --use-conditional-ln \
  --frozen-blocks 6 --seed 0 \
  --output-dir phase2_vivor4/runs/<session>/local_results/by_task/uwb-industrial/ft2/s0
```

Mode mapping for `--frozen-blocks`:

- LP: `--frozen-blocks 8` (all blocks frozen)
- FT2: `--frozen-blocks 6` (last 2 blocks trainable)
- LoRA: `--frozen-blocks 8 --lora-rank 32 --lora-alpha 64`

## Aggregating reproduction results

After a session completes:

```bash
python phase2_vivor4/scripts/summarize_local_results.py --session-root phase2_vivor4/runs/<session>
python phase2_vivor4/scripts/compare_with_official.py   --session-root phase2_vivor4/runs/<session>
python phase2_vivor4/scripts/plot_local_detailed_eval.py --session-root phase2_vivor4/runs/<session>
```

---

# Modulation Subset Study Commands

These commands belong only to the reduced-data follow-up. They live under
their own session folder prefix `modulation_subset_*` and never get mixed
into the reproduction session aggregates.

Study commands folder on the lab machine:

```bash
~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands
```

Use:

- `launch_1pct.sh`, `launch_3pct.sh`, `launch_5pct.sh`, `launch_10pct.sh`,
  `launch_50pct.sh` — exact two-task study launchers for each subset
- `run_modulation_subset_fresh.sh` — fresh timestamped run for one task or a
  custom rerun without touching completed study sessions

## On the lab machine

Exact study launchers, both tasks together (rml + radcom):

```bash
bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/launch_1pct.sh
bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/launch_3pct.sh
bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/launch_5pct.sh
bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/launch_10pct.sh
bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/launch_50pct.sh
```

Fresh single-task runs, `rml`:

```bash
bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/run_modulation_subset_fresh.sh 1pct 0.01 rml
bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/run_modulation_subset_fresh.sh 3pct 0.03 rml
bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/run_modulation_subset_fresh.sh 5pct 0.05 rml
bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/run_modulation_subset_fresh.sh 10pct 0.10 rml
bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/run_modulation_subset_fresh.sh 50pct 0.50 rml
```

Fresh single-task runs, `radcom`:

```bash
bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/run_modulation_subset_fresh.sh 1pct 0.01 radcom
bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/run_modulation_subset_fresh.sh 3pct 0.03 radcom
bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/run_modulation_subset_fresh.sh 5pct 0.05 radcom
bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/run_modulation_subset_fresh.sh 10pct 0.10 radcom
bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/run_modulation_subset_fresh.sh 50pct 0.50 radcom
```

Fresh two-task reruns with new session roots:

```bash
bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/run_modulation_subset_fresh.sh 1pct 0.01 rml radcom
bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/run_modulation_subset_fresh.sh 3pct 0.03 rml radcom
bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/run_modulation_subset_fresh.sh 5pct 0.05 rml radcom
bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/run_modulation_subset_fresh.sh 10pct 0.10 rml radcom
bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/run_modulation_subset_fresh.sh 50pct 0.50 rml radcom
```

Single-mode or single-seed examples:

```bash
bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/run_modulation_subset_fresh.sh 5pct 0.05 rml -- --modes lora --seeds 0
bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/run_modulation_subset_fresh.sh 10pct 0.10 radcom -- --modes ft2 --seeds 1
```

## From the Mac, executing on the lab machine

Study launchers, both tasks together:

```bash
ssh ngwn06 'bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/launch_1pct.sh'
ssh ngwn06 'bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/launch_3pct.sh'
ssh ngwn06 'bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/launch_5pct.sh'
ssh ngwn06 'bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/launch_10pct.sh'
ssh ngwn06 'bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/launch_50pct.sh'
```

Fresh single-task runs, `rml`:

```bash
ssh ngwn06 'bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/run_modulation_subset_fresh.sh 1pct 0.01 rml'
ssh ngwn06 'bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/run_modulation_subset_fresh.sh 3pct 0.03 rml'
ssh ngwn06 'bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/run_modulation_subset_fresh.sh 5pct 0.05 rml'
ssh ngwn06 'bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/run_modulation_subset_fresh.sh 10pct 0.10 rml'
ssh ngwn06 'bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/run_modulation_subset_fresh.sh 50pct 0.50 rml'
```

Fresh single-task runs, `radcom`:

```bash
ssh ngwn06 'bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/run_modulation_subset_fresh.sh 1pct 0.01 radcom'
ssh ngwn06 'bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/run_modulation_subset_fresh.sh 3pct 0.03 radcom'
ssh ngwn06 'bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/run_modulation_subset_fresh.sh 5pct 0.05 radcom'
ssh ngwn06 'bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/run_modulation_subset_fresh.sh 10pct 0.10 radcom'
ssh ngwn06 'bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/run_modulation_subset_fresh.sh 50pct 0.50 radcom'
```

Fresh two-task reruns with new session roots:

```bash
ssh ngwn06 'bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/run_modulation_subset_fresh.sh 1pct 0.01 rml radcom'
ssh ngwn06 'bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/run_modulation_subset_fresh.sh 3pct 0.03 rml radcom'
ssh ngwn06 'bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/run_modulation_subset_fresh.sh 5pct 0.05 rml radcom'
ssh ngwn06 'bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/run_modulation_subset_fresh.sh 10pct 0.10 rml radcom'
ssh ngwn06 'bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/run_modulation_subset_fresh.sh 50pct 0.50 rml radcom'
```

Single-mode or single-seed example:

```bash
ssh ngwn06 'bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/run_modulation_subset_fresh.sh 5pct 0.05 rml -- --modes lora --seeds 0'
ssh ngwn06 'bash ~/wavesfm_vivor4_m2/phase2_vivor4/experiments/modulation_subset_study_20260424/commands/run_modulation_subset_fresh.sh 10pct 0.10 radcom -- --modes ft2 --seeds 1'
```

## Quick monitoring

Find the newest fresh `rml` run on the lab machine:

```bash
ls -td /local/data0/maheenfs/wavesfm_vivor4_m2/phase2_vivor4/runs/modulation_subset_rml_* | head -n 1
```

Tail the supervisor log for the newest fresh `rml` run:

```bash
tail -f "$(ls -td /local/data0/maheenfs/wavesfm_vivor4_m2/phase2_vivor4/runs/modulation_subset_rml_* | head -n 1)/supervisor.log"
```

Tail the training log for the newest fresh `rml` run:

```bash
tail -f "$(ls -td /local/data0/maheenfs/wavesfm_vivor4_m2/phase2_vivor4/runs/modulation_subset_rml_* | head -n 1)/local_results/by_task/rml/lp/s0/log.txt"
```
