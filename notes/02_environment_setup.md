# Environment Setup

This note documents the hardware and software environment in which the
reproduction and the modulation subset study were run. Every result in this
repository was produced on the configuration described here.

Anyone wanting to reproduce should plug in their own GPU host, their own
account, and their own paths — none of the personal access details (host
names, usernames, SSH config) are included.

---

## Two-machine workflow

Work was split between a laptop used for editing/coordination and a GPU
server used for training:

| Role | Used for |
|------|----------|
| Development laptop | Code editing, dataset acquisition, preprocessing coordination, report writing, dashboard viewing, sync orchestration |
| GPU server         | Full benchmark runs, long unattended training, heavy preprocessing |

The laptop is always the canonical editing source. Code and configuration
changes are made on the laptop and synced to the GPU host. Results flow the
other direction — the GPU host produces checkpoints, logs, and plots that
are synced back to the laptop.

---

## GPU host

| Component | Used here |
|-----------|-----------|
| GPU       | NVIDIA GeForce RTX 5090 |
| GPU memory | 32 GB |
| CPU       | x86_64, 24 cores |
| RAM       | 62 GB |
| OS        | Ubuntu 24.04 |
| Driver / CUDA | NVIDIA driver 575.x, CUDA 12.4 |

Heavy directories (`datasets_raw/`, `datasets_h5/`, `phase2_vivor4/runs/`)
were kept on a fast local data partition and symlinked into the workspace
so the home directory stayed small.

---

## Python environment (GPU host and laptop)

```bash
conda create -n wavesfm python=3.10 -y
conda activate wavesfm
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu124
pip install timm==1.0.11 h5py==3.15.1 DeepMIMOv3 matplotlib==3.10.7
pip install -r wavesfm/requirements.txt
```

| Package      | Version |
|--------------|---------|
| Python       | 3.10    |
| PyTorch      | 2.9.1   |
| torchvision  | 0.24.1  |
| CUDA toolkit | 12.4    |
| timm         | 1.0.11  |
| h5py         | 3.15.1  |
| DeepMIMOv3   | latest  |
| matplotlib   | 3.10.7  |

On a Mac/Apple Silicon laptop, the same environment also runs end-to-end on
MPS for development; the AMP / GradScaler path is gated to CUDA only (see
`notes/03_code_changes.md`).

---

## Sync workflow

Code and notes are pushed from the laptop to the GPU host with `rsync`,
excluding the heavy data caches and dataset archives. Results are pulled
back the same way after a session completes. Substitute your own host name,
username, and remote root for the placeholders below:

```bash
# Laptop -> GPU host  (push code and config)
rsync -avz --exclude '__pycache__' --exclude '*.pyc' \
    --exclude 'datasets_*' --exclude 'checkpoints/' --exclude 'dataset_sources/' \
    /path/to/wavesfm_vivor4_m2/ <user>@<lab-host>:~/wavesfm_vivor4_m2/

# GPU host -> Laptop  (pull results)
rsync -avz <user>@<lab-host>:~/wavesfm_vivor4_m2/phase2_vivor4/runs/ \
    /path/to/wavesfm_vivor4_m2/phase2_vivor4/runs/
```

A convenience script `phase2_vivor4/scripts/sync_second_try_to_<lab-host>.sh`
wraps the same operations for specific session pushes.

### Rules of thumb

- Always sync **before** starting a new training session (push latest code).
- Always sync **after** a session completes (pull results back).
- Never edit code directly on the GPU host -- edit on the laptop, then sync.
- Large data (raw datasets, preprocessed caches, checkpoints) lives
  permanently on the GPU host and is not synced back.

---

## Running training sessions on the GPU host

Inside a `tmux` session on the GPU host:

```bash
tmux new -s wavesfm
conda activate wavesfm
cd ~/wavesfm_vivor4_m2

# Full reproduction sweep
python phase2_vivor4/scripts/run_all_tasks.py

# Or the supervised tracker (queues + thermal/pressure pauses)
python phase2_vivor4/scripts/wait_for_radcom_and_run_next.py \
    --session-root phase2_vivor4/runs/<your_session_label> \
    --tasks rfp rml radcom interf sensing deepmimo-los deepmimo-beam pos uwb-indoor uwb-industrial \
    --modes lp ft2 lora --seeds 0 1 2
```

Detach with `Ctrl-B D`, reattach with `tmux attach -t wavesfm`.

---

## Live monitoring

The harness writes a live HTML dashboard at
`phase2_vivor4/automation_logs/dashboard.html` during each session. It
tracks GPU temperature (with automatic pause/resume above a threshold),
training progress, and session completion state. Forward the local port
over SSH to view it from the laptop browser.

---

## Personal access details have been removed

This note intentionally does **not** include:

- Specific lab host names or fully-qualified domain names
- Specific account user names
- SSH config blocks tied to a personal account
- Storage symlink targets that reveal a specific home directory layout

Anyone reproducing this work should configure their own GPU host, their own
SSH access, and their own storage layout. The `<user>` and `<lab-host>`
placeholders that appear throughout the run logs in `phase2_vivor4/runs/`
are stand-ins for whatever was used at the time and have no meaning outside
that machine.
