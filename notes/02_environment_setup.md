# Environment Setup

This project uses two machines: a MacBook Pro for development and a university
lab server for GPU training. This note documents how they are set up and how
work is coordinated between them.

---

## Machine Roles

| Machine | Role | Used for |
|---------|------|----------|
| MacBook Pro | Development host | Code editing, dataset acquisition, preprocessing coordination, report writing, dashboard viewing, sync orchestration |
| `ngwn06` (York EECS lab) | GPU execution host | Full benchmark runs, long unattended training, heavy preprocessing |

The MacBook is always the canonical editing source. Code and configuration changes
are made on the Mac and synced to the lab. Results flow the other direction: the
lab produces checkpoints, logs, and plots that are synced back to the Mac.

---

## Lab Server Details

- **Host:** `ngwn06.eecs.yorku.ca`
- **User:** `maheenfs`
- **GPU:** NVIDIA GeForce RTX 5090
- **CPU:** 24 cores
- **RAM:** 62 GB
- **Home workspace:** `/home/maheenfs/wavesfm_vivor4_m2`
- **Heavy storage:** `/local/data0/maheenfs/wavesfm_vivor4_m2`
  (home has limited quota; large files go here via symlinks)

### SSH Configuration

Add to `~/.ssh/config` on Mac:

```
# Direct access (on campus or VPN)
Host ngwn06
    HostName ngwn06.eecs.yorku.ca
    User maheenfs

# If behind NAT, proxy through Indigo
Host ngwn06-via-indigo
    HostName ngwn06.eecs.yorku.ca
    User maheenfs
    ProxyJump maheenfs@indigo.eecs.yorku.ca
```

First-time setup:
```bash
ssh-copy-id ngwn06        # install SSH key
ssh ngwn06                 # test connection
```

### Python Environment on Lab

```bash
conda create -n wavesfm python=3.10 -y
conda activate wavesfm
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu124
pip install timm==1.0.11 h5py==3.15.1 DeepMIMOv3 matplotlib==3.10.7
```

### Storage Protocol

Heavy directories are symlink-backed to `/local/data0/`:
```bash
# On ngwn06:
ln -s /local/data0/maheenfs/wavesfm_vivor4_m2/datasets_raw ~/wavesfm_vivor4_m2/datasets_raw
ln -s /local/data0/maheenfs/wavesfm_vivor4_m2/datasets_h5  ~/wavesfm_vivor4_m2/datasets_h5
ln -s /local/data0/maheenfs/wavesfm_vivor4_m2/phase2_vivor4/runs ~/wavesfm_vivor4_m2/phase2_vivor4/runs
```

This keeps the home directory small while giving fast local-disk access to large
files during training.

---

## Sync Workflow

### Mac → Lab (push code and config)

```bash
rsync -avz --exclude '__pycache__' --exclude '*.pyc' --exclude 'datasets_*' \
    --exclude 'checkpoints/' --exclude 'dataset_sources/' \
    /Users/maheenfatima/Desktop/wavesfm_vivor4_m2/ \
    ngwn06:~/wavesfm_vivor4_m2/
```

### Lab → Mac (pull results)

```bash
rsync -avz ngwn06:/local/data0/maheenfs/wavesfm_vivor4_m2/phase2_vivor4/runs/ \
    /Users/maheenfatima/Desktop/wavesfm_vivor4_m2/phase2_vivor4/runs/
```

A convenience script `phase2_vivor4/scripts/sync_second_try_to_ngwn06.sh` wraps
common sync operations for specific sessions.

### Rules

- Always sync **before** starting a new lab session (push latest code)
- Always sync **after** a session completes (pull results back)
- Never edit code directly on the lab machine -- edit on Mac, then sync
- Large data (datasets, checkpoints) lives permanently on the lab machine and
  is not synced back unless specifically needed

---

## Running Experiments on the Lab

### Start a session

```bash
ssh ngwn06
tmux new -s wavesfm
conda activate wavesfm
cd ~/wavesfm_vivor4_m2
python phase2_vivor4/scripts/run_all_tasks.py  # or a specific session launch script
```

Detach with `Ctrl-B D`. Reattach with `tmux attach -t wavesfm`.

### Monitor from Mac

The harness generates a live HTML dashboard at
`phase2_vivor4/automation_logs/dashboard.html` (now archived). During active runs
this was viewable through an SSH tunnel:

```bash
ssh -L 8080:localhost:8080 ngwn06
# then open http://localhost:8080/dashboard.html on Mac
```

The dashboard tracked GPU temperature (with automatic pause/resume at high temps),
training progress, and session completion state.

---

## Software Versions

| Package      | Version  |
|--------------|----------|
| Python       | 3.10     |
| PyTorch      | 2.9.1    |
| torchvision  | 0.24.1   |
| CUDA         | 12.4     |
| timm         | 1.0.11   |
| h5py         | 3.15.1   |
| DeepMIMOv3   | latest   |
| matplotlib   | 3.10.7   |
