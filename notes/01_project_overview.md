# WavesFM Project Overview

## What WavesFM Is

WavesFM is a multimodal wireless foundation model developed by Ahmed Aboulfotouh
and Hatem Abou-Zeid at the University of Calgary. The core idea is that one shared
pretrained model can learn useful representations across many different wireless
signal types, replacing the traditional approach of building a separate model for
each task.

The architecture is straightforward:

```
Input → Modality-Specific Adapter → Shared ViT Encoder → Task Head → Output
```

- For **image-like** wireless data (spectrograms, CSI, resource grids), a vision
  patch embedding splits the input into 16x16 patches.
- For **IQ** data (raw in-phase/quadrature time series), a segment-based tokenizer
  splits the signal into length-16 segments via 1D convolution.

Both adapters produce tokens in the same 256-dimensional space, which the shared
8-block ViT encoder then processes. A lightweight linear head maps the encoder
output to task-specific predictions.

The model is pretrained using masked wireless modeling (analogous to masked
autoencoders in vision): 70% of tokens are randomly masked and the model learns
to reconstruct them. Pretraining uses 3,200 spectrogram samples and 3,200 IQ
samples over 800 epochs. After pretraining, the decoder is discarded and the
encoder is reused for downstream tasks.

**Paper:** [Multimodal Wireless Foundation Models](https://arxiv.org/abs/2511.15162)
(arXiv:2511.15162v2, Feb 2026)

**Official site:** https://wavesfm.waveslab.ai/

## Why It Matters

Wireless ML is unusually fragmented. Each subfield (modulation recognition, indoor
positioning, interference detection, beam management) typically has its own model
architecture and training pipeline. If a single pretrained backbone genuinely
transfers across all of these, it would:

- Reduce per-task engineering effort
- Show that shared structure exists across wireless problems
- Make transfer learning practical for wireless AI
- Advance the AI-native 6G vision

The importance of this claim is precisely why it needs careful verification. A
foundation model is only valuable if its results hold up independently.

## Model Configuration

The benchmark uses the **Small** variant (all official results use this):

| Parameter       | Encoder | Decoder |
|-----------------|---------|---------|
| Blocks          | 8       | 4       |
| Embed dimension | 256     | 128     |
| Hidden dimension| 1024    | 512     |
| Attention heads | 8       | 16      |
| Parameters      | 6.32M   | 0.79M   |
| Patch/segment   | 16      | --      |

This is relatively compact (~7M total params), which makes the transfer learning
claims more impressive.

## Transfer Learning Modes

The benchmark evaluates three adaptation regimes:

- **LP (Linear Probe):** Encoder frozen. Only the task head and input projections
  are trained. Tests how good the pretrained features already are.

- **FT2 (Partial Fine-Tuning):** First 6 of 8 blocks frozen; last 2 blocks +
  head + projections trainable. Allows limited feature adaptation.

- **LoRA (Low-Rank Adaptation):** Low-rank adapters (rank 32, alpha 64) inserted
  into Q/V projections of every attention block. Encoder stays frozen. Efficient
  adaptation with ~0.3M task-specific parameters (5x fewer than FT2).

Note: the paper reports LoRA alpha=32, but the current website/code uses alpha=64.
This reproduction follows the current release (alpha=64).

## Downstream Tasks

The benchmark spans 11 task IDs across two modality families:

### Classification Tasks (metric: per-class accuracy / PCA)

| Task ID         | Dataset         | Classes | Modality | What it does                    |
|-----------------|-----------------|---------|----------|---------------------------------|
| `sensing`       | EfficientFi HAS | 6       | Vision   | WiFi CSI human activity sensing |
| `rfs`           | CommRad RF      | 20      | Vision   | RF signal type classification   |
| `radcom`        | RadCom OTA      | 9       | IQ       | Modulation/signal classification|
| `rml`           | RadioML 2022    | 11      | IQ       | Automatic modulation recognition|
| `rfp`           | POWDER RFP      | 4       | IQ       | RF device fingerprinting        |
| `interf`        | ICARUS          | 3       | IQ       | Interference detection/class.   |
| `deepmimo-los`  | DeepMIMO        | 2       | Vision   | LoS/NLoS classification         |
| `deepmimo-beam` | DeepMIMO        | 64      | Vision   | Beam index prediction           |

### Regression Tasks (metric: mean distance error in meters)

| Task ID           | Dataset             | Modality | What it does                 |
|-------------------|---------------------|----------|------------------------------|
| `pos`             | 5G NR Positioning   | Vision   | UE location from CSI         |
| `uwb-indoor`      | UWB Indoor          | IQ       | Indoor positioning from CIR  |
| `uwb-industrial`  | UWB Industrial      | IQ       | Industrial localization      |

### Metric Definitions

**PCA (Per-Class Accuracy):** Mean of per-class recall values. Treats every class
equally regardless of sample count. This is the primary metric for all
classification tasks and the basis for best-checkpoint selection during training.

**Mean Distance Error:** Average Euclidean distance between predicted and true
coordinates (in meters, after denormalization). The code also reports median,
p75, and p90 errors.

### Public Website vs. Local Task IDs

The website splits some tasks into subtask rows:
- ICARUS → INTD (detection) + INTC (classification)
- RADCOM → signal-type accuracy + modulation accuracy

Locally these are unified as `interf` and `radcom`, with auxiliary metrics
(`det_acc`, `mod_acc`, `sig_acc`) logged alongside the primary PCA.

## This Repository

This repo is **not** the full pretraining pipeline. It contains:
- Preprocessing scripts that convert raw datasets into HDF5 caches
- The pretrained checkpoint (`wavesfm-v1p0.pth`)
- Fine-tuning and evaluation code for downstream tasks
- A local benchmark harness for systematic reproduction

In other words, this is a **benchmark and transfer-learning** repo.
