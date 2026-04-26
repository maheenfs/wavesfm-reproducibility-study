from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import textwrap
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str((Path(__file__).resolve().parents[2] / "phase2_vivor4" / "automation_logs" / ".mplconfig").resolve()),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str((Path(__file__).resolve().parents[2] / "phase2_vivor4" / "automation_logs" / ".cache").resolve()),
)

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors as mcolors
from torch.utils.data import DataLoader, Subset

from benchmark_config import LOCAL_RESULTS_ROOT, OFFICIAL_RESULTS_ROOT, PHASE2_ROOT, TASK_SPECS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT / "wavesfm"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import build_datasets  # noqa: E402
from lora import create_lora_model  # noqa: E402
from main_finetune import _state_has_lora_keys, build_model, parse_args as parse_train_args  # noqa: E402
from utils import trim_blocks  # noqa: E402


CHECKPOINT_RE = re.compile(r"checkpoint_(\d+)\.pth$")
PLOTS_ROOT = PHASE2_ROOT / "plots" / "detailed_eval"
OFFICIAL_JSON = OFFICIAL_RESULTS_ROOT / "official_results_all.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate local detailed-evaluation plots from completed WavesFM runs.")
    p.add_argument("--results-root", type=Path, default=LOCAL_RESULTS_ROOT)
    p.add_argument("--output-root", type=Path, default=PLOTS_ROOT)
    p.add_argument("--official-json", type=Path, default=OFFICIAL_JSON)
    p.add_argument("--tasks", nargs="+", default=sorted(TASK_SPECS.keys()))
    p.add_argument("--modes", nargs="+", default=["lp", "ft2", "lora"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument(
        "--selection",
        choices=["best-per-mode", "all-completed"],
        default="best-per-mode",
        help="Choose one best completed seed per task/mode, or emit plots for every completed seed.",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def temporary_argv(argv: list[str]):
    old = sys.argv[:]
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = old


def latest_logged_epoch(log_path: Path) -> int | None:
    if not log_path.exists():
        return None
    for line in reversed(log_path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        epoch = payload.get("epoch")
        if isinstance(epoch, int):
            return epoch
    return None


def latest_epoch_checkpoint(run_dir: Path) -> Path | None:
    checkpoints: list[tuple[int, Path]] = []
    for path in run_dir.glob("checkpoint_*.pth"):
        match = CHECKPOINT_RE.fullmatch(path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda item: item[0])
    return checkpoints[-1][1]


def pick_eval_checkpoint(run_dir: Path) -> Path:
    best = run_dir / "best.pth"
    if best.exists():
        return best
    latest = latest_epoch_checkpoint(run_dir)
    if latest is None:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")
    return latest


def run_completed(meta: dict, run_dir: Path) -> bool:
    expected_epochs = int(meta["expected_epochs"])
    latest_epoch = latest_logged_epoch(run_dir / "log.txt")
    if latest_epoch is not None and latest_epoch + 1 >= expected_epochs:
        return True
    return (run_dir / f"checkpoint_{expected_epochs - 1:03d}.pth").exists()


def best_entry(log_path: Path) -> dict | None:
    if not log_path.exists():
        return None
    last_best = None
    best = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        cur = payload.get("best_metric")
        if cur is None:
            continue
        if last_best is None or cur != last_best:
            last_best = cur
            best = payload
    return best


def parse_train_namespace(command: list[str]):
    argv = ["main_finetune.py"]
    if len(command) >= 3:
        argv.extend(command[2:])
    with temporary_argv(argv):
        args = parse_train_args()
    return args


def higher_is_better(task: str) -> bool:
    return TASK_SPECS[task]["task_type"] == "classification"


def load_official_index(path: Path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks", {})
    rows = {}
    for task, info in tasks.items():
        primary_metric = info["primary_metric"]
        for mode, metrics in info["modes"].items():
            rows[(task, mode)] = {
                "task_name": info.get("task_name"),
                "dataset_name": info.get("dataset_name"),
                "task_type": info.get("task_type"),
                "primary_metric": primary_metric,
                "official_value": metrics.get(primary_metric),
                "reported_split": info.get("reported_split"),
                "metrics": metrics,
            }
    return rows


def _unwrap_subset_with_indices(ds):
    if not isinstance(ds, Subset):
        return ds, None
    indices = list(ds.indices)
    base = ds.dataset
    while isinstance(base, Subset):
        indices = [base.indices[i] for i in indices]
        base = base.dataset
    return base, np.asarray(indices, dtype=np.int64)


def load_label_names(ds, num_outputs: int) -> list[str]:
    base, _ = _unwrap_subset_with_indices(ds)
    labels = getattr(base, "labels", None)
    if labels:
        return [str(x) for x in labels]
    h5_path = getattr(base, "h5_path", None)
    if h5_path:
        with h5py.File(h5_path, "r") as h5:
            raw = h5.attrs.get("labels", None)
            if raw:
                return list(json.loads(raw))
            raw = h5.attrs.get("labels_los", None)
            if raw and num_outputs == 2:
                return list(json.loads(raw))
    return [str(i) for i in range(num_outputs)]


def load_model(eval_args, task_info, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = build_model(eval_args, task_info)
    if getattr(eval_args, "trim_blocks", None) is not None:
        model = trim_blocks(model, eval_args.trim_blocks)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    if getattr(eval_args, "lora", False) or _state_has_lora_keys(state):
        model = create_lora_model(model, lora_rank=eval_args.lora_rank, lora_alpha=eval_args.lora_alpha)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def metric_caption(task: str, mode: str, selected_label: str, local_values: list[float], official: dict | None) -> str:
    primary_metric = str(TASK_SPECS[task]["primary_metric"])
    mean_val = statistics.fmean(local_values)
    std_val = statistics.pstdev(local_values) if len(local_values) > 1 else 0.0
    caption = f"Local {primary_metric}: {mean_val:.4f}"
    if len(local_values) > 1:
        caption += f" +/- {std_val:.4f}"
    caption += f" across {len(local_values)} seed(s); plotted {selected_label}"
    if official and official.get("official_value") is not None:
        caption += f" | Official {primary_metric}: {float(official['official_value']):.4f}"
    return caption


def display_mode(mode: str) -> str:
    return {"lp": "LP", "ft2": "FT2", "lora": "LoRA"}.get(mode, mode.upper())


def format_percent(value: float | None) -> str | None:
    if value is None:
        return None
    text = f"{float(value):.2f}".rstrip("0").rstrip(".")
    return f"{text}%"


def local_eval_split_label(train_args) -> str:
    return "provided eval file" if getattr(train_args, "val_path", None) else "validation split"


def build_experiment_info(meta: dict, train_args, *, full_train_size: int, eval_size: int, task_type: str) -> dict:
    requested_fraction = meta.get("train_subset_fraction")
    requested_fraction = float(requested_fraction) if requested_fraction is not None else None
    requested_size = meta.get("train_subset_size")
    requested_size = int(requested_size) if requested_size is not None else None

    if requested_fraction is not None:
        train_size_used = max(1, int(round(full_train_size * requested_fraction)))
    elif requested_size is not None:
        train_size_used = min(requested_size, full_train_size)
    else:
        train_size_used = full_train_size

    subset_applied = train_size_used < full_train_size
    subset_percent = (100.0 * train_size_used / full_train_size) if full_train_size > 0 else None
    subset_percent_label = format_percent(subset_percent)
    eval_label = local_eval_split_label(train_args)
    sampling_label = (
        "stratified subset after split"
        if subset_applied and task_type == "classification"
        else "random subset after split"
        if subset_applied
        else "full post-split train set"
    )
    train_text = (
        f"{train_size_used:,} / {full_train_size:,} ({subset_percent_label})"
        if subset_percent_label is not None
        else f"{train_size_used:,} / {full_train_size:,}"
    )
    label = f"train subset {subset_percent_label}" if subset_applied and subset_percent_label else "full training split"
    short_badge = f"{subset_percent_label} subset" if subset_applied and subset_percent_label else "full train"
    return {
        "kind": "train_subset_study" if subset_applied else "full_data_session",
        "label": label,
        "short_badge": short_badge,
        "subset_applied": subset_applied,
        "train_subset_fraction": requested_fraction,
        "train_subset_percent": subset_percent,
        "train_subset_size": train_size_used if subset_applied else None,
        "train_split_size": full_train_size,
        "train_size_used": train_size_used,
        "train_text": train_text,
        "eval_size": eval_size,
        "eval_label": eval_label,
        "sampling_label": sampling_label,
    }


def apply_header(fig, title: str, subtitle: str) -> None:
    wrapped = textwrap.wrap(subtitle, width=90) or [subtitle]
    fig.suptitle(title, x=0.08, y=0.975, ha="left", fontsize=13, fontweight="bold")
    top = 0.93
    for idx, line in enumerate(wrapped[:3]):
        fig.text(0.08, top - (idx * 0.032), line, ha="left", va="top", fontsize=9, color="#43505a")
    fig.subplots_adjust(top=0.82 - max(0, len(wrapped[:3]) - 1) * 0.03)


def truncated_cmap(name: str, start: float, stop: float, n: int = 256):
    base = plt.get_cmap(name)
    colors = base(np.linspace(start, stop, n))
    return mcolors.LinearSegmentedColormap.from_list(f"{name}_trunc", colors)


def muted_confusion_cmap():
    colors = ["#fbfdfd", "#e8f6f6", "#cdebec", "#96d6dc", "#4db7c7", "#2b83b5", "#1f5b9c", "#173a74"]
    return mcolors.LinearSegmentedColormap.from_list("muted_confusion", colors)


def draw_badges(fig, badges: list[str]) -> None:
    x = 0.975
    y = 0.968
    for badge in reversed([str(b).strip() for b in badges if str(b).strip()]):
        width = 0.018 + 0.0105 * len(badge)
        x -= width
        fig.text(
            x,
            y,
            badge,
            ha="left",
            va="top",
            fontsize=8,
            color="#4b5965",
            bbox=dict(boxstyle="round,pad=0.32,rounding_size=0.9", fc="#eef3f4", ec="#d7e1e3", lw=0.8),
        )
        x -= 0.008


def mean_metric(rows_by_mode: list[dict], key: str) -> float | None:
    values = []
    for row in rows_by_mode:
        val = row.get("best_entry", {}).get("val", {}).get(key)
        if val is not None:
            values.append(float(val))
    if not values:
        return None
    return statistics.fmean(values)


def format_metric_row(prefix: str, metrics: list[tuple[str, float | int | None]]) -> str:
    parts = []
    for label, value in metrics:
        if value is None:
            continue
        if isinstance(value, str):
            parts.append(f"{label}: {value}")
            continue
        if isinstance(value, int):
            parts.append(f"{label}: {value}")
        else:
            parts.append(f"{label}: {float(value):.2f}")
    return f"{prefix}  " + "    ".join(parts)


def build_run_setup_row(experiment: dict) -> str:
    return (
        "Run setup  "
        f"Train split: {experiment['train_text']}    "
        f"Eval split: {experiment['eval_size']:,} ({experiment['eval_label']})    "
        f"Sampling: {experiment['sampling_label']}"
    )


def build_classification_header(
    task: str,
    mode: str,
    plot_label: str,
    rows_by_mode: list[dict],
    official: dict | None,
    experiment: dict,
) -> dict:
    local_acc1 = mean_metric(rows_by_mode, "acc1")
    local_acc3 = mean_metric(rows_by_mode, "acc3")
    local_pca = mean_metric(rows_by_mode, "pca")
    official_metrics = (official or {}).get("metrics", {})

    dataset_name = (official or {}).get("dataset_name") or TASK_SPECS[task]["display_name"]
    task_name = (official or {}).get("task_name") or TASK_SPECS[task]["display_name"]
    task_type = (official or {}).get("task_type") or TASK_SPECS[task]["task_type"]
    reported_split = (official or {}).get("reported_split")

    local_row = format_metric_row(
        "Local",
        [
            ("Acc@1", local_acc1),
            ("Acc@3", local_acc3),
            ("PCA (mean per-class accuracy)", local_pca),
        ],
    )

    setup_row = build_run_setup_row(experiment)

    official_row = None
    if official_metrics:
        official_row = format_metric_row(
            "Official",
            [
                ("Acc@1", official_metrics.get("acc1")),
                ("Acc@3", official_metrics.get("acc3")),
                ("PCA (mean per-class accuracy)", official_metrics.get("pca")),
                (
                    "Reported samples",
                    (
                        f"{int(official_metrics['test_samples'])} ({reported_split})"
                        if official_metrics.get("test_samples") is not None and reported_split
                        else official_metrics.get("test_samples")
                    ),
                ),
            ],
        )

    return {
        "title": f"{dataset_name} ({display_mode(mode)})",
        "subtitle": task_name,
        "badges": [display_mode(mode), task_type, plot_label, experiment["short_badge"]],
        "protocol": reported_split,
        "rows": [line for line in (local_row, setup_row, official_row) if line],
    }


def build_position_header(
    task: str,
    mode: str,
    plot_label: str,
    rows_by_mode: list[dict],
    official: dict | None,
    experiment: dict,
) -> dict:
    local_mean = mean_metric(rows_by_mode, "mean_error")
    local_median = mean_metric(rows_by_mode, "median_error")
    local_p90 = mean_metric(rows_by_mode, "p90_error")
    official_metrics = (official or {}).get("metrics", {})

    dataset_name = (official or {}).get("dataset_name") or TASK_SPECS[task]["display_name"]
    task_name = (official or {}).get("task_name") or TASK_SPECS[task]["display_name"]
    task_type = (official or {}).get("task_type") or TASK_SPECS[task]["task_type"]
    reported_split = (official or {}).get("reported_split")

    local_row = format_metric_row(
        "Local",
        [
            ("Mean error", local_mean),
            ("Median error", local_median),
            ("P90 error", local_p90),
        ],
    )

    setup_row = build_run_setup_row(experiment)

    official_row = None
    if official_metrics:
        official_row = format_metric_row(
            "Official",
            [
                ("Mean error", official_metrics.get("mean_error")),
                ("Median error", official_metrics.get("median_error")),
                ("P90 error", official_metrics.get("p90_error")),
                (
                    "Reported samples",
                    (
                        f"{int(official_metrics['test_samples'])} ({reported_split})"
                        if official_metrics.get("test_samples") is not None and reported_split
                        else official_metrics.get("test_samples")
                    ),
                ),
            ],
        )

    return {
        "title": f"{dataset_name} ({display_mode(mode)})",
        "subtitle": task_name,
        "badges": [display_mode(mode), task_type, plot_label, experiment["short_badge"]],
        "protocol": reported_split,
        "rows": [line for line in (local_row, setup_row, official_row) if line],
    }


def apply_official_header(fig, header: dict) -> None:
    draw_badges(fig, header.get("badges", []))
    fig.suptitle(header["title"], x=0.08, y=0.975, ha="left", fontsize=15, fontweight="bold", color="#17222b")
    fig.text(0.08, 0.943, header["subtitle"], ha="left", va="top", fontsize=9.5, color="#6e7b86")
    y = 0.913
    if header.get("protocol"):
        fig.text(0.08, y, "Dataset protocol", ha="left", va="top", fontsize=8.6, color="#2f6e9f")
        y -= 0.029
    for idx, row in enumerate(header.get("rows", [])):
        color = "#28333c" if idx == 0 else "#55636e"
        fig.text(0.08, y, row, ha="left", va="top", fontsize=8.6, color=color)
        y -= 0.029
    top = max(0.73, y - 0.015)
    fig.subplots_adjust(top=top)


def iter_completed_runs(results_root: Path, tasks: list[str], modes: list[str], seeds: list[int]) -> list[dict]:
    rows = []
    for task in tasks:
        for mode in modes:
            for seed in seeds:
                run_dir = results_root / task / mode / f"s{seed}"
                meta_path = run_dir / "metadata.json"
                if not meta_path.exists():
                    continue
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if not run_completed(meta, run_dir):
                    continue
                best = best_entry(run_dir / "log.txt")
                if best is None:
                    continue
                rows.append(
                    {
                        "task": task,
                        "mode": mode,
                        "seed": seed,
                        "run_dir": run_dir,
                        "metadata": meta,
                        "best_entry": best,
                        "best_metric": float(best["best_metric"]),
                    }
                )
    return rows


def select_runs(run_rows: list[dict], selection: str) -> list[dict]:
    if selection == "all-completed":
        return sorted(run_rows, key=lambda item: (item["task"], item["mode"], item["seed"]))

    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in run_rows:
        grouped.setdefault((row["task"], row["mode"]), []).append(row)

    selected = []
    for (task, mode), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda item: item["seed"])
        key_fn = (lambda item: item["best_metric"]) if higher_is_better(task) else (lambda item: -item["best_metric"])
        selected.append(max(rows, key=key_fn))
    return selected


def plot_confusion_matrix(
    task: str,
    mode: str,
    plot_label: str,
    out_prefix: Path,
    caption: str,
    conf: np.ndarray,
    conf_norm: np.ndarray,
    labels: list[str],
    header: dict | None = None,
) -> list[str]:
    num_classes = conf.shape[0]
    fig_size = max(6.0, min(18.0, 3.0 + 0.45 * num_classes))
    annotate = num_classes <= 20
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    cmap = muted_confusion_cmap()
    norm = mcolors.PowerNorm(gamma=0.6, vmin=0.0, vmax=1.0)
    im = ax.imshow(conf_norm, cmap=cmap, norm=norm)
    if header is None:
        apply_header(fig, f"{TASK_SPECS[task]['display_name']} | {mode} | {plot_label}", caption)
    else:
        apply_official_header(fig, header)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    numeric_ticks = [str(i) for i in range(num_classes)]
    ax.set_xticklabels(numeric_ticks, fontsize=8)
    ax.set_yticklabels(numeric_ticks, fontsize=8)
    ax.set_xticks(np.arange(-0.5, num_classes, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, num_classes, 1), minor=True)
    ax.grid(which="minor", color="#fffaf3", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    if annotate:
        font_size = 8 if num_classes <= 10 else 7 if num_classes <= 15 else 6
        for i in range(num_classes):
            for j in range(num_classes):
                value = conf_norm[i, j]
                text = f"{value:.2f}"
                color = "white" if norm(value) >= 0.62 else "#152536"
                ax.text(j, i, text, ha="center", va="center", fontsize=font_size, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    outputs = []
    for suffix in (".png", ".svg"):
        path = out_prefix.with_suffix(suffix)
        fig.savefig(path, dpi=220, bbox_inches="tight")
        outputs.append(str(path))
    np.savez_compressed(out_prefix.with_suffix(".npz"), confusion=conf, confusion_norm=conf_norm, labels=np.asarray(labels))
    outputs.append(str(out_prefix.with_suffix(".npz")))
    label_map_path = out_prefix.with_name(out_prefix.name + "_label_map.json")
    label_map = {str(i): str(label) for i, label in enumerate(labels)}
    label_map_path.write_text(json.dumps(label_map, indent=2) + "\n", encoding="utf-8")
    outputs.append(str(label_map_path))
    plt.close(fig)
    return outputs


def collect_position_errors(
    model: torch.nn.Module,
    val_ds,
    task_info,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    coord_min = task_info.coord_min.to(device)
    coord_max = task_info.coord_max.to(device)

    def denorm(x: torch.Tensor) -> torch.Tensor:
        return (x + 1) * 0.5 * (coord_max - coord_min) + coord_min

    errors = []
    with torch.no_grad():
        for batch in loader:
            samples = batch[0].to(device)
            targets = batch[1].to(device)
            outputs = model(samples)
            pred = denorm(outputs)
            true = denorm(targets)
            dist = torch.linalg.norm(pred - true, dim=-1)
            errors.append(dist.detach().cpu().numpy())
    if not errors:
        return np.empty((0,), dtype=np.float64)
    return np.concatenate(errors, axis=0).astype(np.float64)


def plot_position_error_distribution(
    task: str,
    mode: str,
    plot_label: str,
    out_prefix: Path,
    caption: str,
    values: np.ndarray,
    header: dict | None = None,
    *,
    bins: int | None = None,
) -> list[str]:
    if values.size == 0:
        return []
    bins = bins if bins is not None else min(60, max(20, int(np.sqrt(values.size))))
    hist, edges = np.histogram(values, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean_val = float(values.mean())
    median_val = float(np.median(values))
    p90 = float(np.quantile(values, 0.90))

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(centers, hist, color="#2f6ba8", linewidth=2.2)
    ax.fill_between(centers, hist, color="#67a6df", alpha=0.35)
    ax.axvline(mean_val, color="#d34d43", linestyle="--", linewidth=1.8)
    ax.text(
        0.98,
        0.95,
        f"mean={mean_val:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        color="#d34d43",
    )
    if header is None:
        apply_header(fig, f"{TASK_SPECS[task]['display_name']} | {mode} | {plot_label}", caption)
    else:
        apply_official_header(fig, header)
    ax.set_xlabel("Position error distance")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.22)

    outputs = []
    for suffix in (".png", ".svg"):
        path = out_prefix.with_suffix(suffix)
        fig.savefig(path, dpi=220, bbox_inches="tight")
        outputs.append(str(path))
    np.savez_compressed(
        out_prefix.with_suffix(".npz"),
        errors=values,
        mean_error=mean_val,
        median_error=median_val,
        p90_error=p90,
    )
    outputs.append(str(out_prefix.with_suffix(".npz")))
    plt.close(fig)
    return outputs


def compute_confusion_from_arrays(
    targets: np.ndarray,
    preds: np.ndarray,
    num_outputs: int,
) -> tuple[np.ndarray, np.ndarray]:
    conf = np.zeros((num_outputs, num_outputs), dtype=np.int64)
    for target, pred in zip(targets.tolist(), preds.tolist()):
        conf[int(target), int(pred)] += 1
    row_sums = conf.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        conf_norm = np.divide(conf, row_sums, out=np.zeros_like(conf, dtype=float), where=row_sums != 0)
    return conf, conf_norm


def compute_confusion_stats(
    model: torch.nn.Module,
    val_ds,
    num_outputs: int,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    targets_all = []
    preds_all = []
    with torch.no_grad():
        for batch in loader:
            samples = batch[0].to(device)
            targets = batch[1].to(device).long()
            outputs = model(samples)
            preds = outputs.argmax(dim=1)
            targets_all.append(targets.cpu().numpy())
            preds_all.append(preds.cpu().numpy())
    targets_np = np.concatenate(targets_all, axis=0) if targets_all else np.empty((0,), dtype=np.int64)
    preds_np = np.concatenate(preds_all, axis=0) if preds_all else np.empty((0,), dtype=np.int64)
    conf, conf_norm = compute_confusion_from_arrays(targets_np, preds_np, num_outputs)
    labels = load_label_names(val_ds, num_outputs)
    return conf, conf_norm, labels


def collect_rml_outputs(
    model: torch.nn.Module,
    val_ds,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    base, idxs = _unwrap_subset_with_indices(val_ds)
    snr_by_index = getattr(base, "snr_by_index", None)
    if snr_by_index is None:
        raise ValueError("RML plotting requires snr_by_index metadata.")

    snrs = np.asarray(snr_by_index, dtype=np.int16)
    if idxs is not None:
        snrs = snrs[idxs]

    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    offset = 0
    all_targets = []
    all_preds = []
    all_snrs = []
    with torch.no_grad():
        for batch in loader:
            samples = batch[0].to(device)
            targets = batch[1].to(device).long()
            outputs = model(samples)
            preds = outputs.argmax(dim=1)
            batch_snrs = snrs[offset : offset + len(targets)]
            offset += len(targets)
            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_snrs.append(np.asarray(batch_snrs, dtype=np.int16))

    targets = np.concatenate(all_targets, axis=0) if all_targets else np.empty((0,), dtype=np.int64)
    preds = np.concatenate(all_preds, axis=0) if all_preds else np.empty((0,), dtype=np.int64)
    snrs = np.concatenate(all_snrs, axis=0) if all_snrs else np.empty((0,), dtype=np.int16)
    num_outputs = int(max(np.max(targets, initial=0), np.max(preds, initial=0)) + 1) if targets.size else 0
    labels = load_label_names(val_ds, max(1, num_outputs))
    return targets, preds, snrs, labels


def accuracy_by_snr(
    targets: np.ndarray,
    preds: np.ndarray,
    snrs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    levels = sorted({int(x) for x in snrs.tolist()})
    acc = []
    for level in levels:
        mask = snrs == level
        if not np.any(mask):
            continue
        acc.append(float(np.mean(preds[mask] == targets[mask])))
    return np.asarray(levels, dtype=np.int16), np.asarray(acc, dtype=np.float64)


def pick_selected_snrs(snrs: np.ndarray, limit: int = 2) -> list[int]:
    available = sorted({int(x) for x in snrs.tolist()})
    if not available:
        return []
    selected: list[int] = []
    for preferred in (0, 20):
        if preferred in available and preferred not in selected:
            selected.append(preferred)
    if len(selected) < limit:
        fallback = [available[len(available) // 2], available[-1], available[0]]
        for value in fallback:
            if value not in selected:
                selected.append(value)
            if len(selected) >= limit:
                break
    return selected[:limit]


def draw_confusion_axis(ax, conf_norm: np.ndarray, *, title: str | None = None) -> None:
    num_classes = conf_norm.shape[0]
    cmap = muted_confusion_cmap()
    norm = mcolors.PowerNorm(gamma=0.6, vmin=0.0, vmax=1.0)
    im = ax.imshow(conf_norm, cmap=cmap, norm=norm)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    tick_labels = [str(i) for i in range(num_classes)]
    ax.set_xticklabels(tick_labels, fontsize=7)
    ax.set_yticklabels(tick_labels, fontsize=7)
    ax.set_xticks(np.arange(-0.5, num_classes, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, num_classes, 1), minor=True)
    ax.grid(which="minor", color="#fffaf3", linestyle="-", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    if title:
        ax.set_title(title, fontsize=8.5, color="#55636e", pad=4)
    if num_classes <= 20:
        font_size = 7 if num_classes <= 11 else 6
        for i in range(num_classes):
            for j in range(num_classes):
                value = conf_norm[i, j]
                color = "white" if norm(value) >= 0.62 else "#152536"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=font_size, color=color)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_rml_overview(
    task: str,
    mode: str,
    plot_label: str,
    out_prefix: Path,
    caption: str,
    levels: np.ndarray,
    acc: np.ndarray,
    selected_confusions: list[tuple[int, np.ndarray, np.ndarray]],
    labels: list[str],
    header: dict | None = None,
) -> list[str]:
    fig = plt.figure(figsize=(10.6, 9.4))
    if header is None:
        apply_header(fig, f"{TASK_SPECS[task]['display_name']} | {mode} | {plot_label}", caption)
    else:
        apply_official_header(fig, header)

    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.42, wspace=0.28)
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.plot(levels, acc, marker="o", color="#3b78b3", linewidth=2.0, markersize=4.5)
    ax_top.set_ylim(0.0, 1.0)
    ax_top.grid(True, alpha=0.18)
    ax_top.set_xlabel("SNR")
    ax_top.set_ylabel("Accuracy")

    fig.text(0.08, 0.47, "Confusion matrices at selected SNRs", ha="left", va="bottom", fontsize=9, color="#55636e")
    axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    payload = {}
    for ax, item in zip(axes, selected_confusions):
        snr_value, conf, conf_norm = item
        draw_confusion_axis(ax, conf_norm, title=f"SNR {snr_value}")
        payload[f"confusion_snr_{snr_value}"] = conf
        payload[f"confusion_norm_snr_{snr_value}"] = conf_norm
    for ax in axes[len(selected_confusions) :]:
        ax.axis("off")

    outputs = []
    for suffix in (".png", ".svg"):
        path = out_prefix.with_suffix(suffix)
        fig.savefig(path, dpi=220, bbox_inches="tight")
        outputs.append(str(path))
    np.savez_compressed(
        out_prefix.with_suffix(".npz"),
        snr=np.asarray(levels),
        accuracy=acc,
        selected_snr=np.asarray([item[0] for item in selected_confusions], dtype=np.int16),
        labels=np.asarray(labels),
        **payload,
    )
    outputs.append(str(out_prefix.with_suffix(".npz")))
    label_map_path = out_prefix.with_name(out_prefix.name + "_label_map.json")
    label_map = {str(i): str(label) for i, label in enumerate(labels)}
    label_map_path.write_text(json.dumps(label_map, indent=2) + "\n", encoding="utf-8")
    outputs.append(str(label_map_path))
    plt.close(fig)
    return outputs


def generate_plots_for_run(
    row: dict,
    rows_by_mode: list[dict],
    official_index: dict[tuple[str, str], dict],
    output_root: Path,
    *,
    num_workers: int,
    device: torch.device,
    overwrite: bool,
) -> dict:
    task = row["task"]
    mode = row["mode"]
    seed = row["seed"]
    meta = row["metadata"]
    train_args = parse_train_namespace(meta["command"])
    local_values = [float(item["best_metric"]) for item in rows_by_mode]
    official = official_index.get((task, mode))

    plot_dir = output_root / task / mode
    ensure_dir(plot_dir)

    manifests = []
    checkpoint = pick_eval_checkpoint(row["run_dir"])
    train_args.device = str(device)
    train_args.num_workers = num_workers

    train_ds, val_ds, task_info = build_datasets(
        train_args.task,
        train_args.train_path,
        val_path=train_args.val_path,
        val_split=train_args.val_split,
        stratified_split=train_args.stratified_split,
        seed=train_args.seed,
        deepmimo_n_beams=train_args.deepmimo_n_beams,
    )
    experiment = build_experiment_info(
        meta,
        train_args,
        full_train_size=len(train_ds),
        eval_size=len(val_ds),
        task_type=task_info.target_type,
    )
    del train_ds
    model = load_model(train_args, task_info, checkpoint, device)

    if task_info.target_type == "classification":
        if task == "rml":
            plot_label = "all_seeds" if len(rows_by_mode) > 1 else f"s{seed}"
            header = build_classification_header(task, mode, plot_label, rows_by_mode, official, experiment)
            caption = metric_caption(task, mode, plot_label, local_values, official)
            prefix = plot_dir / f"{plot_label}_rml_overview"
            if overwrite or not prefix.with_suffix(".png").exists():
                parts_targets = []
                parts_preds = []
                parts_snrs = []
                labels = None
                source_rows = rows_by_mode if len(rows_by_mode) > 1 else [row]
                for seed_row in source_rows:
                    seed_meta = seed_row["metadata"]
                    seed_args = parse_train_namespace(seed_meta["command"])
                    seed_args.device = str(device)
                    seed_args.num_workers = num_workers
                    seed_train_ds, seed_val_ds, seed_task_info = build_datasets(
                        seed_args.task,
                        seed_args.train_path,
                        val_path=seed_args.val_path,
                        val_split=seed_args.val_split,
                        stratified_split=seed_args.stratified_split,
                        seed=seed_args.seed,
                        deepmimo_n_beams=seed_args.deepmimo_n_beams,
                    )
                    del seed_train_ds
                    seed_model = load_model(seed_args, seed_task_info, pick_eval_checkpoint(seed_row["run_dir"]), device)
                    targets_np, preds_np, snrs_np, labels = collect_rml_outputs(
                        seed_model,
                        seed_val_ds,
                        batch_size=int(seed_args.batch_size),
                        num_workers=num_workers,
                        device=device,
                    )
                    parts_targets.append(targets_np)
                    parts_preds.append(preds_np)
                    parts_snrs.append(snrs_np)
                targets_np = np.concatenate(parts_targets, axis=0)
                preds_np = np.concatenate(parts_preds, axis=0)
                snrs_np = np.concatenate(parts_snrs, axis=0)
                levels, acc = accuracy_by_snr(targets_np, preds_np, snrs_np)
                num_outputs = len(labels or [])
                selected_confusions = []
                for snr_value in pick_selected_snrs(snrs_np):
                    mask = snrs_np == snr_value
                    conf, conf_norm = compute_confusion_from_arrays(targets_np[mask], preds_np[mask], num_outputs)
                    selected_confusions.append((snr_value, conf, conf_norm))
                paths = plot_rml_overview(
                    task,
                    mode,
                    plot_label,
                    prefix,
                    caption,
                    levels,
                    acc,
                    selected_confusions,
                    labels or [],
                    header=header,
                )
                if paths:
                    manifests.append({"kind": "rml_overview", "paths": paths})
        else:
            header = build_classification_header(
                task,
                mode,
                "all_seeds" if len(rows_by_mode) > 1 else f"s{seed}",
                rows_by_mode,
                official,
                experiment,
            )
            if len(rows_by_mode) > 1:
                confs = []
                conf_norms = []
                labels = None
                for seed_row in rows_by_mode:
                    seed_meta = seed_row["metadata"]
                    seed_args = parse_train_namespace(seed_meta["command"])
                    seed_args.device = str(device)
                    seed_args.num_workers = num_workers
                    seed_train_ds, seed_val_ds, seed_task_info = build_datasets(
                        seed_args.task,
                        seed_args.train_path,
                        val_path=seed_args.val_path,
                        val_split=seed_args.val_split,
                        stratified_split=seed_args.stratified_split,
                        seed=seed_args.seed,
                        deepmimo_n_beams=seed_args.deepmimo_n_beams,
                    )
                    del seed_train_ds
                    seed_model = load_model(seed_args, seed_task_info, pick_eval_checkpoint(seed_row["run_dir"]), device)
                    conf, conf_norm, labels = compute_confusion_stats(
                        seed_model,
                        seed_val_ds,
                        seed_task_info.num_outputs,
                        batch_size=int(seed_args.batch_size),
                        num_workers=num_workers,
                        device=device,
                    )
                    confs.append(conf)
                    conf_norms.append(conf_norm)
                plot_label = "all_seeds"
                caption = metric_caption(task, mode, plot_label, local_values, official)
                prefix = plot_dir / f"{plot_label}_confusion_matrix"
                conf = np.sum(confs, axis=0)
                conf_norm = np.mean(conf_norms, axis=0)
                assert labels is not None
            else:
                plot_label = f"s{seed}"
                caption = metric_caption(task, mode, plot_label, local_values, official)
                prefix = plot_dir / f"{plot_label}_confusion_matrix"
                conf, conf_norm, labels = compute_confusion_stats(
                    model,
                    val_ds,
                    task_info.num_outputs,
                    batch_size=int(train_args.batch_size),
                    num_workers=num_workers,
                    device=device,
                )
            if overwrite or not prefix.with_suffix(".png").exists():
                paths = plot_confusion_matrix(
                    task,
                    mode,
                    plot_label,
                    prefix,
                    caption,
                    conf,
                    conf_norm,
                    labels,
                    header=header,
                )
                manifests.append({"kind": "confusion_matrix", "paths": paths})
    elif task_info.target_type == "position":
        plot_label = "all_seeds" if len(rows_by_mode) > 1 else f"s{seed}"
        header = build_position_header(task, mode, plot_label, rows_by_mode, official, experiment)
        caption = metric_caption(task, mode, plot_label, local_values, official)
        prefix = plot_dir / f"{plot_label}_position_error_density"
        if overwrite or not prefix.with_suffix(".png").exists():
            if len(rows_by_mode) > 1:
                errors_parts = []
                for seed_row in rows_by_mode:
                    seed_meta = seed_row["metadata"]
                    seed_args = parse_train_namespace(seed_meta["command"])
                    seed_args.device = str(device)
                    seed_args.num_workers = num_workers
                    seed_train_ds, seed_val_ds, seed_task_info = build_datasets(
                        seed_args.task,
                        seed_args.train_path,
                        val_path=seed_args.val_path,
                        val_split=seed_args.val_split,
                        stratified_split=seed_args.stratified_split,
                        seed=seed_args.seed,
                        deepmimo_n_beams=seed_args.deepmimo_n_beams,
                    )
                    del seed_train_ds
                    seed_model = load_model(seed_args, seed_task_info, pick_eval_checkpoint(seed_row["run_dir"]), device)
                    errors_parts.append(
                        collect_position_errors(
                            seed_model,
                            seed_val_ds,
                            seed_task_info,
                            batch_size=int(seed_args.batch_size),
                            num_workers=num_workers,
                            device=device,
                        )
                    )
                values = np.concatenate(errors_parts, axis=0) if errors_parts else np.empty((0,), dtype=np.float64)
            else:
                values = collect_position_errors(
                    model,
                    val_ds,
                    task_info,
                    batch_size=int(train_args.batch_size),
                    num_workers=num_workers,
                    device=device,
                )
            paths = plot_position_error_distribution(
                task,
                mode,
                plot_label,
                prefix,
                caption,
                values,
                header=header,
            )
            manifests.append({"kind": "position_error_density", "paths": paths})

    return {
        "task": task,
        "mode": mode,
        "seed": seed,
        "checkpoint": str(checkpoint),
        "caption": locals().get("caption"),
        "output_dir": str(plot_dir),
        "experiment": experiment,
        "data_splits": {
            "train_split_size": experiment["train_split_size"],
            "train_size_used": experiment["train_size_used"],
            "eval_size": experiment["eval_size"],
            "eval_label": experiment["eval_label"],
        },
        "plots": manifests,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    ensure_dir(args.output_root)
    official_index = load_official_index(args.official_json)
    completed = iter_completed_runs(args.results_root, args.tasks, args.modes, args.seeds)

    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in completed:
        grouped.setdefault((row["task"], row["mode"]), []).append(row)

    selected = select_runs(completed, args.selection)
    manifest = {
        "status": "completed",
        "selection": args.selection,
        "device": str(device),
        "generated_at_utc": utc_now(),
        "results_root": str(args.results_root),
        "output_root": str(args.output_root),
        "total_task_modes": len({(task, mode) for task in args.tasks for mode in args.modes}),
        "completed_task_modes": len({(row["task"], row["mode"]) for row in completed}),
        "generated_runs": 0,
        "experiment_labels": [],
        "generated": [],
    }
    for row in selected:
        rows_by_mode = grouped[(row["task"], row["mode"])]
        generated = generate_plots_for_run(
            row,
            rows_by_mode,
            official_index,
            args.output_root,
            num_workers=args.num_workers,
            device=device,
            overwrite=args.overwrite,
        )
        manifest["generated"].append(generated)
        print(f"[plot] {row['task']} / {row['mode']} / s{row['seed']}")
    manifest["generated_runs"] = len(manifest["generated"])
    manifest["experiment_labels"] = sorted(
        {str(item["experiment"]["label"]) for item in manifest["generated"] if item.get("experiment")}
    )

    manifest_path = args.output_root / "plot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[done] wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
