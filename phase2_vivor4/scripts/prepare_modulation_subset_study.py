from __future__ import annotations

import argparse
import csv
import json
import platform
import stat
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

try:
    from benchmark_config import CHECKPOINT_PATH, PHASE2_ROOT, RUNS_ROOT, TASK_SPECS, session_layout
except ModuleNotFoundError:  # pragma: no cover - package import path
    from .benchmark_config import CHECKPOINT_PATH, PHASE2_ROOT, RUNS_ROOT, TASK_SPECS, session_layout


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ORDER = ("rml", "radcom")
MODE_ORDER = ("lp", "ft2", "lora")
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_SUBSET_FRACTIONS = (0.01, 0.03, 0.05, 0.10, 0.50)
VAL_SPLIT = 0.2
NUM_WORKERS = 4
SAVE_EVERY = 5
BASE_LR = 1e-3
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 5
LOCAL_FULL_BASELINES = {
    ("rml", "lp"): 50.39,
    ("rml", "ft2"): 55.14,
    ("rml", "lora"): 56.42,
    ("radcom", "lp"): 90.12,
    ("radcom", "ft2"): 94.61,
    ("radcom", "lora"): 94.07,
}
OFFICIAL_FULL_BASELINES = {
    ("rml", "lp"): 50.39,
    ("rml", "ft2"): 55.16,
    ("rml", "lora"): 56.49,
    ("radcom", "lp"): 90.10,
    ("radcom", "ft2"): 94.53,
    ("radcom", "lora"): 93.78,
}
PARAM_COUNTS = {
    ("rml", "lp"): {"total": 6335243, "trainable": 17163},
    ("rml", "ft2"): {"total": 6335243, "trainable": 1596683},
    ("rml", "lora"): {"total": 6597387, "trainable": 279307},
    ("radcom", "lp"): {"total": 6334729, "trainable": 16649},
    ("radcom", "ft2"): {"total": 6334729, "trainable": 1596169},
    ("radcom", "lora"): {"total": 6596873, "trainable": 278793},
}
MODE_DEFINITIONS = {
    "lp": {
        "mode_label": "LP",
        "what_trains": "Task head, IQ tokenizer/input projection, conditional LN",
        "what_stays_frozen": "All shared encoder blocks",
        "frozen_blocks": "8 / 8",
        "lora_rank": "",
        "lora_alpha": "",
        "notes": "Linear-probe style transfer with frozen encoder features.",
    },
    "ft2": {
        "mode_label": "FT2",
        "what_trains": "Last 2 encoder blocks, task head, IQ tokenizer/input projection, conditional LN",
        "what_stays_frozen": "First 6 shared encoder blocks",
        "frozen_blocks": "6 / 8",
        "lora_rank": "",
        "lora_alpha": "",
        "notes": "Partial fine-tuning with limited backbone adaptation.",
    },
    "lora": {
        "mode_label": "LoRA",
        "what_trains": "LoRA adapters on q/v projections, task head, IQ tokenizer/input projection, conditional LN",
        "what_stays_frozen": "Base shared encoder weights",
        "frozen_blocks": "8 / 8 base blocks",
        "lora_rank": "32",
        "lora_alpha": "64",
        "notes": "Parameter-efficient adaptation; base encoder stays frozen.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the modulation subset study scaffold.")
    parser.add_argument(
        "--study-root",
        type=Path,
        default=None,
        help="Study output root. Defaults to phase2_vivor4/experiments/modulation_subset_study_<date>.",
    )
    parser.add_argument(
        "--subset-fractions",
        type=float,
        nargs="+",
        default=list(DEFAULT_SUBSET_FRACTIONS),
        help="Subset fractions to scaffold.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--session-stamp", default=datetime.now().strftime("%Y%m%d"))
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def short_host() -> str:
    return (platform.node().split(".", 1)[0] or "host").replace(" ", "-")


def percent_text(fraction: float) -> str:
    text = f"{fraction * 100.0:.2f}".rstrip("0").rstrip(".")
    return f"{text}%"


def subset_label(fraction: float) -> str:
    text = f"{fraction * 100.0:.2f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"{text}pct"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    write_text(path, json.dumps(payload, indent=2) + "\n")


def format_shape(shape: tuple[int, ...]) -> str:
    return " x ".join(str(int(dim)) for dim in shape)


def format_int(value: int) -> str:
    return f"{int(value):,}"


def format_percent_number(value: float) -> str:
    return f"{value:.3f}"


def compute_split_sizes(total_size: int) -> tuple[int, int]:
    val_size = max(1, int(total_size * VAL_SPLIT))
    train_size = max(1, total_size - val_size)
    return train_size, val_size


def compute_subset_size(train_size: int, fraction: float) -> int:
    return max(1, int(round(train_size * fraction)))


def cache_path_for_task(task: str) -> Path:
    if task == "rml":
        return PROJECT_ROOT / "datasets_h5" / "rml22.h5"
    if task == "radcom":
        return PROJECT_ROOT / "datasets_h5" / "radcom.h5"
    raise KeyError(task)


def load_dataset_info(task: str) -> dict:
    path = cache_path_for_task(task)
    with h5py.File(path, "r") as h5:
        total_size = int(h5["sample"].shape[0])
        sample_shape = tuple(int(v) for v in h5["sample"].shape[1:])
        label_counts = np.bincount(np.asarray(h5["label"], dtype=np.int64))
        labels_attr = h5.attrs.get("labels")
        label_pairs_attr = h5.attrs.get("label_pairs")
        if labels_attr:
            labels = json.loads(labels_attr)
        elif label_pairs_attr:
            labels = [f"{mod} + {sig}" for mod, sig in json.loads(label_pairs_attr)]
        else:
            labels = [str(idx) for idx in range(int(label_counts.size))]
        attr_mean_key = "mean" if "mean" in h5.attrs else "mu" if "mu" in h5.attrs else None
        attr_std_key = "std" if "std" in h5.attrs else None
        norm_detail = "Normalized IQ cache"
        if attr_mean_key and attr_std_key:
            norm_detail += f"; {attr_mean_key}/{attr_std_key} stored in H5 attrs"
        if int(label_counts.min()) == int(label_counts.max()):
            balance_note = f"Balanced; {format_int(int(label_counts[0]))} samples/class"
        else:
            balance_note = (
                f"Imbalanced; min {format_int(int(label_counts.min()))}, "
                f"max {format_int(int(label_counts.max()))} samples/class"
            )

    train_size, val_size = compute_split_sizes(total_size)
    if task == "rml":
        label_meaning = "11-way modulation label"
        output_type = "11-way class logits"
        secondary_metrics = "acc1, acc3, pca"
    else:
        label_meaning = "9-way joint modulation + signal-type label"
        output_type = "9-way joint class logits"
        secondary_metrics = "acc1, acc3, pca, mod_acc, sig_acc"

    return {
        "task": task,
        "task_name": TASK_SPECS[task]["display_name"],
        "dataset_name": TASK_SPECS[task]["dataset_name"],
        "modality": "IQ",
        "task_type": "classification",
        "label_output_meaning": label_meaning,
        "num_classes": len(labels),
        "labels": " | ".join(str(item) for item in labels),
        "input_shape": format_shape(sample_shape),
        "output_type": output_type,
        "total_size": total_size,
        "train_size": train_size,
        "validation_size": val_size,
        "primary_metric": "pca",
        "secondary_metrics": secondary_metrics,
        "loss_function": "CrossEntropyLoss",
        "class_balance_notes": balance_note,
        "normalization_notes": norm_detail,
        "cache_path": str(path),
    }


def dataset_table_rows(dataset_info: dict[str, dict]) -> list[dict]:
    rows = []
    for task in TASK_ORDER:
        info = dataset_info[task]
        rows.append(
            {
                "Task ID": info["task"],
                "Task Name": info["task_name"],
                "Dataset Name": info["dataset_name"],
                "Modality": info["modality"],
                "Task Type": info["task_type"],
                "Label / Output Meaning": info["label_output_meaning"],
                "# Classes": info["num_classes"],
                "Labels / Classes": info["labels"],
                "Input Shape": info["input_shape"],
                "Output Type": info["output_type"],
                "Total Size": format_int(info["total_size"]),
                "Train Size": format_int(info["train_size"]),
                "Validation Size": format_int(info["validation_size"]),
                "Primary Metric": info["primary_metric"],
                "Secondary Metrics": info["secondary_metrics"],
                "Loss Function": info["loss_function"],
                "Class Balance Notes": info["class_balance_notes"],
                "Normalization / Preprocessing": info["normalization_notes"],
                "Cache Path": info["cache_path"],
            }
        )
    return rows


def mode_table_rows() -> list[dict]:
    rows = []
    for mode in MODE_ORDER:
        definition = MODE_DEFINITIONS[mode]
        rml_counts = PARAM_COUNTS[("rml", mode)]
        radcom_counts = PARAM_COUNTS[("radcom", mode)]
        rows.append(
            {
                "Mode": definition["mode_label"],
                "What Trains": definition["what_trains"],
                "What Stays Frozen": definition["what_stays_frozen"],
                "Frozen Blocks": definition["frozen_blocks"],
                "LoRA Rank": definition["lora_rank"],
                "LoRA Alpha": definition["lora_alpha"],
                "RML Total Params": format_int(rml_counts["total"]),
                "RML Trainable Params": format_int(rml_counts["trainable"]),
                "RML Trainable %": format_percent_number(100.0 * rml_counts["trainable"] / rml_counts["total"]),
                "RADCOM Total Params": format_int(radcom_counts["total"]),
                "RADCOM Trainable Params": format_int(radcom_counts["trainable"]),
                "RADCOM Trainable %": format_percent_number(100.0 * radcom_counts["trainable"] / radcom_counts["total"]),
                "Checkpoint Used": str(CHECKPOINT_PATH),
                "Checkpoint Selection Metric": "best validation pca",
                "Notes": definition["notes"],
            }
        )
    return rows


def preferred_python_executable(repo_root: Path) -> str:
    venv_python = repo_root / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return "python3"


def tracker_args(session_root: Path, fraction: float, seeds: list[int]) -> list[str]:
    return [
        "--session-root",
        str(session_root),
        "--tasks",
        "rml",
        "radcom",
        "--modes",
        "lp",
        "ft2",
        "lora",
        "--seeds",
        *[str(seed) for seed in seeds],
        "--num-workers",
        str(NUM_WORKERS),
        "--save-every",
        str(SAVE_EVERY),
        "--train-subset-fraction",
        str(fraction),
    ]


def tracker_command(repo_root: Path, session_root: Path, fraction: float, seeds: list[int]) -> list[str]:
    return [
        preferred_python_executable(repo_root),
        "phase2_vivor4/scripts/wait_for_radcom_and_run_next.py",
        *tracker_args(session_root, fraction, seeds),
    ]


def supervisor_command(repo_root: Path, session_root: Path, fraction: float, seeds: list[int]) -> list[str]:
    return [
        preferred_python_executable(repo_root),
        "phase2_vivor4/scripts/run_tracker_supervised.py",
        "--session-root",
        str(session_root),
        "--label",
        session_root.name,
        "--restart-delay-seconds",
        "20",
        "--poll-seconds",
        "15",
        "--stale-seconds",
        "900",
        "--startup-grace-seconds",
        "180",
        "--graceful-stop-seconds",
        "20",
        "--max-restarts",
        "50",
        "--",
        *tracker_args(session_root, fraction, seeds),
    ]


def build_run_config_rows(
    dataset_info: dict[str, dict],
    subset_fractions: list[float],
    seeds: list[int],
    session_roots: dict[float, Path],
    launch_scripts: dict[float, Path],
) -> list[dict]:
    rows: list[dict] = []
    for fraction in subset_fractions:
        session_root = session_roots[fraction]
        for task in TASK_ORDER:
            info = dataset_info[task]
            subset_size = compute_subset_size(info["train_size"], fraction)
            effective_batch = int(TASK_SPECS[task]["batch_size"])
            effective_lr = BASE_LR * effective_batch / 256.0
            for mode in MODE_ORDER:
                counts = PARAM_COUNTS[(task, mode)]
                rows.append(
                    {
                        "Task": task,
                        "Dataset": info["dataset_name"],
                        "Mode": mode,
                        "Subset %": percent_text(fraction),
                        "Sampling Size": format_int(subset_size),
                        "Sampling Policy": "stratified subset after train/validation split",
                        "Train Size Used": format_int(subset_size),
                        "Train Split Size": format_int(info["train_size"]),
                        "Validation Size": format_int(info["validation_size"]),
                        "Total Size": format_int(info["total_size"]),
                        "Seeds": ",".join(str(seed) for seed in seeds),
                        "Batch Size": str(TASK_SPECS[task]["batch_size"]),
                        "Accum Steps": "1",
                        "Effective Batch Size": str(effective_batch),
                        "Epochs": str(TASK_SPECS[task]["epochs"]),
                        "Loss Function": "CrossEntropyLoss",
                        "Optimizer": "AdamW",
                        "Base LR (blr)": f"{BASE_LR:.0e}",
                        "Effective LR": f"{effective_lr:.4f}",
                        "Weight Decay": str(WEIGHT_DECAY),
                        "Warmup Epochs": str(WARMUP_EPOCHS),
                        "Label Smoothing": "0.0",
                        "Class Weights": "no",
                        "Validation Split Strategy": "random 80/20 split",
                        "Trainable Params": format_int(counts["trainable"]),
                        "Trainable %": format_percent_number(100.0 * counts["trainable"] / counts["total"]),
                        "Checkpoint Used": str(CHECKPOINT_PATH),
                        "Primary Metric": "pca",
                        "Secondary Metrics": info["secondary_metrics"],
                        "Session Root": str(session_root),
                        "Launch Script": str(launch_scripts[fraction]),
                    }
                )
    return rows


def build_results_rows(
    dataset_info: dict[str, dict],
    subset_fractions: list[float],
    session_roots: dict[float, Path],
) -> list[dict]:
    rows: list[dict] = []
    for fraction in subset_fractions:
        session_root = session_roots[fraction]
        for task in TASK_ORDER:
            info = dataset_info[task]
            subset_size = compute_subset_size(info["train_size"], fraction)
            for mode in MODE_ORDER:
                rows.append(
                    {
                        "Task": task,
                        "Mode": mode,
                        "Subset %": percent_text(fraction),
                        "Train Size Used": format_int(subset_size),
                        "Validation Size": format_int(info["validation_size"]),
                        "Session Root": str(session_root),
                        "Primary Metric Mean": "",
                        "Primary Metric Std": "",
                        "Acc1 Mean": "",
                        "Acc1 Std": "",
                        "Acc3 Mean": "",
                        "Acc3 Std": "",
                        "Macro-F1 Mean": "",
                        "Macro-F1 Std": "",
                        "Mod Acc Mean": "",
                        "Sig Acc Mean": "",
                        "Best Epoch Mean": "",
                        "Best Epoch Std": "",
                        "Training Time / Run": "",
                        "Total Time": "",
                        "Peak GPU Memory": "",
                        "Peak Host RAM": "",
                        "Status": "planned",
                        "Notes": "",
                    }
                )
    return rows


def build_comparison_rows(
    dataset_info: dict[str, dict],
    subset_fractions: list[float],
    session_roots: dict[float, Path],
) -> list[dict]:
    rows: list[dict] = []
    for fraction in subset_fractions:
        session_root = session_roots[fraction]
        for task in TASK_ORDER:
            subset_size = compute_subset_size(dataset_info[task]["train_size"], fraction)
            for mode in MODE_ORDER:
                rows.append(
                    {
                        "Task": task,
                        "Mode": mode,
                        "Subset %": percent_text(fraction),
                        "Train Size Used": format_int(subset_size),
                        "Session Root": str(session_root),
                        "Subset Primary Metric": "",
                        "Local Full-Data Baseline": f"{LOCAL_FULL_BASELINES[(task, mode)]:.2f}",
                        "Official Baseline": f"{OFFICIAL_FULL_BASELINES[(task, mode)]:.2f}",
                        "Delta vs Local Full-Data": "",
                        "Delta vs Official": "",
                        "Retention vs Local Full-Data (%)": "",
                        "Retention vs Official (%)": "",
                        "Relative Time vs Full-Data": "",
                        "Status": "planned",
                        "Notes": "",
                    }
                )
    return rows


def build_runtime_rows(
    dataset_info: dict[str, dict],
    subset_fractions: list[float],
    seeds: list[int],
    session_roots: dict[float, Path],
) -> list[dict]:
    rows: list[dict] = []
    total_runs = len(TASK_ORDER) * len(MODE_ORDER) * len(seeds)
    for fraction in subset_fractions:
        session_root = session_roots[fraction]
        rows.append(
            {
                "Subset %": percent_text(fraction),
                "Session Root": str(session_root),
                "Runs Planned": str(total_runs),
                "Runs Completed": "",
                "Tasks": ",".join(TASK_ORDER),
                "Modes": ",".join(MODE_ORDER),
                "Seeds": ",".join(str(seed) for seed in seeds),
                "RML Train Size Used": format_int(compute_subset_size(dataset_info["rml"]["train_size"], fraction)),
                "RADCOM Train Size Used": format_int(compute_subset_size(dataset_info["radcom"]["train_size"], fraction)),
                "Total Session Time": "",
                "Plot Stage Time": "",
                "Summary Stage Time": "",
                "Comparison Stage Time": "",
                "Peak GPU Memory": "",
                "Peak Host RAM": "",
                "GPU Type": "",
                "CPU / Worker Count": str(NUM_WORKERS),
                "Status": "planned",
                "Notes": "",
            }
        )
    return rows


def build_baseline_rows() -> list[dict]:
    rows = []
    for task in TASK_ORDER:
        for mode in MODE_ORDER:
            rows.append(
                {
                    "Task": task,
                    "Mode": mode,
                    "Local Full-Data Baseline": f"{LOCAL_FULL_BASELINES[(task, mode)]:.2f}",
                    "Official Baseline": f"{OFFICIAL_FULL_BASELINES[(task, mode)]:.2f}",
                }
            )
    return rows


def session_readme_text(session_root: Path, fraction: float) -> str:
    return (
        f"# Modulation Subset Session\n\n"
        f"Session root: `{session_root}`\n\n"
        f"This prepared session is reserved for the `{percent_text(fraction)}` modulation subset experiment.\n"
        f"The tracker will write run outputs, plots, summaries, comparisons, and manifests here.\n"
    )


def launch_script_text(repo_root: Path, session_root: Path, fraction: float, seeds: list[int]) -> str:
    cmd = supervisor_command(repo_root, session_root, fraction, seeds)
    rendered = " \\\n  ".join(cmd)
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        f'cd "{repo_root}"\n\n'
        f"{rendered}\n"
    )


def launch_all_text(launch_scripts: list[Path]) -> str:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for script in launch_scripts:
        lines.append(f'bash "{script}"')
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def make_executable(path: Path) -> None:
    current = path.stat().st_mode
    path.chmod(current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def prepare_session_dirs(session_root: Path, fraction: float) -> None:
    layout = session_layout(session_root)
    for key in (
        "session_root",
        "imports_root",
        "results_root",
        "summary_root",
        "comparison_root",
        "plots_root",
        "detailed_plots_root",
    ):
        ensure_dir(layout[key])
    write_text(session_root / "README.md", session_readme_text(session_root, fraction))


def study_readme_text(study_root: Path, session_roots: dict[float, Path]) -> str:
    rows = []
    for fraction, session_root in session_roots.items():
        rows.append(f"- `{percent_text(fraction)}` -> `{session_root}`")
    return (
        "# Modulation Subset Study\n\n"
        "This scaffold was generated to prepare the `rml` + `radcom` reduced-data fine-tuning study.\n\n"
        "Contents:\n"
        "- `tables/`: prefilled experiment tables and result placeholders.\n"
        "- `commands/`: launch scripts for each subset level plus a convenience launcher.\n"
        "- `results/`: baseline references and result-holder files.\n"
        "- `study_manifest.json`: machine-readable study plan.\n\n"
        "Planned sessions:\n"
        + "\n".join(rows)
        + "\n"
    )


def results_readme_text() -> str:
    return (
        "# Results Folder\n\n"
        "This folder holds study-level result sheets and baseline references.\n"
        "Run outputs themselves are written inside the planned session roots under `phase2_vivor4/runs/`.\n"
    )


def main() -> None:
    args = parse_args()
    study_root = (
        args.study_root.expanduser().resolve()
        if args.study_root is not None
        else (PHASE2_ROOT / "experiments" / f"modulation_subset_study_{args.session_stamp}").resolve()
    )
    subset_fractions = [float(value) for value in args.subset_fractions]
    seeds = [int(seed) for seed in args.seeds]
    host = short_host()

    tables_dir = study_root / "tables"
    commands_dir = study_root / "commands"
    results_dir = study_root / "results"
    ensure_dir(tables_dir)
    ensure_dir(commands_dir)
    ensure_dir(results_dir)

    dataset_info = {task: load_dataset_info(task) for task in TASK_ORDER}
    session_roots: dict[float, Path] = {}
    launch_scripts: dict[float, Path] = {}
    session_rows: list[dict] = []

    for fraction in subset_fractions:
        label = subset_label(fraction)
        session_root = (RUNS_ROOT / f"modulation_subset_{label}_{args.session_stamp}_{host}").resolve()
        session_roots[fraction] = session_root
        prepare_session_dirs(session_root, fraction)
        launch_path = commands_dir / f"launch_{label}.sh"
        write_text(launch_path, launch_script_text(PROJECT_ROOT, session_root, fraction, seeds))
        make_executable(launch_path)
        launch_scripts[fraction] = launch_path
        session_rows.append(
            {
                "Subset %": percent_text(fraction),
                "Subset Label": label,
                "Session Root": str(session_root),
                "Launch Script": str(launch_path),
                "Tracker Command": " ".join(tracker_command(PROJECT_ROOT, session_root, fraction, seeds)),
                "Status": "planned",
            }
        )

    launch_all_path = commands_dir / "launch_all_subsets.sh"
    write_text(launch_all_path, launch_all_text([launch_scripts[fraction] for fraction in subset_fractions]))
    make_executable(launch_all_path)

    write_csv(tables_dir / "table_a_dataset_task_summary.csv", dataset_table_rows(dataset_info))
    write_csv(tables_dir / "table_b_mode_definition.csv", mode_table_rows())
    write_csv(
        tables_dir / "table_c_run_configuration_matrix.csv",
        build_run_config_rows(dataset_info, subset_fractions, seeds, session_roots, launch_scripts),
    )
    write_csv(
        tables_dir / "table_d_results_by_task_mode_subset.csv",
        build_results_rows(dataset_info, subset_fractions, session_roots),
    )
    write_csv(
        tables_dir / "table_e_comparison_vs_full_data.csv",
        build_comparison_rows(dataset_info, subset_fractions, session_roots),
    )
    write_csv(
        tables_dir / "table_f_runtime_summary.csv",
        build_runtime_rows(dataset_info, subset_fractions, seeds, session_roots),
    )
    write_csv(results_dir / "full_data_baselines.csv", build_baseline_rows())
    write_csv(results_dir / "planned_sessions.csv", session_rows)
    write_text(study_root / "README.md", study_readme_text(study_root, session_roots))
    write_text(results_dir / "README.md", results_readme_text())

    manifest = {
        "study_root": str(study_root),
        "generated_at_utc": utc_now(),
        "tasks": list(TASK_ORDER),
        "modes": list(MODE_ORDER),
        "seeds": seeds,
        "subset_fractions": subset_fractions,
        "launch_all_script": str(launch_all_path),
        "tables": {
            "dataset_task_summary": str(tables_dir / "table_a_dataset_task_summary.csv"),
            "mode_definition": str(tables_dir / "table_b_mode_definition.csv"),
            "run_configuration_matrix": str(tables_dir / "table_c_run_configuration_matrix.csv"),
            "results_by_task_mode_subset": str(tables_dir / "table_d_results_by_task_mode_subset.csv"),
            "comparison_vs_full_data": str(tables_dir / "table_e_comparison_vs_full_data.csv"),
            "runtime_summary": str(tables_dir / "table_f_runtime_summary.csv"),
        },
        "results": {
            "full_data_baselines": str(results_dir / "full_data_baselines.csv"),
            "planned_sessions": str(results_dir / "planned_sessions.csv"),
        },
        "sessions": session_rows,
    }
    write_json(study_root / "study_manifest.json", manifest)

    print(f"[done] study scaffold ready at {study_root}")
    print(f"[done] launch scripts ready in {commands_dir}")
    print(f"[done] tables ready in {tables_dir}")


if __name__ == "__main__":
    main()
