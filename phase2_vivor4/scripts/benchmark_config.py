from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT / "wavesfm"
RAW_ROOT = PROJECT_ROOT / "datasets_raw"
CACHE_ROOT = PROJECT_ROOT / "datasets_h5"
PHASE2_ROOT = PROJECT_ROOT / "phase2_vivor4"
RUNS_ROOT = PHASE2_ROOT / "runs"
LOCAL_RESULTS_ROOT = PHASE2_ROOT / "local_results" / "by_task"
LOCAL_SUMMARY_ROOT = PHASE2_ROOT / "local_results" / "summaries"
OFFICIAL_RESULTS_ROOT = PHASE2_ROOT / "official_results"
OFFICIAL_TASK_ROOT = OFFICIAL_RESULTS_ROOT / "by_task"
COMPARISON_ROOT = PHASE2_ROOT / "comparisons"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "wavesfm-v1p0.pth"
CURRENT_SESSION_LINK = PHASE2_ROOT / "current"
CURRENT_SESSION_META = PHASE2_ROOT / "current_session.json"

OFFICIAL_RESULTS_URL = "https://waveslab.ai/detailed-eval/"
OFFICIAL_RESULTS_UPDATED = "2026-01-23"

OFFICIAL_MODES = ("lp", "ft2", "lora")
ALL_MODES = ("lp", "ft2", "lora", "strict", "sl")
DEFAULT_NUM_WORKERS = 4
DEFAULT_SAVE_EVERY = 5
DEEPMIMO_SCENARIO_NAMES = (
    "city_18_denver",
    "city_15_indianapolis",
    "city_19_oklahoma",
    "city_12_fortworth",
    "city_11_santaclara",
    "city_7_sandiego",
)


def session_layout(session_root: Path) -> dict[str, Path]:
    session_root = session_root.expanduser().resolve()
    imports_root = session_root / "imports"
    local_results_root = session_root / "local_results" / "by_task"
    summary_root = session_root / "local_results" / "summaries"
    comparison_root = session_root / "comparisons"
    plots_root = session_root / "plots"
    detailed_plots_root = plots_root / "detailed_eval"
    return {
        "session_root": session_root,
        "imports_root": imports_root,
        "results_root": local_results_root,
        "summary_root": summary_root,
        "comparison_root": comparison_root,
        "plots_root": plots_root,
        "detailed_plots_root": detailed_plots_root,
        "plot_manifest_path": plots_root / "plot_manifest.json",
        "summary_runs_json": summary_root / "local_results_runs.json",
        "summary_agg_json": summary_root / "local_results_aggregated.json",
        "summary_manifest_path": summary_root / "summary_manifest.json",
        "comparison_json": comparison_root / "local_vs_official.json",
        "comparison_manifest_path": comparison_root / "comparison_manifest.json",
        "manifest_path": session_root / "session_manifest.json",
        "preflight_report_path": session_root / "preflight_report.json",
    }


def preferred_runtime_device() -> str:
    forced = os.environ.get("WAVESFM_FORCE_DEVICE", "").strip().lower()
    if forced:
        if forced in {"cpu", "mps", "cuda"}:
            return forced
        raise ValueError(f"Unsupported WAVESFM_FORCE_DEVICE={forced!r}")
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


CACHE_SPECS = {
    "has": {
        "display_name": "EfficientFi Human Activity Sensing (HAS)",
        "cache_name": "has.h5",
        "expected_raw": "datasets_raw/has/NTU-Fi_HAR",
    },
    "rfs": {
        "display_name": "CommRad RF Signals (RFS)",
        "cache_name": "rfs.h5",
        "expected_raw": "datasets_raw/rfs",
    },
    "pos": {
        "display_name": "5G NR Positioning (POS)",
        "cache_name": "nrpos-outdoor.h5",
        "expected_raw": "datasets_raw/pos",
    },
    "uwb-indoor": {
        "display_name": "UWB Indoor Positioning and Tracking",
        "cache_name": "environment0.h5",
        "expected_raw": "datasets_raw/uwb_indoor",
    },
    "uwb-industrial": {
        "display_name": "UWB Industrial Localization",
        "cache_name": "ipin-train.h5",
        "expected_raw": "datasets_raw/uwb_industrial/industrial_training.pkl",
    },
    "radcom": {
        "display_name": "RADCOM",
        "cache_name": "radcom.h5",
        "expected_raw": "datasets_raw/radcom/<source>.h5",
    },
    "rml": {
        "display_name": "RML2022",
        "cache_name": "rml22.h5",
        "expected_raw": "datasets_raw/rml22/RML22.01A",
    },
    "rfp": {
        "display_name": "POWDER RF Fingerprinting",
        "cache_name": "rfp.h5",
        "expected_raw": "datasets_raw/rfp/GlobecomPOWDER",
    },
    "interf": {
        "display_name": "ICARUS Interference",
        "cache_name": "icarus.h5",
        "expected_raw": "datasets_raw/icarus",
    },
    "deepmimo": {
        "display_name": "DeepMIMO",
        "cache_name": "deepmimo.h5",
        "expected_raw": "datasets_raw/deepmimo",
    },
    "lwm-beam-challenge": {
        "display_name": "LWM Beam Challenge",
        "cache_name": "lwm-beam-challenge.h5",
        "expected_raw": "datasets_raw/deepmimo/lwm_beam_labels/beam_prediction_challenge",
    },
}


TASK_SPECS = {
    "sensing": {
        "display_name": "Human Activity Sensing",
        "dataset_name": "EfficientFi (HAS)",
        "cache_id": "has",
        "task_type": "classification",
        "primary_metric": "pca",
        "batch_size": 256,
        "epochs": 100,
        "smoothing": 0.1,
        "stratified_split": True,
    },
    "rfs": {
        "display_name": "RF Signal Classification",
        "dataset_name": "CommRad RF (RFS)",
        "cache_id": "rfs",
        "task_type": "classification",
        "primary_metric": "pca",
        "batch_size": 256,
        "epochs": 100,
        "smoothing": 0.05,
        "stratified_split": True,
        "class_weights": True,
    },
    "pos": {
        "display_name": "5G NR Positioning",
        "dataset_name": "5G NR Positioning (POS)",
        "cache_id": "pos",
        "task_type": "position",
        "primary_metric": "mean_error",
        "batch_size": 256,
        "epochs": 100,
    },
    "radcom": {
        "display_name": "RADCOM Signal Classification",
        "dataset_name": "RADCOM",
        "cache_id": "radcom",
        "task_type": "classification",
        "primary_metric": "pca",
        "batch_size": 2048,
        "epochs": 50,
    },
    "uwb-indoor": {
        "display_name": "UWB Indoor Positioning",
        "dataset_name": "UWB Indoor Positioning and Tracking",
        "cache_id": "uwb-indoor",
        "task_type": "position",
        "primary_metric": "mean_error",
        "batch_size": 256,
        "epochs": 50,
    },
    "uwb-industrial": {
        "display_name": "UWB Industrial Localization",
        "dataset_name": "UWB Industrial Localization",
        "cache_id": "uwb-industrial",
        "task_type": "position",
        "primary_metric": "mean_error",
        "batch_size": 512,
        "epochs": 50,
    },
    "rml": {
        "display_name": "Modulation Classification",
        "dataset_name": "RML2022",
        "cache_id": "rml",
        "task_type": "classification",
        "primary_metric": "pca",
        "batch_size": 2048,
        "epochs": 50,
    },
    "rfp": {
        "display_name": "RF Fingerprinting",
        "dataset_name": "POWDER (RFP)",
        "cache_id": "rfp",
        "task_type": "classification",
        "primary_metric": "pca",
        "batch_size": 256,
        "epochs": 10,
        "smoothing": 0.1,
    },
    "interf": {
        "display_name": "Interference Classification",
        "dataset_name": "ICARUS (INTD/INTC)",
        "cache_id": "interf",
        "task_type": "classification",
        "primary_metric": "pca",
        "batch_size": 256,
        "epochs": 35,
        "smoothing": 0.02,
        "stratified_split": True,
        "class_weights": True,
        "accum_steps": 2,
    },
    "deepmimo-los": {
        "display_name": "LOS/NLOS Classification",
        "dataset_name": "DeepMIMO",
        "cache_id": "deepmimo",
        "task_type": "classification",
        "primary_metric": "pca",
        "batch_size": 256,
        "epochs": 100,
        "stratified_split": True,
        "class_weights": True,
        "vis_img_size": 32,
    },
    "deepmimo-beam": {
        "display_name": "Beam Prediction",
        "dataset_name": "DeepMIMO",
        "cache_id": "deepmimo",
        "task_type": "classification",
        "primary_metric": "pca",
        "batch_size": 256,
        "epochs": 100,
        "stratified_split": True,
        "class_weights": True,
        "vis_img_size": 32,
        "deepmimo_n_beams": 64,
    },
    "lwm-beam-challenge": {
        "display_name": "Beam Prediction",
        "dataset_name": "LWM Beam Challenge",
        "cache_id": "lwm-beam-challenge",
        "task_type": "classification",
        "primary_metric": "pca",
        "batch_size": 256,
        "epochs": 100,
        "stratified_split": True,
        "class_weights": True,
        "vis_img_size": 32,
        "official_comparison_task": "deepmimo-beam",
        "hypothesis_note": (
            "Third-run hypothesis task: compare official LWM beam-challenge train labels "
            "against the published WavesFM DeepMIMO beam detailed-eval numbers."
        ),
    },
}


OFFICIAL_RESULTS = {
    "sensing": {
        "dataset_protocol_url": "https://waveslab.ai/docs/datasets/has/",
        "reported_split": "20% split",
        "modes": {
            "ft2": {"acc1": 99.31, "acc3": 100.00, "pca": 99.31, "test_samples": 240},
            "lora": {"acc1": 98.47, "acc3": 100.00, "pca": 98.47, "test_samples": 240},
            "lp": {"acc1": 94.44, "acc3": 100.00, "pca": 94.42, "test_samples": 240},
        },
    },
    "rfs": {
        "dataset_protocol_url": "https://waveslab.ai/docs/datasets/rfs/",
        "reported_split": "20% split",
        "modes": {
            "ft2": {"acc1": 84.53, "acc3": 93.37, "pca": 83.41, "test_samples": 724},
            "lora": {"acc1": 86.46, "acc3": 93.51, "pca": 84.49, "test_samples": 724},
            "lp": {"acc1": 55.52, "acc3": 83.33, "pca": 42.85, "test_samples": 724},
        },
    },
    "pos": {
        "dataset_protocol_url": "https://waveslab.ai/docs/datasets/pos/",
        "reported_split": "20% split",
        "modes": {
            "ft2": {"mean_error": 1.24, "median_error": 0.87, "p90_error": 2.69, "test_samples": 2325},
            "lora": {"mean_error": 1.45, "median_error": 0.99, "p90_error": 3.09, "test_samples": 2325},
            "lp": {"mean_error": 3.17, "median_error": 2.77, "p90_error": 5.92, "test_samples": 2325},
        },
    },
    "radcom": {
        "dataset_protocol_url": "https://waveslab.ai/docs/datasets/radcom/",
        "reported_split": "20% split",
        "modes": {
            "ft2": {"acc1": 94.53, "acc3": 99.99, "pca": 94.53, "test_samples": 113400},
            "lora": {"acc1": 93.78, "acc3": 99.99, "pca": 93.78, "test_samples": 113400},
            "lp": {"acc1": 90.12, "acc3": 99.97, "pca": 90.10, "test_samples": 113400},
        },
    },
    "uwb-indoor": {
        "dataset_protocol_url": "https://waveslab.ai/docs/datasets/uwb_indoor/",
        "reported_split": "20% split",
        "modes": {
            "ft2": {"mean_error": 0.83, "median_error": 0.62, "p90_error": 1.66, "test_samples": 3160},
            "lora": {"mean_error": 0.65, "median_error": 0.50, "p90_error": 1.30, "test_samples": 3160},
            "lp": {"mean_error": 1.43, "median_error": 1.24, "p90_error": 2.65, "test_samples": 3160},
        },
    },
    "uwb-industrial": {
        "dataset_protocol_url": "https://waveslab.ai/docs/datasets/uwb_industrial/",
        "reported_split": "20% split",
        "modes": {
            "ft2": {"mean_error": 0.88, "median_error": 0.72, "p90_error": 1.62, "test_samples": 8052},
            "lora": {"mean_error": 0.74, "median_error": 0.63, "p90_error": 1.33, "test_samples": 8052},
            "lp": {"mean_error": 2.71, "median_error": 2.50, "p90_error": 4.69, "test_samples": 8052},
        },
    },
    "rml": {
        "dataset_protocol_url": "https://waveslab.ai/docs/datasets/rml2022/",
        "reported_split": "20% split",
        "modes": {
            "ft2": {"acc1": 55.25, "acc3": 78.57, "pca": 55.16, "test_samples": 92400},
            "lora": {"acc1": 56.59, "acc3": 78.59, "pca": 56.49, "test_samples": 92400},
            "lp": {"acc1": 50.49, "acc3": 74.34, "pca": 50.39, "test_samples": 92400},
        },
    },
    "rfp": {
        "dataset_protocol_url": "https://waveslab.ai/docs/datasets/rfp/",
        "reported_split": "20% split",
        "modes": {
            "ft2": {"acc1": 99.70, "acc3": 100.00, "pca": 99.70, "test_samples": 208894},
            "lora": {"acc1": 99.82, "acc3": 100.00, "pca": 99.82, "test_samples": 208894},
            "lp": {"acc1": 98.96, "acc3": 100.00, "pca": 98.97, "test_samples": 208894},
        },
    },
    "interf": {
        "dataset_protocol_url": "https://waveslab.ai/docs/datasets/icarus/",
        "reported_split": "20% split",
        "modes": {
            "ft2": {"acc1": 75.00, "acc3": 100.00, "pca": 78.94, "test_samples": 500},
            "lora": {"acc1": 74.87, "acc3": 100.00, "pca": 78.94, "test_samples": 500},
            "lp": {"acc1": 67.47, "acc3": 100.00, "pca": 71.50, "test_samples": 500},
        },
    },
    "deepmimo-los": {
        "dataset_protocol_url": "https://waveslab.ai/docs/datasets/deepmimo/",
        "reported_split": "20% split",
        "modes": {
            "ft2": {"acc1": 95.66, "acc3": 100.00, "pca": 95.46, "test_samples": 2968},
            "lora": {"acc1": 95.60, "acc3": 100.00, "pca": 95.24, "test_samples": 2968},
            "lp": {"acc1": 95.35, "acc3": 100.00, "pca": 95.02, "test_samples": 2968},
        },
    },
    "deepmimo-beam": {
        "dataset_protocol_url": "https://waveslab.ai/docs/datasets/deepmimo/",
        "reported_split": "20% split",
        "modes": {
            "ft2": {"acc1": 80.50, "acc3": 94.42, "pca": 78.38, "test_samples": 2968},
            "lora": {"acc1": 79.68, "acc3": 93.82, "pca": 77.38, "test_samples": 2968},
            "lp": {"acc1": 70.66, "acc3": 90.96, "pca": 67.67, "test_samples": 2968},
        },
    },
}


def cache_path_for_task(task: str, cache_root: Path = CACHE_ROOT) -> Path:
    cache_id = TASK_SPECS[task]["cache_id"]
    return cache_root / CACHE_SPECS[cache_id]["cache_name"]


def resolved_cache_path_for_task(
    task: str,
    cache_root: Path = CACHE_ROOT,
    *,
    radcom_cache: Path | None = None,
    path_overrides: dict[str, Path] | None = None,
) -> Path:
    if task not in TASK_SPECS:
        raise KeyError(f"Unknown task: {task}")
    if path_overrides and task in path_overrides:
        return Path(path_overrides[task]).expanduser().resolve()
    cache_id = TASK_SPECS[task]["cache_id"]
    if cache_id == "radcom" and radcom_cache is not None:
        return Path(radcom_cache).expanduser().resolve()
    return cache_path_for_task(task, cache_root)


def _ordered_unique_tasks(tasks: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for task in tasks:
        if task not in TASK_SPECS:
            raise KeyError(f"Unknown task: {task}")
        if task in seen:
            continue
        seen.add(task)
        ordered.append(task)
    return ordered


def task_run_readiness(
    task: str,
    cache_root: Path = CACHE_ROOT,
    *,
    radcom_cache: Path | None = None,
    path_overrides: dict[str, Path] | None = None,
    allow_build_from_raw: bool = False,
) -> tuple[bool, str]:
    cache_path = resolved_cache_path_for_task(
        task,
        cache_root,
        radcom_cache=radcom_cache,
        path_overrides=path_overrides,
    )
    cache_exists = cache_path.exists()
    cache_size_ok = True
    if cache_exists and cache_path.is_file():
        try:
            cache_size_ok = cache_path.stat().st_size > 0
        except OSError:
            cache_size_ok = False
    if cache_exists and cache_size_ok:
        return True, f"cache ready: {cache_path.name}"

    cache_id = TASK_SPECS[task]["cache_id"]
    raw_path = discover_raw_input(cache_id)
    raw_exists = raw_path.exists()
    if allow_build_from_raw and raw_exists:
        return True, f"cache missing; raw available at {raw_path}"

    detail = f"missing cache: {cache_path.name}"
    if raw_exists:
        detail += f"; raw available at {raw_path}"
    else:
        detail += f"; raw missing: {raw_path}"
    return False, detail


def classify_requested_tasks(
    tasks: Iterable[str],
    cache_root: Path = CACHE_ROOT,
    *,
    radcom_cache: Path | None = None,
    path_overrides: dict[str, Path] | None = None,
    allow_build_from_raw_tasks: set[str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    runnable: list[str] = []
    blocked: dict[str, str] = {}
    allow_build = set(allow_build_from_raw_tasks or set())
    for task in _ordered_unique_tasks(tasks):
        ok, detail = task_run_readiness(
            task,
            cache_root,
            radcom_cache=radcom_cache,
            path_overrides=path_overrides,
            allow_build_from_raw=task in allow_build,
        )
        if ok:
            runnable.append(task)
        else:
            blocked[task] = detail
    return runnable, blocked


def expected_raw_path(cache_id: str, raw_root: Path = RAW_ROOT) -> Path:
    if cache_id == "has":
        return raw_root / "has" / "NTU-Fi_HAR"
    if cache_id == "rfs":
        return raw_root / "rfs"
    if cache_id == "pos":
        return raw_root / "pos"
    if cache_id == "uwb-indoor":
        return raw_root / "uwb_indoor"
    if cache_id == "uwb-industrial":
        return raw_root / "uwb_industrial" / "industrial_training.pkl"
    if cache_id == "radcom":
        return raw_root / "radcom"
    if cache_id == "rml":
        return raw_root / "rml22" / "RML22.01A"
    if cache_id == "rfp":
        return raw_root / "rfp" / "GlobecomPOWDER"
    if cache_id == "interf":
        return raw_root / "icarus"
    if cache_id == "deepmimo":
        return raw_root / "deepmimo"
    if cache_id == "lwm-beam-challenge":
        return raw_root / "deepmimo" / "lwm_beam_labels" / "beam_prediction_challenge"
    raise KeyError(f"Unknown cache id: {cache_id}")


def _first_match(candidates: Iterable[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def _looks_like_deepmimo_scenario_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any((path / scenario).is_dir() for scenario in DEEPMIMO_SCENARIO_NAMES)


def _discover_deepmimo_scenario_root(root: Path) -> Path | None:
    if _looks_like_deepmimo_scenario_root(root):
        return root
    if not root.exists():
        return None
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if _looks_like_deepmimo_scenario_root(child):
            return child
    return None


def discover_raw_input(cache_id: str, raw_root: Path = RAW_ROOT) -> Path:
    if cache_id == "has":
        path = _first_match(
            [
                raw_root / "has" / "NTU-Fi_HAR",
                raw_root / "NTU-Fi_HAR",
            ]
        )
        return path or expected_raw_path(cache_id, raw_root)

    if cache_id == "uwb-industrial":
        direct = raw_root / "uwb_industrial" / "industrial_training.pkl"
        if direct.exists():
            return direct
        pkls = sorted((raw_root / "uwb_industrial").glob("*.pkl"))
        return pkls[0] if pkls else direct

    if cache_id == "rml":
        default = raw_root / "rml22" / "RML22.01A"
        if default.exists():
            return default
        files = sorted(p for p in (raw_root / "rml22").glob("*") if p.is_file())
        return files[0] if files else default

    if cache_id == "radcom":
        root = raw_root / "radcom"
        files = sorted(p for p in root.glob("*") if p.is_file())
        return files[0] if files else root / "radcom_source.h5"

    if cache_id == "rfp":
        default = raw_root / "rfp" / "GlobecomPOWDER"
        if default.exists():
            return default
        root = raw_root / "rfp"
        if root.exists() and any(root.iterdir()):
            return root
        return default

    if cache_id == "deepmimo":
        default = raw_root / "deepmimo"
        scenario_root = _discover_deepmimo_scenario_root(default)
        if scenario_root is not None:
            return scenario_root
        path = _first_match(
            [
                default / "deepmimo_data.p",
                default / "deepmimo_data 2.p",
                raw_root / "deepmimo_data.p",
                raw_root / "deepmimo_data 2.p",
            ]
        )
        if path is not None:
            return path
        if default.exists() and any(default.iterdir()):
            return default
        return default

    if cache_id == "lwm-beam-challenge":
        default = raw_root / "deepmimo" / "lwm_beam_labels" / "beam_prediction_challenge"
        train = default / "train" / "bp_data_train.p"
        labels = default / "train" / "bp_label_train.p"
        if train.exists() and labels.exists():
            return default
        return default

    return expected_raw_path(cache_id, raw_root)


def build_preprocess_command(
    cache_id: str,
    raw_input: Path,
    output_path: Path,
    repo_root: Path = REPO_ROOT,
) -> list[str]:
    py = sys.executable
    if cache_id == "has":
        return [
            py,
            str(repo_root / "preprocessing" / "preprocess_csi_sensing.py"),
            "--data-path",
            str(raw_input),
            "--output",
            str(output_path),
            "--overwrite",
        ]
    if cache_id == "rfs":
        return [
            py,
            str(repo_root / "preprocessing" / "preprocess_rfs.py"),
            "--data-path",
            str(raw_input),
            "--output",
            str(output_path),
            "--overwrite",
        ]
    if cache_id == "pos":
        return [
            py,
            str(repo_root / "preprocessing" / "preprocess_nr_positioning.py"),
            "--data-path",
            str(raw_input),
            "--scene",
            "outdoor",
            "--output",
            str(output_path),
            "--overwrite",
        ]
    if cache_id == "uwb-indoor":
        return [
            py,
            str(repo_root / "preprocessing" / "preprocess_uwb_loc.py"),
            "--data-path",
            str(raw_input),
            "--environment",
            "environment0",
            "--output",
            str(output_path),
        ]
    if cache_id == "uwb-industrial":
        return [
            py,
            str(repo_root / "preprocessing" / "preprocess_ipin_loc.py"),
            "--data-path",
            str(raw_input),
            "--output",
            str(output_path),
            "--overwrite",
        ]
    if cache_id == "radcom":
        return [
            py,
            str(repo_root / "preprocessing" / "preprocess_radcom.py"),
            "--input",
            str(raw_input),
            "--output",
            str(output_path),
            "--batch-size",
            "1024",
            "--overwrite",
        ]
    if cache_id == "rml":
        return [
            py,
            str(repo_root / "preprocessing" / "preprocess_rml.py"),
            "--data-file",
            str(raw_input),
            "--version",
            "2022",
            "--output",
            str(output_path),
            "--overwrite",
        ]
    if cache_id == "rfp":
        return [
            py,
            str(repo_root / "preprocessing" / "preprocess_rfp.py"),
            "--data-path",
            str(raw_input),
            "--output",
            str(output_path),
            "--overwrite",
        ]
    if cache_id == "interf":
        return [
            py,
            str(repo_root / "preprocessing" / "preprocess_icarus.py"),
            "--data-path",
            str(raw_input),
            "--output",
            str(output_path),
            "--overwrite",
        ]
    if cache_id == "deepmimo":
        cmd = [
            py,
            str(repo_root / "preprocessing" / "preprocess_deepmimo.py"),
            "--output",
            str(output_path),
            "--n-beams",
            "64",
            "--n-beams-list",
            "16,32,64",
            "--resize-size",
            "32",
            "--overwrite",
        ]
        if raw_input.is_file():
            cmd[2:2] = ["--data-pickle", str(raw_input)]
        else:
            cmd[2:2] = ["--dataset-folder", str(raw_input)]
        return cmd
    if cache_id == "lwm-beam-challenge":
        return [
            py,
            str(repo_root / "preprocessing" / "preprocess_lwm_beam_challenge.py"),
            "--train-data",
            str(raw_input / "train" / "bp_data_train.p"),
            "--train-labels",
            str(raw_input / "train" / "bp_label_train.p"),
            "--test-data",
            str(raw_input / "test" / "bp_data_test.p"),
            "--output",
            str(output_path),
            "--overwrite",
        ]
    raise KeyError(f"Unknown cache id: {cache_id}")


def build_train_command(
    task: str,
    mode: str,
    seed: int,
    cache_path: Path,
    output_dir: Path,
    ckpt_path: Path,
    repo_root: Path = REPO_ROOT,
    num_workers: int = 2,
    val_split: float = 0.2,
    save_every: int = DEFAULT_SAVE_EVERY,
    resume_path: Path | None = None,
    train_subset_fraction: float | None = None,
    train_subset_size: int | None = None,
) -> list[str]:
    spec = TASK_SPECS[task]
    cmd = [
        sys.executable,
        str(repo_root / "main_finetune.py"),
        "--task",
        task,
        "--train-data",
        str(cache_path),
        "--output-dir",
        str(output_dir),
        "--save-every",
        str(save_every),
        "--batch-size",
        str(spec["batch_size"]),
        "--num-workers",
        str(num_workers),
        "--epochs",
        str(spec["epochs"]),
        "--seed",
        str(seed),
        "--val-split",
        str(val_split),
        "--model",
        "vit_multi_small",
        "--warmup-epochs",
        "5",
        "--device",
        preferred_runtime_device(),
        "--use-conditional-ln",
    ]

    if mode != "sl":
        cmd += ["--finetune", str(ckpt_path)]

    if resume_path is not None:
        cmd += ["--resume", str(resume_path)]

    if mode == "ft2":
        cmd += ["--frozen-blocks", "6"]
    elif mode == "lora":
        cmd += ["--lora", "--lora-rank", "32", "--lora-alpha", "64"]
    elif mode == "strict":
        cmd += ["--strict-probe"]
    elif mode == "sl":
        cmd += ["--sl-baseline"]
    elif mode != "lp":
        raise ValueError(f"Unsupported mode: {mode}")

    if spec.get("stratified_split"):
        cmd.append("--stratified-split")
    if spec.get("class_weights"):
        cmd.append("--class-weights")
    if spec.get("smoothing") is not None:
        cmd += ["--smoothing", str(spec["smoothing"])]
    if spec.get("accum_steps") is not None:
        cmd += ["--accum-steps", str(spec["accum_steps"])]
    if spec.get("vis_img_size") is not None:
        cmd += ["--vis-img-size", str(spec["vis_img_size"])]
    if spec.get("deepmimo_n_beams") is not None:
        cmd += ["--deepmimo-n-beams", str(spec["deepmimo_n_beams"])]
    if train_subset_fraction is not None:
        cmd += ["--train-subset-fraction", str(train_subset_fraction)]
    if train_subset_size is not None:
        cmd += ["--train-subset-size", str(train_subset_size)]

    return cmd
