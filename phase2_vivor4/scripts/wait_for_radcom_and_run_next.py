from __future__ import annotations

import argparse
import html
import json
import os
import platform
import pty
import re
import select
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from _workspace_entrypoint import ensure_project_python
except ModuleNotFoundError:  # pragma: no cover - optional local launcher helper
    def ensure_project_python(_: Path) -> None:
        return

ensure_project_python(Path(__file__).resolve())

import h5py

try:
    import torch
except Exception:  # pragma: no cover - dashboard can still render without torch
    torch = None

try:
    from benchmark_config import (
        CACHE_SPECS,
        COMPARISON_ROOT,
        CURRENT_SESSION_LINK,
        CURRENT_SESSION_META,
        classify_requested_tasks,
        DEFAULT_NUM_WORKERS,
        DEFAULT_SAVE_EVERY,
        LOCAL_SUMMARY_ROOT,
        OFFICIAL_RESULTS_ROOT,
        resolved_cache_path_for_task,
        RUNS_ROOT,
        TASK_SPECS,
        cache_path_for_task,
        discover_raw_input,
        session_layout,
    )
except ModuleNotFoundError:  # pragma: no cover - package import path
    from .benchmark_config import (
        CACHE_SPECS,
        COMPARISON_ROOT,
        CURRENT_SESSION_LINK,
        CURRENT_SESSION_META,
        classify_requested_tasks,
        DEFAULT_NUM_WORKERS,
        DEFAULT_SAVE_EVERY,
        LOCAL_SUMMARY_ROOT,
        OFFICIAL_RESULTS_ROOT,
        resolved_cache_path_for_task,
        RUNS_ROOT,
        TASK_SPECS,
        cache_path_for_task,
        discover_raw_input,
        session_layout,
    )

PHASE2_ROOT = PROJECT_ROOT / "phase2_vivor4"
CACHE_ROOT = PROJECT_ROOT / "datasets_h5"
SESSION_ROOT = PHASE2_ROOT
RESULTS_ROOT = PHASE2_ROOT / "local_results" / "by_task"
LOG_ROOT = PHASE2_ROOT / "automation_logs"
STATUS_PATH = LOG_ROOT / "after_radcom_status.json"
RUN_LOG_PATH = LOG_ROOT / "after_radcom_run.log"
DASHBOARD_PATH = LOG_ROOT / "dashboard.html"
PLOT_MANIFEST_PATH = PHASE2_ROOT / "plots" / "plot_manifest.json"
SUMMARY_RUNS_JSON = LOCAL_SUMMARY_ROOT / "local_results_runs.json"
SUMMARY_AGG_JSON = LOCAL_SUMMARY_ROOT / "local_results_aggregated.json"
SUMMARY_MANIFEST_PATH = LOCAL_SUMMARY_ROOT / "summary_manifest.json"
COMPARE_JSON = COMPARISON_ROOT / "local_vs_official.json"
COMPARISON_MANIFEST_PATH = COMPARISON_ROOT / "comparison_manifest.json"
OFFICIAL_JSON = OFFICIAL_RESULTS_ROOT / "official_results_all.json"
AUTO_REFRESH_SECONDS = 5

DEFAULT_TASKS = [
    "sensing",
    "pos",
    "uwb-indoor",
    "uwb-industrial",
    "radcom",
    "rml",
    "rfp",
    "interf",
    "deepmimo-los",
    "deepmimo-beam",
]
DEFAULT_MODES = ["lp", "ft2", "lora"]
DEFAULT_SEEDS = [0, 1, 2]

REQUIRED_RADCOM_KEYS = {"sample", "label", "modulation", "signal_type", "snr"}
REQUIRED_RADCOM_ATTRS = {"class_weights", "label_pairs", "mean", "std"}

RADCOM_PASS1_RE = re.compile(r"Pass 1: read\+write:\s+(\d+)%.*?(\d+)/(\d+)")
RADCOM_PASS2_RE = re.compile(r"Pass 2: normalize:\s+(\d+)%.*?(\d+)/(\d+)")
RUN_RE = re.compile(r"\[run\] task=(\S+) mode=(\S+) seed=(\d+)")
TRAIN_CONFIG_RE = re.compile(r"\[train\]\s+epochs=(\d+)\s+base_lr=.*?accum_steps=(\d+)")
TRAIN_STEP_RE = re.compile(
    r"\[train\]\s+epoch=([0-9]+(?:\.[0-9]+)?)\s+step=(\d+)/(\d+)(?:\s+loss=([0-9.eE+-]+)\s+lr=([0-9.eE+-]+))?"
)
DONE_RE = re.compile(r"\[done\]\s+training time\s+([^|]+)\|\s+best\s+(\S+)=([^\s]+)")
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
CHECKPOINT_RE = re.compile(r"checkpoint_(\d+)\.pth$")

PHASE_PARALLEL = "parallel_ready_benchmarks_and_radcom_preprocessing"
PHASE_READY_BENCH = "running_ready_benchmarks_after_radcom_preprocessing"
PHASE_WAIT_RADCOM = "waiting_for_radcom_after_ready_benchmarks"
PHASE_RADCOM_BENCH = "running_radcom_benchmarks"
PHASE_PLOTS = "rendering_detailed_eval_plots"
PHASE_SUMMARY = "summarizing_results"
PHASE_COMPARE = "comparing_with_official"
PHASE_COMPLETED = "completed"
PHASE_ERROR = "error"

_RUNTIME_HARDWARE_CACHE: dict | None = None
_SYSTEM_SNAPSHOT_CACHE: tuple[float, dict] | None = None
ACTIVE_CHILD_PROCS: dict[int, dict] = {}
_SIGNAL_HANDLERS_INSTALLED = False
_TRACKER_TERMINATION_IN_PROGRESS = False


class TrackerTermination(RuntimeError):
    pass


def now_ts() -> float:
    return time.time()


def default_session_name() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    host = platform.node().split(".", 1)[0] or "host"
    return f"{stamp}-{host}"


def resolve_session_root(session_root: Path | None, reuse_current_session: bool) -> Path:
    if session_root is not None:
        return session_root.expanduser().resolve()
    if reuse_current_session and CURRENT_SESSION_META.exists():
        data = safe_read_json(CURRENT_SESSION_META)
        if isinstance(data, dict) and data.get("session_root"):
            return Path(str(data["session_root"])).expanduser().resolve()
    candidate = RUNS_ROOT / default_session_name()
    idx = 1
    while candidate.exists():
        candidate = RUNS_ROOT / f"{default_session_name()}-{idx}"
        idx += 1
    return candidate.resolve()


def update_current_session_pointer(session_root: Path) -> None:
    session_root = session_root.resolve()
    PHASE2_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "session_root": str(session_root),
        "updated_at_utc": utc_now(),
        "host": platform.node(),
    }
    atomic_write(CURRENT_SESSION_META, json.dumps(payload, indent=2) + "\n")
    if CURRENT_SESSION_LINK.exists() or CURRENT_SESSION_LINK.is_symlink():
        if CURRENT_SESSION_LINK.is_symlink() or CURRENT_SESSION_LINK.is_file():
            CURRENT_SESSION_LINK.unlink()
        else:
            return
    try:
        CURRENT_SESSION_LINK.symlink_to(session_root, target_is_directory=True)
    except OSError:
        pass


def configure_session_paths(session_root: Path) -> dict[str, Path]:
    global SESSION_ROOT, RESULTS_ROOT, PLOT_MANIFEST_PATH, SUMMARY_RUNS_JSON, SUMMARY_AGG_JSON
    global SUMMARY_MANIFEST_PATH, COMPARE_JSON, COMPARISON_MANIFEST_PATH
    SESSION_ROOT = session_root.resolve()
    layout = session_layout(SESSION_ROOT)
    RESULTS_ROOT = layout["results_root"]
    PLOT_MANIFEST_PATH = layout["plot_manifest_path"]
    SUMMARY_RUNS_JSON = layout["summary_runs_json"]
    SUMMARY_AGG_JSON = layout["summary_agg_json"]
    SUMMARY_MANIFEST_PATH = layout["summary_manifest_path"]
    COMPARE_JSON = layout["comparison_json"]
    COMPARISON_MANIFEST_PATH = layout["comparison_manifest_path"]
    return layout


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def human_duration(seconds: float | None) -> str:
    if seconds is None:
        return "estimating"
    seconds = max(0, int(round(seconds)))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours}h {minutes}m {secs}s"
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def iso_to_ts(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def safe_read_json(path: Path, *, fresh_after_ts: float | None = None):
    if not path.exists():
        return None
    if fresh_after_ts is not None and path.stat().st_mtime + 1e-6 < fresh_after_ts:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def parse_local_timestamp(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def status_badge(status: str) -> str:
    mapping = {
        "done": "done",
        "completed": "done",
        "validated": "done",
        "ok": "done",
        "active": "active",
        "running": "active",
        "ready": "ready",
        "armed": "done",
        "paused": "partial",
        "stopping": "error",
        "partial": "partial",
        "pending": "pending",
        "blocked": "blocked",
        "needs_restart": "restart",
        "restartable": "restart",
        "error": "error",
        "failed": "error",
        "unknown": "pending",
    }
    return mapping.get(status, "pending")


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def safe_stat(path: Path) -> os.stat_result | None:
    try:
        return path.stat()
    except OSError:
        return None


def format_bytes(size_bytes: int | float | None) -> str:
    if size_bytes is None:
        return "n/a"
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def format_percent_value(value: float | None) -> str:
    if value is None:
        return "n/a"
    text = f"{float(value):.2f}".rstrip("0").rstrip(".")
    return f"{text}%"


def build_experiment_context(
    tasks: list[str],
    train_subset_fraction: float | None,
    train_subset_size: int | None,
) -> dict:
    subset_percent = float(train_subset_fraction) * 100.0 if train_subset_fraction is not None else None
    is_subset = train_subset_fraction is not None or train_subset_size is not None
    task_scope = ", ".join(tasks) if tasks else "none"
    task_set = set(tasks)

    if is_subset:
        short_label = (
            f"{format_percent_value(subset_percent)} subset"
            if subset_percent is not None
            else f"{int(train_subset_size or 0):,}-sample subset"
        )
        control_label = (
            f"{format_percent_value(subset_percent)} of the training split"
            if subset_percent is not None
            else f"{int(train_subset_size or 0):,} train samples"
        )
        label = "Modulation subset study" if task_set == {"rml", "radcom"} else "Train-subset experiment"
        kind = "train_subset_study"
        sampling_policy = "Subset applied after the train/validation split; classification tasks use stratified sampling."
        storage_policy = (
            "All outputs stay under this session root and should remain separate from the main reproduction artifacts."
        )
    else:
        short_label = "full train split"
        control_label = "100% of the post-split training set"
        label = "Full-data session"
        kind = "full_data_session"
        sampling_policy = "Uses the full post-split training set."
        storage_policy = "All outputs stay under this session root."

    return {
        "kind": kind,
        "label": label,
        "short_label": short_label,
        "control_label": control_label,
        "task_scope": list(tasks),
        "task_scope_label": task_scope,
        "is_subset": is_subset,
        "train_subset_fraction": train_subset_fraction,
        "train_subset_percent": subset_percent,
        "train_subset_size": train_subset_size,
        "sampling_policy": sampling_policy,
        "storage_policy": storage_policy,
    }


def file_freshness(path: Path, *, fresh_after_ts: float | None = None) -> dict:
    stat = safe_stat(path)
    exists = stat is not None
    updated_ts = float(stat.st_mtime) if stat is not None else None
    updated_at_utc = datetime.fromtimestamp(updated_ts, tz=timezone.utc).isoformat() if updated_ts is not None else None
    age_seconds = max(0.0, now_ts() - updated_ts) if updated_ts is not None else None
    fresh = bool(exists and (fresh_after_ts is None or (updated_ts is not None and updated_ts >= fresh_after_ts - 1e-6)))
    return {
        "path": str(path),
        "exists": exists,
        "fresh": fresh,
        "size_bytes": int(stat.st_size) if stat is not None else None,
        "updated_ts": updated_ts,
        "updated_at_utc": updated_at_utc,
        "age_seconds": age_seconds,
    }


def read_run_metadata(task: str, mode: str, seed: int) -> dict | None:
    path = run_output_dir(task, mode, seed) / "metadata.json"
    data = safe_read_json(path)
    return data if isinstance(data, dict) else None


def classify_run_origin(metadata: dict | None) -> dict:
    host = platform.node().split(".", 1)[0] or platform.node()
    if not metadata:
        return {"label": "unknown", "status": "pending", "host": None}
    launch_host = str(metadata.get("launch_host") or "")
    run_origin = str(metadata.get("run_origin") or "fresh")
    normalized_host = launch_host.split(".", 1)[0] if launch_host else None
    launch_history = []
    for item in metadata.get("launch_history") or []:
        if isinstance(item, str) and item:
            launch_history.append(item.split(".", 1)[0])
    foreign_hosts = sorted({item for item in launch_history if item and item != host})
    if foreign_hosts and normalized_host == host:
        label = "lab resumed (transferred)" if run_origin == "resume" else "lab fresh (transferred)"
        return {"label": label, "status": "restart" if run_origin == "resume" else "blocked", "host": normalized_host}
    if normalized_host and normalized_host != host:
        return {"label": f"transferred ({normalized_host})", "status": "blocked", "host": normalized_host}
    if run_origin == "resume":
        return {"label": "lab resumed", "status": "restart", "host": normalized_host}
    if run_origin == "fresh":
        return {"label": "lab fresh", "status": "done", "host": normalized_host}
    return {"label": run_origin, "status": "pending", "host": normalized_host}


def append_log(source: str, message: str) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    with RUN_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"[{utc_now()}] [{source}] {message}\n")


def append_gpu_guard_action(state: dict, message: str) -> None:
    guard = state.setdefault("gpu_guard", {})
    actions = list(guard.get("actions") or [])
    actions.append({"time_utc": utc_now(), "message": message})
    guard["actions"] = actions[-12:]
    guard["last_action"] = message
    append_event(state, message)


def gpu_guard_number(guard: dict, key: str, default: float) -> float:
    value = guard.get(key)
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def clean_message(text: str) -> str:
    text = ANSI_RE.sub("", text)
    return text.strip()


def run_output_dir(task: str, mode: str, seed: int) -> Path:
    return RESULTS_ROOT / task / mode / f"s{seed}"


def ordered_tasks(tasks: list[str]) -> list[str]:
    task_list = list(tasks)
    ready = [task for task in task_list if task != "radcom"]
    if "radcom" in task_list:
        ready.append("radcom")
    return ready


def best_exists(task: str, mode: str, seed: int) -> bool:
    return (run_output_dir(task, mode, seed) / "best.pth").exists()


def latest_logged_epoch(run_dir: Path) -> int | None:
    log_path = run_dir / "log.txt"
    if not log_path.exists():
        return None
    lines = log_path.read_text(encoding="utf-8").splitlines()
    for line in reversed(lines):
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


def checkpoint_epoch(path: Path | None) -> int | None:
    if path is None:
        return None
    match = CHECKPOINT_RE.fullmatch(path.name)
    if match:
        return int(match.group(1))
    if path.name != "best.pth" or torch is None:
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    epoch = payload.get("epoch") if isinstance(payload, dict) else None
    return int(epoch) if isinstance(epoch, int) else None


def resume_checkpoint(task: str, mode: str, seed: int) -> Path | None:
    run_dir = run_output_dir(task, mode, seed)
    latest_epoch = latest_epoch_checkpoint(run_dir)
    if latest_epoch is not None:
        return latest_epoch
    best = run_dir / "best.pth"
    if best.exists():
        return best
    return None


def run_completed(task: str, mode: str, seed: int) -> bool:
    run_dir = run_output_dir(task, mode, seed)
    expected_epochs = int(TASK_SPECS[task]["epochs"])
    latest_epoch = latest_logged_epoch(run_dir)
    if latest_epoch is not None and latest_epoch + 1 >= expected_epochs:
        return True
    return (run_dir / f"checkpoint_{expected_epochs - 1:03d}.pth").exists()


def build_run_plan(tasks: list[str], modes: list[str], seeds: list[int]) -> list[dict]:
    plan: list[dict] = []
    for task in ordered_tasks(tasks):
        for mode in modes:
            for seed in seeds:
                completed = run_completed(task, mode, seed)
                resume_ckpt = resume_checkpoint(task, mode, seed)
                metadata = read_run_metadata(task, mode, seed)
                origin = classify_run_origin(metadata)
                plan.append(
                    {
                        "id": f"{task}|{mode}|{seed}",
                        "task": task,
                        "mode": mode,
                        "seed": seed,
                        "status": "completed" if completed else "pending",
                        "skip_expected": completed,
                        "resume_available": (str(resume_ckpt) if (resume_ckpt is not None and not completed) else None),
                        "run_origin": origin["label"],
                        "run_origin_status": origin["status"],
                        "launch_host": origin["host"],
                    }
                )
    return plan


def radcom_ready(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "radcom cache missing"
    try:
        with h5py.File(path, "r") as h5:
            keys = set(h5.keys())
            attrs = set(h5.attrs.keys())
            missing_keys = sorted(REQUIRED_RADCOM_KEYS - keys)
            missing_attrs = sorted(REQUIRED_RADCOM_ATTRS - attrs)
            if missing_keys:
                return False, f"missing keys: {missing_keys}"
            if missing_attrs:
                return False, f"missing attrs: {missing_attrs}"
            if h5["sample"].shape[0] == 0:
                return False, "empty sample dataset"
    except Exception as exc:
        return False, f"open failed: {type(exc).__name__}: {exc}"
    return True, "ready"


def run_ref(entry: dict | None) -> dict | None:
    if not entry:
        return None
    return {
        "task": entry["task"],
        "mode": entry["mode"],
        "seed": entry["seed"],
        "status": entry.get("status"),
        "skip_expected": bool(entry.get("skip_expected", False)),
        "resume_available": entry.get("resume_available"),
        "run_origin": entry.get("run_origin"),
        "run_origin_status": entry.get("run_origin_status"),
    }


def format_run_label(entry: dict | None) -> str:
    if not entry:
        return "none"
    return f"{entry['task']} / {entry['mode']} / s{entry['seed']}"


def format_run_progress(current_run: dict | None) -> str:
    if not current_run:
        return "idle"
    if current_run.get("epochs") and current_run.get("steps_per_epoch") and current_run.get("epoch") is not None:
        return (
            f"{format_run_label(current_run)} | "
            f"epoch {int(current_run['epoch']) + 1}/{int(current_run['epochs'])} | "
            f"step {int(current_run['step'])}/{int(current_run['steps_per_epoch'])}"
        )
    return format_run_label(current_run)


def append_event(state: dict, message: str) -> None:
    state["recent_events"].append({"time_utc": utc_now(), "message": message})
    state["recent_events"] = state["recent_events"][-50:]


def summarize_sequence_changes(label: str, previous: list, current: list) -> list[str]:
    prev = list(previous or [])
    curr = list(current or [])
    changes: list[str] = []
    added = [item for item in curr if item not in prev]
    removed = [item for item in prev if item not in curr]
    if added:
        changes.append(f"{label} added: {', '.join(str(item) for item in added)}")
    if removed:
        changes.append(f"{label} removed: {', '.join(str(item) for item in removed)}")
    if prev and curr and not added and not removed and prev != curr:
        changes.append(f"{label} order changed: {' -> '.join(str(item) for item in curr)}")
    return changes


def current_launch_order(plan: list[dict]) -> list[str]:
    order: list[str] = []
    for item in plan:
        task = str(item["task"])
        if not order or order[-1] != task:
            order.append(task)
    return order


def carry_forward_dashboard_context(state: dict, previous_state: dict | None) -> None:
    if not isinstance(previous_state, dict):
        state["plan"]["change_messages"] = []
        return

    prev_tracker = previous_state.get("tracker")
    if isinstance(prev_tracker, dict):
        state["tracker"]["restart_count"] = int(prev_tracker.get("restart_count") or 0) + 1
        state["tracker"]["previous_pid"] = prev_tracker.get("pid")
        state["tracker"]["last_restart_from_phase"] = previous_state.get("phase")
    runtime_history = previous_state.get("runtime_history")
    if isinstance(runtime_history, dict):
        state["runtime_history"] = {
            "interval_seconds": float(runtime_history.get("interval_seconds") or 30.0),
            "last_sample_ts": runtime_history.get("last_sample_ts"),
            "samples": list(runtime_history.get("samples") or [])[-120:],
        }

    prev_events = previous_state.get("recent_events")
    if isinstance(prev_events, list):
        state["recent_events"] = list(prev_events[-20:])

    previous_tasks = previous_state.get("tasks") or []
    previous_modes = previous_state.get("modes") or []
    previous_seeds = previous_state.get("seeds") or []
    previous_plan = list((previous_state.get("benchmarks") or {}).get("run_plan") or [])
    current_plan = list(state["benchmarks"]["run_plan"])

    changes: list[str] = []
    changes.extend(summarize_sequence_changes("Tasks", previous_tasks, state["tasks"]))
    changes.extend(summarize_sequence_changes("Modes", previous_modes, state["modes"]))
    changes.extend(summarize_sequence_changes("Seeds", previous_seeds, state["seeds"]))

    previous_launch_order = current_launch_order(previous_plan)
    current_plan_order = current_launch_order(current_plan)
    if previous_launch_order and current_plan_order and previous_launch_order != current_plan_order:
        changes.append(f"Launch order updated: {' -> '.join(current_plan_order)}")

    previous_total_runs = len(previous_plan)
    current_total_runs = len(current_plan)
    if previous_total_runs and previous_total_runs != current_total_runs:
        changes.append(f"Run plan size changed: {previous_total_runs} -> {current_total_runs}")

    previous_phase = previous_state.get("phase")
    if previous_phase and previous_phase != state["phase"]:
        changes.append(f"Tracker restarted from prior phase: {previous_phase}")

    restartable_runs = sum(1 for item in current_plan if item.get("status") == "pending" and item.get("resume_available"))
    if restartable_runs:
        changes.append(f"Detected {restartable_runs} restartable runs with checkpoints.")

    state["plan"]["change_messages"] = changes[-8:]
    for message in state["plan"]["change_messages"]:
        append_event(state, message)


def init_state(
    tasks: list[str],
    modes: list[str],
    seeds: list[int],
    radcom_cache: Path,
    *,
    num_workers: int,
    save_every: int,
    train_subset_fraction: float | None = None,
    train_subset_size: int | None = None,
) -> dict:
    plan = build_run_plan(tasks, modes, seeds)
    started = utc_now()
    expected_skips = sum(1 for item in plan if item["skip_expected"])
    nonradcom_total = sum(1 for item in plan if item["task"] != "radcom")
    radcom_total = sum(1 for item in plan if item["task"] == "radcom")
    layout = session_layout(SESSION_ROOT)
    return {
        "started_at_utc": started,
        "updated_at_utc": started,
        "state": "initializing",
        "phase": "setup",
        "tracker": {
            "pid": os.getpid(),
            "started_at_ts": now_ts(),
            "started_at_utc": started,
            "restart_count": 0,
            "previous_pid": None,
            "last_restart_from_phase": None,
        },
        "runtime_history": {
            "interval_seconds": 30.0,
            "last_sample_ts": None,
            "samples": [],
        },
        "gpu_guard": {
            "status": "armed",
            "last_check_utc": None,
            "last_temp_c": None,
            "poll_interval_seconds": 5.0,
            "pause_threshold_c": 86.0,
            "resume_threshold_c": 78.0,
            "critical_threshold_c": 92.0,
            "pause_min_seconds": 300.0,
            "hot_streak": 0,
            "cool_streak": 0,
            "critical_streak": 0,
            "paused": False,
            "pause_started_at_ts": None,
            "pause_started_at_utc": None,
            "pause_until_ts": None,
            "pause_reason": None,
            "stop_requested_at_ts": None,
            "stop_requested_at_utc": None,
            "stop_reason": None,
            "last_action": "armed",
            "options": [
                "Check airflow, fans, and ambient room temperature on the lab host.",
                "Stop other competing GPU jobs and background CUDA workloads.",
                "Reduce concurrent host load and keep the machine on AC power.",
                "Wait for the GPU temperature to return to a safe range before resuming.",
            ],
            "actions": [],
        },
        "parallel_mode": True,
        "pipeline_eta_seconds": None,
        "session_root": str(SESSION_ROOT),
        "session_manifest_path": str(layout["manifest_path"]),
        "results_root": str(layout["results_root"]),
        "tasks": tasks,
        "modes": modes,
        "seeds": seeds,
        "plan": {
            "task_order": list(tasks),
            "launch_order": current_launch_order(plan),
            "modes": list(modes),
            "seeds": list(seeds),
            "change_messages": [],
        },
        "num_workers": int(num_workers),
        "save_every": int(save_every),
        "train_subset_fraction": train_subset_fraction,
        "train_subset_size": train_subset_size,
        "status_path": str(STATUS_PATH),
        "dashboard_path": str(DASHBOARD_PATH),
        "run_log_path": str(RUN_LOG_PATH),
        "last_output": "",
        "last_error": None,
        "recent_events": [],
        "preflight": {
            "status": "pending",
            "started_at_utc": None,
            "completed_at_utc": None,
            "report_path": str(layout["preflight_report_path"]),
            "summary": None,
        },
        "current": {
            "label": "Preparing automation",
            "command": None,
            "started_at_ts": now_ts(),
            "started_at_utc": started,
            "progress_percent": 0.0,
            "progress_label": "starting",
            "eta_seconds": None,
        },
        "radcom": {
            "cache_path": str(radcom_cache),
            "status": "pending",
            "ready": False,
            "current_pass": None,
            "current": 0,
            "total": 0,
            "progress_percent": 0.0,
            "eta_seconds": None,
            "pass1_started_at_ts": None,
            "pass2_started_at_ts": None,
            "completed_at_ts": None,
            "completed_at_utc": None,
            "duration_seconds": None,
            "process_command": None,
        },
        "benchmarks": {
            "status": "pending",
            "stage": "pending",
            "started_at_ts": None,
            "started_at_utc": None,
            "completed_at_ts": None,
            "completed_at_utc": None,
            "duration_seconds": None,
            "total_runs": len(plan),
            "nonradcom_total_runs": nonradcom_total,
            "radcom_total_runs": radcom_total,
            "completed_runs": 0,
            "completed_nonradcom_runs": 0,
            "completed_radcom_runs": 0,
            "expected_skip_runs": expected_skips,
            "durations": [],
            "durations_by_task": {},
            "durations_by_task_mode": {},
            "avg_run_seconds": None,
            "duration_estimate_seconds": None,
            "benchmark_eta_seconds": None,
            "eta_nonradcom_seconds": None,
            "eta_radcom_runs_seconds": None,
            "progress_percent": 0.0,
            "current_run": None,
            "next_run": None,
            "next_runs": [],
            "process_command": None,
            "run_plan": plan,
        },
        "summary": {
            "status": "pending",
            "started_at_ts": None,
            "started_at_utc": None,
            "completed_at_ts": None,
            "completed_at_utc": None,
            "duration_seconds": None,
            "eta_seconds": None,
            "total_units": len(tasks) * len(modes),
            "completed_units": 0,
            "progress_percent": 0.0,
            "current_task": None,
            "current_mode": None,
            "current_seed": None,
            "current_item_label": None,
            "runs_json_path": str(SUMMARY_RUNS_JSON),
            "aggregated_json_path": str(SUMMARY_AGG_JSON),
            "output_root": str(layout["summary_root"]),
            "manifest_path": str(layout["summary_manifest_path"]),
        },
        "plots": {
            "status": "pending",
            "started_at_ts": None,
            "started_at_utc": None,
            "completed_at_ts": None,
            "completed_at_utc": None,
            "duration_seconds": None,
            "eta_seconds": None,
            "total_units": len(tasks) * len(modes),
            "completed_units": 0,
            "progress_percent": 0.0,
            "current_task": None,
            "current_mode": None,
            "current_seed": None,
            "current_item_label": None,
            "output_root": str(layout["plots_root"]),
            "manifest_path": str(PLOT_MANIFEST_PATH),
        },
        "comparison": {
            "status": "pending",
            "started_at_ts": None,
            "started_at_utc": None,
            "completed_at_ts": None,
            "completed_at_utc": None,
            "duration_seconds": None,
            "eta_seconds": None,
            "total_units": len(tasks) * len(modes),
            "completed_units": 0,
            "progress_percent": 0.0,
            "current_task": None,
            "current_mode": None,
            "current_seed": None,
            "current_item_label": None,
            "json_path": str(COMPARE_JSON),
            "official_json_path": str(OFFICIAL_JSON),
            "output_root": str(layout["comparison_root"]),
            "manifest_path": str(layout["comparison_manifest_path"]),
        },
        "overview": {},
    }


def refresh_plan_views(state: dict) -> None:
    bench = state["benchmarks"]
    plan = bench["run_plan"]
    pending = [item for item in plan if item["status"] == "pending"]
    bench["next_run"] = run_ref(pending[0]) if pending else None
    bench["next_runs"] = [run_ref(item) for item in pending[:5]]
    bench["completed_runs"] = sum(1 for item in plan if item["status"] in {"completed", "skipped"})
    bench["completed_nonradcom_runs"] = sum(
        1 for item in plan if item["task"] != "radcom" and item["status"] in {"completed", "skipped"}
    )
    bench["completed_radcom_runs"] = sum(
        1 for item in plan if item["task"] == "radcom" and item["status"] in {"completed", "skipped"}
    )


def ensure_current(state: dict, label: str, command: str | None) -> None:
    current = state["current"]
    if current.get("label") != label or current.get("command") != command:
        state["current"] = {
            "label": label,
            "command": command,
            "started_at_ts": now_ts(),
            "started_at_utc": utc_now(),
            "progress_percent": 0.0,
            "progress_label": "starting",
            "eta_seconds": None,
        }


def update_radcom_eta(state: dict) -> None:
    radcom = state["radcom"]
    current = int(radcom.get("current") or 0)
    total = int(radcom.get("total") or 0)
    cur_pass = radcom.get("current_pass")
    if not cur_pass or current <= 0 or total <= 0:
        radcom["eta_seconds"] = None
        return
    if cur_pass == "Pass 1: read+write":
        start = radcom.get("pass1_started_at_ts")
        if not start:
            radcom["eta_seconds"] = None
            return
        elapsed = max(1.0, now_ts() - float(start))
        rate = current / elapsed
        remaining_batches = (total - current) + total
        radcom["eta_seconds"] = remaining_batches / max(rate, 1e-8)
        radcom["progress_percent"] = 50.0 * current / total
    else:
        start = radcom.get("pass2_started_at_ts")
        if not start:
            radcom["eta_seconds"] = None
            return
        elapsed = max(1.0, now_ts() - float(start))
        rate = current / elapsed
        remaining_batches = total - current
        radcom["eta_seconds"] = remaining_batches / max(rate, 1e-8)
        radcom["progress_percent"] = 50.0 + (50.0 * current / total)


def update_benchmark_eta(state: dict) -> None:
    bench = state["benchmarks"]
    current_run = bench.get("current_run")

    durations = list(bench.get("durations", []))
    avg_run_seconds = statistics.fmean(durations) if durations else None

    duration_estimate = avg_run_seconds
    if duration_estimate is None and current_run:
        reference_total = estimate_reference_run_seconds(bench, str(current_run["task"]), str(current_run["mode"]))
        if reference_total is not None:
            duration_estimate = float(reference_total)
        else:
            launch_finished_units, _, fraction = current_run_launch_progress(current_run)
            if fraction > 0.0 and (launch_finished_units >= 10 or fraction >= 0.005):
                elapsed = max(1.0, now_ts() - float(current_run["started_at_ts"]))
                duration_estimate = elapsed / fraction

    bench["avg_run_seconds"] = avg_run_seconds
    bench["duration_estimate_seconds"] = duration_estimate

    nonrad_pending = [
        item
        for item in bench["run_plan"]
        if item["task"] != "radcom" and item["status"] == "pending" and not item["skip_expected"]
    ]
    rad_pending = [
        item
        for item in bench["run_plan"]
        if item["task"] == "radcom" and item["status"] == "pending" and not item["skip_expected"]
    ]

    nonrad_estimates = [estimate_entry_run_seconds(bench, item) for item in nonrad_pending]
    rad_estimates = [estimate_entry_run_seconds(bench, item) for item in rad_pending]
    nonrad_eta = None if any(val is None for val in nonrad_estimates) else float(sum(nonrad_estimates))
    rad_eta = None if any(val is None for val in rad_estimates) else float(sum(rad_estimates))
    if current_run:
        current_remaining = estimate_current_run_remaining_seconds(bench, current_run)
        if current_run["task"] == "radcom":
            rad_eta = None if current_remaining is None or rad_eta is None else float(current_remaining) + rad_eta
        else:
            nonrad_eta = None if current_remaining is None or nonrad_eta is None else float(current_remaining) + nonrad_eta

    bench["eta_nonradcom_seconds"] = nonrad_eta
    bench["eta_radcom_runs_seconds"] = rad_eta
    if nonrad_eta is not None or rad_eta is not None:
        bench["benchmark_eta_seconds"] = float(nonrad_eta or 0.0) + float(rad_eta or 0.0)
    else:
        bench["benchmark_eta_seconds"] = None

    current_fraction = 0.0
    if current_run:
        current_fraction = float(current_run.get("progress_fraction") or 0.0)
    total_runs = max(1, int(bench["total_runs"]))
    bench["progress_percent"] = 100.0 * (float(bench["completed_runs"]) + current_fraction) / total_runs


def update_pipeline_eta(state: dict) -> None:
    bench = state["benchmarks"]
    radcom = state["radcom"]
    phase = state["phase"]

    if phase == PHASE_PLOTS:
        state["pipeline_eta_seconds"] = state["plots"].get("eta_seconds")
        return
    if phase == PHASE_SUMMARY:
        state["pipeline_eta_seconds"] = state["summary"].get("eta_seconds")
        return
    if phase == PHASE_COMPARE:
        state["pipeline_eta_seconds"] = state["comparison"].get("eta_seconds")
        return
    if phase == PHASE_COMPLETED:
        state["pipeline_eta_seconds"] = 0.0
        return

    nonrad_eta = bench.get("eta_nonradcom_seconds")
    rad_runs_eta = bench.get("eta_radcom_runs_seconds")
    radcom_prep_eta = None if radcom.get("ready") else radcom.get("eta_seconds")

    if phase == PHASE_RADCOM_BENCH:
        state["pipeline_eta_seconds"] = rad_runs_eta
        return

    if nonrad_eta is None and rad_runs_eta is None and radcom_prep_eta is None:
        state["pipeline_eta_seconds"] = None
        return

    wait_for_radcom = 0.0
    if not radcom.get("ready"):
        if radcom_prep_eta is None:
            state["pipeline_eta_seconds"] = None
            return
        wait_for_radcom = max(float(radcom_prep_eta) - float(nonrad_eta or 0.0), 0.0)

    state["pipeline_eta_seconds"] = float(nonrad_eta or 0.0) + wait_for_radcom + float(rad_runs_eta or 0.0)


def active_commands(state: dict) -> str | None:
    commands: list[str] = []
    radcom_cmd = state["radcom"].get("process_command")
    bench_cmd = state["benchmarks"].get("process_command")
    if radcom_cmd and state["radcom"].get("status") == "running":
        commands.append(radcom_cmd)
    if bench_cmd and state["benchmarks"].get("status") == "running":
        commands.append(bench_cmd)
    if not commands:
        return None
    return " || ".join(commands)


def run_capture(cmd: list[str], *, cwd: Path | None = None) -> str | None:
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    output = (proc.stdout or proc.stderr or "").strip()
    return output or None


def build_session_manifest(state: dict, tracker_cmd: list[str]) -> dict:
    session_root = Path(str(state["session_root"]))
    layout = session_layout(session_root)
    experiment = build_experiment_context(
        list(state["tasks"]),
        state.get("train_subset_fraction"),
        state.get("train_subset_size"),
    )
    return {
        "session_root": str(session_root),
        "generated_at_utc": utc_now(),
        "host": platform.node(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "tracker_command": tracker_cmd,
        "tracker_command_shell": shlex.join(tracker_cmd),
        "paths": {
            "status_path": str(STATUS_PATH),
            "dashboard_path": str(DASHBOARD_PATH),
            "run_log_path": str(RUN_LOG_PATH),
            "results_root": str(layout["results_root"]),
            "plots_root": str(layout["plots_root"]),
            "plots_manifest_path": str(layout["plot_manifest_path"]),
            "summary_root": str(layout["summary_root"]),
            "summary_runs_json": str(layout["summary_runs_json"]),
            "summary_agg_json": str(layout["summary_agg_json"]),
            "summary_manifest_path": str(layout["summary_manifest_path"]),
            "comparison_root": str(layout["comparison_root"]),
            "comparison_json": str(layout["comparison_json"]),
            "comparison_manifest_path": str(layout["comparison_manifest_path"]),
            "official_json": str(OFFICIAL_JSON),
            "checkpoint_path": str(PROJECT_ROOT / "checkpoints" / "wavesfm-v1p0.pth"),
            "cache_root": str(CACHE_ROOT),
        },
        "experiment": {
            "kind": experiment["kind"],
            "label": experiment["label"],
            "short_label": experiment["short_label"],
            "task_scope": experiment["task_scope"],
            "task_scope_label": experiment["task_scope_label"],
            "train_subset_fraction": experiment["train_subset_fraction"],
            "train_subset_percent": experiment["train_subset_percent"],
            "train_subset_size": experiment["train_subset_size"],
            "control_label": experiment["control_label"],
            "sampling_policy": experiment["sampling_policy"],
            "storage_policy": experiment["storage_policy"],
            "applies_after_train_validation_split": bool(experiment["is_subset"]),
        },
        "plan": {
            "tasks": list(state["tasks"]),
            "modes": list(state["modes"]),
            "seeds": list(state["seeds"]),
        },
        "runtime_hardware": runtime_hardware(),
        "system_snapshot": system_runtime_snapshot(refresh_after_seconds=0),
        "git": {
            "head": run_capture(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"]),
            "branch": run_capture(["git", "-C", str(PROJECT_ROOT), "rev-parse", "--abbrev-ref", "HEAD"]),
            "status_short": run_capture(["git", "-C", str(PROJECT_ROOT), "status", "--short"]),
        },
        "nvidia_smi": run_capture(["nvidia-smi"]),
    }


def write_session_manifest(state: dict, tracker_cmd: list[str], *, preflight_report: dict | None = None) -> None:
    manifest = build_session_manifest(state, tracker_cmd)
    if preflight_report is not None:
        manifest["preflight"] = preflight_report
    atomic_write(Path(str(state["session_manifest_path"])), json.dumps(manifest, indent=2) + "\n")


def build_data_readiness(state: dict) -> dict:
    grouped: dict[str, list[str]] = {}
    for task in ordered_tasks(state["tasks"]):
        grouped.setdefault(str(TASK_SPECS[task]["cache_id"]), []).append(task)

    rows: list[dict] = []
    ready_tasks = 0
    blocked_tasks = 0
    for cache_id, tasks in grouped.items():
        representative = tasks[0]
        spec = CACHE_SPECS[cache_id]
        cache_path = (
            Path(str(state["radcom"]["cache_path"]))
            if cache_id == "radcom"
            else cache_path_for_task(representative)
        )
        raw_path = discover_raw_input(cache_id)
        stamp = file_freshness(cache_path)
        if cache_id == "radcom":
            ready, detail = radcom_ready(cache_path)
        elif stamp["exists"] and (stamp["size_bytes"] or 0) > 0:
            ready, detail = True, "ready"
        else:
            ready, detail = False, "cache missing or empty"
        status = "done" if ready else "blocked"
        if not ready:
            detail = f"{detail}; raw {'present' if raw_path.exists() else 'missing'} at {raw_path}"
            blocked_tasks += len(tasks)
        else:
            ready_tasks += len(tasks)
        rows.append(
            {
                "cache_id": cache_id,
                "display_name": spec["display_name"],
                "cache_name": spec["cache_name"],
                "status": status,
                "detail": detail,
                "tasks": list(tasks),
                "task_labels": [TASK_SPECS[task]["display_name"] for task in tasks],
                "cache": stamp,
                "raw_path": str(raw_path),
                "raw_exists": raw_path.exists(),
            }
        )

    selection_blocked = state.get("blocked_tasks") or {}
    requested_tasks = ordered_tasks(state.get("requested_tasks") or state["tasks"])
    for task in requested_tasks:
        if task not in selection_blocked:
            continue
        cache_id = str(TASK_SPECS[task]["cache_id"])
        spec = CACHE_SPECS[cache_id]
        cache_path = resolved_cache_path_for_task(
            task,
            radcom_cache=Path(str(state["radcom"]["cache_path"])),
        )
        raw_path = discover_raw_input(cache_id)
        rows.append(
            {
                "cache_id": cache_id,
                "display_name": spec["display_name"],
                "cache_name": spec["cache_name"],
                "status": "blocked",
                "detail": str(selection_blocked[task]),
                "tasks": [task],
                "task_labels": [TASK_SPECS[task]["display_name"]],
                "cache": file_freshness(cache_path),
                "raw_path": str(raw_path),
                "raw_exists": raw_path.exists(),
            }
        )
        blocked_tasks += 1
    return {
        "rows": rows,
        "total_groups": len(rows),
        "ready_groups": sum(1 for row in rows if row["status"] == "done"),
        "blocked_groups": sum(1 for row in rows if row["status"] != "done"),
        "ready_tasks": ready_tasks,
        "blocked_tasks": blocked_tasks,
    }


def build_artifact_freshness(state: dict) -> dict:
    session_started_ts = iso_to_ts(state.get("started_at_utc"))
    checks = [
        ("Session Manifest", Path(str(state["session_manifest_path"])), session_started_ts, "session manifest"),
        ("Preflight Report", Path(str(state["preflight"]["report_path"])), session_started_ts, "preflight"),
        ("Plot Manifest", Path(str(state["plots"]["manifest_path"])), state["plots"].get("started_at_ts") or session_started_ts, "plots"),
        (
            "Summary Aggregate",
            Path(str(state["summary"]["aggregated_json_path"])),
            state["summary"].get("started_at_ts") or session_started_ts,
            "summary",
        ),
        (
            "Comparison JSON",
            Path(str(state["comparison"]["json_path"])),
            state["comparison"].get("started_at_ts") or session_started_ts,
            "comparison",
        ),
    ]
    rows = []
    for label, path, fresh_after_ts, section in checks:
        stamp = file_freshness(path, fresh_after_ts=fresh_after_ts)
        if stamp["exists"] and stamp["fresh"]:
            status = "done"
            detail = "fresh for current session"
        elif stamp["exists"]:
            status = "blocked"
            detail = "present but older than current session/phase"
        else:
            expected_status = state.get(section, {}).get("status") if isinstance(state.get(section), dict) else None
            status = "error" if expected_status == "completed" else "pending"
            detail = "not created yet"
        rows.append({"label": label, "status": status, "detail": detail, "file": stamp})
    return {
        "rows": rows,
        "fresh_count": sum(1 for row in rows if row["status"] == "done"),
        "stale_count": sum(1 for row in rows if row["status"] == "blocked"),
        "missing_count": sum(1 for row in rows if row["status"] in {"pending", "error"}),
    }


def build_resume_safety_panel(state: dict) -> dict:
    current_run = state["benchmarks"].get("current_run")
    target = current_run
    if target is None:
        target = next(
            (
                item
                for item in state["benchmarks"]["run_plan"]
                if item.get("status") == "pending" and item.get("resume_available")
            ),
            None,
        )
    if target is None:
        return {
            "status": "pending",
            "title": "No active or restartable run",
            "detail": "Resume safety updates when a run is active or a checkpoint-backed restart is pending.",
        }

    task = str(target["task"])
    mode = str(target["mode"])
    seed = int(target["seed"])
    run_dir = run_output_dir(task, mode, seed)
    latest_log_epoch = latest_logged_epoch(run_dir)
    latest_epoch_ckpt = latest_epoch_checkpoint(run_dir)
    resume_path = Path(str(target["resume_available"])) if target.get("resume_available") else resume_checkpoint(task, mode, seed)
    resume_epoch = checkpoint_epoch(resume_path)
    best_epoch = checkpoint_epoch((run_dir / "best.pth") if (run_dir / "best.pth").exists() else None)
    stale_log_tail = (
        latest_log_epoch is not None and resume_epoch is not None and latest_log_epoch > resume_epoch
    )
    stale_best = best_epoch is not None and resume_epoch is not None and best_epoch > resume_epoch
    trimmed_logs = sorted(run_dir.glob("log_pretrim_*.txt"))
    quarantined_best = sorted(run_dir.glob("best_pretrim_epoch*.pth"))
    status = "error" if stale_log_tail or stale_best else "done" if resume_path else "pending"
    detail = (
        "stale log or best checkpoint detected"
        if status == "error"
        else "resume state looks consistent"
        if resume_path
        else "fresh run"
    )
    return {
        "status": status,
        "title": format_run_label(target),
        "detail": detail,
        "latest_log_epoch": latest_log_epoch,
        "resume_checkpoint": str(resume_path) if resume_path is not None else None,
        "resume_epoch": resume_epoch,
        "latest_epoch_checkpoint": str(latest_epoch_ckpt) if latest_epoch_ckpt is not None else None,
        "latest_epoch_checkpoint_label": latest_epoch_ckpt.name if latest_epoch_ckpt is not None else "n/a",
        "best_epoch": best_epoch,
        "stale_log_tail": stale_log_tail,
        "stale_best": stale_best,
        "trimmed_logs": [path.name for path in trimmed_logs[-3:]],
        "quarantined_best": [path.name for path in quarantined_best[-3:]],
    }


def build_command_panel(state: dict) -> dict:
    session_root = Path(str(state["session_root"]))
    stored_num_workers = state.get("num_workers")
    num_workers = int(DEFAULT_NUM_WORKERS if stored_num_workers is None else stored_num_workers)
    save_every = int(state.get("save_every") or DEFAULT_SAVE_EVERY)
    train_subset_fraction = state.get("train_subset_fraction")
    train_subset_size = state.get("train_subset_size")
    tracker_cmd = [
        sys.executable,
        "phase2_vivor4/scripts/wait_for_radcom_and_run_next.py",
        "--session-root",
        str(session_root),
        "--tasks",
        *[str(task) for task in state["tasks"]],
        "--modes",
        *[str(mode) for mode in state["modes"]],
        "--seeds",
        *[str(seed) for seed in state["seeds"]],
        "--radcom-cache",
        str(Path(str(state["radcom"]["cache_path"]))),
        "--num-workers",
        str(num_workers),
        "--save-every",
        str(save_every),
    ]
    if train_subset_fraction is not None:
        tracker_cmd += ["--train-subset-fraction", str(train_subset_fraction)]
    if train_subset_size is not None:
        tracker_cmd += ["--train-subset-size", str(train_subset_size)]
    preflight_cmd = [
        sys.executable,
        "phase2_vivor4/scripts/preflight_check.py",
        "--session-root",
        str(session_root),
        "--tasks",
        *[str(task) for task in state["tasks"]],
        "--modes",
        *[str(mode) for mode in state["modes"]],
        "--seeds",
        *[str(seed) for seed in state["seeds"]],
        "--radcom-cache",
        str(Path(str(state["radcom"]["cache_path"]))),
        "--report-json",
        str(Path(str(state["preflight"]["report_path"]))),
    ]
    host = platform.node()
    ssh_base = (
        "ssh "
        "-o ServerAliveInterval=30 "
        "-o ServerAliveCountMax=3 "
        "-o ExitOnForwardFailure=yes "
        "-N -L 8765:localhost:8765 "
        f"{host}"
    )
    autossh_cmd = (
        "autossh -M 0 "
        "-o ServerAliveInterval=30 "
        "-o ServerAliveCountMax=3 "
        "-o ExitOnForwardFailure=yes "
        "-N -L 8765:localhost:8765 "
        f"{host}"
    )
    resilient_tunnel = (
        "while true; do "
        f"{ssh_base}; "
        "date; "
        'echo "dashboard tunnel lost; retrying in 2s"; '
        "sleep 2; "
        "done"
    )
    return {
        "tracker_restart": shlex.join(tracker_cmd),
        "preflight": shlex.join(preflight_cmd),
        "tmux_attach": "tmux attach -t wavesfm",
        "dashboard_server": f"cd {shlex.quote(str(LOG_ROOT))} && {shlex.quote(sys.executable)} -m http.server 8765",
        "dashboard_tunnel": ssh_base,
        "dashboard_tunnel_resilient": resilient_tunnel,
        "dashboard_tunnel_autossh": autossh_cmd,
        "dashboard_tunnel_helper": f"bash phase2_vivor4/scripts/dashboard_tunnel_watch.sh {host}",
        "dashboard_url": "http://localhost:8765/dashboard.html",
        "session_root": str(session_root),
        "results_root": str(RESULTS_ROOT),
    }


def run_preflight_check(state: dict) -> dict:
    report_path = Path(str(state["preflight"]["report_path"]))
    cmd = [
        sys.executable,
        str(PHASE2_ROOT / "scripts" / "preflight_check.py"),
        "--session-root",
        str(Path(str(state["session_root"]))),
        "--tasks",
        *[str(task) for task in state["tasks"]],
        "--modes",
        *[str(mode) for mode in state["modes"]],
        "--seeds",
        *[str(seed) for seed in state["seeds"]],
        "--radcom-cache",
        str(Path(str(state["radcom"]["cache_path"]))),
        "--report-json",
        str(report_path),
    ]
    state["preflight"]["status"] = "running"
    state["preflight"]["started_at_utc"] = utc_now()
    state["preflight"]["summary"] = shlex.join(cmd)
    append_event(state, "Running preflight validation.")
    write_state(state)
    proc = subprocess.run(cmd, check=False, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    report = safe_read_json(report_path)
    if not isinstance(report, dict):
        report = {
            "status": "error",
            "summary": "preflight report was not written",
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    state["preflight"]["completed_at_utc"] = utc_now()
    state["preflight"]["status"] = "completed" if proc.returncode == 0 else "error"
    state["preflight"]["summary"] = str(report.get("summary") or "preflight completed")
    append_event(state, f"Preflight {state['preflight']['status']}: {state['preflight']['summary']}")
    write_state(state)
    if proc.returncode != 0:
        raise RuntimeError(f"Preflight failed: {state['preflight']['summary']}")
    return report


def mean_or_none(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def current_run_launch_progress(current_run: dict | None) -> tuple[int, int, float]:
    if not current_run:
        return 0, 0, 0.0
    steps_per_epoch = int(current_run.get("steps_per_epoch") or 0)
    epochs = int(current_run.get("epochs") or 0)
    if steps_per_epoch <= 0 or epochs <= 0:
        return 0, 0, 0.0
    resume_epoch = int(current_run.get("resume_epoch") or 0)
    epoch_idx = int(current_run.get("epoch") or 0)
    step_idx = int(current_run.get("step") or 0)
    launch_total_units = max((epochs - resume_epoch) * steps_per_epoch, 0)
    launch_finished_units = max(((epoch_idx - resume_epoch) * steps_per_epoch) + step_idx, 0)
    launch_fraction = (launch_finished_units / launch_total_units) if launch_total_units > 0 else 0.0
    return launch_finished_units, launch_total_units, float(launch_fraction)


def estimate_reference_run_seconds(bench: dict, task: str, mode: str) -> float | None:
    mode_values = list((bench.get("durations_by_task_mode") or {}).get(f"{task}|{mode}", []))
    if mode_values:
        return mean_or_none(mode_values)
    task_values = list((bench.get("durations_by_task") or {}).get(task, []))
    if task_values:
        return mean_or_none(task_values)
    global_values = list(bench.get("durations", []))
    if global_values:
        return mean_or_none(global_values)
    duration_estimate = bench.get("duration_estimate_seconds")
    if duration_estimate is not None:
        return float(duration_estimate)
    return None


def estimate_entry_run_seconds(bench: dict, entry: dict) -> float | None:
    return estimate_reference_run_seconds(bench, str(entry["task"]), str(entry["mode"]))


def estimate_current_run_remaining_seconds(bench: dict, current_run: dict | None) -> float | None:
    if not current_run:
        return None
    eta_now = current_run.get("eta_seconds")
    if eta_now is not None:
        return float(eta_now)
    reference_total = estimate_reference_run_seconds(bench, str(current_run["task"]), str(current_run["mode"]))
    fraction_total = float(current_run.get("progress_fraction") or 0.0)
    if reference_total is not None and fraction_total > 0.0:
        return max(float(reference_total) * (1.0 - fraction_total), 0.0)
    launch_finished_units, _, launch_fraction = current_run_launch_progress(current_run)
    if launch_fraction > 0.0 and (launch_finished_units >= 10 or launch_fraction >= 0.005):
        elapsed = max(1.0, now_ts() - float(current_run["started_at_ts"]))
        total_est = elapsed / launch_fraction
        return max(total_est - elapsed, 0.0)
    return None


def eta_confidence(bench: dict, current_run: dict | None, eta_tracking: dict | None, *, scope: str) -> dict:
    tracking = eta_tracking or {}
    samples = list(tracking.get("samples") or [])
    sample_count = len(samples)
    durations = list(bench.get("durations") or [])
    durations_by_task = bench.get("durations_by_task") or {}
    durations_by_task_mode = bench.get("durations_by_task_mode") or {}
    launch_finished_units, _, launch_fraction = current_run_launch_progress(current_run)

    score = 0.0
    reasons: list[str] = []

    if scope == "current":
        if launch_fraction >= 0.50:
            score += 0.55
            reasons.append("active run is past halfway")
        elif launch_fraction >= 0.15:
            score += 0.40
            reasons.append("active run has meaningful progress")
        elif launch_finished_units >= 10 or launch_fraction >= 0.03:
            score += 0.22
            reasons.append("active run has early progress")
        if current_run and current_run.get("steps_per_epoch"):
            score += 0.15
            reasons.append("run structure is known")
        if sample_count >= 2:
            score += 0.15
            reasons.append("multiple ETA check samples exist")
        elif sample_count >= 1:
            score += 0.08
            reasons.append("one ETA check sample exists")
    else:
        completed_runs = len(durations)
        task_coverage = sum(1 for values in durations_by_task.values() if values)
        mode_coverage = sum(1 for values in durations_by_task_mode.values() if values)
        if completed_runs >= 12:
            score += 0.40
            reasons.append(f"{completed_runs} completed runs on disk")
        elif completed_runs >= 6:
            score += 0.28
            reasons.append(f"{completed_runs} completed runs on disk")
        elif completed_runs >= 3:
            score += 0.18
            reasons.append(f"{completed_runs} completed runs on disk")
        elif completed_runs >= 1:
            score += 0.10
            reasons.append("some completed runs exist")
        if task_coverage >= 3:
            score += 0.25
            reasons.append(f"history covers {task_coverage} tasks")
        elif task_coverage >= 2:
            score += 0.18
            reasons.append(f"history covers {task_coverage} tasks")
        elif task_coverage >= 1:
            score += 0.10
            reasons.append("history covers one task")
        if mode_coverage >= 4:
            score += 0.10
            reasons.append(f"history covers {mode_coverage} task-mode pairs")
        elif mode_coverage >= 2:
            score += 0.06
            reasons.append(f"history covers {mode_coverage} task-mode pairs")
        if launch_fraction >= 0.15:
            score += 0.10
            reasons.append("current run has enough progress to calibrate")
        elif launch_finished_units >= 10 or launch_fraction >= 0.03:
            score += 0.05
            reasons.append("current run has some calibration progress")
        if sample_count >= 2:
            score += 0.08
            reasons.append("ETA checks have started comparing forecast vs elapsed time")
        elif sample_count >= 1:
            score += 0.04
            reasons.append("one ETA check sample exists")

    if score >= 0.72:
        level = "high"
    elif score >= 0.40:
        level = "medium"
    else:
        level = "low"

    if reasons:
        summary = "; ".join(reasons[:2])
    else:
        summary = "not enough history yet"

    return {"level": level, "summary": summary, "score": round(score, 2)}


def update_current_summary(state: dict) -> None:
    phase = state["phase"]
    bench = state["benchmarks"]
    radcom = state["radcom"]
    plots = state["plots"]
    summary = state["summary"]
    comparison = state["comparison"]
    pipeline_eta = state.get("pipeline_eta_seconds")
    current_run = bench.get("current_run")
    current_run_eta = estimate_current_run_remaining_seconds(bench, current_run)

    if phase == PHASE_PARALLEL:
        ensure_current(state, "Running ready benchmarks while preprocessing radcom", active_commands(state))
        if current_run:
            state["current"]["progress_label"] = format_run_progress(current_run)
        else:
            rad_label = radcom.get("current_pass") or "waiting for first radcom progress"
            state["current"]["progress_label"] = f"radcom: {rad_label}"
        state["current"]["progress_percent"] = bench.get("progress_percent", 0.0)
        state["current"]["eta_seconds"] = pipeline_eta if pipeline_eta is not None else current_run_eta
        return

    if phase == PHASE_READY_BENCH:
        ensure_current(state, "Running benchmark queue", active_commands(state))
        state["current"]["progress_label"] = format_run_progress(current_run)
        state["current"]["progress_percent"] = bench.get("progress_percent", 0.0)
        state["current"]["eta_seconds"] = pipeline_eta if pipeline_eta is not None else current_run_eta
        return

    if phase == PHASE_WAIT_RADCOM:
        ensure_current(state, "Waiting for radcom cache", active_commands(state))
        rad_label = radcom.get("current_pass") or "waiting"
        state["current"]["progress_label"] = (
            f"{bench['completed_nonradcom_runs']}/{bench['nonradcom_total_runs']} ready-task runs done | "
            f"{rad_label}: {radcom.get('current', 0)}/{radcom.get('total', 0)}"
        )
        state["current"]["progress_percent"] = bench.get("progress_percent", 0.0)
        state["current"]["eta_seconds"] = pipeline_eta if pipeline_eta is not None else current_run_eta
        return

    if phase == PHASE_RADCOM_BENCH:
        ensure_current(state, "Running benchmark queue", active_commands(state))
        state["current"]["progress_label"] = format_run_progress(current_run)
        state["current"]["progress_percent"] = bench.get("progress_percent", 0.0)
        state["current"]["eta_seconds"] = pipeline_eta if pipeline_eta is not None else current_run_eta
        return

    if phase == PHASE_SUMMARY:
        ensure_current(state, "Summarizing local results", state["current"].get("command"))
        total_units = int(summary.get("total_units") or expected_task_mode_units(state))
        completed_units = int(summary.get("completed_units") or 0)
        item_label = str(summary.get("current_item_label") or "waiting for first task-mode summary")
        state["current"]["progress_percent"] = float(summary.get("progress_percent") or 0.0)
        state["current"]["progress_label"] = f"{completed_units}/{total_units} task-modes done | {item_label}"
        state["current"]["eta_seconds"] = summary.get("eta_seconds")
        return

    if phase == PHASE_PLOTS:
        ensure_current(state, "Refreshing detailed-evaluation plots", state["current"].get("command"))
        total_units = int(plots.get("total_units") or expected_task_mode_units(state))
        completed_units = int(plots.get("completed_units") or 0)
        item_label = str(plots.get("current_item_label") or "waiting for first plot bundle")
        state["current"]["progress_percent"] = float(plots.get("progress_percent") or 0.0)
        state["current"]["progress_label"] = f"{completed_units}/{total_units} task-modes done | {item_label}"
        state["current"]["eta_seconds"] = plots.get("eta_seconds")
        return

    if phase == PHASE_COMPARE:
        ensure_current(state, "Comparing local results with official references", state["current"].get("command"))
        total_units = int(comparison.get("total_units") or expected_task_mode_units(state))
        completed_units = int(comparison.get("completed_units") or 0)
        item_label = str(comparison.get("current_item_label") or "waiting for first task-mode comparison")
        state["current"]["progress_percent"] = float(comparison.get("progress_percent") or 0.0)
        state["current"]["progress_label"] = f"{completed_units}/{total_units} task-modes done | {item_label}"
        state["current"]["eta_seconds"] = comparison.get("eta_seconds")
        return

    if phase == PHASE_COMPLETED:
        ensure_current(state, "Pipeline completed", None)
        state["current"]["progress_percent"] = 100.0
        state["current"]["progress_label"] = "completed"
        state["current"]["eta_seconds"] = 0.0
        return


def write_state(state: dict) -> None:
    update_runtime_history(state)
    refresh_plan_views(state)
    update_benchmark_eta(state)
    update_postprocess_progress(state)
    update_pipeline_eta(state)
    update_current_summary(state)
    update_eta_tracking(state)
    state["overview"] = build_overview(state)
    state["updated_at_utc"] = utc_now()
    atomic_write(STATUS_PATH, json.dumps(state, indent=2) + "\n")
    atomic_write(DASHBOARD_PATH, render_dashboard(state))


def step_statuses(state: dict) -> list[tuple[str, str]]:
    steps = [
        ("RadCom Cache", "pending"),
        ("Benchmarks", "pending"),
        ("Plots", "pending"),
        ("Summary", "pending"),
        ("Comparison", "pending"),
    ]

    radcom = state["radcom"]
    bench = state["benchmarks"]
    phase = state["phase"]

    if radcom.get("ready"):
        steps[0] = ("RadCom Cache", "done")
    elif radcom.get("status") == "running":
        steps[0] = ("RadCom Cache", "active")

    bench_done = all(item["status"] in {"completed", "skipped"} for item in bench["run_plan"])
    bench_active = bench.get("status") == "running" or (
        not bench_done and phase in {PHASE_PARALLEL, PHASE_READY_BENCH, PHASE_WAIT_RADCOM, PHASE_RADCOM_BENCH}
    )
    if bench_done:
        steps[1] = ("Benchmarks", "done")
    elif bench_active:
        steps[1] = ("Benchmarks", "active")

    if phase == PHASE_SUMMARY:
        steps[3] = ("Summary", "active")
    elif phase in {PHASE_COMPARE, PHASE_COMPLETED}:
        steps[3] = ("Summary", "done")

    if phase == PHASE_PLOTS:
        steps[2] = ("Plots", "active")
    elif phase in {PHASE_SUMMARY, PHASE_COMPARE, PHASE_COMPLETED}:
        steps[2] = ("Plots", "done")

    if phase == PHASE_COMPARE:
        steps[4] = ("Comparison", "active")
    elif phase == PHASE_COMPLETED:
        steps[4] = ("Comparison", "done")

    if phase == PHASE_ERROR:
        for idx, (label, status) in enumerate(steps):
            if status == "pending":
                steps[idx] = (label, "error")
                break
    return steps


def benchmark_counts(entries: list[dict]) -> dict[str, int]:
    counts = {
        "total": len(entries),
        "done": 0,
        "active": 0,
        "pending": 0,
        "needs_restart": 0,
        "error": 0,
    }
    for item in entries:
        status = item.get("status")
        if status in {"completed", "skipped"}:
            counts["done"] += 1
        elif status == "running":
            counts["active"] += 1
        elif status == "error":
            counts["error"] += 1
        elif item.get("resume_available"):
            counts["needs_restart"] += 1
        else:
            counts["pending"] += 1
    counts["remaining"] = counts["total"] - counts["done"]
    return counts


def benchmark_rollup(counts: dict[str, int]) -> str:
    total = counts["total"]
    if total and counts["done"] == total:
        return "done"
    if counts["error"] > 0:
        return "error"
    if counts["active"] > 0 or counts["done"] > 0:
        return "active"
    if counts["needs_restart"] > 0:
        return "needs_restart"
    return "pending"


def artifact_cutoff_ts(state: dict, section: str | None = None) -> float | None:
    if section and state.get(section, {}).get("started_at_ts"):
        return float(state[section]["started_at_ts"])
    return iso_to_ts(state.get("started_at_utc"))


def expected_task_mode_units(state: dict) -> int:
    return max(len(state.get("tasks") or []), 0) * max(len(state.get("modes") or []), 0)


def format_task_mode_item(task: str | None, mode: str | None, seed: int | None = None) -> str | None:
    if not task or not mode:
        return None
    if seed is None:
        return f"{task} / {mode}"
    return f"{task} / {mode} / s{seed}"


def latest_manifest_item_label(section_name: str, payload: dict) -> str | None:
    if section_name == "plots":
        generated = list(payload.get("generated") or [])
        if not generated:
            return None
        last = generated[-1]
        seed = last.get("seed")
        return format_task_mode_item(str(last.get("task") or ""), str(last.get("mode") or ""), int(seed) if seed is not None else None)

    entries = list(payload.get("entries") or [])
    if not entries:
        return None
    last = entries[-1]
    return format_task_mode_item(str(last.get("task") or ""), str(last.get("mode") or ""))


def refresh_postprocess_section(state: dict, section_name: str) -> None:
    section = state.get(section_name)
    if not isinstance(section, dict):
        return

    manifest_path = section.get("manifest_path")
    payload = None
    if manifest_path:
        payload = safe_read_json(Path(str(manifest_path)), fresh_after_ts=artifact_cutoff_ts(state, section_name))

    total_default = int(section.get("total_units") or 0) or expected_task_mode_units(state)
    total_units = total_default
    completed_units = int(section.get("completed_units") or 0)

    if isinstance(payload, dict):
        total_units = int(payload.get("total_task_modes") or payload.get("total_units") or total_units or 0)
        if section_name == "plots":
            completed_units = int(payload.get("completed_task_modes") or payload.get("completed_units") or len(payload.get("generated") or []))
        else:
            completed_units = int(payload.get("completed_task_modes") or payload.get("completed_units") or len(payload.get("entries") or []))
        section["current_task"] = payload.get("current_task")
        section["current_mode"] = payload.get("current_mode")
        current_seed = payload.get("current_seed")
        section["current_seed"] = int(current_seed) if current_seed is not None else None
        section["current_item_label"] = (
            format_task_mode_item(section.get("current_task"), section.get("current_mode"), section.get("current_seed"))
            or latest_manifest_item_label(section_name, payload)
        )
        if payload.get("status") == "completed" and section.get("status") == "running":
            section["eta_seconds"] = 0.0
    else:
        section.setdefault("current_task", None)
        section.setdefault("current_mode", None)
        section.setdefault("current_seed", None)
        section.setdefault("current_item_label", None)
        if section_name == "plots":
            plot_root = Path(str(section.get("manifest_path") or "")).parent
            completed_pairs: set[tuple[str, str]] = set()
            latest_path: Path | None = None
            latest_mtime = -1.0
            if plot_root.exists():
                for path in plot_root.rglob("*"):
                    if not path.is_file():
                        continue
                    rel_parts = path.relative_to(plot_root).parts
                    if len(rel_parts) < 3 or rel_parts[0] == "by_task":
                        continue
                    task, mode = rel_parts[0], rel_parts[1]
                    completed_pairs.add((task, mode))
                    mtime = path.stat().st_mtime
                    if mtime >= latest_mtime:
                        latest_mtime = mtime
                        latest_path = path
            completed_units = len(completed_pairs)
            if latest_path is not None:
                rel_parts = latest_path.relative_to(plot_root).parts
                section["current_task"] = rel_parts[0]
                section["current_mode"] = rel_parts[1]
                section["current_item_label"] = format_task_mode_item(rel_parts[0], rel_parts[1])
        elif section_name == "summary":
            rows = safe_read_json(Path(str(section.get("aggregated_json_path") or "")), fresh_after_ts=artifact_cutoff_ts(state, section_name))
            if isinstance(rows, list):
                completed_units = len(rows)
                if rows:
                    last = rows[-1]
                    section["current_task"] = last.get("task")
                    section["current_mode"] = last.get("mode")
                    section["current_item_label"] = format_task_mode_item(
                        str(last.get("task") or ""),
                        str(last.get("mode") or ""),
                    )
        elif section_name == "comparison":
            rows = safe_read_json(Path(str(section.get("json_path") or "")), fresh_after_ts=artifact_cutoff_ts(state, section_name))
            if isinstance(rows, list):
                completed_units = len(rows)
                if rows:
                    last = rows[-1]
                    section["current_task"] = last.get("task")
                    section["current_mode"] = last.get("mode")
                    section["current_item_label"] = format_task_mode_item(
                        str(last.get("task") or ""),
                        str(last.get("mode") or ""),
                    )

    total_units = max(total_units, 0)
    completed_units = max(0, min(completed_units, total_units if total_units > 0 else completed_units))
    section["total_units"] = total_units
    section["completed_units"] = completed_units

    if section.get("status") == "completed":
        section["progress_percent"] = 100.0
        section["eta_seconds"] = 0.0
        return

    if total_units > 0:
        section["progress_percent"] = 100.0 * completed_units / total_units
    else:
        section["progress_percent"] = 0.0

    started_at_ts = section.get("started_at_ts")
    if section.get("status") == "running" and started_at_ts and total_units > 0 and 0 < completed_units < total_units:
        elapsed = max(1.0, now_ts() - float(started_at_ts))
        seconds_per_unit = elapsed / completed_units
        section["eta_seconds"] = max(seconds_per_unit * (total_units - completed_units), 0.0)
    elif section.get("status") == "running" and completed_units >= total_units > 0:
        section["eta_seconds"] = 0.0
    elif section.get("status") != "running":
        section["eta_seconds"] = 0.0 if section.get("status") == "completed" else None


def update_postprocess_progress(state: dict) -> None:
    for section_name in ("plots", "summary", "comparison"):
        refresh_postprocess_section(state, section_name)


def load_summary_index(state: dict) -> dict[tuple[str, str], dict]:
    rows = safe_read_json(SUMMARY_AGG_JSON, fresh_after_ts=artifact_cutoff_ts(state, "summary"))
    if not isinstance(rows, list):
        return {}
    return {
        (str(row.get("task")), str(row.get("mode"))): row
        for row in rows
        if isinstance(row, dict) and row.get("task") and row.get("mode")
    }


def load_compare_index(state: dict) -> dict[tuple[str, str], dict]:
    rows = safe_read_json(COMPARE_JSON, fresh_after_ts=artifact_cutoff_ts(state, "comparison"))
    if not isinstance(rows, list):
        return {}
    return {
        (str(row.get("task")), str(row.get("mode"))): row
        for row in rows
        if isinstance(row, dict) and row.get("task") and row.get("mode")
    }


def collect_run_snapshots(state: dict) -> dict[tuple[str, str, int], dict]:
    snapshots: dict[tuple[str, str, int], dict] = {}
    for task in state["tasks"]:
        primary_metric = str(TASK_SPECS[task]["primary_metric"])
        for mode in state["modes"]:
            for seed in state["seeds"]:
                run_dir = run_output_dir(task, mode, seed)
                log_path = run_dir / "log.txt"
                snapshot = {
                    "task": task,
                    "mode": mode,
                    "seed": seed,
                    "primary_metric": primary_metric,
                    "best_metric": None,
                    "best_key": None,
                    "duration_seconds": None,
                    "first_ts": None,
                    "last_ts": None,
                    "completed": False,
                    "has_metrics": False,
                    "run_origin": "unknown",
                    "run_origin_status": "pending",
                    "launch_host": None,
                }
                metadata = read_run_metadata(task, mode, seed)
                origin = classify_run_origin(metadata)
                snapshot["run_origin"] = origin["label"]
                snapshot["run_origin_status"] = origin["status"]
                snapshot["launch_host"] = origin["host"]
                if not log_path.exists():
                    snapshots[(task, mode, seed)] = snapshot
                    continue

                first_ts = None
                last_ts = None
                last_best = None
                best_entry = None
                for line in log_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    ts = parse_local_timestamp(payload.get("timestamp"))
                    if ts is not None:
                        if first_ts is None:
                            first_ts = ts
                        last_ts = ts

                    cur = payload.get("best_metric")
                    if cur is None:
                        continue
                    if last_best is None or cur != last_best:
                        last_best = cur
                        best_entry = payload

                if first_ts is not None and last_ts is not None and last_ts >= first_ts:
                    snapshot["duration_seconds"] = last_ts - first_ts
                snapshot["first_ts"] = first_ts
                snapshot["last_ts"] = last_ts
                snapshot["completed"] = run_completed(task, mode, seed)
                if best_entry is not None:
                    snapshot["has_metrics"] = True
                    snapshot["best_metric"] = best_entry.get("best_metric")
                    snapshot["best_key"] = best_entry.get("best_key") or primary_metric

                snapshots[(task, mode, seed)] = snapshot
    return snapshots


def seed_benchmark_history_from_disk(state: dict) -> None:
    bench = state["benchmarks"]
    snapshots = collect_run_snapshots(state)
    durations: list[float] = []
    durations_by_task: dict[str, list[float]] = {}
    durations_by_task_mode: dict[str, list[float]] = {}

    for (task, mode, seed), snapshot in snapshots.items():
        if not snapshot.get("completed"):
            continue
        duration = snapshot.get("duration_seconds")
        if not isinstance(duration, (int, float)) or duration <= 0:
            continue
        duration = float(duration)
        durations.append(duration)
        durations_by_task.setdefault(task, []).append(duration)
        durations_by_task_mode.setdefault(f"{task}|{mode}", []).append(duration)

    bench["durations"] = durations
    bench["durations_by_task"] = durations_by_task
    bench["durations_by_task_mode"] = durations_by_task_mode


def update_eta_tracking(state: dict) -> None:
    bench = state["benchmarks"]
    tracking = state.setdefault(
        "eta_tracking",
        {
            "interval_seconds": 600.0,
            "last_sample_ts": None,
            "samples": [],
            "current_run_alignment": None,
            "pipeline_alignment": None,
        },
    )

    current_run = bench.get("current_run")
    if not current_run:
        return

    sample_ts = now_ts()
    interval = float(tracking.get("interval_seconds") or 600.0)
    last_sample_ts = tracking.get("last_sample_ts")
    if last_sample_ts is not None and (sample_ts - float(last_sample_ts)) < interval:
        return

    current_eta = current_run.get("eta_seconds")
    pipeline_eta = state.get("pipeline_eta_seconds")
    samples = list(tracking.get("samples") or [])

    if samples:
        prev = samples[-1]
        elapsed = max(sample_ts - float(prev["ts"]), 1.0)
        if prev.get("current_run_eta_seconds") is not None and current_eta is not None:
            predicted_drop = float(prev["current_run_eta_seconds"]) - float(current_eta)
            tracking["current_run_alignment"] = {
                "predicted_drop_seconds": predicted_drop,
                "actual_elapsed_seconds": elapsed,
                "drift_seconds": predicted_drop - elapsed,
            }
        if prev.get("pipeline_eta_seconds") is not None and pipeline_eta is not None:
            predicted_drop = float(prev["pipeline_eta_seconds"]) - float(pipeline_eta)
            tracking["pipeline_alignment"] = {
                "predicted_drop_seconds": predicted_drop,
                "actual_elapsed_seconds": elapsed,
                "drift_seconds": predicted_drop - elapsed,
            }

    samples.append(
        {
            "ts": sample_ts,
            "time_utc": utc_now(),
            "current_run_eta_seconds": current_eta,
            "pipeline_eta_seconds": pipeline_eta,
            "current_progress_fraction": current_run.get("progress_fraction"),
            "current_task": current_run.get("task"),
            "current_mode": current_run.get("mode"),
            "current_seed": current_run.get("seed"),
        }
    )
    tracking["samples"] = samples[-12:]
    tracking["last_sample_ts"] = sample_ts


def build_overview(state: dict) -> dict:
    bench = state["benchmarks"]
    current_run = bench.get("current_run")
    summary_index = load_summary_index(state)
    compare_index = load_compare_index(state)
    run_snapshots = collect_run_snapshots(state)
    avg_run_seconds = bench.get("duration_estimate_seconds")
    seeds_per_mode = len(state["seeds"])

    overview = {
        "timers": {
            "pipeline_elapsed_seconds": max(0.0, now_ts() - float(iso_to_ts(state.get("started_at_utc")) or now_ts())),
            "pipeline_eta_seconds": state.get("pipeline_eta_seconds"),
            "current_phase_elapsed_seconds": max(
                0.0, now_ts() - float(state.get("current", {}).get("started_at_ts") or now_ts())
            ),
            "current_run_elapsed_seconds": (
                max(0.0, now_ts() - float(current_run.get("started_at_ts") or now_ts())) if current_run else None
            ),
            "current_run_eta_seconds": current_run.get("eta_seconds") if current_run else None,
            "benchmark_elapsed_seconds": (
                max(0.0, now_ts() - float(bench.get("started_at_ts") or now_ts())) if bench.get("started_at_ts") else None
            ),
            "benchmark_eta_seconds": bench.get("benchmark_eta_seconds"),
            "radcom_elapsed_seconds": (
                state["radcom"].get("duration_seconds")
                if state["radcom"].get("duration_seconds") is not None
                else (
                    max(0.0, now_ts() - float(state["radcom"].get("pass1_started_at_ts") or now_ts()))
                    if state["radcom"].get("pass1_started_at_ts")
                    else None
                )
            ),
            "radcom_eta_seconds": state["radcom"].get("eta_seconds"),
            "plots_elapsed_seconds": (
                state["plots"].get("duration_seconds")
                if state["plots"].get("duration_seconds") is not None
                else (
                    max(0.0, now_ts() - float(state["plots"].get("started_at_ts") or now_ts()))
                    if state["plots"].get("started_at_ts")
                    else None
                )
            ),
            "plots_eta_seconds": state["plots"].get("eta_seconds"),
            "summary_elapsed_seconds": (
                state["summary"].get("duration_seconds")
                if state["summary"].get("duration_seconds") is not None
                else (
                    max(0.0, now_ts() - float(state["summary"].get("started_at_ts") or now_ts()))
                    if state["summary"].get("started_at_ts")
                    else None
                )
            ),
            "summary_eta_seconds": state["summary"].get("eta_seconds"),
            "comparison_elapsed_seconds": (
                state["comparison"].get("duration_seconds")
                if state["comparison"].get("duration_seconds") is not None
                else (
                    max(0.0, now_ts() - float(state["comparison"].get("started_at_ts") or now_ts()))
                    if state["comparison"].get("started_at_ts")
                    else None
                )
            ),
            "comparison_eta_seconds": state["comparison"].get("eta_seconds"),
        },
        "task_rows": [],
        "burnup": {},
        "task_forecasts": [],
        "task_counts": {"total": 0, "done": 0, "active": 0, "pending": 0, "needs_restart": 0, "error": 0},
        "summary_counts": {
            "total": 0,
            "done": 0,
            "partial": 0,
            "ready": 0,
            "active": 0,
            "blocked": 0,
            "pending": 0,
            "error": 0,
        },
        "comparison_counts": {
            "total": 0,
            "done": 0,
            "partial": 0,
            "ready": 0,
            "active": 0,
            "blocked": 0,
            "pending": 0,
            "error": 0,
        },
        "data_readiness": build_data_readiness(state),
        "artifact_freshness": build_artifact_freshness(state),
        "resume_safety": build_resume_safety_panel(state),
        "commands": build_command_panel(state),
    }

    for task in ordered_tasks(state["tasks"]):
        task_entries = [item for item in bench["run_plan"] if item["task"] == task]
        task_counts = benchmark_counts(task_entries)
        task_status = benchmark_rollup(task_counts)
        overview["task_counts"]["total"] += 1
        overview["task_counts"][task_status] += 1

        task_eta = None
        task_eta_parts = [
            estimate_entry_run_seconds(bench, item)
            for item in task_entries
            if item["status"] == "pending" and not item["skip_expected"]
        ]
        if task_eta_parts and not any(part is None for part in task_eta_parts):
            task_eta = float(sum(task_eta_parts))
        if current_run and current_run.get("task") == task:
            current_eta = estimate_current_run_remaining_seconds(bench, current_run)
            if current_eta is not None:
                task_eta = float(current_eta) if task_eta is None else float(current_eta) + task_eta

        mode_rows = []
        task_duration_seconds = 0.0
        task_duration_present = False
        for mode in state["modes"]:
            mode_entries = [item for item in task_entries if item["mode"] == mode]
            mode_counts = benchmark_counts(mode_entries)
            mode_status = benchmark_rollup(mode_counts)
            summary_row = summary_index.get((task, mode))
            compare_row = compare_index.get((task, mode))
            mode_snapshots = [run_snapshots[(task, mode, seed)] for seed in state["seeds"]]
            metric_values = [
                float(item["best_metric"])
                for item in mode_snapshots
                if item.get("has_metrics") and item.get("best_metric") is not None
            ]
            duration_values = [
                float(item["duration_seconds"])
                for item in mode_snapshots
                if item.get("duration_seconds") is not None
            ]
            mode_duration_seconds = sum(duration_values) if duration_values else None
            if mode_duration_seconds is not None:
                task_duration_seconds += mode_duration_seconds
                task_duration_present = True
            local_metric = None
            if metric_values:
                local_metric = {
                    "key": str(mode_snapshots[0].get("best_key") or TASK_SPECS[task]["primary_metric"]),
                    "mean": statistics.fmean(metric_values),
                    "std": statistics.pstdev(metric_values) if len(metric_values) > 1 else 0.0,
                    "n": len(metric_values),
                }

            if summary_row is not None:
                n_seeds = int(summary_row.get("n_seeds") or 0)
                summary_status = "done" if n_seeds >= seeds_per_mode else "partial"
            elif mode_counts["done"] == seeds_per_mode:
                if state["summary"]["status"] == "running":
                    summary_status = "active"
                elif state["summary"]["status"] == "completed":
                    summary_status = "error"
                else:
                    summary_status = "ready"
            elif mode_counts["done"] > 0 or mode_counts["active"] > 0 or mode_counts["needs_restart"] > 0:
                summary_status = "blocked"
            else:
                summary_status = "pending"

            if compare_row is not None:
                compare_status = "done" if compare_row.get("status") == "ok" else "partial"
            elif summary_status in {"done", "partial"}:
                if state["comparison"]["status"] == "running":
                    compare_status = "active"
                elif state["comparison"]["status"] == "completed":
                    compare_status = "error"
                else:
                    compare_status = "ready"
            elif summary_status == "ready":
                compare_status = "blocked"
            elif summary_status in {"active", "blocked"}:
                compare_status = "blocked"
            else:
                compare_status = "pending"

            overview["summary_counts"]["total"] += 1
            overview["summary_counts"][summary_status] += 1
            overview["comparison_counts"]["total"] += 1
            overview["comparison_counts"][compare_status] += 1

            mode_eta = None
            mode_eta_parts = [
                estimate_entry_run_seconds(bench, item)
                for item in mode_entries
                if item["status"] == "pending" and not item["skip_expected"]
            ]
            if mode_eta_parts and not any(part is None for part in mode_eta_parts):
                mode_eta = float(sum(mode_eta_parts))
            if current_run and current_run.get("task") == task and current_run.get("mode") == mode:
                current_eta = estimate_current_run_remaining_seconds(bench, current_run)
                if current_eta is not None:
                    mode_eta = float(current_eta) if mode_eta is None else float(current_eta) + mode_eta

            mode_rows.append(
                {
                    "mode": mode,
                    "benchmark_status": mode_status,
                    "benchmark_counts": mode_counts,
                    "summary_status": summary_status,
                    "comparison_status": compare_status,
                    "eta_seconds": mode_eta,
                    "n_seeds_summarized": int(summary_row.get("n_seeds") or 0) if summary_row else 0,
                    "comparison_row_status": compare_row.get("status") if compare_row else None,
                    "local_metric": local_metric,
                    "duration_seconds": mode_duration_seconds,
                }
            )

        overview["task_rows"].append(
            {
                "task": task,
                "display_name": TASK_SPECS[task]["display_name"],
                "status": task_status,
                "counts": task_counts,
                "eta_seconds": task_eta,
                "mode_rows": mode_rows,
                "summary_done_modes": sum(1 for item in mode_rows if item["summary_status"] == "done"),
                "comparison_done_modes": sum(1 for item in mode_rows if item["comparison_status"] == "done"),
                "duration_seconds": task_duration_seconds if task_duration_present else None,
                "primary_metric": TASK_SPECS[task]["primary_metric"],
            }
        )

    overview["burnup"] = build_burnup_panel(state, run_snapshots)
    overview["task_forecasts"] = build_task_forecasts(state)
    return overview


def user_epoch(epoch: int | None) -> int | None:
    if epoch is None:
        return None
    return int(epoch) + 1


def metric_log_key(task: str) -> str:
    primary = str(TASK_SPECS[task]["primary_metric"])
    return {"mean_error": "mean_distance_error"}.get(primary, primary)


def metric_label(task: str) -> str:
    primary = str(TASK_SPECS[task]["primary_metric"])
    return {
        "pca": "PCA",
        "mean_error": "Mean error",
    }.get(primary, primary)


def compact_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    if abs(value) >= 10:
        return f"{value:.2f}"
    if abs(value) >= 1:
        return f"{value:.3f}"
    return f"{value:.4f}"


def compact_rate(value: float | None, unit: str) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f} {unit}"


def runtime_hardware() -> dict:
    global _RUNTIME_HARDWARE_CACHE
    if _RUNTIME_HARDWARE_CACHE is not None:
        return _RUNTIME_HARDWARE_CACHE

    label = "CPU"
    detail = "No accelerator detected"
    badge = "pending"
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            label = "CUDA"
            detail = torch.cuda.get_device_name(idx)
            badge = "done"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            label = "MPS"
            detail = "Apple Metal backend"
            badge = "ready"
        else:
            detail = "No accelerator detected"
            badge = "partial"
        torch_version = getattr(torch, "__version__", "unknown")
    except Exception as exc:
        torch_version = f"unavailable ({type(exc).__name__})"
        badge = "pending"

    _RUNTIME_HARDWARE_CACHE = {
        "label": label,
        "detail": detail,
        "badge": badge,
        "cpu_count": os.cpu_count(),
        "machine": platform.machine(),
        "python": ".".join(map(str, sys.version_info[:3])),
        "torch": torch_version,
    }
    return _RUNTIME_HARDWARE_CACHE


def system_runtime_snapshot(*, refresh_after_seconds: float = 10.0) -> dict:
    global _SYSTEM_SNAPSHOT_CACHE
    now = now_ts()
    if _SYSTEM_SNAPSHOT_CACHE is not None and (now - _SYSTEM_SNAPSHOT_CACHE[0]) < refresh_after_seconds:
        return _SYSTEM_SNAPSHOT_CACHE[1]

    cpu_count = os.cpu_count() or 1
    load1, load5, load15 = os.getloadavg()
    snapshot = {
        "captured_at_ts": now,
        "load1": load1,
        "load5": load5,
        "load15": load15,
        "load1_pct": 100.0 * load1 / cpu_count,
        "load5_pct": 100.0 * load5 / cpu_count,
        "load15_pct": 100.0 * load15 / cpu_count,
        "power_source": "unknown",
        "battery_percent": None,
        "battery_state": None,
        "tracker_pid": os.getpid(),
        "trainer_pid": None,
        "trainer_cpu": None,
        "trainer_mem": None,
        "trainer_etime": None,
        "memory_total_gb": None,
        "memory_free_gb": None,
        "memory_available_gb": None,
        "memory_compressed_gb": None,
        "workspace_disk_root": str(SESSION_ROOT),
        "workspace_disk_total_gb": None,
        "workspace_disk_free_gb": None,
        "workspace_disk_used_pct": None,
        "gpu_name": None,
        "gpu_util": None,
        "gpu_mem_used_mb": None,
        "gpu_mem_total_mb": None,
        "gpu_temp_c": None,
        "pressure_note": "Basic system snapshot",
    }

    try:
        out = subprocess.check_output(["pmset", "-g", "batt"], text=True, stderr=subprocess.DEVNULL)
        first = out.splitlines()[0].strip() if out.splitlines() else ""
        if "AC Power" in first:
            snapshot["power_source"] = "AC"
        elif "Battery Power" in first:
            snapshot["power_source"] = "Battery"
        match = re.search(r"(\d+)%", out)
        if match:
            snapshot["battery_percent"] = int(match.group(1))
        lower = out.lower()
        if "charging" in lower:
            snapshot["battery_state"] = "charging"
        elif "discharging" in lower:
            snapshot["battery_state"] = "discharging"
        elif "charged" in lower:
            snapshot["battery_state"] = "charged"
    except Exception:
        pass

    if sys.platform.startswith("linux"):
        try:
            power_root = Path("/sys/class/power_supply")
            mains = sorted(power_root.glob("A*C*/online")) + sorted(power_root.glob("AC*/online"))
            batteries = sorted(power_root.glob("BAT*/capacity"))
            statuses = sorted(power_root.glob("BAT*/status"))
            for path in mains:
                value = path.read_text(encoding="utf-8").strip()
                if value == "1":
                    snapshot["power_source"] = "AC"
                    break
            if batteries:
                snapshot["battery_percent"] = int(batteries[0].read_text(encoding="utf-8").strip())
            if statuses:
                snapshot["battery_state"] = statuses[0].read_text(encoding="utf-8").strip().lower()
        except Exception:
            pass

    try:
        out = subprocess.check_output(
            ["ps", "-Ao", "pid,%cpu,%mem,etime,command"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        for line in out.splitlines():
            if "wavesfm/main_finetune.py" not in line:
                continue
            parts = line.strip().split(None, 4)
            if len(parts) < 5:
                continue
            pid_text, cpu, mem, etime, _cmd = parts
            try:
                snapshot["trainer_pid"] = int(pid_text)
            except ValueError:
                snapshot["trainer_pid"] = None
            snapshot["trainer_cpu"] = float(cpu)
            snapshot["trainer_mem"] = float(mem)
            snapshot["trainer_etime"] = etime
            break
    except Exception:
        pass

    try:
        total_bytes = int(
            subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, stderr=subprocess.DEVNULL).strip()
        )
        vm = subprocess.check_output(["vm_stat"], text=True, stderr=subprocess.DEVNULL)
        page_size = 4096
        m = re.search(r"page size of (\d+) bytes", vm)
        if m:
            page_size = int(m.group(1))
        values = {}
        for line in vm.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            value = value.strip().rstrip(".")
            value = value.replace(".", "")
            try:
                values[key.strip()] = int(value)
            except ValueError:
                continue
        free_pages = values.get("Pages free", 0) + values.get("Pages speculative", 0)
        compressed_pages = values.get("Pages occupied by compressor", 0)
        snapshot["memory_total_gb"] = total_bytes / (1024 ** 3)
        snapshot["memory_free_gb"] = (free_pages * page_size) / (1024 ** 3)
        snapshot["memory_compressed_gb"] = (compressed_pages * page_size) / (1024 ** 3)
        if snapshot["memory_free_gb"] is not None:
            if snapshot["memory_free_gb"] < 1.0:
                snapshot["pressure_note"] = "Low free memory; the machine may start to feel heavy."
            elif snapshot["memory_free_gb"] < 2.0:
                snapshot["pressure_note"] = "Memory is getting tight; avoid opening heavy apps."
            else:
                snapshot["pressure_note"] = "Memory headroom looks acceptable."
    except Exception:
        pass

    if snapshot["memory_total_gb"] is None and sys.platform.startswith("linux"):
        try:
            meminfo = {}
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                parts = value.strip().split()
                if not parts:
                    continue
                meminfo[key] = int(parts[0]) * 1024
            total_bytes = meminfo.get("MemTotal")
            free_bytes = meminfo.get("MemFree")
            available_bytes = meminfo.get("MemAvailable")
            if total_bytes is not None:
                snapshot["memory_total_gb"] = total_bytes / (1024 ** 3)
            if free_bytes is not None:
                snapshot["memory_free_gb"] = free_bytes / (1024 ** 3)
            if available_bytes is not None:
                snapshot["memory_available_gb"] = available_bytes / (1024 ** 3)

            headroom = snapshot["memory_available_gb"]
            if headroom is None:
                headroom = snapshot["memory_free_gb"]
            if headroom is not None:
                if headroom < 2.0:
                    snapshot["pressure_note"] = "Low available memory on the lab host."
                elif headroom < 4.0:
                    snapshot["pressure_note"] = "Memory headroom is moderate; avoid competing jobs."
                else:
                    snapshot["pressure_note"] = "Linux memory headroom looks acceptable."
        except Exception:
            pass

    try:
        disk_root = RESULTS_ROOT if RESULTS_ROOT.exists() else SESSION_ROOT
        usage = shutil.disk_usage(disk_root)
        snapshot["workspace_disk_root"] = str(disk_root)
        snapshot["workspace_disk_total_gb"] = usage.total / (1024 ** 3)
        snapshot["workspace_disk_free_gb"] = usage.free / (1024 ** 3)
        snapshot["workspace_disk_used_pct"] = 100.0 * (usage.used / usage.total) if usage.total else None
    except Exception:
        pass

    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if out:
            first = out.splitlines()[0]
            name, util, mem_used, mem_total, temp = [part.strip() for part in first.split(",", 4)]
            snapshot["gpu_name"] = name
            snapshot["gpu_util"] = float(util)
            snapshot["gpu_mem_used_mb"] = float(mem_used)
            snapshot["gpu_mem_total_mb"] = float(mem_total)
            snapshot["gpu_temp_c"] = float(temp)
    except Exception:
        pass

    _SYSTEM_SNAPSHOT_CACHE = (now, snapshot)
    return snapshot


def update_runtime_history(state: dict) -> None:
    history = state.setdefault(
        "runtime_history",
        {
            "interval_seconds": 30.0,
            "last_sample_ts": None,
            "samples": [],
        },
    )
    interval = float(history.get("interval_seconds") or 30.0)
    last_sample_ts = history.get("last_sample_ts")
    current_ts = now_ts()
    if last_sample_ts is not None and (current_ts - float(last_sample_ts)) < interval:
        return

    snapshot = system_runtime_snapshot(refresh_after_seconds=0)
    samples = list(history.get("samples") or [])
    samples.append(
        {
            "ts": current_ts,
            "time_utc": utc_now(),
            "gpu_util": snapshot.get("gpu_util"),
            "gpu_mem_used_mb": snapshot.get("gpu_mem_used_mb"),
            "gpu_mem_total_mb": snapshot.get("gpu_mem_total_mb"),
            "gpu_temp_c": snapshot.get("gpu_temp_c"),
            "trainer_cpu": snapshot.get("trainer_cpu"),
            "trainer_mem": snapshot.get("trainer_mem"),
            "load1_pct": snapshot.get("load1_pct"),
            "workspace_disk_free_gb": snapshot.get("workspace_disk_free_gb"),
            "workspace_disk_used_pct": snapshot.get("workspace_disk_used_pct"),
        }
    )
    history["samples"] = samples[-120:]
    history["last_sample_ts"] = current_ts


def format_wall_time(ts: float | None) -> str:
    if ts is None:
        return "n/a"
    try:
        return datetime.fromtimestamp(float(ts)).astimezone().strftime("%b %d %H:%M")
    except Exception:
        return "n/a"


def svg_burnup_chart(
    events: list[dict],
    *,
    total: int,
    start_ts: float,
    end_ts: float,
    color: str,
    fill: str,
    width: int = 460,
    height: int = 190,
) -> str:
    if total <= 0:
        total = 1
    if end_ts <= start_ts:
        end_ts = start_ts + 1.0
    pad_left = 42
    pad_right = 16
    pad_top = 18
    pad_bottom = 30
    inner_w = width - pad_left - pad_right
    inner_h = height - pad_top - pad_bottom

    def x_for(ts: float) -> float:
        return pad_left + ((ts - start_ts) / (end_ts - start_ts)) * inner_w

    def y_for(count: float) -> float:
        return pad_top + (1.0 - (count / total)) * inner_h

    grid_lines = []
    grid_labels = []
    for frac in (0.0, 0.5, 1.0):
        value = int(round(total * frac))
        y = y_for(value)
        grid_lines.append(
            f"<line x1='{pad_left:.1f}' y1='{y:.1f}' x2='{width-pad_right:.1f}' y2='{y:.1f}' stroke='#e7dccd' stroke-width='1'></line>"
        )
        grid_labels.append(
            f"<text x='{pad_left-8:.1f}' y='{y+4:.1f}' text-anchor='end' font-size='11' fill='#7a6958'>{value}</text>"
        )

    x_axis_labels = [
        f"<text x='{pad_left:.1f}' y='{height-8:.1f}' text-anchor='start' font-size='11' fill='#7a6958'>{html.escape(format_wall_time(start_ts))}</text>",
        f"<text x='{width-pad_right:.1f}' y='{height-8:.1f}' text-anchor='end' font-size='11' fill='#7a6958'>now</text>",
    ]

    step_points: list[tuple[float, float]] = [(pad_left, y_for(0))]
    area_points: list[tuple[float, float]] = [(pad_left, y_for(0))]
    current_count = 0
    milestone_circles: list[str] = []
    recent = events[-6:]
    for event in events:
        ts = float(event["ts"])
        count = int(event["count"])
        x = x_for(ts)
        prev_y = y_for(current_count)
        new_y = y_for(count)
        step_points.append((x, prev_y))
        step_points.append((x, new_y))
        area_points.append((x, prev_y))
        area_points.append((x, new_y))
        current_count = count
    step_points.append((width - pad_right, y_for(current_count)))
    area_points.append((width - pad_right, y_for(current_count)))
    area_points.append((width - pad_right, height - pad_bottom))
    area_points.append((pad_left, height - pad_bottom))

    for event in recent:
        x = x_for(float(event["ts"]))
        y = y_for(int(event["count"]))
        title = html.escape(f"{event['label']} | {event['count']}/{total} | {format_wall_time(float(event['ts']))}")
        milestone_circles.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='3.6' fill='{color}' stroke='white' stroke-width='1.2'><title>{title}</title></circle>"
        )

    step_poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in step_points)
    area_poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in area_points)
    latest_y = y_for(current_count)
    latest_label = f"{current_count}/{total}"

    return (
        f"<svg viewBox='0 0 {width} {height}' class='burnup-chart' aria-hidden='true'>"
        f"{''.join(grid_lines)}"
        f"{''.join(grid_labels)}"
        f"<line x1='{pad_left:.1f}' y1='{height-pad_bottom:.1f}' x2='{width-pad_right:.1f}' y2='{height-pad_bottom:.1f}' stroke='#cdbba4' stroke-width='1.2'></line>"
        f"<line x1='{pad_left:.1f}' y1='{pad_top:.1f}' x2='{pad_left:.1f}' y2='{height-pad_bottom:.1f}' stroke='#cdbba4' stroke-width='1.2'></line>"
        f"<polygon points=\"{area_poly}\" fill='{fill}' opacity='0.9'></polygon>"
        f"<polyline points=\"{step_poly}\" fill='none' stroke='{color}' stroke-width='2.8' stroke-linecap='round' stroke-linejoin='round'></polyline>"
        f"{''.join(milestone_circles)}"
        f"<text x='{width-pad_right:.1f}' y='{max(pad_top+10, latest_y-8):.1f}' text-anchor='end' font-size='12' font-weight='700' fill='{color}'>{html.escape(latest_label)}</text>"
        f"{''.join(x_axis_labels)}"
        "</svg>"
    )


def trend_sparkline(values: list[float], *, lower_is_better: bool) -> str:
    if not values:
        return "<div class='muted'>n/a</div>"
    if len(values) == 1:
        values = values * 2
    width = 180
    height = 52
    pad = 6
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        hi = lo + 1.0
    points = []
    for idx, value in enumerate(values):
        x = pad + (idx * (width - 2 * pad) / max(len(values) - 1, 1))
        frac = (value - lo) / (hi - lo)
        y = height - pad - frac * (height - 2 * pad)
        points.append(f"{x:.1f},{y:.1f}")
    stroke = "#b56731" if lower_is_better else "#2f6ba8"
    fill = "#f3e5d5" if lower_is_better else "#dce9f8"
    return (
        f"<svg viewBox='0 0 {width} {height}' class='sparkline' aria-hidden='true'>"
        f"<polyline points=\"{' '.join(points)}\" fill='none' stroke='{stroke}' stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round'></polyline>"
        f"<circle cx='{points[-1].split(',')[0]}' cy='{points[-1].split(',')[1]}' r='3.2' fill='{stroke}'></circle>"
        f"<rect x='0.5' y='0.5' width='{width-1}' height='{height-1}' rx='10' ry='10' fill='none' stroke='{fill}' stroke-width='1'></rect>"
        "</svg>"
    )


def evaluate_run_health(task: str, entries: list[dict]) -> dict:
    if len(entries) < 2:
        return {"label": "warming up", "status": "pending", "message": "Collecting epoch history."}
    key = metric_log_key(task)
    latest = float(entries[-1].get("val", {}).get(key) or 0.0)
    prev = float(entries[-2].get("val", {}).get(key) or 0.0)
    latest_loss = float(entries[-1].get("val", {}).get("loss") or 0.0)
    recent_losses = [float(item.get("val", {}).get("loss") or 0.0) for item in entries[-4:] if item.get("val")]
    lower_is_better = TASK_SPECS[task]["task_type"] == "position"
    delta = latest - prev
    wrong_way = delta > 0.10 if lower_is_better else delta < -0.50
    loss_spike = False
    if len(recent_losses) >= 2:
        baseline = min(recent_losses[:-1]) if len(recent_losses) > 1 else recent_losses[0]
        if baseline > 0:
            loss_spike = latest_loss > baseline * 1.15
    if wrong_way or loss_spike:
        return {
            "label": "watch",
            "status": "partial",
            "message": (
                f"{metric_label(task)} moved {'up' if lower_is_better else 'down'} by {compact_float(abs(delta))}; "
                f"val loss {compact_float(latest_loss)}"
            ),
        }
    improved = delta < -0.01 if lower_is_better else delta > 0.05
    if improved:
        return {
            "label": "improving",
            "status": "done",
            "message": f"{metric_label(task)} improved to {compact_float(latest)}.",
        }
    return {
        "label": "stable",
        "status": "active",
        "message": f"Recent epochs are moving within a normal range around {compact_float(latest)}.",
    }


def detect_epoch_anomalies(task: str, entries: list[dict]) -> dict:
    if len(entries) < 2:
        return {"by_epoch": {}, "items": []}
    task_type = TASK_SPECS[task]["task_type"]
    primary_key = metric_log_key(task)
    lower_is_better = task_type == "position"
    by_epoch: dict[int, list[str]] = {}
    items: list[dict] = []
    for prev, cur in zip(entries, entries[1:]):
        issues: list[str] = []
        prev_val = prev.get("val") or {}
        cur_val = cur.get("val") or {}
        prev_metric = prev_val.get(primary_key)
        cur_metric = cur_val.get(primary_key)
        if prev_metric is not None and cur_metric is not None:
            delta = float(cur_metric) - float(prev_metric)
            metric_jump = delta > 0.10 if lower_is_better else delta < -0.50
            if metric_jump:
                direction = "up" if lower_is_better else "down"
                issues.append(f"{metric_label(task)} moved {direction} by {compact_float(abs(delta))}")
        prev_val_loss = prev_val.get("loss")
        cur_val_loss = cur_val.get("loss")
        if prev_val_loss not in (None, 0) and cur_val_loss is not None:
            if float(cur_val_loss) > float(prev_val_loss) * 1.15:
                issues.append(f"val loss rose to {compact_float(float(cur_val_loss))}")
        prev_train = (prev.get("train") or {}).get("loss")
        cur_train = (cur.get("train") or {}).get("loss")
        if prev_train not in (None, 0) and cur_train is not None:
            if float(cur_train) > float(prev_train) * 1.20:
                issues.append(f"train loss rose to {compact_float(float(cur_train))}")
        if not issues:
            continue
        epoch_display = user_epoch(cur.get("epoch"))
        if epoch_display is None:
            continue
        by_epoch[epoch_display] = issues
        for issue in issues:
            items.append({"epoch": epoch_display, "message": issue})
    return {"by_epoch": by_epoch, "items": items[-6:]}


def build_single_run_command(
    task: str,
    mode: str,
    seed: int,
    *,
    num_workers: int,
    save_every: int,
    train_subset_fraction: float | None = None,
    train_subset_size: int | None = None,
    force_restart: bool = False,
) -> str:
    cmd = [
        sys.executable,
        "phase2_vivor4/scripts/run_all_tasks.py",
        "--tasks",
        task,
        "--modes",
        mode,
        "--seeds",
        str(seed),
        "--num-workers",
        str(num_workers),
        "--save-every",
        str(save_every),
    ]
    if train_subset_fraction is not None:
        cmd += ["--train-subset-fraction", str(train_subset_fraction)]
    if train_subset_size is not None:
        cmd += ["--train-subset-size", str(train_subset_size)]
    if force_restart:
        cmd.append("--force-restart")
    return " ".join(cmd)


def load_failure_panel(state: dict) -> dict:
    stored_num_workers = state.get("num_workers")
    num_workers = int(DEFAULT_NUM_WORKERS if stored_num_workers is None else stored_num_workers)
    save_every = int(state.get("save_every") or DEFAULT_SAVE_EVERY)
    train_subset_fraction = state.get("train_subset_fraction")
    train_subset_size = state.get("train_subset_size")
    errored = [item for item in state["benchmarks"]["run_plan"] if item.get("status") == "error"]
    restartable = [
        item
        for item in state["benchmarks"]["run_plan"]
        if item.get("status") == "pending" and item.get("resume_available")
    ]
    if errored:
        item = errored[0]
        return {
            "status": "error",
            "title": f"Run error in {format_run_label(item)}",
            "message": state.get("last_error") or "The run plan contains an errored item.",
            "resume_command": build_single_run_command(
                item["task"],
                item["mode"],
                int(item["seed"]),
                num_workers=num_workers,
                save_every=save_every,
                train_subset_fraction=train_subset_fraction,
                train_subset_size=train_subset_size,
            ),
            "restart_command": build_single_run_command(
                item["task"],
                item["mode"],
                int(item["seed"]),
                num_workers=num_workers,
                save_every=save_every,
                train_subset_fraction=train_subset_fraction,
                train_subset_size=train_subset_size,
                force_restart=True,
            ),
        }
    if restartable:
        item = restartable[0]
        return {
            "status": "restart",
            "title": f"Restartable run available: {format_run_label(item)}",
            "message": f"Checkpoint found at {item['resume_available']}",
            "resume_command": build_single_run_command(
                item["task"],
                item["mode"],
                int(item["seed"]),
                num_workers=num_workers,
                save_every=save_every,
                train_subset_fraction=train_subset_fraction,
                train_subset_size=train_subset_size,
            ),
            "restart_command": build_single_run_command(
                item["task"],
                item["mode"],
                int(item["seed"]),
                num_workers=num_workers,
                save_every=save_every,
                train_subset_fraction=train_subset_fraction,
                train_subset_size=train_subset_size,
                force_restart=True,
            ),
        }
    return {
        "status": "done",
        "title": "No failures or restart debt",
        "message": "No errored runs recorded and no pending restartable runs detected.",
        "resume_command": None,
        "restart_command": None,
    }


def build_burnup_panel(state: dict, run_snapshots: dict[tuple[str, str, int], dict]) -> dict:
    start_ts = iso_to_ts(state.get("started_at_utc")) or now_ts()
    end_ts = now_ts()

    run_events_raw: list[tuple[float, str]] = []
    for (task, mode, seed), snapshot in run_snapshots.items():
        if snapshot.get("completed") and snapshot.get("last_ts") is not None:
            run_events_raw.append((float(snapshot["last_ts"]), f"{task} / {mode} / s{seed}"))
    run_events_raw.sort()
    run_events = [{"ts": ts, "label": label, "count": idx} for idx, (ts, label) in enumerate(run_events_raw, start=1)]

    task_events_raw: list[tuple[float, str]] = []
    for task in state["tasks"]:
        task_runs = [run_snapshots[(task, mode, seed)] for mode in state["modes"] for seed in state["seeds"]]
        if task_runs and all(item.get("completed") and item.get("last_ts") is not None for item in task_runs):
            task_events_raw.append((max(float(item["last_ts"]) for item in task_runs), task))
    task_events_raw.sort()
    task_events = [{"ts": ts, "label": label, "count": idx} for idx, (ts, label) in enumerate(task_events_raw, start=1)]

    pipeline_elapsed_hours = max((end_ts - start_ts) / 3600.0, 1e-9)
    runs_done = len(run_events)
    tasks_done = len(task_events)
    total_runs = int(state["benchmarks"]["total_runs"])
    total_tasks = len(state["tasks"])
    run_rate = runs_done / pipeline_elapsed_hours if runs_done else 0.0
    task_rate = tasks_done / pipeline_elapsed_hours if tasks_done else 0.0

    milestones: list[dict] = []
    for event in run_events:
        milestones.append(
            {
                "ts": event["ts"],
                "kind": "run",
                "label": event["label"],
                "count_text": f"{event['count']}/{total_runs}",
                "percent": 100.0 * event["count"] / max(1, total_runs),
            }
        )
    for event in task_events:
        milestones.append(
            {
                "ts": event["ts"],
                "kind": "task",
                "label": event["label"],
                "count_text": f"{event['count']}/{total_tasks}",
                "percent": 100.0 * event["count"] / max(1, total_tasks),
            }
        )
    milestones.sort(key=lambda item: float(item["ts"]), reverse=True)

    expected_finish = None
    if state.get("pipeline_eta_seconds") is not None:
        expected_finish = end_ts + float(state["pipeline_eta_seconds"])

    return {
        "runs_done": runs_done,
        "runs_total": total_runs,
        "run_percent": 100.0 * runs_done / max(1, total_runs),
        "run_rate_per_hour": run_rate,
        "tasks_done": tasks_done,
        "tasks_total": total_tasks,
        "task_percent": 100.0 * tasks_done / max(1, total_tasks),
        "task_rate_per_hour": task_rate,
        "first_completion_ts": min((item["ts"] for item in milestones), default=None),
        "latest_completion_ts": max((item["ts"] for item in milestones), default=None),
        "expected_finish_ts": expected_finish,
        "run_chart": svg_burnup_chart(
            run_events,
            total=total_runs,
            start_ts=start_ts,
            end_ts=end_ts,
            color="#1f6f8b",
            fill="#d8ebf1",
        ),
        "task_chart": svg_burnup_chart(
            task_events,
            total=total_tasks,
            start_ts=start_ts,
            end_ts=end_ts,
            color="#1b7f6b",
            fill="#d8efe8",
        ),
        "recent_milestones": milestones[:8],
    }


def forecast_band_for_task(state: dict, task: str) -> dict | None:
    bench = state["benchmarks"]
    current_run = bench.get("current_run")
    entries = [item for item in bench["run_plan"] if item["task"] == task and item["status"] == "pending" and not item["skip_expected"]]
    optimistic = 0.0
    expected = 0.0
    slow = 0.0
    known = False

    if current_run and current_run.get("task") == task:
        current_eta = estimate_current_run_remaining_seconds(bench, current_run)
        if current_eta is not None:
            expected += float(current_eta)
            optimistic += float(current_eta) * 0.85
            slow += float(current_eta) * 1.20
            known = True

    for item in entries:
        values = list((bench.get("durations_by_task_mode") or {}).get(f"{task}|{item['mode']}", []))
        if not values:
            values = list((bench.get("durations_by_task") or {}).get(task, []))
        if not values:
            values = list(bench.get("durations", []))
        if not values:
            continue
        mean = statistics.fmean(values)
        std = statistics.pstdev(values) if len(values) > 1 else max(mean * 0.15, 60.0)
        optimistic += max(mean - std, mean * 0.70)
        expected += mean
        slow += max(mean + std, mean * 1.30)
        known = True

    if not known:
        return None
    return {"task": task, "optimistic": optimistic, "expected": expected, "slow": slow}


def build_task_forecasts(state: dict) -> list[dict]:
    rows: list[dict] = []
    for task in ordered_tasks(state["tasks"]):
        forecast = forecast_band_for_task(state, task)
        if forecast is None:
            continue
        counts = benchmark_counts([item for item in state["benchmarks"]["run_plan"] if item["task"] == task])
        if counts["done"] == counts["total"]:
            continue
        forecast["display_name"] = TASK_SPECS[task]["display_name"]
        forecast["done"] = counts["done"]
        forecast["total"] = counts["total"]
        rows.append(forecast)
    rows.sort(key=lambda item: float(item["expected"]), reverse=True)
    return rows[:6]


def load_active_run_monitor(state: dict, *, limit: int = 4) -> dict | None:
    current_run = state.get("benchmarks", {}).get("current_run")
    if not current_run:
        return None
    task = str(current_run["task"])
    mode = str(current_run["mode"])
    seed = int(current_run["seed"])
    run_dir = run_output_dir(task, mode, seed)
    log_path = run_dir / "log.txt"
    history: list[dict] = []
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload.get("epoch"), int) and isinstance(payload.get("val"), dict):
                history.append(payload)

    current_display_epoch = user_epoch(current_run.get("epoch"))
    recent_completed = history[-(limit + 4) :]
    visible_completed = []
    for item in reversed(recent_completed):
        display_epoch = user_epoch(item.get("epoch"))
        if current_display_epoch is not None and display_epoch is not None and display_epoch >= current_display_epoch:
            continue
        visible_completed.append(item)
        if len(visible_completed) >= max(limit - 1, 0):
            break

    task_type = TASK_SPECS[task]["task_type"]
    primary_key = metric_log_key(task)
    lower_is_better = task_type == "position"
    primary_values = [float(item.get("val", {}).get(primary_key) or 0.0) for item in history[-8:] if item.get("val")]
    val_loss_values = [float(item.get("val", {}).get("loss") or 0.0) for item in history[-8:] if item.get("val")]
    health = evaluate_run_health(task, history[-4:])
    anomalies = detect_epoch_anomalies(task, history[-8:])
    latest_validated = history[-1] if history else None
    latest_val = latest_validated.get("val", {}) if latest_validated else {}
    latest_primary = latest_val.get(primary_key)
    latest_val_loss = latest_val.get("loss")
    best_primary = None
    best_val_loss = None
    if history:
        primary_candidates = [item.get("val", {}).get(primary_key) for item in history if item.get("val", {}).get(primary_key) is not None]
        loss_candidates = [item.get("val", {}).get("loss") for item in history if item.get("val", {}).get("loss") is not None]
        if primary_candidates:
            best_primary = min(primary_candidates) if lower_is_better else max(primary_candidates)
        if loss_candidates:
            best_val_loss = min(loss_candidates)

    rows = []
    rows.append(
        {
            "epoch": current_display_epoch,
            "status": "running",
            "step": (
                f"{current_run.get('step')}/{current_run.get('steps_per_epoch')}"
                if current_run.get("step") is not None and current_run.get("steps_per_epoch") is not None
                else "n/a"
            ),
            "lr": current_run.get("latest_lr"),
            "train_loss": current_run.get("latest_train_loss"),
            "val_loss": None,
            "primary": None,
            "secondary_1": None,
            "secondary_2": None,
            "anomaly": None,
        }
    )
    for item in visible_completed:
        val = item.get("val", {})
        epoch_display = user_epoch(item.get("epoch"))
        if task_type == "classification":
            secondary_1 = val.get("acc1")
            secondary_2 = val.get("acc3")
        else:
            secondary_1 = val.get("median_distance_error")
            secondary_2 = val.get("p90_distance_error")
        rows.append(
            {
                "epoch": epoch_display,
                "status": "validated",
                "step": "done",
                "lr": item.get("lr"),
                "train_loss": item.get("train", {}).get("loss"),
                "val_loss": val.get("loss"),
                "primary": val.get(primary_key),
                "secondary_1": secondary_1,
                "secondary_2": secondary_2,
                "anomaly": anomalies["by_epoch"].get(epoch_display),
            }
        )

    if task_type == "classification":
        secondary_labels = ("Acc@1", "Acc@3")
        reading_guide = {
            "summary": "Higher task metrics are better. Stable or rising PCA/accuracy with flat or falling validation loss is the normal pattern.",
            "watch": "Watch for PCA dropping by about 0.5 or more, validation loss jumping by 15% or more, or two validated epochs in a row moving the wrong way.",
        }
    else:
        secondary_labels = ("Median", "P90")
        reading_guide = {
            "summary": "Lower mean error is better. Flat or slowly falling mean error with flat validation loss is the normal late-training pattern.",
            "watch": "Watch for mean error rising by about 0.10 or more, validation loss jumping by 15% or more, or both metrics worsening across multiple validated epochs.",
        }

    elapsed = max(1.0, now_ts() - float(current_run.get("started_at_ts") or now_ts()))
    steps_per_epoch = current_run.get("steps_per_epoch")
    epoch_idx = current_run.get("epoch")
    resume_epoch = int(current_run.get("resume_epoch") or 0)
    steps_done = None
    epochs_done = None
    steps_per_min = None
    epochs_per_hour = None
    seconds_per_step = None
    if steps_per_epoch and epoch_idx is not None and current_run.get("step") is not None:
        steps_done = max(((int(epoch_idx) - resume_epoch) * int(steps_per_epoch)) + int(current_run["step"]), 0)
        epochs_done = max((int(epoch_idx) - resume_epoch) + (int(current_run["step"]) / int(steps_per_epoch)), 0.0)
        if steps_done > 0:
            steps_per_min = steps_done / (elapsed / 60.0)
            seconds_per_step = elapsed / steps_done
        if epochs_done and epochs_done > 0:
            epochs_per_hour = epochs_done / (elapsed / 3600.0)

    run_dir = run_output_dir(task, mode, seed)
    latest_ckpt = latest_epoch_checkpoint(run_dir)
    latest_ckpt_label = latest_ckpt.name if latest_ckpt is not None else "n/a"
    latest_ckpt_age = None
    if latest_ckpt is not None:
        latest_ckpt_age = max(0.0, now_ts() - latest_ckpt.stat().st_mtime)
    timeline = recent_checkpoint_timeline(run_dir, limit=5)
    next_ckpt_epoch = None
    next_ckpt_eta = None
    save_every = int(state.get("save_every") or DEFAULT_SAVE_EVERY)
    total_epochs = current_run.get("epochs")
    if total_epochs and current_display_epoch is not None and steps_per_epoch and epoch_idx is not None and current_run.get("step") is not None:
        next_ckpt_epoch = min(
            ((int(current_display_epoch) + save_every - 1) // save_every) * save_every,
            int(total_epochs),
        )
        if next_ckpt_epoch <= int(total_epochs):
            if epochs_per_hour and epochs_per_hour > 0:
                epoch_rate = epochs_per_hour / 3600.0
                remaining_epoch_units = max(
                    ((next_ckpt_epoch - 1) - int(epoch_idx)) + ((int(steps_per_epoch) - int(current_run["step"])) / int(steps_per_epoch)),
                    0.0,
                )
                next_ckpt_eta = remaining_epoch_units / epoch_rate if epoch_rate > 0 else None

    current_anomaly = None
    if rows and history:
        latest_val_loss = history[-1].get("val", {}).get("loss")
        current_train_loss = current_run.get("latest_train_loss")
        if latest_val_loss not in (None, 0) and current_train_loss is not None:
            if float(current_train_loss) > float(latest_val_loss) * 1.35:
                current_anomaly = f"train loss {compact_float(float(current_train_loss))} is well above the last validated loss"

    return {
        "task": task,
        "mode": mode,
        "seed": seed,
        "task_type": task_type,
        "primary_label": metric_label(task),
        "secondary_labels": secondary_labels,
        "rows": rows,
        "health": health,
        "anomalies": anomalies["items"],
        "current_anomaly": current_anomaly,
        "primary_sparkline": trend_sparkline(primary_values, lower_is_better=lower_is_better),
        "val_loss_sparkline": trend_sparkline(val_loss_values, lower_is_better=True),
        "throughput": {
            "steps_per_min": steps_per_min,
            "epochs_per_hour": epochs_per_hour,
            "seconds_per_step": seconds_per_step,
        },
        "current_vs_best": {
            "latest_primary": latest_primary,
            "best_primary": best_primary,
            "latest_val_loss": latest_val_loss,
            "best_val_loss": best_val_loss,
            "primary_delta_from_best": (
                None
                if latest_primary is None or best_primary is None
                else float(latest_primary) - float(best_primary)
            ),
            "val_loss_delta_from_best": (
                None
                if latest_val_loss is None or best_val_loss is None
                else float(latest_val_loss) - float(best_val_loss)
            ),
        },
        "reading_guide": reading_guide,
        "checkpoint": {
            "latest_label": latest_ckpt_label,
            "latest_age_seconds": latest_ckpt_age,
            "next_epoch": next_ckpt_epoch,
            "next_eta_seconds": next_ckpt_eta,
            "save_every": int(state.get("save_every") or DEFAULT_SAVE_EVERY),
            "timeline": timeline,
        },
    }


def recent_checkpoint_timeline(run_dir: Path, *, limit: int = 5) -> list[dict]:
    rows: list[dict] = []
    checkpoints: list[Path] = []
    for path in run_dir.glob("checkpoint_*.pth"):
        match = CHECKPOINT_RE.fullmatch(path.name)
        if match:
            checkpoints.append(path)
    checkpoints.sort(key=lambda path: path.stat().st_mtime if path.exists() else 0.0)
    for path in checkpoints[-limit:]:
        match = CHECKPOINT_RE.fullmatch(path.name)
        epoch_idx = int(match.group(1)) if match else None
        rows.append(
            {
                "label": path.name,
                "epoch": user_epoch(epoch_idx),
                "updated_at_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
                "age_seconds": max(0.0, now_ts() - path.stat().st_mtime),
            }
        )
    return rows


def build_disk_and_checkpoint_watch(system_snapshot: dict, active_monitor: dict | None) -> dict:
    disk_free_gb = system_snapshot.get("workspace_disk_free_gb")
    disk_used_pct = system_snapshot.get("workspace_disk_used_pct")
    disk_status = "done"
    disk_message = "Workspace disk headroom looks acceptable."
    if disk_free_gb is not None and disk_free_gb < 10.0:
        disk_status = "error"
        disk_message = "Workspace disk is critically low."
    elif disk_free_gb is not None and disk_free_gb < 25.0:
        disk_status = "partial"
        disk_message = "Workspace disk is getting tight."
    elif disk_used_pct is not None and disk_used_pct >= 95.0:
        disk_status = "error"
        disk_message = "Workspace disk usage is above 95%."
    elif disk_used_pct is not None and disk_used_pct >= 90.0:
        disk_status = "partial"
        disk_message = "Workspace disk usage is above 90%."

    checkpoint_status = "pending"
    checkpoint_message = "No active run checkpoint timeline yet."
    checkpoint_age = None
    checkpoint_expected = None
    if active_monitor is not None:
        checkpoint_age = active_monitor["checkpoint"].get("latest_age_seconds")
        checkpoint_expected = active_monitor["checkpoint"].get("next_eta_seconds")
        checkpoint_status = "done"
        checkpoint_message = "Checkpoint cadence looks normal."
        if checkpoint_age is not None:
            if checkpoint_expected is not None and checkpoint_age > max(float(checkpoint_expected) * 3.0, 900.0):
                checkpoint_status = "error"
                checkpoint_message = "Checkpoint age is well beyond the next expected save window."
            elif checkpoint_age > 5400.0:
                checkpoint_status = "error"
                checkpoint_message = "No fresh checkpoint has appeared for more than 90 minutes."
            elif checkpoint_age > 1800.0:
                checkpoint_status = "partial"
                checkpoint_message = "Checkpoint age is getting high; verify the run is still advancing."

    overall = checkpoint_status if status_badge(checkpoint_status) == "error" else disk_status
    if status_badge(disk_status) == "error":
        overall = disk_status
    elif status_badge(checkpoint_status) == "partial" or status_badge(disk_status) == "partial":
        overall = "partial"
    elif status_badge(checkpoint_status) == "pending" and status_badge(disk_status) == "done":
        overall = "partial"

    return {
        "status": overall,
        "disk_status": disk_status,
        "disk_message": disk_message,
        "checkpoint_status": checkpoint_status,
        "checkpoint_message": checkpoint_message,
        "disk_free_gb": disk_free_gb,
        "disk_used_pct": disk_used_pct,
        "checkpoint_age_seconds": checkpoint_age,
        "checkpoint_expected_seconds": checkpoint_expected,
        "disk_root": system_snapshot.get("workspace_disk_root"),
    }


def build_process_liveness_panel(state: dict, system_snapshot: dict, current_run: dict | None) -> dict:
    tracker = state.get("tracker") or {}
    tracker_restart_count = int(tracker.get("restart_count") or 0)
    tracker_pid = tracker.get("pid")
    trainer_pid = system_snapshot.get("trainer_pid")
    trainer_status = "running" if trainer_pid else "pending"
    trainer_message = "Trainer process detected." if trainer_pid else "No live trainer process detected right now."
    if current_run is None:
        trainer_status = "pending"
        trainer_message = "No active training run."

    return {
        "tracker_pid": tracker_pid,
        "tracker_started_at_utc": tracker.get("started_at_utc"),
        "tracker_restart_count": tracker_restart_count,
        "tracker_status": "done",
        "tracker_message": "Tracker process is running in this session.",
        "previous_pid": tracker.get("previous_pid"),
        "last_restart_from_phase": tracker.get("last_restart_from_phase"),
        "trainer_pid": trainer_pid,
        "trainer_status": trainer_status,
        "trainer_message": trainer_message,
        "trainer_launch_count": int(current_run.get("launch_count") or 0) if current_run else 0,
        "trainer_uptime": system_snapshot.get("trainer_etime"),
        "initial_launch_host": current_run.get("initial_launch_host") if current_run else None,
    }


def render_dashboard(state: dict) -> str:
    current = state["current"]
    bench = state["benchmarks"]
    radcom = state["radcom"]
    plots = state["plots"]
    summary = state["summary"]
    comparison = state["comparison"]
    preflight = state.get("preflight") or {}
    plan_info = state.get("plan") or {}
    overview = state.get("overview") or build_overview(state)
    data_readiness = overview.get("data_readiness") or {}
    artifact_freshness = overview.get("artifact_freshness") or {}
    resume_safety = overview.get("resume_safety") or {}
    commands_panel = overview.get("commands") or {}
    timers = overview["timers"]
    current_progress = float(current.get("progress_percent") or 0.0)
    bench_progress = float(bench.get("progress_percent") or 0.0)
    radcom_progress = float(radcom.get("progress_percent") or 0.0)
    plots_progress = float(plots.get("progress_percent") or 0.0)
    summary_progress = float(summary.get("progress_percent") or 0.0)
    comparison_progress = float(comparison.get("progress_percent") or 0.0)
    current_eta = human_duration(current.get("eta_seconds"))
    pipeline_eta = human_duration(state.get("pipeline_eta_seconds"))
    bench_eta = human_duration(bench.get("benchmark_eta_seconds"))
    current_elapsed = human_duration(timers.get("current_phase_elapsed_seconds"))
    current_run = bench.get("current_run")
    current_run_label = format_run_label(current_run)
    current_run_progress = format_run_progress(current_run)
    current_run_launch_kind = (
        "resume"
        if current_run and current_run.get("launch_kind") == "resume"
        else "fresh"
        if current_run
        else "idle"
    )
    current_run_origin = str(current_run.get("run_origin") or "unknown") if current_run else "n/a"
    next_run = bench.get("next_run")
    current_run_elapsed = human_duration(timers.get("current_run_elapsed_seconds"))
    active_monitor = load_active_run_monitor(state)
    hardware = runtime_hardware()
    system_snapshot = system_runtime_snapshot()
    failure_panel = load_failure_panel(state)
    runtime_history = list(((state.get("runtime_history") or {}).get("samples")) or [])
    experiment = build_experiment_context(
        list(state["tasks"]),
        state.get("train_subset_fraction"),
        state.get("train_subset_size"),
    )
    results_root = str(state.get("results_root") or RESULTS_ROOT)
    plots_output_root = str(plots.get("output_root") or Path(str(plots.get("manifest_path") or PLOT_MANIFEST_PATH)).parent)
    summary_output_root = str(
        summary.get("output_root") or Path(str(summary.get("manifest_path") or SUMMARY_MANIFEST_PATH)).parent
    )
    comparison_output_root = str(
        comparison.get("output_root") or Path(str(comparison.get("manifest_path") or COMPARISON_MANIFEST_PATH)).parent
    )
    disk_checkpoint_watch = build_disk_and_checkpoint_watch(system_snapshot, active_monitor)
    process_liveness = build_process_liveness_panel(state, system_snapshot, current_run)
    gpu_guard = state.get("gpu_guard") or {}
    gpu_util_values = [float(item["gpu_util"]) for item in runtime_history if item.get("gpu_util") is not None]
    gpu_mem_values = [float(item["gpu_mem_used_mb"]) for item in runtime_history if item.get("gpu_mem_used_mb") is not None]
    trainer_cpu_values = [float(item["trainer_cpu"]) for item in runtime_history if item.get("trainer_cpu") is not None]
    disk_free_values = [float(item["workspace_disk_free_gb"]) for item in runtime_history if item.get("workspace_disk_free_gb") is not None]
    gpu_util_sparkline = trend_sparkline(gpu_util_values[-12:], lower_is_better=False)
    gpu_mem_sparkline = trend_sparkline(gpu_mem_values[-12:], lower_is_better=False)
    trainer_cpu_sparkline = trend_sparkline(trainer_cpu_values[-12:], lower_is_better=False)
    disk_free_sparkline = trend_sparkline(disk_free_values[-12:], lower_is_better=True)
    checkpoint_timeline_rows = (
        "".join(
            f"<tr><td>{html.escape(str(item.get('epoch') or 'n/a'))}</td><td>{html.escape(item['label'])}</td><td>{html.escape(human_duration(item.get('age_seconds')))}</td><td>{html.escape(str(item.get('updated_at_utc') or 'n/a'))}</td></tr>"
            for item in (active_monitor["checkpoint"].get("timeline") or [])
        )
        if active_monitor is not None
        else ""
    )
    gpu_guard_actions_html = "".join(
        f"<li><span>{html.escape(str(item.get('time_utc') or 'n/a'))}</span><span>{html.escape(str(item.get('message') or ''))}</span></li>"
        for item in list(gpu_guard.get("actions") or [])[-5:]
    ) or "<li><span>none</span><span>No GPU safety actions recorded.</span></li>"
    gpu_guard_options_html = "".join(
        f"<li><span>option</span><span>{html.escape(str(item))}</span></li>"
        for item in list(gpu_guard.get("options") or [])
    )
    pause_remaining = None
    if gpu_guard.get("pause_until_ts") is not None:
        pause_remaining = max(0.0, float(gpu_guard.get("pause_until_ts") or 0.0) - now_ts())
    burnup = overview.get("burnup") or {}
    task_forecasts = overview.get("task_forecasts") or []
    eta_tracking = state.get("eta_tracking") or {}
    current_run_alignment = eta_tracking.get("current_run_alignment")
    pipeline_alignment = eta_tracking.get("pipeline_alignment")
    current_run_conf = eta_confidence(bench, current_run, eta_tracking, scope="current")
    pipeline_conf = eta_confidence(bench, current_run, eta_tracking, scope="pipeline")
    task_counts = overview["task_counts"]
    summary_counts = overview["summary_counts"]
    comparison_counts = overview["comparison_counts"]
    radcom_complete = bool(radcom.get("ready") and radcom.get("status") == "completed")
    phase_display = {
        PHASE_PARALLEL: "Ready Benchmarks + RadCom Preprocessing",
        PHASE_READY_BENCH: "Benchmark Queue",
        PHASE_WAIT_RADCOM: "Waiting For RadCom Cache",
        PHASE_RADCOM_BENCH: "Benchmark Queue",
        PHASE_PLOTS: "Refreshing Detailed-Eval Plots",
        PHASE_SUMMARY: "Summarizing Local Results",
        PHASE_COMPARE: "Comparing With Official Results",
        PHASE_COMPLETED: "Completed",
        PHASE_ERROR: "Error",
    }.get(state["phase"], state["phase"])
    phase_intro = (
        "This session is a reduced-data train-subset experiment. Outputs stay isolated under this session root. "
        if experiment["is_subset"]
        else ""
    )
    phase_intro += (
        "RadCom preprocessing is complete. The remaining work is the normal benchmark, plotting, summary, and comparison pipeline."
        if radcom_complete
        else "Ready benchmarks start immediately while radcom preprocessing runs in the background. RadCom training joins at the end."
    )
    status_endpoint = STATUS_PATH.name

    def alignment_text(alignment: dict | None) -> str:
        if not alignment:
            return "waiting for the next interval sample"
        drift = alignment.get("drift_seconds")
        if drift is None:
            return "waiting for the next interval sample"
        drift = float(drift)
        if abs(drift) <= 300:
            return f"on track (drift {human_duration(abs(drift))})"
        if drift > 0:
            return f"running faster than predicted by about {human_duration(abs(drift))}"
        return f"running slower than predicted by about {human_duration(abs(drift))}"

    def pill(label: str, status: str) -> str:
        return f"<span class='pill {status_badge(status)}'>{html.escape(label)}</span>"

    def counts_text(counts: dict[str, int], *, summary_mode: bool = False) -> str:
        if summary_mode:
            return (
                f"done {counts['done']} | partial {counts['partial']} | ready {counts['ready']} | "
                f"blocked {counts['blocked']} | pending {counts['pending']} | error {counts['error']}"
            )
        return (
            f"done {counts['done']} | active {counts['active']} | restart {counts['needs_restart']} | "
            f"pending {counts['pending']} | error {counts['error']}"
        )

    def section_counts_text(section: dict) -> str:
        completed_units = int(section.get("completed_units") or 0)
        total_units = int(section.get("total_units") or expected_task_mode_units(state))
        return f"{completed_units} / {total_units}"

    def section_focus_text(section: dict, idle_text: str) -> str:
        item = section.get("current_item_label")
        if item:
            return str(item)
        return idle_text

    next_runs_html = "\n".join(
        (
            "<li>"
            f"<span>{html.escape(format_run_label(item))}<br><span class='muted'>{html.escape(str(item.get('run_origin') or 'unknown'))}</span></span>"
            f"<span>{pill('restart', 'needs_restart') if item and item.get('resume_available') else pill('pending', 'pending')}</span>"
            "</li>"
        )
        for item in bench.get("next_runs", [])
    ) or "<li><span>none</span><span></span></li>"
    step_html = []
    for label, status in step_statuses(state):
        step_html.append(
            f'<div class="step {status_badge(status)}"><span>{html.escape(label)}</span><strong>{html.escape(status)}</strong></div>'
        )

    event_html = "\n".join(
        f"<li><span>{html.escape(item['time_utc'])}</span><span>{html.escape(item['message'])}</span></li>"
        for item in reversed(state["recent_events"][-14:])
    ) or "<li><span>no events yet</span><span></span></li>"

    error_html = (
        f"<div class='error-box'>{html.escape(state['last_error'])}</div>"
        if state.get("last_error")
        else "<div class='ok-box'>No errors recorded.</div>"
    )

    queue_cells = []
    for item in bench.get("run_plan", []):
        cell_status = (
            "done"
            if item["status"] in {"completed", "skipped"}
            else "active"
            if item["status"] == "running"
            else "restart"
            if item.get("resume_available")
            else "pending"
        )
        radcom_flag = " radcom-cell" if (not radcom_complete and item["task"] == "radcom") else ""
        queue_cells.append(
            f"<div class='queue-cell {cell_status}{radcom_flag}' title='{html.escape(format_run_label(item))}: {html.escape(item['status'])}'></div>"
        )
    queue_map_html = "".join(queue_cells) or "<div class='muted'>No queue entries.</div>"

    task_cards = []
    for row in overview["task_rows"]:
        mode_html = []
        for mode_row in row["mode_rows"]:
            benchmark_counts_text = counts_text(mode_row["benchmark_counts"])
            summary_label = mode_row["summary_status"]
            if mode_row["summary_status"] in {"done", "partial"}:
                summary_label = f"{mode_row['summary_status']} ({mode_row['n_seeds_summarized']}/{len(state['seeds'])} seeds)"
            compare_label = mode_row["comparison_status"]
            if mode_row["comparison_row_status"]:
                compare_label = f"{mode_row['comparison_status']} ({mode_row['comparison_row_status']})"
            metric_html = ""
            if mode_row["local_metric"] is not None:
                metric = mode_row["local_metric"]
                metric_html = (
                    f"<div class='metric-line'><strong>local {html.escape(str(metric['key']))}</strong>"
                    f"<span>{metric['mean']:.4f}"
                    f"{(' ± ' + format(metric['std'], '.4f')) if metric['n'] > 1 else ''}"
                    f" across {metric['n']}/{len(state['seeds'])} seeds</span></div>"
                )
            mode_html.append(
                f"""
                <div class="mode-row" data-mode="{html.escape(mode_row['mode'])}" data-bench-status="{html.escape(mode_row['benchmark_status'])}" data-summary-status="{html.escape(mode_row['summary_status'])}" data-compare-status="{html.escape(mode_row['comparison_status'])}">
                  <div class="mode-head">
                    <strong>{html.escape(mode_row['mode'])}</strong>
                    <div class="mode-pills">
                      {pill(mode_row['benchmark_status'], mode_row['benchmark_status'])}
                      {pill(summary_label, mode_row['summary_status'])}
                      {pill(compare_label, mode_row['comparison_status'])}
                    </div>
                  </div>
                  <div class="mode-meta">
                    <span>benchmarks: {html.escape(benchmark_counts_text)}</span>
                    <span>eta: {html.escape(human_duration(mode_row['eta_seconds']))}</span>
                    <span>time: {html.escape(human_duration(mode_row['duration_seconds']))}</span>
                  </div>
                  {metric_html}
                </div>
                """
            )
        task_class = status_badge(row["status"])
        task_cards.append(
            f"""
            <section class="task-card {task_class}" data-task="{html.escape(row['task'])}" data-status="{html.escape(row['status'])}">
              <div class="task-head">
                <div>
                  <h3>{html.escape(row['task'])}</h3>
                  <p>{html.escape(row['display_name'])}</p>
                </div>
                <div class="task-side">
                  {pill(row['status'], row['status'])}
                  <strong>{row['counts']['done']} / {row['counts']['total']}</strong>
                </div>
              </div>
              <div class="task-meta">
                <span>benchmarks: {html.escape(counts_text(row['counts']))}</span>
                <span>summary done: {row['summary_done_modes']} / {len(state['modes'])}</span>
                <span>compared: {row['comparison_done_modes']} / {len(state['modes'])}</span>
                <span>eta: {html.escape(human_duration(row['eta_seconds']))}</span>
                <span>time: {html.escape(human_duration(row['duration_seconds']))}</span>
                <span>primary metric: {html.escape(str(row['primary_metric']))}</span>
              </div>
              <div class="mode-grid">
                {''.join(mode_html)}
              </div>
            </section>
            """
        )

    run_done = bench["completed_runs"]
    run_remaining = bench["total_runs"] - bench["completed_runs"]
    run_restartable = sum(1 for item in bench["run_plan"] if item["status"] == "pending" and item.get("resume_available"))
    run_pending_fresh = sum(1 for item in bench["run_plan"] if item["status"] == "pending" and not item.get("resume_available"))
    run_errors = sum(1 for item in bench["run_plan"] if item["status"] == "error")
    run_active = 1 if current_run else 0
    stacked_total = max(1, bench["total_runs"])
    stacked_done = 100.0 * run_done / stacked_total
    stacked_active = 100.0 * run_active / stacked_total
    stacked_restart = 100.0 * run_restartable / stacked_total
    stacked_pending = max(0.0, 100.0 - stacked_done - stacked_active - stacked_restart)
    queue_legend = (
        "green = finished, blue = active, purple = restartable, tan = pending"
        if radcom_complete
        else "green = finished, blue = active, purple = restartable, tan = pending, orange inner border = RadCom"
    )
    experiment_strip = " ".join(
        f"<span class='pill ready'>{html.escape(message)}</span>"
        for message in (
            experiment["label"],
            experiment["short_label"],
            experiment["control_label"],
            f"tasks {experiment['task_scope_label']}",
        )
    )
    session_change_strip = " ".join(
        f"<span class='pill ready'>{html.escape(message)}</span>"
        for message in (plan_info.get("change_messages") or [])
    ) or "<span class='pill pending'>No plan deltas since the last tracker session.</span>"
    plan_changes_html = "\n".join(
        f"<li><span>plan</span><span>{html.escape(message)}</span></li>"
        for message in (plan_info.get("change_messages") or [])
    ) or "<li><span>plan</span><span>No recent plan changes detected.</span></li>"
    data_readiness_html = "".join(
        f"""
        <div class="mode-row">
          <div class="mode-head">
            <strong>{html.escape(row['cache_name'])}</strong>
            <div class="mode-pills">
              {pill(row['status'], row['status'])}
              {pill(', '.join(row['tasks']), 'ready' if row['status']=='done' else 'blocked')}
            </div>
          </div>
          <div class="mode-meta">
            <span>{html.escape(row['display_name'])}</span>
            <span>{html.escape(format_bytes(row['cache']['size_bytes']))}</span>
            <span>{html.escape(str(row['cache']['updated_at_utc'] or 'n/a'))}</span>
          </div>
          <div class="muted">{html.escape(row['detail'])}</div>
          <div class="muted">cache {html.escape(row['cache']['path'])}</div>
        </div>
        """
        for row in data_readiness.get("rows", [])
    ) or "<div class='muted'>No cache dependencies recorded.</div>"
    artifact_html = "".join(
        f"""
        <div class="mode-row">
          <div class="mode-head">
            <strong>{html.escape(row['label'])}</strong>
            <div class="mode-pills">{pill(row['status'], row['status'])}</div>
          </div>
          <div class="mode-meta">
            <span>{html.escape(format_bytes(row['file']['size_bytes']))}</span>
            <span>{html.escape(human_duration(row['file']['age_seconds']))}</span>
            <span>{html.escape(str(row['file']['updated_at_utc'] or 'n/a'))}</span>
          </div>
          <div class="muted">{html.escape(row['detail'])}</div>
          <div class="muted">{html.escape(row['file']['path'])}</div>
        </div>
        """
        for row in artifact_freshness.get("rows", [])
    ) or "<div class='muted'>No tracked artifacts yet.</div>"
    command_html = "\n".join(
        f"""
        <div class="info" style="margin-top: 12px;">
          <h3>{html.escape(label)}</h3>
          <code>{html.escape(value)}</code>
        </div>
        """
        for label, value in (
            ("Preflight", commands_panel.get("preflight", "n/a")),
            ("Tracker Restart", commands_panel.get("tracker_restart", "n/a")),
            ("Tmux Attach", commands_panel.get("tmux_attach", "n/a")),
            ("Dashboard Server", commands_panel.get("dashboard_server", "n/a")),
            ("Mac Tunnel", commands_panel.get("dashboard_tunnel", "n/a")),
            ("Mac Tunnel (Resilient)", commands_panel.get("dashboard_tunnel_resilient", "n/a")),
            ("Mac Tunnel (autossh)", commands_panel.get("dashboard_tunnel_autossh", "n/a")),
            ("Mac Tunnel Helper (local repo)", commands_panel.get("dashboard_tunnel_helper", "n/a")),
            ("Dashboard URL", commands_panel.get("dashboard_url", "n/a")),
        )
    )

    monitor_html = ""
    if active_monitor is not None:
        health = active_monitor["health"]
        secondary_1_label, secondary_2_label = active_monitor["secondary_labels"]
        epoch_rows = []
        for row in active_monitor["rows"]:
            anomaly_html = ""
            if row.get("anomaly"):
                anomaly_html = " ".join(pill("anomaly", "partial") + f" {html.escape(msg)}" for msg in row["anomaly"])
            epoch_rows.append(
                f"""
                <tr class="{status_badge(row['status'])}">
                  <td>{html.escape(str(row['epoch']) if row['epoch'] is not None else 'n/a')}</td>
                  <td>{pill(str(row['status']), row['status'])}</td>
                  <td>{html.escape(str(row['step']))}</td>
                  <td>{html.escape(compact_float(row['train_loss']))}</td>
                  <td>{html.escape(compact_float(row['val_loss']))}</td>
                  <td>{html.escape(compact_float(row['primary']))}</td>
                  <td>{html.escape(compact_float(row['secondary_1']))}</td>
                  <td>{html.escape(compact_float(row['secondary_2']))}</td>
                  <td>{html.escape(compact_float(row['lr']))}</td>
                  <td>{anomaly_html or "<span class='muted'>none</span>"}</td>
                </tr>
                """
            )
        anomaly_items = active_monitor.get("anomalies") or []
        anomaly_list_html = "".join(
            f"<li><span>epoch {item['epoch']}</span><span>{html.escape(item['message'])}</span></li>" for item in anomaly_items
        ) or "<li><span>none</span><span>No recent anomaly markers.</span></li>"
        monitor_html = f"""
        <div class="card" id="monitor" style="margin-bottom: 18px;">
          <div class="task-head">
            <div>
              <h2>Active Run Monitor</h2>
              <p class="muted">Current running epoch plus the previous 3 validated epochs for {html.escape(format_run_label(current_run))}.</p>
            </div>
            <div class="task-side">
              {pill(health['label'], health['status'])}
              <span class="muted">{html.escape(health['message'])}</span>
            </div>
          </div>
          <div class="monitor-grid">
            <div class="spark-card">
              <h3>{html.escape(active_monitor['primary_label'])} Trend</h3>
              {active_monitor['primary_sparkline']}
              <div class="chart-note">
                <strong>How to read:</strong> {html.escape(active_monitor['reading_guide']['summary'])}
              </div>
            </div>
            <div class="spark-card">
              <h3>Validation Loss Trend</h3>
              {active_monitor['val_loss_sparkline']}
              <div class="chart-note">
                <strong>Watch for:</strong> {html.escape(active_monitor['reading_guide']['watch'])}
              </div>
            </div>
            <div class="spark-card">
              <h3>Current Run Check</h3>
              <div class="task-meta">
                <span>task {html.escape(active_monitor['task'])}</span>
                <span>mode {html.escape(active_monitor['mode'])}</span>
                <span>seed {active_monitor['seed']}</span>
                <span>type {html.escape(active_monitor['task_type'])}</span>
              </div>
              <div class="task-meta">
                <span>steps/min {html.escape(compact_rate(active_monitor['throughput']['steps_per_min'], 'spm'))}</span>
                <span>epochs/hour {html.escape(compact_rate(active_monitor['throughput']['epochs_per_hour'], 'eph'))}</span>
                <span>sec/step {html.escape(compact_rate(active_monitor['throughput']['seconds_per_step'], 's'))}</span>
              </div>
              <div class="chart-note">
                current {html.escape(active_monitor['primary_label'])}: {html.escape(compact_float(active_monitor['current_vs_best']['latest_primary']))} |
                best: {html.escape(compact_float(active_monitor['current_vs_best']['best_primary']))} |
                gap: {html.escape(compact_float(active_monitor['current_vs_best']['primary_delta_from_best']))}
              </div>
            </div>
            <div class="spark-card">
              <h3>Checkpoint</h3>
              <div class="task-meta">
                <span>latest {html.escape(active_monitor['checkpoint']['latest_label'])}</span>
                <span>age {html.escape(human_duration(active_monitor['checkpoint']['latest_age_seconds']))}</span>
                <span>save every {active_monitor['checkpoint']['save_every']}</span>
              </div>
              <div class="task-meta">
                <span>next epoch {html.escape(str(active_monitor['checkpoint']['next_epoch'] or 'n/a'))}</span>
                <span>ETA {html.escape(human_duration(active_monitor['checkpoint']['next_eta_seconds']))}</span>
              </div>
            </div>
            <div class="spark-card">
              <h3>Checkpoint Timeline</h3>
              <div class="task-meta">
                <span>resume from {html.escape(Path(str(current_run.get('resume_source'))).name if current_run and current_run.get('resume_source') else 'fresh start')}</span>
                <span>launches {html.escape(str(current_run.get('launch_count') if current_run else 0))}</span>
              </div>
              <div class="epoch-table-wrap" style="margin-top:8px;">
                <table class="epoch-table">
                  <thead>
                    <tr>
                      <th>Epoch</th>
                      <th>Checkpoint</th>
                      <th>Age</th>
                      <th>Updated</th>
                    </tr>
                  </thead>
                  <tbody>
                    {checkpoint_timeline_rows or "<tr><td colspan='4' class='muted'>No checkpoint files yet.</td></tr>"}
                  </tbody>
                </table>
              </div>
            </div>
            <div class="spark-card">
              <h3>Anomaly Markers</h3>
              <ul>
                {anomaly_list_html}
              </ul>
              <div class="muted" style="margin-top:8px;">{html.escape(active_monitor.get('current_anomaly') or 'No live anomaly on the running step.')}</div>
            </div>
            <div class="spark-card">
              <h3>Validation Loss Check</h3>
              <div class="task-meta">
                <span>current {html.escape(compact_float(active_monitor['current_vs_best']['latest_val_loss']))}</span>
                <span>best {html.escape(compact_float(active_monitor['current_vs_best']['best_val_loss']))}</span>
                <span>gap {html.escape(compact_float(active_monitor['current_vs_best']['val_loss_delta_from_best']))}</span>
              </div>
              <div class="chart-note">
                Flat or falling validation loss is healthy. A repeated rise together with worse validation metrics is a stronger warning sign than one noisy step.
              </div>
            </div>
          </div>
          <div class="epoch-table-wrap">
            <table class="epoch-table">
              <thead>
                <tr>
                  <th>Epoch</th>
                  <th>Status</th>
                  <th>Step</th>
                  <th>Train loss</th>
                  <th>Val loss</th>
                  <th>{html.escape(active_monitor['primary_label'])}</th>
                  <th>{html.escape(secondary_1_label)}</th>
                  <th>{html.escape(secondary_2_label)}</th>
                  <th>LR</th>
                  <th>Markers</th>
                </tr>
              </thead>
              <tbody>
                {''.join(epoch_rows)}
              </tbody>
            </table>
          </div>
        </div>
        """

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>WavesFM Automation Tracker</title>
  <style>
    :root {{
      --bg: #f4efe7;
      --panel: #fffaf3;
      --panel-2: #fbf6ee;
      --ink: #1e2a33;
      --muted: #65727d;
      --line: #dccfbf;
      --accent: #1b7f6b;
      --bench: #b56731;
      --warn: #c46a2d;
      --error: #a63f3f;
      --done: #2a6f3a;
      --ready: #2f6ba8;
      --restart: #7b5eb5;
      --blocked: #8e6b42;
      --shadow: 0 16px 34px rgba(37, 42, 46, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background:
        radial-gradient(circle at top left, #fff7ec 0, #f4efe7 40%),
        linear-gradient(180deg, #efe6d8 0, #f6f1ea 100%);
      color: var(--ink);
    }}
    .toolbar {{
      position: sticky;
      top: 0;
      z-index: 20;
      background: rgba(244, 239, 231, 0.92);
      backdrop-filter: blur(8px);
      border-bottom: 1px solid var(--line);
    }}
    .toolbar-inner {{
      max-width: 1380px;
      margin: 0 auto;
      padding: 10px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .toolbar .stats {{
      display: flex;
      gap: 14px;
      color: var(--muted);
      font-size: 13px;
      flex-wrap: wrap;
    }}
    .toolbar .actions {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    button {{
      appearance: none;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      border-radius: 999px;
      padding: 8px 12px;
      font: inherit;
      cursor: pointer;
    }}
    button.primary {{
      background: #eef8f5;
      border-color: #9ccbbb;
    }}
    input,
    select {{
      appearance: none;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      border-radius: 12px;
      padding: 8px 10px;
      font: inherit;
      min-width: 140px;
    }}
    .filter-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }}
    .hidden-by-filter {{
      display: none !important;
    }}
    .wrap {{
      max-width: 1380px;
      margin: 0 auto;
      padding: 28px 20px 40px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 18px;
      margin-bottom: 18px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1.2fr 1fr;
      gap: 18px;
      margin-bottom: 18px;
    }}
    .grid-3 {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 18px;
      margin-bottom: 18px;
    }}
    .grid-4 {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 18px;
      margin-bottom: 18px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: var(--shadow);
      padding: 18px 20px;
    }}
    .connection-health.connected {{
      border-color: #b8d4bf;
      background: #f8fcf8;
    }}
    .connection-health.reconnecting {{
      border-color: #c9b7ea;
      background: #fbf7ff;
    }}
    .connection-health.lost {{
      border-color: #e0baba;
      background: #fff8f8;
    }}
    .connection-health.stale {{
      border-color: #e2c7a4;
      background: #fffaf4;
    }}
    h1, h2, h3, p {{ margin: 0; }}
    h1 {{ font-size: 32px; line-height: 1.1; margin-bottom: 10px; }}
    h2 {{ font-size: 18px; margin-bottom: 12px; }}
    h3 {{ font-size: 15px; margin-bottom: 8px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-top: 14px;
    }}
    .mini {{
      background: #f9f3e9;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
    }}
    .mini strong {{
      display: block;
      font-size: 20px;
      margin-top: 4px;
    }}
    .steps {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-top: 14px;
    }}
    .step {{
      border-radius: 14px;
      padding: 12px;
      border: 1px solid var(--line);
      background: #f9f3e9;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .step.active {{ border-color: var(--ready); background: #eef4fb; }}
    .step.done {{ border-color: var(--done); background: #edf7ef; }}
    .step.error {{ border-color: var(--error); background: #fbefef; }}
    .step.ready {{ border-color: var(--ready); background: #eef4fb; }}
    .step.restart {{ border-color: var(--restart); background: #f3eefb; }}
    .bar {{
      margin-top: 12px;
      height: 14px;
      border-radius: 999px;
      background: #eadfce;
      overflow: hidden;
    }}
    .bar > span {{
      display: block;
      height: 100%;
      width: 100%;
      background: linear-gradient(90deg, #1b7f6b, #58b39d);
    }}
    .bar.bench > span {{
      background: linear-gradient(90deg, var(--bench), #de9b58);
    }}
    .bar.radcom > span {{
      background: linear-gradient(90deg, #1b7f6b, #58b39d);
    }}
    .labels {{
      display: flex;
      justify-content: space-between;
      margin-top: 8px;
      color: var(--muted);
      font-size: 14px;
      gap: 10px;
    }}
    .status-line {{
      margin-top: 10px;
      font-size: 17px;
    }}
    .status-line strong {{ font-size: 22px; }}
    .two-col {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-top: 14px;
    }}
    .info {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      background: #fbf7f0;
    }}
    code {{
      font-family: "SFMono-Regular", Menlo, Monaco, monospace;
      font-size: 12px;
      word-break: break-all;
    }}
    ul {{
      list-style: none;
      margin: 0;
      padding: 0;
      display: grid;
      gap: 8px;
    }}
    li {{
      display: grid;
      grid-template-columns: 230px 1fr;
      gap: 12px;
      padding: 10px 0;
      border-bottom: 1px solid #ece1d2;
      font-size: 14px;
    }}
    li:last-child {{ border-bottom: 0; }}
    .queue li {{
      grid-template-columns: 1fr 120px;
    }}
    .ok-box, .error-box {{
      border-radius: 14px;
      padding: 12px;
      margin-top: 12px;
    }}
    .ok-box {{ background: #edf7ef; border: 1px solid #c7dfce; }}
    .error-box {{ background: #fbefef; border: 1px solid #e0baba; color: var(--error); }}
    .muted {{ color: var(--muted); }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 12px;
      border: 1px solid var(--line);
      background: #f8f1e6;
      white-space: nowrap;
    }}
    .pill.done {{ background: #edf7ef; border-color: #b8d4bf; color: var(--done); }}
    .pill.active {{ background: #eef4fb; border-color: #b4cae5; color: var(--ready); }}
    .pill.ready {{ background: #eef4fb; border-color: #b4cae5; color: var(--ready); }}
    .pill.partial {{ background: #fcf4ea; border-color: #e2c7a4; color: var(--warn); }}
    .pill.blocked {{ background: #f6f0e8; border-color: #d3b892; color: var(--blocked); }}
    .pill.restart {{ background: #f3eefb; border-color: #c9b7ea; color: var(--restart); }}
    .pill.pending {{ background: #f8f1e6; border-color: var(--line); color: var(--muted); }}
    .pill.error {{ background: #fbefef; border-color: #e0baba; color: var(--error); }}
    .timeline {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }}
    .stacked-bar {{
      display: flex;
      height: 16px;
      border-radius: 999px;
      overflow: hidden;
      background: #eadfce;
      margin-top: 10px;
    }}
    .stacked-bar > span {{ display: block; height: 100%; }}
    .stacked-bar .done {{ background: linear-gradient(90deg, #2a6f3a, #4ea45f); }}
    .stacked-bar .active {{ background: linear-gradient(90deg, #2f6ba8, #67a6df); }}
    .stacked-bar .restart {{ background: linear-gradient(90deg, #7b5eb5, #a58ae1); }}
    .stacked-bar .pending {{ background: linear-gradient(90deg, #c7b59e, #e1d2c1); }}
    .queue-map {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(18px, 1fr));
      gap: 6px;
      margin-top: 12px;
    }}
    .queue-cell {{
      height: 18px;
      border-radius: 6px;
      background: #e8ddcf;
      border: 1px solid #dbcbb7;
    }}
    .queue-cell.done {{ background: #dcefe1; border-color: #9ec8aa; }}
    .queue-cell.active {{ background: #dce9f8; border-color: #8db5df; }}
    .queue-cell.restart {{ background: #e9e0f9; border-color: #c4b0e8; }}
    .queue-cell.pending {{ background: #efe4d5; border-color: #d8c3aa; }}
    .queue-cell.error {{ background: #f8dddd; border-color: #ddaaaa; }}
    .queue-cell.radcom-cell {{ box-shadow: inset 0 0 0 2px #b56731; }}
    .task-list {{
      display: grid;
      gap: 14px;
    }}
    .task-card {{
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px;
      border-left-width: 7px;
    }}
    .task-card.done {{ border-left-color: var(--done); }}
    .task-card.active {{ border-left-color: var(--ready); }}
    .task-card.pending {{ border-left-color: #c9b8a2; }}
    .task-card.restart {{ border-left-color: var(--restart); }}
    .task-card.error {{ border-left-color: var(--error); }}
    .metric-line {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-top: 10px;
      padding-top: 10px;
      border-top: 1px dashed #e5d7c3;
      font-size: 13px;
    }}
    .task-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: start;
    }}
    .task-head p {{
      margin-top: 4px;
      color: var(--muted);
      font-size: 14px;
    }}
    .task-side {{
      display: flex;
      flex-direction: column;
      align-items: end;
      gap: 8px;
    }}
    .task-meta, .mode-meta {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 13px;
      margin-top: 10px;
    }}
    .mode-grid {{
      display: grid;
      gap: 10px;
      margin-top: 14px;
    }}
    .mode-row {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      background: white;
    }}
    .mode-head {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: start;
    }}
    .mode-pills {{
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      justify-content: end;
    }}
    .monitor-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-top: 14px;
      margin-bottom: 14px;
    }}
    .spark-card {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      background: white;
    }}
    .sparkline {{
      width: 100%;
      height: 56px;
      display: block;
      margin-top: 8px;
    }}
    .mini-chart {{
      width: 100%;
      height: 132px;
      display: block;
      margin-top: 8px;
    }}
    .burnup-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-top: 14px;
    }}
    .burnup-summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-top: 14px;
    }}
    .burnup-card {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      background: white;
    }}
    .burnup-card strong {{
      display: block;
      margin-top: 4px;
      font-size: 22px;
    }}
    .burnup-chart {{
      width: 100%;
      height: 180px;
      display: block;
      margin-top: 8px;
    }}
    .chart-note {{
      color: var(--muted);
      font-size: 13px;
      margin-top: 8px;
      line-height: 1.45;
    }}
    .runtime-badge {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: white;
      margin-top: 12px;
    }}
    .forecast-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    .forecast-table th,
    .forecast-table td {{
      padding: 10px 12px;
      border-bottom: 1px solid #ece1d2;
      text-align: left;
      white-space: nowrap;
    }}
    .forecast-table thead th {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      background: #faf4eb;
    }}
    .chart-legend {{
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
    }}
    .legend-chip {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}
    .legend-dot {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      display: inline-block;
    }}
    .epoch-table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: white;
    }}
    .epoch-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    .epoch-table th,
    .epoch-table td {{
      padding: 10px 12px;
      border-bottom: 1px solid #ece1d2;
      text-align: left;
      white-space: nowrap;
    }}
    .epoch-table thead th {{
      position: sticky;
      top: 0;
      background: #faf4eb;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .epoch-table tbody tr.running {{
      background: #eef4fb;
    }}
    .epoch-table tbody tr.active {{
      background: #eef4fb;
    }}
    .epoch-table tbody tr:last-child td {{
      border-bottom: 0;
    }}
    @media (max-width: 980px) {{
      .hero, .grid, .grid-3, .grid-4, .timeline {{ grid-template-columns: 1fr; }}
      .meta, .steps, .two-col {{ grid-template-columns: 1fr; }}
      .burnup-grid, .burnup-summary {{ grid-template-columns: 1fr; }}
      .monitor-grid {{ grid-template-columns: 1fr; }}
      li {{ grid-template-columns: 1fr; gap: 4px; }}
      .queue li {{ grid-template-columns: 1fr; }}
      .task-head, .mode-head {{ flex-direction: column; }}
      .task-side, .mode-pills {{ align-items: start; justify-content: start; }}
    }}
  </style>
</head>
<body data-updated-at="{html.escape(state['updated_at_utc'])}" data-status-endpoint="{html.escape(status_endpoint)}" data-pipeline-state="{html.escape(state['state'])}">
  <div class="toolbar">
    <div class="toolbar-inner">
      <div class="stats">
        <span>last tracker update: <strong id="since-update">calculating</strong></span>
        <span>ssh tunnel / dashboard link: <strong id="toolbar-connection-status">checking</strong></span>
        <span>page age: <strong id="page-age">0s</strong></span>
        <span>next connection check: <strong id="refresh-countdown">{AUTO_REFRESH_SECONDS}s</strong></span>
        <span>pipeline elapsed: <strong>{html.escape(human_duration(timers.get('pipeline_elapsed_seconds')))}</strong></span>
      </div>
      <div class="actions">
        <button class="primary" onclick="manualRefresh()">Check Now</button>
        <button onclick="toggleAutoRefresh()" id="auto-refresh-button">Pause Auto Refresh</button>
        <button onclick="jumpToId('overview')">Progress Check</button>
        <button onclick="jumpToId('connection-health')">Connection</button>
        <button onclick="jumpToId('monitor')">Run Monitor</button>
        <button onclick="jumpToId('burnup')">Burnup</button>
        <button onclick="jumpToId('failures')">Failures</button>
        <button onclick="jumpToId('tasks')">Tasks</button>
        <button onclick="jumpToId('queue')">Queue</button>
        <button onclick="jumpToId('events')">Events</button>
      </div>
    </div>
  </div>
  <div class="wrap">
    <div class="hero">
      <section class="card" id="failures">
        <h3>WavesFM Pipeline</h3>
        <h1>Pipeline Tracker</h1>
        <p>{html.escape(phase_intro)}</p>
        <div class="meta">
          <div class="mini">
            <h3>State</h3>
            <strong>{html.escape(state['state'])}</strong>
          </div>
          <div class="mini">
            <h3>Phase</h3>
            <strong>{html.escape(phase_display)}</strong>
          </div>
          <div class="mini">
            <h3>Pipeline ETA</h3>
            <strong>{html.escape(pipeline_eta)}</strong>
          </div>
          <div class="mini">
            <h3>Updated</h3>
            <strong>{html.escape(state['updated_at_utc'])}</strong>
          </div>
        </div>
        <div class="steps">
          {''.join(step_html)}
        </div>
      </section>
      <section class="card" id="burnup">
        <h3>Files</h3>
        <div class="two-col">
          <div class="info">
            <h3>Status JSON</h3>
            <code>{html.escape(state['status_path'])}</code>
          </div>
          <div class="info">
            <h3>Run Log</h3>
            <code>{html.escape(state['run_log_path'])}</code>
          </div>
          <div class="info">
            <h3>Dashboard</h3>
            <code>{html.escape(state['dashboard_path'])}</code>
          </div>
          <div class="info">
            <h3>Last Output</h3>
            <code>{html.escape(state.get('last_output') or 'n/a')}</code>
          </div>
        </div>
      </section>
    </div>

    <div class="card" style="margin-bottom: 18px;">
      <h2>Session Diff</h2>
      <div class="labels">
        <span>session root {html.escape(str(state['session_root']))}</span>
        <span>manifest {html.escape(str(state['session_manifest_path']))}</span>
      </div>
      <div class="labels" style="margin-top: 10px; gap: 8px;">
        {experiment_strip}
      </div>
      <div class="labels" style="margin-top: 10px; gap: 8px;">
        {session_change_strip}
      </div>
      <div class="muted" style="margin-top: 8px;">{html.escape(experiment['sampling_policy'])}</div>
      <div class="muted" style="margin-top: 4px;">{html.escape(experiment['storage_policy'])}</div>
    </div>

    <div class="grid-4" id="overview">
      <section class="card">
        <h2>Progress Check</h2>
        <div class="status-line"><strong>{run_done} / {bench['total_runs']}</strong> benchmark runs done</div>
        <div class="labels">
          <span>remaining {run_remaining}</span>
          <span>{bench_progress:.1f}%</span>
        </div>
        <div class="stacked-bar">
          <span class="done" style="width:{stacked_done:.2f}%"></span>
          <span class="active" style="width:{stacked_active:.2f}%"></span>
          <span class="restart" style="width:{stacked_restart:.2f}%"></span>
          <span class="pending" style="width:{stacked_pending:.2f}%"></span>
        </div>
        <div class="task-meta">
          <span>active {run_active}</span>
          <span>restartable {run_restartable}</span>
          <span>pipeline eta {html.escape(pipeline_eta)}</span>
        </div>
      </section>
      <section class="card">
        <h2>Experiment</h2>
        <div class="status-line"><strong>{html.escape(experiment['label'])}</strong></div>
        <div class="task-meta">
          <span>{html.escape(experiment['short_label'])}</span>
          <span>{html.escape(experiment['control_label'])}</span>
          <span>tasks {html.escape(experiment['task_scope_label'])}</span>
        </div>
        <div class="muted" style="margin-top:8px;">{html.escape(experiment['sampling_policy'])}</div>
        <div class="muted" style="margin-top:4px;">{html.escape(experiment['storage_policy'])}</div>
      </section>
      <section class="card">
        <h2>Task Status</h2>
        <div class="status-line"><strong>{task_counts['done']} / {task_counts['total']}</strong> tasks done</div>
        <div class="task-meta">
          <span>active {task_counts['active']}</span>
          <span>restart {task_counts['needs_restart']}</span>
          <span>pending {task_counts['pending']}</span>
          <span>error {task_counts['error']}</span>
        </div>
      </section>
      <section class="card">
        <h2>Plot Status</h2>
        <div class="status-line"><strong>{section_counts_text(plots)}</strong> plot task-modes done</div>
        <div class="task-meta">
          <span>{pill(str(plots.get('status') or 'pending'), str(plots.get('status') or 'pending'))}</span>
          <span>elapsed {html.escape(human_duration(timers.get('plots_elapsed_seconds')))}</span>
          <span>ETA {html.escape(human_duration(timers.get('plots_eta_seconds')))}</span>
        </div>
        <div class="bar"><span style="width:{plots_progress:.2f}%"></span></div>
        <div class="muted" style="margin-top:8px;">{html.escape(section_focus_text(plots, 'waiting for plot generation'))}</div>
      </section>
      <section class="card">
        <h2>Summary Status</h2>
        <div class="status-line"><strong>{summary_counts['done']} / {summary_counts['total']}</strong> task-mode summaries done</div>
        <div class="task-meta">
          <span>{pill(str(summary.get('status') or 'pending'), str(summary.get('status') or 'pending'))}</span>
          <span>phase {section_counts_text(summary)}</span>
          <span>active {summary_counts['active']}</span>
          <span>partial {summary_counts['partial']}</span>
          <span>ready {summary_counts['ready']}</span>
          <span>blocked {summary_counts['blocked']}</span>
          <span>pending {summary_counts['pending']}</span>
        </div>
        <div class="bar"><span style="width:{summary_progress:.2f}%"></span></div>
        <div class="muted" style="margin-top:8px;">{html.escape(section_focus_text(summary, 'waiting for summary generation'))}</div>
      </section>
      <section class="card">
        <h2>Comparison Status</h2>
        <div class="status-line"><strong>{comparison_counts['done']} / {comparison_counts['total']}</strong> task-mode comparisons done</div>
        <div class="task-meta">
          <span>{pill(str(comparison.get('status') or 'pending'), str(comparison.get('status') or 'pending'))}</span>
          <span>phase {section_counts_text(comparison)}</span>
          <span>active {comparison_counts['active']}</span>
          <span>partial {comparison_counts['partial']}</span>
          <span>ready {comparison_counts['ready']}</span>
          <span>blocked {comparison_counts['blocked']}</span>
          <span>pending {comparison_counts['pending']}</span>
        </div>
        <div class="bar"><span style="width:{comparison_progress:.2f}%"></span></div>
        <div class="muted" style="margin-top:8px;">{html.escape(section_focus_text(comparison, 'waiting for comparison generation'))}</div>
      </section>
    </div>

    <div class="grid-3" style="margin-bottom: 18px;">
      <section class="card">
        <h2>Plan Snapshot</h2>
        <div class="task-meta">
          <span>launch order {html.escape(' -> '.join(str(item) for item in (plan_info.get('launch_order') or state['tasks'])))}</span>
          <span>modes {html.escape(', '.join(str(item) for item in (plan_info.get('modes') or state['modes'])))}</span>
          <span>seeds {html.escape(', '.join('s' + str(item) for item in (plan_info.get('seeds') or state['seeds'])))}</span>
          <span>runs {bench['total_runs']}</span>
        </div>
        <div class="task-meta">
          <span>phase {html.escape(phase_display)}</span>
          <span>updated {html.escape(state['updated_at_utc'])}</span>
          <span>preflight {html.escape(str(preflight.get('status') or 'pending'))}</span>
          <span>session {html.escape(Path(str(state['session_root'])).name)}</span>
        </div>
      </section>
      <section class="card">
        <h2>Data Readiness</h2>
        <div class="task-meta">
          <span>ready groups {data_readiness.get('ready_groups', 0)} / {data_readiness.get('total_groups', 0)}</span>
          <span>ready tasks {data_readiness.get('ready_tasks', 0)}</span>
          <span>blocked tasks {data_readiness.get('blocked_tasks', 0)}</span>
        </div>
        <div class="mode-grid" style="margin-top: 12px;">
          {data_readiness_html}
        </div>
      </section>
      <section class="card">
        <h2>Artifact Freshness</h2>
        <div class="task-meta">
          <span>fresh {artifact_freshness.get('fresh_count', 0)}</span>
          <span>stale {artifact_freshness.get('stale_count', 0)}</span>
          <span>missing {artifact_freshness.get('missing_count', 0)}</span>
        </div>
        <div class="mode-grid" style="margin-top: 12px;">
          {artifact_html}
        </div>
      </section>
      <section class="card">
        <h2>Commands</h2>
        <div class="task-meta">
          <span>session root {html.escape(commands_panel.get('session_root', 'n/a'))}</span>
          <span>results root {html.escape(commands_panel.get('results_root', 'n/a'))}</span>
        </div>
        {command_html}
      </section>
      <section class="card">
        <h2>Resume Safety</h2>
        <div class="status-line"><strong>{html.escape(resume_safety.get('title', 'n/a'))}</strong></div>
        <div class="labels">
          <span>{pill(str(resume_safety.get('status', 'pending')), str(resume_safety.get('status', 'pending')))}</span>
          <span>{html.escape(str(resume_safety.get('detail', 'n/a')))}</span>
        </div>
        <div class="task-meta">
          <span>latest log epoch {html.escape(str(resume_safety.get('latest_log_epoch', 'n/a')))}</span>
          <span>resume epoch {html.escape(str(resume_safety.get('resume_epoch', 'n/a')))}</span>
          <span>best epoch {html.escape(str(resume_safety.get('best_epoch', 'n/a')))}</span>
        </div>
        <div class="task-meta">
          <span>stale log {html.escape(str(resume_safety.get('stale_log_tail', False)))}</span>
          <span>stale best {html.escape(str(resume_safety.get('stale_best', False)))}</span>
        </div>
        <div class="info" style="margin-top: 12px;">
          <h3>Resume Checkpoint</h3>
          <code>{html.escape(str(resume_safety.get('resume_checkpoint') or 'n/a'))}</code>
        </div>
        <div class="task-meta">
          <span>trim backups {html.escape(', '.join(resume_safety.get('trimmed_logs') or ['none']))}</span>
          <span>best backups {html.escape(', '.join(resume_safety.get('quarantined_best') or ['none']))}</span>
        </div>
      </section>
      <section class="card">
        <h2>Preflight</h2>
        <div class="status-line"><strong>{html.escape(str(preflight.get('status') or 'pending'))}</strong></div>
        <div class="task-meta">
          <span>started {html.escape(str(preflight.get('started_at_utc') or 'n/a'))}</span>
          <span>completed {html.escape(str(preflight.get('completed_at_utc') or 'n/a'))}</span>
        </div>
        <div class="muted" style="margin-top: 10px;">{html.escape(str(preflight.get('summary') or 'Preflight has not run yet.'))}</div>
        <div class="info" style="margin-top: 12px;">
          <h3>Report Path</h3>
          <code>{html.escape(str(preflight.get('report_path') or 'n/a'))}</code>
        </div>
      </section>
      <section class="card connection-health connected" id="connection-health">
        <h2>SSH Tunnel / Dashboard Link</h2>
        <div class="status-line"><strong id="connection-status">checking...</strong></div>
        <div class="task-meta">
          <span>tracker heartbeat <strong id="tracker-heartbeat">checking</strong></span>
          <span>retries <strong id="connection-retries">0</strong></span>
        </div>
        <div class="two-col">
          <div class="info">
            <h3>Connected Since</h3>
            <strong id="connection-since">checking...</strong>
            <div class="muted" id="connection-last-success">last successful check pending</div>
          </div>
          <div class="info">
            <h3>Connection Loss</h3>
            <strong id="connection-loss-since">none recorded</strong>
            <div class="muted" id="connection-loss-reason">No tunnel or dashboard failure recorded in this browser session.</div>
          </div>
          <div class="info">
            <h3>Recovery</h3>
            <strong id="connection-recovery-state">armed</strong>
            <div class="muted" id="connection-recovery-note">Automatic retries will keep probing the status JSON and reload the page when a fresh snapshot becomes reachable again.</div>
          </div>
          <div class="info">
            <h3>Retry Backoff</h3>
            <strong id="connection-next-retry">next probe pending</strong>
            <div class="muted" id="connection-same-error">same-error streak 0</div>
          </div>
          <div class="info">
            <h3>Last Tracker Snapshot Seen</h3>
            <strong id="tracker-last-update-wall">{html.escape(state['updated_at_utc'])}</strong>
            <div class="muted" id="tracker-last-update-age">age calculating</div>
          </div>
          <div class="info">
            <h3>Last Known Phase</h3>
            <strong id="connection-last-known-phase">{html.escape(phase_display)}</strong>
            <div class="muted" id="connection-last-known-work">{html.escape(current.get('label') or 'n/a')}</div>
          </div>
          <div class="info">
            <h3>Last Known Training State</h3>
            <strong id="connection-last-known-run">{html.escape(current_run_label)}</strong>
            <div class="muted" id="connection-last-known-progress">{html.escape(current_run_progress)}</div>
            <div class="muted" id="connection-last-known-validation">{html.escape((current_run.get('last_validation') if current_run else None) or 'n/a')}</div>
          </div>
          <div class="info">
            <h3>Mac Tunnel (Resilient)</h3>
            <code>{html.escape(commands_panel.get('dashboard_tunnel_resilient', commands_panel.get('dashboard_tunnel', 'n/a')))}</code>
          </div>
          <div class="info">
            <h3>Mac Tunnel (autossh)</h3>
            <code>{html.escape(commands_panel.get('dashboard_tunnel_autossh', 'n/a'))}</code>
          </div>
          <div class="info">
            <h3>Dashboard Server Command</h3>
            <code>{html.escape(commands_panel.get('dashboard_server', 'n/a'))}</code>
          </div>
        </div>
        <div class="chart-note">This is browser-side monitoring of the Mac-to-remote dashboard path. The page preserves the last known training state and keeps retrying the status endpoint, but it cannot recreate the SSH tunnel itself. Use the resilient Mac tunnel command or the `autossh` variant above if you want the tunnel to reconnect on its own.</div>
      </section>
      <section class="card">
        <h2>Runtime Hardware</h2>
        <div class="runtime-badge">
          <div>
            <strong>{html.escape(hardware['label'])}</strong>
            <div class="muted">{html.escape(hardware['detail'])}</div>
          </div>
          {pill(hardware['label'], hardware['badge'])}
        </div>
        <div class="task-meta">
          <span>CPU cores {html.escape(str(hardware['cpu_count']))}</span>
          <span>machine {html.escape(str(hardware['machine']))}</span>
          <span>python {html.escape(str(hardware['python']))}</span>
          <span>torch {html.escape(str(hardware['torch']))}</span>
        </div>
      </section>
      <section class="card">
        <h2>System Pressure</h2>
        <div class="task-meta">
          <span>power {html.escape(str(system_snapshot['power_source']))}</span>
          <span>battery {html.escape((str(system_snapshot['battery_percent']) + '%') if system_snapshot['battery_percent'] is not None else 'n/a')}</span>
          <span>state {html.escape(str(system_snapshot['battery_state'] or 'n/a'))}</span>
        </div>
        <div class="task-meta">
          <span>load 1m {system_snapshot['load1_pct']:.0f}%</span>
          <span>load 5m {system_snapshot['load5_pct']:.0f}%</span>
          <span>load 15m {system_snapshot['load15_pct']:.0f}%</span>
        </div>
        <div class="task-meta">
          <span>trainer cpu {html.escape(compact_rate(system_snapshot['trainer_cpu'], '%'))}</span>
          <span>trainer mem {html.escape(compact_rate(system_snapshot['trainer_mem'], '%'))}</span>
          <span>trainer up {html.escape(str(system_snapshot['trainer_etime'] or 'n/a'))}</span>
        </div>
        <div class="task-meta">
          <span>free mem {html.escape(compact_rate(system_snapshot['memory_free_gb'], 'GB'))}</span>
          <span>available {html.escape(compact_rate(system_snapshot['memory_available_gb'], 'GB'))}</span>
          <span>compressed {html.escape(compact_rate(system_snapshot['memory_compressed_gb'], 'GB'))}</span>
          <span>total {html.escape(compact_rate(system_snapshot['memory_total_gb'], 'GB'))}</span>
        </div>
        <div class="task-meta">
          <span>gpu {html.escape(str(system_snapshot['gpu_name'] or 'n/a'))}</span>
          <span>gpu util {html.escape(compact_rate(system_snapshot['gpu_util'], '%'))}</span>
          <span>gpu mem {html.escape(compact_rate(system_snapshot['gpu_mem_used_mb'], 'MB'))} / {html.escape(compact_rate(system_snapshot['gpu_mem_total_mb'], 'MB'))}</span>
          <span>gpu temp {html.escape(compact_rate(system_snapshot['gpu_temp_c'], 'C'))}</span>
        </div>
        <div class="chart-note">{html.escape(system_snapshot['pressure_note'])}</div>
      </section>
      <section class="card">
        <h2>Process Liveness</h2>
        <div class="task-meta">
          <span>{pill('tracker', process_liveness['tracker_status'])}</span>
          <span>pid {html.escape(str(process_liveness['tracker_pid'] or 'n/a'))}</span>
          <span>restarts {html.escape(str(process_liveness['tracker_restart_count']))}</span>
        </div>
        <div class="muted" style="margin-top:8px;">{html.escape(process_liveness['tracker_message'])}</div>
        <div class="task-meta">
          <span>started {html.escape(str(process_liveness['tracker_started_at_utc'] or 'n/a'))}</span>
          <span>previous pid {html.escape(str(process_liveness['previous_pid'] or 'n/a'))}</span>
          <span>from phase {html.escape(str(process_liveness['last_restart_from_phase'] or 'fresh start'))}</span>
        </div>
        <div class="task-meta" style="margin-top:12px;">
          <span>{pill('trainer', process_liveness['trainer_status'])}</span>
          <span>pid {html.escape(str(process_liveness['trainer_pid'] or 'n/a'))}</span>
          <span>launches {html.escape(str(process_liveness['trainer_launch_count']))}</span>
        </div>
        <div class="muted" style="margin-top:8px;">{html.escape(process_liveness['trainer_message'])}</div>
        <div class="task-meta">
          <span>uptime {html.escape(str(process_liveness['trainer_uptime'] or 'n/a'))}</span>
          <span>initial host {html.escape(str(process_liveness['initial_launch_host'] or 'n/a'))}</span>
        </div>
      </section>
      <section class="card">
        <h2>GPU Safety Guard</h2>
        <div class="task-meta">
          <span>{pill(str(gpu_guard.get('status') or 'armed'), str(gpu_guard.get('status') or 'pending'))}</span>
          <span>temp {html.escape(compact_rate(gpu_guard.get('last_temp_c'), 'C'))}</span>
          <span>checked {html.escape(str(gpu_guard.get('last_check_utc') or 'n/a'))}</span>
        </div>
        <div class="task-meta">
          <span>pause at {html.escape(compact_rate(gpu_guard.get('pause_threshold_c'), 'C'))}</span>
          <span>resume below {html.escape(compact_rate(gpu_guard.get('resume_threshold_c'), 'C'))}</span>
          <span>critical at {html.escape(compact_rate(gpu_guard.get('critical_threshold_c'), 'C'))}</span>
        </div>
        <div class="muted" style="margin-top:8px;">{html.escape(str(gpu_guard.get('pause_reason') or gpu_guard.get('stop_reason') or gpu_guard.get('last_action') or 'GPU safety guard armed.'))}</div>
        <div class="task-meta" style="margin-top:12px;">
          <span>hot streak {html.escape(str(gpu_guard.get('hot_streak') or 0))}</span>
          <span>cool streak {html.escape(str(gpu_guard.get('cool_streak') or 0))}</span>
          <span>critical streak {html.escape(str(gpu_guard.get('critical_streak') or 0))}</span>
        </div>
        <div class="task-meta">
          <span>paused since {html.escape(str(gpu_guard.get('pause_started_at_utc') or 'n/a'))}</span>
          <span>pause remaining {html.escape(human_duration(pause_remaining))}</span>
          <span>stop requested {html.escape(str(gpu_guard.get('stop_requested_at_utc') or 'n/a'))}</span>
        </div>
        <div class="info" style="margin-top: 12px;">
          <h3>Recent Safety Actions</h3>
          <ul>{gpu_guard_actions_html}</ul>
        </div>
        <div class="info" style="margin-top: 12px;">
          <h3>Reduce Risk Options</h3>
          <ul>{gpu_guard_options_html}</ul>
        </div>
      </section>
      <section class="card">
        <h2>Disk And Checkpoint Watch</h2>
        <div class="task-meta">
          <span>{pill('disk', disk_checkpoint_watch['disk_status'])}</span>
          <span>free {html.escape(compact_rate(disk_checkpoint_watch['disk_free_gb'], 'GB'))}</span>
          <span>used {html.escape(compact_rate(disk_checkpoint_watch['disk_used_pct'], '%'))}</span>
        </div>
        <div class="muted" style="margin-top:8px;">{html.escape(disk_checkpoint_watch['disk_message'])}</div>
        <div class="info" style="margin-top: 12px;">
          <h3>Workspace Disk Root</h3>
          <code>{html.escape(str(disk_checkpoint_watch['disk_root'] or 'n/a'))}</code>
        </div>
        <div class="task-meta" style="margin-top:12px;">
          <span>{pill('checkpoint', disk_checkpoint_watch['checkpoint_status'])}</span>
          <span>age {html.escape(human_duration(disk_checkpoint_watch['checkpoint_age_seconds']))}</span>
          <span>expected {html.escape(human_duration(disk_checkpoint_watch['checkpoint_expected_seconds']))}</span>
        </div>
        <div class="muted" style="margin-top:8px;">{html.escape(disk_checkpoint_watch['checkpoint_message'])}</div>
      </section>
      <section class="card">
        <h2>GPU And Host Trends</h2>
        <div class="monitor-grid" style="grid-template-columns: repeat(2, minmax(0, 1fr)); margin-top: 0;">
          <div class="spark-card">
            <h3>GPU Utilization</h3>
            {gpu_util_sparkline}
            <div class="chart-note">Latest {html.escape(compact_rate(system_snapshot['gpu_util'], '%'))}</div>
          </div>
          <div class="spark-card">
            <h3>GPU Memory</h3>
            {gpu_mem_sparkline}
            <div class="chart-note">Latest {html.escape(compact_rate(system_snapshot['gpu_mem_used_mb'], 'MB'))} / {html.escape(compact_rate(system_snapshot['gpu_mem_total_mb'], 'MB'))}</div>
          </div>
          <div class="spark-card">
            <h3>Trainer CPU</h3>
            {trainer_cpu_sparkline}
            <div class="chart-note">Latest {html.escape(compact_rate(system_snapshot['trainer_cpu'], '%'))}</div>
          </div>
          <div class="spark-card">
            <h3>Disk Free</h3>
            {disk_free_sparkline}
            <div class="chart-note">Latest {html.escape(compact_rate(system_snapshot['workspace_disk_free_gb'], 'GB'))}</div>
          </div>
        </div>
      </section>
      <section class="card">
        <h2>Throughput</h2>
        <div class="status-line"><strong>{html.escape(current_run_label)}</strong></div>
        <div class="task-meta">
          <span>steps/min {html.escape(compact_rate(active_monitor['throughput']['steps_per_min'], 'spm') if active_monitor else 'n/a')}</span>
          <span>epochs/hour {html.escape(compact_rate(active_monitor['throughput']['epochs_per_hour'], 'eph') if active_monitor else 'n/a')}</span>
          <span>sec/step {html.escape(compact_rate(active_monitor['throughput']['seconds_per_step'], 's') if active_monitor else 'n/a')}</span>
        </div>
        <div class="muted" style="margin-top:10px;">Use this to spot sudden slowdowns from data-loading stalls or unexpectedly heavy tasks.</div>
      </section>
      <section class="card">
        <h2>Checkpoint Status</h2>
        <div class="task-meta">
          <span>latest {html.escape(active_monitor['checkpoint']['latest_label'] if active_monitor else 'n/a')}</span>
          <span>age {html.escape(human_duration(active_monitor['checkpoint']['latest_age_seconds']) if active_monitor else 'estimating')}</span>
        </div>
        <div class="task-meta">
          <span>next epoch {html.escape(str(active_monitor['checkpoint']['next_epoch']) if active_monitor and active_monitor['checkpoint']['next_epoch'] is not None else 'n/a')}</span>
          <span>ETA {html.escape(human_duration(active_monitor['checkpoint']['next_eta_seconds'] if active_monitor else None))}</span>
          <span>save every {html.escape(str(active_monitor['checkpoint']['save_every']) if active_monitor else str(DEFAULT_SAVE_EVERY))}</span>
        </div>
      </section>
      <section class="card">
        <h2>ETA Check</h2>
        <div class="task-meta">
          <span>sample interval {html.escape(human_duration((eta_tracking.get('interval_seconds') or 600.0)))}</span>
          <span>last sample {html.escape(str((eta_tracking.get('samples') or [{}])[-1].get('time_utc', 'n/a')))}</span>
        </div>
        <div class="info" style="margin-top: 12px;">
          <h3>Current Run</h3>
          <div class="muted">{html.escape(alignment_text(current_run_alignment))}</div>
          <div class="muted">Confidence {html.escape(current_run_conf['level'])}: {html.escape(current_run_conf['summary'])}</div>
        </div>
        <div class="info" style="margin-top: 12px;">
          <h3>Whole Pipeline</h3>
          <div class="muted">{html.escape(alignment_text(pipeline_alignment))}</div>
          <div class="muted">Confidence {html.escape(pipeline_conf['level'])}: {html.escape(pipeline_conf['summary'])}</div>
        </div>
      </section>
      <section class="card">
        <h2>Failure And Restart</h2>
        <div class="{status_badge(failure_panel['status'])}">
          {pill(failure_panel['status'], failure_panel['status'])}
        </div>
        <div class="status-line" style="margin-top:10px;"><strong>{html.escape(failure_panel['title'])}</strong></div>
        <div class="muted" style="margin-top:8px;">{html.escape(failure_panel['message'])}</div>
        <div class="info" style="margin-top: 12px;">
          <h3>Resume Command</h3>
          <code>{html.escape(failure_panel.get('resume_command') or 'n/a')}</code>
        </div>
        <div class="info" style="margin-top: 12px;">
          <h3>Clean Rerun Command</h3>
          <code>{html.escape(failure_panel.get('restart_command') or 'n/a')}</code>
        </div>
      </section>
    </div>

    <div class="card" style="margin-bottom: 18px;">
      <h2>Time Tracking</h2>
      <div class="chart-note" style="margin-bottom: 12px;">
        <strong>Scope key:</strong> Whole Pipeline = everything left until the full automation finishes.
        Benchmark Queue = all remaining task/mode/seed runs only.
        Current Run = just the active training run.
      </div>
      <div class="timeline">
        <div class="info">
          <h3>Whole Pipeline</h3>
          <strong>{html.escape(human_duration(timers.get('pipeline_elapsed_seconds')))}</strong>
          <div class="muted">ETA to all remaining work: {html.escape(human_duration(timers.get('pipeline_eta_seconds')))}</div>
          <div class="muted">Confidence {html.escape(pipeline_conf['level'])}: {html.escape(pipeline_conf['summary'])}</div>
        </div>
        <div class="info">
          <h3>Current Phase Only</h3>
          <strong>{html.escape(current_elapsed)}</strong>
          <div class="muted">ETA for this phase: {html.escape(current_eta)}</div>
        </div>
        <div class="info">
          <h3>Current Run Only</h3>
          <strong>{html.escape(current_run_elapsed)}</strong>
          <div class="muted">ETA for active run: {html.escape(human_duration(timers.get('current_run_eta_seconds')))}</div>
          <div class="muted">Confidence {html.escape(current_run_conf['level'])}: {html.escape(current_run_conf['summary'])}</div>
        </div>
        <div class="info">
          <h3>Benchmark Queue</h3>
          <strong>{html.escape(human_duration(timers.get('benchmark_elapsed_seconds')))}</strong>
          <div class="muted">ETA for benchmark runs: {html.escape(bench_eta)}</div>
          <div class="muted">Confidence {html.escape(pipeline_conf['level'])}: {html.escape(pipeline_conf['summary'])}</div>
        </div>
        {"" if radcom_complete else f'''
        <div class="info">
          <h3>RadCom</h3>
          <strong>{html.escape(human_duration(timers.get('radcom_elapsed_seconds')))}</strong>
          <div class="muted">ETA {html.escape(human_duration(timers.get('radcom_eta_seconds')))}</div>
        </div>
        '''}
        <div class="info">
          <h3>Plots</h3>
          <strong>{html.escape(human_duration(timers.get('plots_elapsed_seconds')))}</strong>
          <div class="muted">ETA {html.escape(human_duration(timers.get('plots_eta_seconds')))}</div>
        </div>
        <div class="info">
          <h3>Summary</h3>
          <strong>{html.escape(human_duration(timers.get('summary_elapsed_seconds')))}</strong>
          <div class="muted">ETA {html.escape(human_duration(timers.get('summary_eta_seconds')))}</div>
        </div>
        <div class="info">
          <h3>Comparison</h3>
          <strong>{html.escape(human_duration(timers.get('comparison_elapsed_seconds')))}</strong>
          <div class="muted">ETA {html.escape(human_duration(timers.get('comparison_eta_seconds')))}</div>
        </div>
        <div class="info">
          <h3>Tracker Refresh</h3>
          <strong>every {AUTO_REFRESH_SECONDS}s</strong>
          <div class="muted">manual override available</div>
        </div>
      </div>
    </div>

    <div class="grid" style="margin-bottom: 18px;">
      <section class="card">
        <h2>Burnup</h2>
        <p class="muted">Cumulative completion over wall-clock time. Each step up means a run or full task finished.</p>
        <div class="burnup-summary">
          <div class="burnup-card">
            <h3>Runs</h3>
            <strong>{burnup.get('runs_done', 0)} / {burnup.get('runs_total', bench['total_runs'])}</strong>
            <div class="muted">{burnup.get('run_percent', 0.0):.1f}% complete</div>
          </div>
          <div class="burnup-card">
            <h3>Tasks</h3>
            <strong>{burnup.get('tasks_done', 0)} / {burnup.get('tasks_total', len(state['tasks']))}</strong>
            <div class="muted">{burnup.get('task_percent', 0.0):.1f}% complete</div>
          </div>
          <div class="burnup-card">
            <h3>Pace</h3>
            <strong>{html.escape(compact_rate(burnup.get('run_rate_per_hour'), 'runs/hr'))}</strong>
            <div class="muted">task pace {html.escape(compact_rate(burnup.get('task_rate_per_hour'), 'tasks/hr'))}</div>
          </div>
          <div class="burnup-card">
            <h3>Latest Milestone</h3>
            <strong>{html.escape(format_wall_time(burnup.get('latest_completion_ts')))}</strong>
            <div class="muted">expected finish {html.escape(format_wall_time(burnup.get('expected_finish_ts')))}</div>
          </div>
        </div>
        <div class="burnup-grid">
          <div class="spark-card">
            <h3>Runs Completed Over Time</h3>
            {burnup.get('run_chart') or "<div class='muted'>n/a</div>"}
            <div class="chart-note">
              Read this as cumulative completed runs. A flat segment means no run finished in that time window. A steeper staircase means faster throughput.
            </div>
          </div>
          <div class="spark-card">
            <h3>Tasks Completed Over Time</h3>
            {burnup.get('task_chart') or "<div class='muted'>n/a</div>"}
            <div class="chart-note">
              This only moves when all 9 runs for a task are done, so it will stay flat for long periods and then jump by one task.
            </div>
          </div>
        </div>
        <div class="epoch-table-wrap" style="margin-top:14px;">
          <table class="forecast-table">
            <thead>
              <tr>
                <th>Time</th>
                <th>Type</th>
                <th>Completed Item</th>
                <th>Cumulative</th>
                <th>%</th>
              </tr>
            </thead>
            <tbody>
              {"".join(
                f"<tr><td>{html.escape(format_wall_time(item['ts']))}</td><td>{pill(item['kind'], 'done' if item['kind']=='task' else 'active')}</td><td>{html.escape(item['label'])}</td><td>{html.escape(item['count_text'])}</td><td>{item['percent']:.1f}%</td></tr>"
                for item in burnup.get('recent_milestones', [])
              ) or "<tr><td colspan='5' class='muted'>No completed milestones yet.</td></tr>"}
            </tbody>
          </table>
        </div>
      </section>
      <section class="card">
        <h2>Per-Task Runtime Forecast</h2>
        <div class="epoch-table-wrap">
          <table class="forecast-table">
            <thead>
              <tr>
                <th>Task</th>
                <th>Done</th>
                <th>Optimistic</th>
                <th>Expected</th>
                <th>Slow</th>
              </tr>
            </thead>
            <tbody>
              {"".join(
                f"<tr><td>{html.escape(item['task'])}<div class='muted'>{html.escape(item['display_name'])}</div></td><td>{item['done']}/{item['total']}</td><td>{html.escape(human_duration(item['optimistic']))}</td><td>{html.escape(human_duration(item['expected']))}</td><td>{html.escape(human_duration(item['slow']))}</td></tr>"
                for item in task_forecasts
              ) or "<tr><td colspan='5' class='muted'>No remaining task forecasts.</td></tr>"}
            </tbody>
          </table>
        </div>
      </section>
    </div>

    <div class="grid">
      <section class="card">
        <h2>Current Pipeline Work</h2>
        <div class="status-line"><strong>{html.escape(current['label'])}</strong></div>
        <div class="labels">
          <span>{html.escape(current.get('progress_label') or 'waiting')}</span>
          <span>{current_progress:.1f}%</span>
        </div>
        <div class="bar"><span style="width:{current_progress:.2f}%"></span></div>
        <div class="two-col">
          <div class="info">
            <h3>Elapsed</h3>
            <strong>{html.escape(current_elapsed)}</strong>
          </div>
          <div class="info">
            <h3>ETA</h3>
            <strong>{html.escape(current_eta)}</strong>
          </div>
        </div>
        <div class="info" style="margin-top: 12px;">
          <h3>Active Command(s)</h3>
          <code>{html.escape(current.get('command') or 'n/a')}</code>
        </div>
      </section>

      <section class="card">
        <h2>Current Benchmark Run</h2>
        <div class="status-line"><strong>{html.escape(current_run_label)}</strong></div>
        <div class="labels">
          <span>{html.escape(current_run_progress)}</span>
          <span>Current run ETA: {html.escape(human_duration(current_run.get('eta_seconds') if current_run else None))}</span>
        </div>
        <div class="muted" style="margin-top:8px;">ETA confidence {html.escape(current_run_conf['level'])}: {html.escape(current_run_conf['summary'])}</div>
        <div class="two-col">
          <div class="info">
            <h3>Run Position</h3>
            <strong>{(current_run.get('queue_number') if current_run else bench['completed_runs'])} / {bench['total_runs']}</strong>
          </div>
          <div class="info">
            <h3>Next Up</h3>
            <strong>{html.escape(format_run_label(next_run))}</strong>
          </div>
          <div class="info">
            <h3>Launch Kind</h3>
            <strong>{html.escape(current_run_launch_kind)}</strong>
          </div>
          <div class="info">
            <h3>Resume Source</h3>
            <strong>{html.escape(Path(str(current_run.get('resume_source'))).name if current_run and current_run.get('resume_source') else 'n/a')}</strong>
          </div>
          <div class="info">
            <h3>Run Origin</h3>
            <strong>{html.escape(current_run_origin)}</strong>
          </div>
        </div>
        <div class="info" style="margin-top: 12px;">
          <h3>Last Validation</h3>
          <code>{html.escape((current_run.get('last_validation') if current_run else None) or 'n/a')}</code>
        </div>
      </section>
    </div>

    {monitor_html}

    <div class="grid-3">
      <section class="card">
        <h2>Benchmarks</h2>
        <div class="status-line"><strong>{bench['completed_runs']} / {bench['total_runs']}</strong> runs processed</div>
        <div class="labels">
          <span>training queue progress</span>
          <span>{bench_progress:.1f}%</span>
        </div>
        <div class="bar bench"><span style="width:{bench_progress:.2f}%"></span></div>
        <div class="two-col">
          <div class="info">
            <h3>Average Finished Run</h3>
            <strong>{html.escape(human_duration(bench.get('avg_run_seconds')))}</strong>
          </div>
          <div class="info">
            <h3>Benchmark ETA</h3>
            <strong>{html.escape(bench_eta)}</strong>
          </div>
          <div class="info">
            <h3>Runs Remaining</h3>
            <strong>{run_remaining}</strong>
          </div>
          <div class="info">
            <h3>Restartable</h3>
            <strong>{run_restartable}</strong>
          </div>
        </div>
      </section>

      {f'''
      <section class="card">
        <h2>Queue Status</h2>
        <div class="status-line"><strong>{bench['completed_runs']} / {bench['total_runs']}</strong> runs completed</div>
        <div class="labels">
          <span>fresh pending {run_pending_fresh}</span>
          <span>restartable {run_restartable}</span>
        </div>
        <div class="two-col">
          <div class="info">
            <h3>Fresh Pending</h3>
            <strong>{run_pending_fresh}</strong>
          </div>
          <div class="info">
            <h3>Errors</h3>
            <strong>{run_errors}</strong>
          </div>
          <div class="info">
            <h3>RadCom Runs Done</h3>
            <strong>{bench['completed_radcom_runs']} / {bench['radcom_total_runs']}</strong>
          </div>
          <div class="info">
            <h3>RadCom Status</h3>
            <strong>{html.escape('ready' if radcom.get('ready') else str(radcom.get('status') or 'pending'))}</strong>
          </div>
        </div>
      </section>
      ''' if not radcom_complete else f'''
      <section class="card">
        <h2>Queue Status</h2>
        <div class="status-line"><strong>{bench['completed_runs']} / {bench['total_runs']}</strong> runs completed</div>
        <div class="labels">
          <span>fresh pending {run_pending_fresh}</span>
          <span>restartable {run_restartable}</span>
        </div>
        <div class="two-col">
          <div class="info">
            <h3>Fresh Pending</h3>
            <strong>{run_pending_fresh}</strong>
          </div>
          <div class="info">
            <h3>Errors</h3>
            <strong>{run_errors}</strong>
          </div>
          <div class="info">
            <h3>Completed Tasks</h3>
            <strong>{task_counts['done']} / {task_counts['total']}</strong>
          </div>
          <div class="info">
            <h3>Summary Ready</h3>
            <strong>{summary_counts['ready']}</strong>
          </div>
        </div>
      </section>
      '''}

      <section class="card">
        <h2>Next Runs</h2>
        <ul class="queue">{next_runs_html}</ul>
      </section>
    </div>

    <div class="grid-3">
      <section class="card" id="queue">
        <h2>Queue Map</h2>
        <div class="task-meta">
          <span>{pill('done', 'done')}</span>
          <span>{pill('active', 'active')}</span>
          <span>{pill('restart', 'needs_restart')}</span>
          <span>{pill('pending', 'pending')}</span>
          <span>{html.escape(queue_legend)}</span>
        </div>
        <div class="queue-map">{queue_map_html}</div>
      </section>
      <section class="card">
        <h2>Status Guide</h2>
        <ul>
          <li><span>done</span><span>all required work for that item exists locally</span></li>
          <li><span>active</span><span>currently running now</span></li>
          <li><span>restart</span><span>not done, but a checkpoint exists so it can resume cleanly</span></li>
          <li><span>blocked</span><span>waiting on an earlier stage. Summary waits for benchmarks; comparison waits for summary.</span></li>
          <li><span>pending</span><span>not started yet and no restart point exists yet</span></li>
          <li><span>error</span><span>the stage finished in a bad state or recorded a failure</span></li>
        </ul>
      </section>
      <section class="card">
        <h2>Stage Files</h2>
        <div class="two-col">
          <div class="info">
            <h3>Run Outputs</h3>
            <div class="task-meta">
              <span>session-scoped benchmark outputs</span>
              <span>{html.escape(experiment['short_label'])}</span>
            </div>
            <div class="muted" style="margin-top:8px;">per-task run directories</div>
            <code>{html.escape(results_root)}</code>
          </div>
          <div class="info">
            <h3>Plots</h3>
            <div class="task-meta">
              <span>{pill(plots['status'], plots['status'])}</span>
              <span>{section_counts_text(plots)}</span>
              <span>elapsed {html.escape(human_duration(timers.get('plots_elapsed_seconds')))}</span>
            </div>
            <div class="bar" style="margin-top:10px;"><span style="width:{plots_progress:.2f}%"></span></div>
            <div class="muted" style="margin-top:8px;">{html.escape(section_focus_text(plots, 'waiting for plot generation'))}</div>
            <div class="muted" style="margin-top:8px;">output root</div>
            <code>{html.escape(plots_output_root)}</code>
            <div class="muted" style="margin-top:8px;">manifest</div>
            <code>{html.escape(plots['manifest_path'])}</code>
          </div>
          <div class="info">
            <h3>Summary</h3>
            <div class="task-meta">
              <span>{pill(summary['status'], summary['status'])}</span>
              <span>{section_counts_text(summary)}</span>
              <span>elapsed {html.escape(human_duration(timers.get('summary_elapsed_seconds')))}</span>
            </div>
            <div class="bar" style="margin-top:10px;"><span style="width:{summary_progress:.2f}%"></span></div>
            <div class="muted" style="margin-top:8px;">{html.escape(section_focus_text(summary, 'waiting for summary generation'))}</div>
            <div class="muted" style="margin-top:8px;">output root</div>
            <code>{html.escape(summary_output_root)}</code>
            <div class="muted" style="margin-top:8px;">manifest</div>
            <code>{html.escape(summary['manifest_path'])}</code>
          </div>
          <div class="info">
            <h3>Comparison</h3>
            <div class="task-meta">
              <span>{pill(comparison['status'], comparison['status'])}</span>
              <span>{section_counts_text(comparison)}</span>
              <span>elapsed {html.escape(human_duration(timers.get('comparison_elapsed_seconds')))}</span>
            </div>
            <div class="bar" style="margin-top:10px;"><span style="width:{comparison_progress:.2f}%"></span></div>
            <div class="muted" style="margin-top:8px;">{html.escape(section_focus_text(comparison, 'waiting for comparison generation'))}</div>
            <div class="muted" style="margin-top:8px;">output root</div>
            <code>{html.escape(comparison_output_root)}</code>
            <div class="muted" style="margin-top:8px;">manifest</div>
            <code>{html.escape(comparison['manifest_path'])}</code>
          </div>
        </div>
      </section>
    </div>

    <section class="card" id="tasks" style="margin-bottom: 18px;">
      <h2>Tasks And Benchmark Status</h2>
      <div class="task-meta">
        <span>task breakdown updates automatically</span>
        <span>summary and comparison columns are per task-mode</span>
        <span>restart means a checkpoint exists and the run can resume cleanly</span>
      </div>
      <div class="filter-row" style="margin-top: 14px;">
        <input id="task-search" type="search" placeholder="filter task name">
        <select id="task-filter">
          <option value="">all tasks</option>
          {"".join(f"<option value='{html.escape(task)}'>{html.escape(task)}</option>" for task in state['tasks'])}
        </select>
        <select id="status-filter">
          <option value="">all benchmark states</option>
          <option value="done">done</option>
          <option value="active">active</option>
          <option value="pending">pending</option>
          <option value="needs_restart">restartable</option>
          <option value="error">error</option>
        </select>
        <select id="mode-filter">
          <option value="">all modes</option>
          {"".join(f"<option value='{html.escape(mode)}'>{html.escape(mode)}</option>" for mode in state['modes'])}
        </select>
        <select id="summary-filter">
          <option value="">all summary states</option>
          <option value="done">done</option>
          <option value="partial">partial</option>
          <option value="ready">ready</option>
          <option value="blocked">blocked</option>
          <option value="pending">pending</option>
          <option value="error">error</option>
        </select>
        <select id="comparison-filter">
          <option value="">all comparison states</option>
          <option value="done">done</option>
          <option value="partial">partial</option>
          <option value="ready">ready</option>
          <option value="blocked">blocked</option>
          <option value="pending">pending</option>
          <option value="error">error</option>
        </select>
      </div>
      <div class="task-list" style="margin-top: 14px;">
        {''.join(task_cards)}
      </div>
    </section>

    <div class="grid">
      <section class="card">
        <h2>Errors</h2>
        {error_html}
      </section>
      <section class="card">
        <h2>Plan Changes</h2>
        <ul>{plan_changes_html}</ul>
      </section>
      <section class="card" id="events">
        <h2>Recent Events</h2>
        <ul>{event_html}</ul>
      </section>
    </div>
  </div>
  <script>
    const REFRESH_SECONDS = {AUTO_REFRESH_SECONDS};
    const STATUS_ENDPOINT = document.body.dataset.statusEndpoint || "after_radcom_status.json";
    const CONNECTION_TIMEOUT_MS = 4000;
    const CONNECTION_STORAGE_KEY = "wavesfm_dashboard_connection_v1";
    const CONNECTION_POLL_MS = Math.max(3000, Math.min(REFRESH_SECONDS * 1000, 10000));
    const CONNECTION_BACKOFF_MIN_MS = 60 * 1000;
    const CONNECTION_BACKOFF_LONG_MS = 15 * 60 * 1000;
    const CONNECTION_BACKOFF_MAX_MS = 20 * 60 * 1000;
    const SCROLL_STORAGE_KEY = `wavesfm_dashboard_scroll_v1:${{window.location.pathname}}`;
    let autoRefresh = true;
    const pageLoadedAt = Date.now();
    const initialTrackerUpdatedAtMs = Date.parse(document.body.dataset.updatedAt || "");
    let probeInFlight = false;
    let nextProbeAtMs = Date.now() + CONNECTION_POLL_MS;
    let probeTimerId = null;
    let scrollSaveTimerId = null;

    function humanAge(totalSeconds) {{
      const seconds = Math.max(0, Math.floor(totalSeconds));
      const hours = Math.floor(seconds / 3600);
      const minutes = Math.floor((seconds % 3600) / 60);
      const secs = seconds % 60;
      if (hours > 0) return `${{hours}}h ${{minutes}}m`;
      if (minutes > 0) return `${{minutes}}m ${{secs}}s`;
      return `${{secs}}s`;
    }}

    function saveScrollPosition() {{
      try {{
        window.localStorage.setItem(SCROLL_STORAGE_KEY, JSON.stringify({{
          x: window.scrollX || 0,
          y: window.scrollY || 0,
          ts: Date.now(),
        }}));
      }} catch (error) {{
        // Browser privacy settings can block localStorage; refresh should still work.
      }}
    }}

    function scheduleScrollSave() {{
      if (scrollSaveTimerId !== null) {{
        return;
      }}
      scrollSaveTimerId = window.setTimeout(() => {{
        scrollSaveTimerId = null;
        saveScrollPosition();
      }}, 120);
    }}

    function applySavedScrollPosition(targetX, targetY) {{
      const maxY = Math.max(0, document.documentElement.scrollHeight - window.innerHeight);
      const clampedY = Math.min(targetY, maxY);
      window.scrollTo({{ left: targetX, top: clampedY, behavior: "auto" }});
      return Math.abs((window.scrollY || 0) - clampedY) <= 4;
    }}

    function queueScrollRestore(targetX, targetY, attempt = 0) {{
      const delayMs = attempt === 0 ? 0 : Math.min(1200, 80 * attempt);
      window.setTimeout(() => {{
        window.requestAnimationFrame(() => {{
          window.requestAnimationFrame(() => {{
            const settled = applySavedScrollPosition(targetX, targetY);
            if (!settled && attempt < 8) {{
              queueScrollRestore(targetX, targetY, attempt + 1);
            }}
          }});
        }});
      }}, delayMs);
    }}

    function restoreScrollPosition() {{
      try {{
        if ("scrollRestoration" in window.history) {{
          window.history.scrollRestoration = "manual";
        }}
        const raw = window.localStorage.getItem(SCROLL_STORAGE_KEY);
        if (!raw) return;
        const saved = JSON.parse(raw);
        const ageMs = Date.now() - Number(saved.ts || 0);
        if (!Number.isFinite(ageMs) || ageMs > 60 * 60 * 1000) return;
        const targetX = Math.max(0, Number(saved.x || 0));
        const targetY = Math.max(0, Number(saved.y || 0));
        queueScrollRestore(targetX, targetY);
      }} catch (error) {{
        // Invalid saved state should never block dashboard rendering.
      }}
    }}

    function defaultConnectionState() {{
      return {{
        status: "connected",
        connectedSinceMs: Date.now(),
        lastSuccessMs: Date.now(),
        lastFailureMs: null,
        lastFailureReason: "",
        lastFailureClass: "",
        reconnectAttempts: 0,
        consecutiveFailures: 0,
        sameFailureStreak: 0,
        cooldownUntilMs: null,
        lastResolutionHint: "",
        lastRecoveredMs: null,
        trackerUpdatedAtMs: Number.isFinite(initialTrackerUpdatedAtMs) ? initialTrackerUpdatedAtMs : null,
        trackerUpdatedAtText: document.body.dataset.updatedAt || "n/a",
        lastKnownPhase: {json.dumps(phase_display)},
        lastKnownWork: {json.dumps(current.get("label") or "n/a")},
        lastKnownRun: {json.dumps(current_run_label)},
        lastKnownProgress: {json.dumps(current_run_progress)},
        lastKnownValidation: {json.dumps((current_run.get("last_validation") if current_run else None) or "n/a")},
      }};
    }}

    function loadConnectionState() {{
      const base = defaultConnectionState();
      try {{
        const raw = window.sessionStorage.getItem(CONNECTION_STORAGE_KEY);
        if (!raw) return base;
        const parsed = JSON.parse(raw);
        const merged = {{ ...base, ...(parsed || {{}}) }};
        if (merged.status && merged.status !== "connected") {{
          merged.lastRecoveredMs = Date.now();
          merged.connectedSinceMs = Date.now();
        }}
        merged.status = "connected";
        merged.lastSuccessMs = Date.now();
        merged.reconnectAttempts = 0;
        merged.consecutiveFailures = 0;
        merged.sameFailureStreak = 0;
        merged.cooldownUntilMs = null;
        merged.lastResolutionHint = "";
        merged.trackerUpdatedAtText = document.body.dataset.updatedAt || merged.trackerUpdatedAtText || "n/a";
        if (!Number.isFinite(merged.trackerUpdatedAtMs) && Number.isFinite(initialTrackerUpdatedAtMs)) {{
          merged.trackerUpdatedAtMs = initialTrackerUpdatedAtMs;
        }}
        return merged;
      }} catch (_err) {{
        return base;
      }}
    }}

    const connectionState = loadConnectionState();

    function saveConnectionState() {{
      try {{
        window.sessionStorage.setItem(CONNECTION_STORAGE_KEY, JSON.stringify(connectionState));
      }} catch (_err) {{
      }}
    }}

    function formatConnectionError(error) {{
      if (typeof navigator !== "undefined" && navigator.onLine === false) return "browser reports offline";
      if (!error) return "unknown fetch failure";
      if (error.name === "AbortError") return "status check timed out";
      if (typeof error.message === "string" && error.message.trim()) return error.message.trim();
      return String(error);
    }}

    function classifyConnectionFailure(reason) {{
      const text = String(reason || "").toLowerCase();
      if (!text) return "unknown";
      if (text.includes("offline")) return "offline";
      if (text.includes("timed out")) return "timeout";
      if (text.includes("http 404")) return "missing";
      if (text.includes("http 5")) return "server";
      if (text.includes("failed to fetch") || text.includes("networkerror")) return "network";
      return "other";
    }}

    function connectionResolutionHint(reasonClass) {{
      if (reasonClass === "offline") return "Local browser/network looks offline. The page will retry and also probe again on the browser online event.";
      if (reasonClass === "missing") return "Status JSON is missing. Check whether the remote dashboard server is serving the automation log directory.";
      if (reasonClass === "server") return "The remote dashboard server returned an error. Restart the remote `python3 -m http.server 8765` process if needed.";
      if (reasonClass === "timeout") return "The dashboard path is timing out. This often means the tunnel or remote server is present but unhealthy.";
      if (reasonClass === "network") return "The browser cannot reach the remote dashboard path. Keep the resilient SSH tunnel or autossh tunnel running on the Mac.";
      return "Repeated browser fetch failures usually mean the tunnel or the remote dashboard server needs attention.";
    }}

    function nextFailureDelayMs(reasonClass, consecutiveFailures, sameFailureStreak) {{
      if (sameFailureStreak >= 8 || consecutiveFailures >= 10) return CONNECTION_BACKOFF_MAX_MS;
      if (sameFailureStreak >= 5 || consecutiveFailures >= 6) return CONNECTION_BACKOFF_LONG_MS;
      if (reasonClass === "offline") return CONNECTION_BACKOFF_MIN_MS;
      if (consecutiveFailures >= 3) return CONNECTION_BACKOFF_MIN_MS;
      return Math.min(CONNECTION_POLL_MS * 2, 10000);
    }}

    function trackerHeartbeat(nowMs) {{
      if (!Number.isFinite(connectionState.trackerUpdatedAtMs)) {{
        return {{ label: "unknown", status: "pending", ageSeconds: null }};
      }}
      const ageSeconds = Math.max(0, (nowMs - connectionState.trackerUpdatedAtMs) / 1000);
      if (ageSeconds <= Math.max(REFRESH_SECONDS * 2, 15)) {{
        return {{ label: "live", status: "done", ageSeconds }};
      }}
      if (ageSeconds <= Math.max(REFRESH_SECONDS * 6, 45)) {{
        return {{ label: "stale", status: "partial", ageSeconds }};
      }}
      return {{ label: "very stale", status: "error", ageSeconds }};
    }}

    function updateConnectionHealth(nowMs = Date.now()) {{
      const card = document.getElementById("connection-health");
      const connectionStatusEl = document.getElementById("connection-status");
      const toolbarStatusEl = document.getElementById("toolbar-connection-status");
      const retriesEl = document.getElementById("connection-retries");
      const connectedSinceEl = document.getElementById("connection-since");
      const lastSuccessEl = document.getElementById("connection-last-success");
      const lossSinceEl = document.getElementById("connection-loss-since");
      const lossReasonEl = document.getElementById("connection-loss-reason");
      const recoveryStateEl = document.getElementById("connection-recovery-state");
      const recoveryNoteEl = document.getElementById("connection-recovery-note");
      const nextRetryEl = document.getElementById("connection-next-retry");
      const sameErrorEl = document.getElementById("connection-same-error");
      const trackerHeartbeatEl = document.getElementById("tracker-heartbeat");
      const trackerLastUpdateWallEl = document.getElementById("tracker-last-update-wall");
      const trackerLastUpdateAgeEl = document.getElementById("tracker-last-update-age");
      const lastKnownPhaseEl = document.getElementById("connection-last-known-phase");
      const lastKnownWorkEl = document.getElementById("connection-last-known-work");
      const lastKnownRunEl = document.getElementById("connection-last-known-run");
      const lastKnownProgressEl = document.getElementById("connection-last-known-progress");
      const lastKnownValidationEl = document.getElementById("connection-last-known-validation");
      const heartbeat = trackerHeartbeat(nowMs);
      const isConnected = connectionState.status === "connected";
      const statusLabel = isConnected
        ? (heartbeat.label === "live" ? "connected" : `connected (${{heartbeat.label}})`)
        : connectionState.status;

      if (card) {{
        card.classList.remove("connected", "reconnecting", "lost", "stale");
        card.classList.add(isConnected ? (heartbeat.label === "live" ? "connected" : "stale") : connectionState.status);
      }}
      if (connectionStatusEl) connectionStatusEl.textContent = statusLabel;
      if (toolbarStatusEl) toolbarStatusEl.textContent = statusLabel;
      if (retriesEl) retriesEl.textContent = String(connectionState.reconnectAttempts || 0);
      if (connectedSinceEl) {{
        connectedSinceEl.textContent = isConnected
          ? humanAge((nowMs - Number(connectionState.connectedSinceMs || nowMs)) / 1000)
          : "awaiting recovery";
      }}
      if (lastSuccessEl) {{
        if (Number.isFinite(connectionState.lastSuccessMs)) {{
          lastSuccessEl.textContent = `last successful check ${{humanAge((nowMs - connectionState.lastSuccessMs) / 1000)}} ago`;
        }} else {{
          lastSuccessEl.textContent = "last successful check pending";
        }}
      }}
      if (lossSinceEl) {{
        if (!Number.isFinite(connectionState.lastFailureMs)) {{
          lossSinceEl.textContent = "none recorded";
        }} else if (isConnected && Number.isFinite(connectionState.lastRecoveredMs) && connectionState.lastRecoveredMs >= connectionState.lastFailureMs) {{
          lossSinceEl.textContent = `recovered ${{humanAge((nowMs - connectionState.lastRecoveredMs) / 1000)}} ago`;
        }} else {{
          lossSinceEl.textContent = humanAge((nowMs - connectionState.lastFailureMs) / 1000);
        }}
      }}
      if (lossReasonEl) {{
        if (!Number.isFinite(connectionState.lastFailureMs)) {{
          lossReasonEl.textContent = "No tunnel or dashboard failure recorded in this browser session.";
        }} else if (isConnected && Number.isFinite(connectionState.lastRecoveredMs) && connectionState.lastRecoveredMs >= connectionState.lastFailureMs) {{
          const outageSeconds = Math.max(0, (connectionState.lastRecoveredMs - connectionState.lastFailureMs) / 1000);
          lossReasonEl.textContent = `Last outage lasted ${{humanAge(outageSeconds)}}. Last recorded reason: ${{connectionState.lastFailureReason || "unknown"}}.`;
        }} else {{
          lossReasonEl.textContent = connectionState.lastFailureReason || "status endpoint unreachable";
        }}
      }}
      if (recoveryStateEl) {{
        recoveryStateEl.textContent = isConnected
          ? (Number.isFinite(connectionState.lastRecoveredMs) ? "recovered" : "stable")
          : (Number.isFinite(connectionState.cooldownUntilMs) && connectionState.cooldownUntilMs > nowMs ? "cooldown" : "retrying");
      }}
      if (recoveryNoteEl) {{
        recoveryNoteEl.textContent = isConnected
          ? "Safe checks keep probing the remote status JSON. A fresh snapshot will trigger a page reload when auto refresh is enabled."
          : `${{connectionState.lastResolutionHint || "Automatic retries are active."}} If this stays red, rerun the resilient Mac tunnel command or restart the remote dashboard server command shown above.`;
      }}
      if (nextRetryEl) {{
        if (isConnected) {{
          nextRetryEl.textContent = "connected";
        }} else if (Number.isFinite(connectionState.cooldownUntilMs) && connectionState.cooldownUntilMs > nowMs) {{
          nextRetryEl.textContent = `retry in ${{humanAge((connectionState.cooldownUntilMs - nowMs) / 1000)}}`;
        }} else {{
          nextRetryEl.textContent = "retry pending";
        }}
      }}
      if (sameErrorEl) {{
        sameErrorEl.textContent = `same-error streak ${{connectionState.sameFailureStreak || 0}} | consecutive failures ${{connectionState.consecutiveFailures || 0}}`;
      }}
      if (trackerHeartbeatEl) trackerHeartbeatEl.textContent = heartbeat.label;
      if (trackerLastUpdateWallEl) trackerLastUpdateWallEl.textContent = connectionState.trackerUpdatedAtText || "n/a";
      if (trackerLastUpdateAgeEl) {{
        trackerLastUpdateAgeEl.textContent = heartbeat.ageSeconds == null
          ? "tracker age unavailable"
          : `age ${{humanAge(heartbeat.ageSeconds)}}`;
      }}
      if (lastKnownPhaseEl) lastKnownPhaseEl.textContent = connectionState.lastKnownPhase || "unknown";
      if (lastKnownWorkEl) lastKnownWorkEl.textContent = connectionState.lastKnownWork || "n/a";
      if (lastKnownRunEl) lastKnownRunEl.textContent = connectionState.lastKnownRun || "idle";
      if (lastKnownProgressEl) lastKnownProgressEl.textContent = connectionState.lastKnownProgress || "n/a";
      if (lastKnownValidationEl) lastKnownValidationEl.textContent = connectionState.lastKnownValidation || "n/a";
      saveConnectionState();
    }}

    function scheduleProbe(delayMs = CONNECTION_POLL_MS) {{
      if (probeTimerId !== null) {{
        window.clearTimeout(probeTimerId);
      }}
      nextProbeAtMs = Date.now() + delayMs;
      probeTimerId = window.setTimeout(() => {{
        probeConnection(false);
      }}, delayMs);
    }}

    async function fetchStatusSnapshot() {{
      const controller = new AbortController();
      const timeoutId = window.setTimeout(() => controller.abort(), CONNECTION_TIMEOUT_MS);
      try {{
        const response = await fetch(`${{STATUS_ENDPOINT}}?_=${{Date.now()}}`, {{
          cache: "no-store",
          signal: controller.signal,
          headers: {{ "Accept": "application/json" }},
        }});
        if (!response.ok) {{
          throw new Error(`status fetch returned HTTP ${{response.status}}`);
        }}
        return await response.json();
      }} finally {{
        window.clearTimeout(timeoutId);
      }}
    }}

    function forceHardReload() {{
      saveScrollPosition();
      const url = new URL(window.location.href);
      url.searchParams.set("_dashboard_reload", String(Date.now()));
      window.location.replace(url.toString());
    }}

    function captureLastKnownState(payload) {{
      const current = payload?.current || {{}};
      const currentRun = payload?.benchmarks?.current_run || null;
      connectionState.lastKnownPhase = payload?.phase || connectionState.lastKnownPhase || "unknown";
      connectionState.lastKnownWork = current?.label || connectionState.lastKnownWork || "n/a";
      if (currentRun) {{
        connectionState.lastKnownRun = `${{currentRun.task}} / ${{currentRun.mode}} / s${{currentRun.seed}}`;
        connectionState.lastKnownProgress = current?.progress_label || connectionState.lastKnownProgress || "n/a";
        connectionState.lastKnownValidation = currentRun?.last_validation || connectionState.lastKnownValidation || "n/a";
      }} else {{
        connectionState.lastKnownRun = "idle";
        connectionState.lastKnownProgress = current?.progress_label || "idle";
        connectionState.lastKnownValidation = "n/a";
      }}
    }}

    function handleProbeSuccess(payload, {{ forceReload = false }} = {{}}) {{
      const nowMs = Date.now();
      const recovered = connectionState.status !== "connected";
      const updatedAtText = payload?.updated_at_utc || connectionState.trackerUpdatedAtText || "n/a";
      const updatedAtMs = Date.parse(updatedAtText);
      connectionState.status = "connected";
      connectionState.lastSuccessMs = nowMs;
      connectionState.reconnectAttempts = 0;
      connectionState.consecutiveFailures = 0;
      connectionState.sameFailureStreak = 0;
      connectionState.cooldownUntilMs = null;
      connectionState.lastResolutionHint = "";
      connectionState.trackerUpdatedAtText = updatedAtText;
      if (Number.isFinite(updatedAtMs)) {{
        connectionState.trackerUpdatedAtMs = updatedAtMs;
      }}
      captureLastKnownState(payload);
      if (recovered) {{
        connectionState.connectedSinceMs = nowMs;
        connectionState.lastRecoveredMs = nowMs;
      }}
      updateConnectionHealth(nowMs);
      const pageUpdatedAt = Date.parse(document.body.dataset.updatedAt || "");
      const hasNewTrackerSnapshot = Number.isFinite(updatedAtMs) && (!Number.isFinite(pageUpdatedAt) || updatedAtMs > pageUpdatedAt);
      if (forceReload || (autoRefresh && (recovered || hasNewTrackerSnapshot))) {{
        forceHardReload();
        return;
      }}
      scheduleProbe(CONNECTION_POLL_MS);
    }}

    function handleProbeFailure(error) {{
      const nowMs = Date.now();
      const reason = formatConnectionError(error);
      const reasonClass = classifyConnectionFailure(reason);
      const previousReason = connectionState.lastFailureReason || "";
      const previousClass = connectionState.lastFailureClass || "";
      if (!Number.isFinite(connectionState.lastFailureMs) || connectionState.status === "connected") {{
        connectionState.lastFailureMs = nowMs;
      }}
      connectionState.lastFailureReason = reason;
      connectionState.lastFailureClass = reasonClass;
      connectionState.reconnectAttempts = Number(connectionState.reconnectAttempts || 0) + 1;
      connectionState.consecutiveFailures = Number(connectionState.consecutiveFailures || 0) + 1;
      if (previousReason === reason && previousClass === reasonClass) {{
        connectionState.sameFailureStreak = Number(connectionState.sameFailureStreak || 0) + 1;
      }} else {{
        connectionState.sameFailureStreak = 1;
      }}
      const delayMs = nextFailureDelayMs(
        reasonClass,
        Number(connectionState.consecutiveFailures || 0),
        Number(connectionState.sameFailureStreak || 0),
      );
      connectionState.cooldownUntilMs = nowMs + delayMs;
      connectionState.lastResolutionHint = connectionResolutionHint(reasonClass);
      connectionState.status = connectionState.reconnectAttempts >= 2 ? "lost" : "reconnecting";
      updateConnectionHealth(nowMs);
      scheduleProbe(delayMs);
    }}

    async function probeConnection(forceReload = false) {{
      if (probeInFlight) return;
      probeInFlight = true;
      try {{
        const payload = await fetchStatusSnapshot();
        handleProbeSuccess(payload, {{ forceReload }});
      }} catch (error) {{
        handleProbeFailure(error);
      }} finally {{
        probeInFlight = false;
      }}
    }}

    function applyTaskFilters() {{
      const taskValue = (document.getElementById("task-filter")?.value || "").toLowerCase();
      const statusValue = document.getElementById("status-filter")?.value || "";
      const modeValue = document.getElementById("mode-filter")?.value || "";
      const summaryValue = document.getElementById("summary-filter")?.value || "";
      const comparisonValue = document.getElementById("comparison-filter")?.value || "";
      const searchValue = (document.getElementById("task-search")?.value || "").trim().toLowerCase();

      document.querySelectorAll(".task-card").forEach((card) => {{
        const task = (card.dataset.task || "").toLowerCase();
        const status = card.dataset.status || "";
        const taskMatch = !taskValue || task === taskValue;
        const searchMatch = !searchValue || task.includes(searchValue) || card.textContent.toLowerCase().includes(searchValue);
        const statusMatch = !statusValue || status === statusValue;

        let visibleModeRows = 0;
        card.querySelectorAll(".mode-row").forEach((row) => {{
          const modeMatch = !modeValue || row.dataset.mode === modeValue;
          const summaryMatch = !summaryValue || row.dataset.summaryStatus === summaryValue;
          const comparisonMatch = !comparisonValue || row.dataset.compareStatus === comparisonValue;
          const visible = modeMatch && summaryMatch && comparisonMatch;
          row.classList.toggle("hidden-by-filter", !visible);
          if (visible) visibleModeRows += 1;
        }});

        const visible = taskMatch && searchMatch && statusMatch && visibleModeRows > 0;
        card.classList.toggle("hidden-by-filter", !visible);
      }});
    }}

    function tick() {{
      const now = Date.now();
      const pageAge = (now - pageLoadedAt) / 1000;
      const updatedAt = Number.isFinite(connectionState.trackerUpdatedAtMs) ? connectionState.trackerUpdatedAtMs : new Date(document.body.dataset.updatedAt).getTime();
      const sinceUpdate = Number.isFinite(updatedAt) ? (now - updatedAt) / 1000 : 0;
      const countdown = Math.max(0, (nextProbeAtMs - now) / 1000);
      const pageAgeEl = document.getElementById("page-age");
      const sinceUpdateEl = document.getElementById("since-update");
      const countdownEl = document.getElementById("refresh-countdown");
      if (pageAgeEl) pageAgeEl.textContent = humanAge(pageAge);
      if (sinceUpdateEl) sinceUpdateEl.textContent = humanAge(sinceUpdate);
      if (countdownEl) countdownEl.textContent = autoRefresh ? humanAge(countdown) : "paused";
      updateConnectionHealth(now);
    }}

    function manualRefresh() {{
      probeConnection(true);
    }}

    function toggleAutoRefresh() {{
      autoRefresh = !autoRefresh;
      const button = document.getElementById("auto-refresh-button");
      if (button) {{
        button.textContent = autoRefresh ? "Pause Auto Refresh" : "Resume Auto Refresh";
      }}
      tick();
    }}

    function jumpToId(id) {{
      const el = document.getElementById(id);
      if (el) {{
        el.scrollIntoView({{behavior: "smooth", block: "start"}});
      }}
    }}

    document.querySelectorAll("#task-filter, #status-filter, #mode-filter, #summary-filter, #comparison-filter, #task-search").forEach((el) => {{
      el.addEventListener("input", applyTaskFilters);
      el.addEventListener("change", applyTaskFilters);
    }});

    window.addEventListener("scroll", scheduleScrollSave, {{ passive: true }});
    window.addEventListener("online", () => probeConnection(false));
    window.addEventListener("beforeunload", saveScrollPosition);
    window.addEventListener("pagehide", saveScrollPosition);
    window.addEventListener("load", restoreScrollPosition);
    window.addEventListener("pageshow", restoreScrollPosition);
    window.addEventListener("offline", () => {{
      const nowMs = Date.now();
      const previousReason = connectionState.lastFailureReason || "";
      const previousClass = connectionState.lastFailureClass || "";
      if (!Number.isFinite(connectionState.lastFailureMs) || connectionState.status === "connected") {{
        connectionState.lastFailureMs = nowMs;
      }}
      connectionState.lastFailureReason = "browser reports offline";
      connectionState.lastFailureClass = "offline";
      connectionState.consecutiveFailures = Number(connectionState.consecutiveFailures || 0) + 1;
      connectionState.sameFailureStreak =
        previousReason === "browser reports offline" && previousClass === "offline"
          ? Number(connectionState.sameFailureStreak || 0) + 1
          : 1;
      connectionState.cooldownUntilMs = nowMs + CONNECTION_BACKOFF_MIN_MS;
      connectionState.lastResolutionHint = connectionResolutionHint("offline");
      connectionState.status = "reconnecting";
      updateConnectionHealth(nowMs);
      scheduleProbe(CONNECTION_BACKOFF_MIN_MS);
    }});
    window.addEventListener("focus", () => probeConnection(false));
    document.addEventListener("visibilitychange", () => {{
      if (!document.hidden) {{
        probeConnection(false);
      }}
    }});

    restoreScrollPosition();
    applyTaskFilters();
    updateConnectionHealth();
    tick();
    scheduleProbe(CONNECTION_POLL_MS);
    window.setInterval(tick, 1000);
    window.setInterval(() => {{
      if (!probeInFlight && Date.now() >= (nextProbeAtMs + 1500)) {{
        probeConnection(false);
      }}
    }}, 1500);
  </script>
</body>
</html>
"""


def find_plan_entry(state: dict, task: str, mode: str, seed: int, *, statuses: tuple[str, ...]) -> dict | None:
    for item in state["benchmarks"]["run_plan"]:
        if item["task"] == task and item["mode"] == mode and item["seed"] == seed and item["status"] in statuses:
            return item
    return None


def mark_run_started(state: dict, task: str, mode: str, seed: int) -> None:
    bench = state["benchmarks"]
    entry = find_plan_entry(state, task, mode, seed, statuses=("pending",))
    if entry is None:
        raise RuntimeError(f"Could not find pending run-plan entry for {task}/{mode}/s{seed}")
    entry["status"] = "running"
    queue_number = next(
        idx + 1
        for idx, item in enumerate(bench["run_plan"])
        if item["task"] == task and item["mode"] == mode and item["seed"] == seed
    )
    resume_epoch = 0
    resume_available = entry.get("resume_available")
    if resume_available:
        restored_epoch = checkpoint_epoch(Path(str(resume_available)))
        if restored_epoch is not None:
            resume_epoch = restored_epoch + 1
    metadata = read_run_metadata(task, mode, seed)
    origin = classify_run_origin(metadata)
    launch_history = list((metadata or {}).get("launch_history") or [])
    bench["current_run"] = {
        "id": entry["id"],
        "task": task,
        "mode": mode,
        "seed": seed,
        "status": "running",
        "started_at_ts": now_ts(),
        "started_at_utc": utc_now(),
        "queue_number": queue_number,
        "epochs": None,
        "accum_steps": None,
        "resume_epoch": resume_epoch,
        "epoch": None,
        "step": None,
        "steps_per_epoch": None,
        "progress_fraction": 0.0,
        "launch_progress_fraction": 0.0,
        "eta_seconds": None,
        "latest_train_loss": None,
        "latest_lr": None,
        "last_validation": None,
        "best_summary": None,
        "launch_kind": "resume" if resume_available else "fresh",
        "resume_source": resume_available,
        "run_origin": origin["label"],
        "launch_host": origin["host"],
        "launch_count": max(len(launch_history), 1 if metadata else 0),
        "initial_launch_host": (metadata or {}).get("initial_launch_host"),
    }
    if resume_available:
        append_event(
            state,
            f"Benchmark resumed: {task} / {mode} / s{seed} from {Path(str(resume_available)).name}",
        )
    else:
        append_event(state, f"Benchmark started fresh: {task} / {mode} / s{seed}")


def finish_current_run(state: dict, status: str) -> None:
    bench = state["benchmarks"]
    current = bench.get("current_run")
    if not current or current.get("status") != "running":
        return
    current["status"] = status
    current["finished_at_utc"] = utc_now()
    current["duration_seconds"] = max(0.0, now_ts() - float(current["started_at_ts"]))
    entry = find_plan_entry(
        state,
        current["task"],
        current["mode"],
        current["seed"],
        statuses=("running",),
    )
    if entry is not None:
        entry["status"] = status
    if status == "completed":
        bench.setdefault("durations", []).append(current["duration_seconds"])
        bench.setdefault("durations_by_task", {}).setdefault(current["task"], []).append(current["duration_seconds"])
        bench.setdefault("durations_by_task_mode", {}).setdefault(
            f"{current['task']}|{current['mode']}", []
        ).append(current["duration_seconds"])
    append_event(state, f"Benchmark {status}: {current['task']} / {current['mode']} / s{current['seed']}")
    bench["current_run"] = None


def parse_radcom_output(message: str, state: dict) -> None:
    if not message:
        return
    state["last_output"] = message
    radcom = state["radcom"]
    m1 = RADCOM_PASS1_RE.search(message)
    m2 = RADCOM_PASS2_RE.search(message)
    if m1:
        _pct, current, total = m1.groups()
        if radcom["pass1_started_at_ts"] is None:
            radcom["pass1_started_at_ts"] = now_ts()
            append_event(state, "Radcom pass 1 started.")
        radcom["status"] = "running"
        radcom["current_pass"] = "Pass 1: read+write"
        radcom["current"] = int(current)
        radcom["total"] = int(total)
        update_radcom_eta(state)
        return
    if m2:
        _pct, current, total = m2.groups()
        if radcom["pass2_started_at_ts"] is None:
            radcom["pass2_started_at_ts"] = now_ts()
            append_event(state, "Radcom pass 2 started.")
        radcom["status"] = "running"
        radcom["current_pass"] = "Pass 2: normalize"
        radcom["current"] = int(current)
        radcom["total"] = int(total)
        update_radcom_eta(state)
        return
    if "Wrote reorganized RadCom cache" in message or ("Wrote" in message and "radcom.h5" in message):
        radcom["status"] = "completed"
        radcom["ready"] = True
        radcom["progress_percent"] = 100.0
        radcom["eta_seconds"] = 0.0
        radcom["completed_at_ts"] = now_ts()
        radcom["completed_at_utc"] = utc_now()
        if radcom.get("pass1_started_at_ts") is not None:
            radcom["duration_seconds"] = max(0.0, now_ts() - float(radcom["pass1_started_at_ts"]))
        append_event(state, "Radcom preprocessing completed.")


def parse_benchmark_output(message: str, state: dict) -> None:
    if not message:
        return
    state["last_output"] = message

    match = RUN_RE.search(message)
    if match:
        task, mode, seed = match.groups()
        existing_done = find_plan_entry(state, task, mode, int(seed), statuses=("completed", "skipped"))
        if existing_done is not None:
            append_event(state, f"Runner revisited completed run: {task} / {mode} / s{seed}")
            return
        finish_current_run(state, "completed")
        mark_run_started(state, task, mode, int(seed))
        return

    if "SKIP (best.pth exists)" in message:
        finish_current_run(state, "skipped")
        return

    current = state["benchmarks"].get("current_run")
    if not current:
        return

    train_config = TRAIN_CONFIG_RE.search(message)
    if train_config:
        epochs, accum_steps = train_config.groups()
        current["epochs"] = int(epochs)
        current["accum_steps"] = int(accum_steps)
        return

    train_step = TRAIN_STEP_RE.search(message)
    if train_step:
        epoch, step, steps_per_epoch, loss_value, lr_value = train_step.groups()
        current["epoch"] = int(float(epoch))
        current["step"] = int(step)
        current["steps_per_epoch"] = int(steps_per_epoch)
        if loss_value is not None:
            current["latest_train_loss"] = float(loss_value)
        if lr_value is not None:
            current["latest_lr"] = float(lr_value)
        if current.get("epochs"):
            total_units = int(current["epochs"]) * int(current["steps_per_epoch"])
            finished_units = (int(current["epoch"]) * int(current["steps_per_epoch"])) + int(current["step"])
            if total_units > 0:
                fraction = finished_units / total_units
                current["progress_fraction"] = min(max(fraction, 0.0), 1.0)
            resume_epoch = int(current.get("resume_epoch") or 0)
            remaining_epochs = max(int(current["epochs"]) - resume_epoch, 1)
            launch_total_units = remaining_epochs * int(current["steps_per_epoch"])
            launch_finished_units = max(
                ((int(current["epoch"]) - resume_epoch) * int(current["steps_per_epoch"])) + int(current["step"]),
                0,
            )
            if launch_total_units > 0:
                launch_fraction = launch_finished_units / launch_total_units
                current["launch_progress_fraction"] = min(max(launch_fraction, 0.0), 1.0)
                min_units_for_eta = max(int(current["steps_per_epoch"]), 10)
                if launch_finished_units >= min_units_for_eta or current["launch_progress_fraction"] >= 0.02:
                    elapsed = max(1.0, now_ts() - float(current["started_at_ts"]))
                    total_est = elapsed / current["launch_progress_fraction"]
                    current["eta_seconds"] = max(total_est - elapsed, 0.0)
                else:
                    current["eta_seconds"] = None
        return

    if message.startswith("[val] "):
        current["last_validation"] = message
        return

    done_match = DONE_RE.search(message)
    if done_match:
        current["best_summary"] = message
        return


def parse_postprocess_output(message: str, state: dict) -> None:
    if not message:
        return
    state["last_output"] = message


def launch_process(cmd: list[str], source: str) -> dict:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    master_fd, slave_fd = pty.openpty()
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
        start_new_session=True,
    )
    os.close(slave_fd)
    append_log(source, "RUN " + " ".join(cmd))
    proc_info = {
        "proc": proc,
        "pgid": proc.pid,
        "master_fd": master_fd,
        "buffer": b"",
        "source": source,
        "cmd": cmd,
        "paused_by_tracker": False,
        "stop_requested_at": None,
        "stop_reason": None,
    }
    register_active_process(proc_info)
    return proc_info


def process_data(proc_info: dict, data: bytes, state: dict, parser) -> None:
    proc_info["buffer"] += data
    while True:
        positions = [idx for idx in (proc_info["buffer"].find(b"\r"), proc_info["buffer"].find(b"\n")) if idx != -1]
        if not positions:
            break
        idx = min(positions)
        chunk = proc_info["buffer"][:idx]
        proc_info["buffer"] = proc_info["buffer"][idx + 1 :]
        message = clean_message(chunk.decode("utf-8", errors="ignore"))
        if message:
            append_log(proc_info["source"], message)
            parser(message, state)


def finalize_process_output(proc_info: dict, state: dict, parser) -> None:
    while True:
        try:
            data = os.read(proc_info["master_fd"], 4096)
        except OSError:
            data = b""
        if not data:
            break
        process_data(proc_info, data, state, parser)
    if proc_info["buffer"]:
        message = clean_message(proc_info["buffer"].decode("utf-8", errors="ignore"))
        proc_info["buffer"] = b""
        if message:
            append_log(proc_info["source"], message)
            parser(message, state)
    try:
        os.close(proc_info["master_fd"])
    except OSError:
        pass
    proc_info["master_fd"] = None
    unregister_active_process(proc_info)


def signal_process_group(proc_info: dict, sig: int) -> bool:
    pgid = proc_info.get("pgid")
    if pgid is None:
        return False
    try:
        os.killpg(int(pgid), sig)
        return True
    except ProcessLookupError:
        return False
    except Exception:
        return False


def register_active_process(proc_info: dict) -> None:
    pgid = proc_info.get("pgid")
    if pgid is None:
        return
    ACTIVE_CHILD_PROCS[int(pgid)] = proc_info


def unregister_active_process(proc_info: dict) -> None:
    pgid = proc_info.get("pgid")
    if pgid is None:
        return
    ACTIVE_CHILD_PROCS.pop(int(pgid), None)


def close_process_fd(proc_info: dict) -> None:
    master_fd = proc_info.get("master_fd")
    if master_fd in (None, -1):
        return
    try:
        os.close(int(master_fd))
    except OSError:
        pass
    proc_info["master_fd"] = None


def terminate_active_children(reason: str, *, grace_seconds: float = 10.0) -> None:
    active = [proc_info for proc_info in ACTIVE_CHILD_PROCS.values()]
    if not active:
        return

    append_log("tracker", f"Stopping {len(active)} active child process group(s): {reason}")
    for proc_info in active:
        proc = proc_info.get("proc")
        if proc is None or proc.poll() is not None:
            close_process_fd(proc_info)
            unregister_active_process(proc_info)
            continue
        append_log(
            "tracker",
            f"SIGTERM child pid={proc.pid} source={proc_info.get('source', 'unknown')}",
        )
        signal_process_group(proc_info, signal.SIGTERM)

    deadline = now_ts() + max(grace_seconds, 0.0)
    while now_ts() < deadline:
        if all(proc_info.get("proc") is None or proc_info["proc"].poll() is not None for proc_info in active):
            break
        time.sleep(0.2)

    for proc_info in active:
        proc = proc_info.get("proc")
        if proc is not None and proc.poll() is None:
            append_log(
                "tracker",
                f"SIGKILL child pid={proc.pid} source={proc_info.get('source', 'unknown')}",
            )
            signal_process_group(proc_info, signal.SIGKILL)

    for proc_info in active:
        close_process_fd(proc_info)
        unregister_active_process(proc_info)


def _handle_tracker_signal(signum: int, _frame) -> None:
    global _TRACKER_TERMINATION_IN_PROGRESS
    signal_name = signal.Signals(signum).name
    if _TRACKER_TERMINATION_IN_PROGRESS:
        raise SystemExit(128 + signum)
    _TRACKER_TERMINATION_IN_PROGRESS = True
    terminate_active_children(f"tracker received {signal_name}")
    raise TrackerTermination(f"Tracker received {signal_name}")


def install_signal_handlers() -> None:
    global _SIGNAL_HANDLERS_INSTALLED
    if _SIGNAL_HANDLERS_INSTALLED:
        return
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handle_tracker_signal)
        except Exception:
            continue
    _SIGNAL_HANDLERS_INSTALLED = True


def request_gpu_guard_stop(state: dict, proc_info: dict, reason: str) -> None:
    guard = state["gpu_guard"]
    if proc_info.get("stop_requested_at") is None:
        proc_info["stop_requested_at"] = now_ts()
        proc_info["stop_reason"] = reason
        guard["status"] = "stopping"
        guard["stop_requested_at_ts"] = proc_info["stop_requested_at"]
        guard["stop_requested_at_utc"] = utc_now()
        guard["stop_reason"] = reason
        state["current"]["progress_label"] = "stopping for gpu safety"
        append_gpu_guard_action(state, reason)
        signal_process_group(proc_info, signal.SIGTERM)


def enforce_gpu_guard(state: dict, bench_proc: dict | None) -> None:
    guard = state.get("gpu_guard")
    if not isinstance(guard, dict):
        return
    if bench_proc is None or state["benchmarks"].get("current_run") is None:
        if not guard.get("paused"):
            guard["status"] = "armed"
        return

    now = now_ts()
    poll_interval = gpu_guard_number(guard, "poll_interval_seconds", 5.0)
    last_check_ts = iso_to_ts(guard.get("last_check_utc"))
    if last_check_ts is not None and (now - float(last_check_ts)) < poll_interval:
        if proc_info_needs_escalation(bench_proc):
            guard["status"] = "stopping"
            maybe_force_kill_after_timeout(state, bench_proc)
        return

    snapshot = system_runtime_snapshot(refresh_after_seconds=0)
    temp_c = snapshot.get("gpu_temp_c")
    guard["last_check_utc"] = utc_now()
    guard["last_temp_c"] = temp_c

    if proc_info_needs_escalation(bench_proc):
        guard["status"] = "stopping"
        maybe_force_kill_after_timeout(state, bench_proc)

    if temp_c is None:
        if bench_proc.get("paused_by_tracker"):
            guard["status"] = "paused"
        elif proc_info_needs_escalation(bench_proc):
            guard["status"] = "stopping"
        else:
            guard["status"] = "pending"
        return

    pause_threshold = gpu_guard_number(guard, "pause_threshold_c", 86.0)
    resume_threshold = gpu_guard_number(guard, "resume_threshold_c", 78.0)
    critical_threshold = gpu_guard_number(guard, "critical_threshold_c", 92.0)
    pause_min_seconds = gpu_guard_number(guard, "pause_min_seconds", 300.0)

    if temp_c >= pause_threshold:
        guard["hot_streak"] = int(guard.get("hot_streak") or 0) + 1
    else:
        guard["hot_streak"] = 0
    if temp_c <= resume_threshold:
        guard["cool_streak"] = int(guard.get("cool_streak") or 0) + 1
    else:
        guard["cool_streak"] = 0
    if temp_c >= critical_threshold:
        guard["critical_streak"] = int(guard.get("critical_streak") or 0) + 1
    else:
        guard["critical_streak"] = 0

    if bench_proc.get("paused_by_tracker"):
        guard["status"] = "paused"
        pause_until_ts = float(guard.get("pause_until_ts") or 0.0)
        if temp_c >= critical_threshold and int(guard.get("critical_streak") or 0) >= 2:
            request_gpu_guard_stop(
                state,
                bench_proc,
                (
                    f"GPU safety stop: temperature stayed critical at {temp_c:.1f}C while training was paused. "
                    f"Training is being stopped to reduce risk to the lab GPU. "
                    f"Reduce thermal load, wait for cooldown, and verify airflow before resuming."
                ),
            )
            return
        if now < pause_until_ts:
            return
        if int(guard.get("cool_streak") or 0) >= 2 and temp_c <= resume_threshold:
            if signal_process_group(bench_proc, signal.SIGCONT):
                bench_proc["paused_by_tracker"] = False
                guard["paused"] = False
                guard["status"] = "armed"
                guard["hot_streak"] = 0
                guard["cool_streak"] = 0
                guard["critical_streak"] = 0
                paused_for = max(0.0, now - float(guard.get("pause_started_at_ts") or now))
                state["current"]["progress_label"] = "gpu guard resumed; waiting for fresh trainer output"
                append_gpu_guard_action(
                    state,
                    (
                        f"GPU safety resume: temperature recovered to {temp_c:.1f}C. "
                        f"Resuming training after a {human_duration(paused_for)} cooling pause."
                    ),
                )
                guard["pause_started_at_ts"] = None
                guard["pause_started_at_utc"] = None
                guard["pause_until_ts"] = None
                guard["pause_reason"] = None
            return
        return

    if temp_c >= critical_threshold and int(guard.get("critical_streak") or 0) >= 2:
        request_gpu_guard_stop(
            state,
            bench_proc,
            (
                f"GPU safety stop: temperature reached {temp_c:.1f}C, above the critical threshold of {critical_threshold:.1f}C. "
                f"Training is being stopped to protect the lab GPU. "
                f"Options: improve cooling/airflow, confirm no other GPU jobs are running, wait for the card to cool, then resume from checkpoint."
            ),
        )
        return

    if temp_c >= pause_threshold and int(guard.get("hot_streak") or 0) >= 3:
        if signal_process_group(bench_proc, signal.SIGSTOP):
            bench_proc["paused_by_tracker"] = True
            guard["paused"] = True
            guard["status"] = "paused"
            guard["pause_started_at_ts"] = now
            guard["pause_started_at_utc"] = utc_now()
            guard["pause_until_ts"] = now + pause_min_seconds
            guard["pause_reason"] = (
                f"GPU temperature reached {temp_c:.1f}C and stayed above the pause threshold of {pause_threshold:.1f}C."
            )
            state["current"]["progress_label"] = f"paused by gpu guard for at least {human_duration(pause_min_seconds)}"
            append_gpu_guard_action(
                state,
                (
                    f"GPU safety pause: temperature reached {temp_c:.1f}C. "
                    f"Pausing training for at least {human_duration(pause_min_seconds)}. "
                    f"Tracker will resume automatically after cooldown once the GPU is below {resume_threshold:.1f}C."
                ),
            )
        return

    guard["status"] = "armed"


def proc_info_needs_escalation(proc_info: dict) -> bool:
    return proc_info.get("stop_requested_at") is not None and proc_info["proc"].poll() is None


def maybe_force_kill_after_timeout(state: dict, proc_info: dict, *, timeout_seconds: float = 15.0) -> None:
    requested_at = proc_info.get("stop_requested_at")
    if requested_at is None or proc_info["proc"].poll() is not None:
        return
    if (now_ts() - float(requested_at)) < timeout_seconds:
        return
    if signal_process_group(proc_info, signal.SIGKILL):
        append_gpu_guard_action(
            state,
            "GPU safety escalation: benchmark queue did not exit after SIGTERM, so the tracker sent SIGKILL.",
        )
    proc_info["stop_requested_at"] = None


def run_parallel_ready_phase(state: dict, python: str, radcom_cache: Path) -> None:
    bench = state["benchmarks"]
    tasks = state["tasks"]
    modes = state["modes"]
    seeds = state["seeds"]
    nonradcom_tasks = [task for task in tasks if task != "radcom"]
    wants_radcom = "radcom" in tasks

    state["state"] = "running"
    state["phase"] = PHASE_PARALLEL
    bench["status"] = "running"
    bench["stage"] = "nonradcom"
    if bench.get("started_at_ts") is None:
        bench["started_at_ts"] = now_ts()
        bench["started_at_utc"] = utc_now()

    radcom_proc = None
    ready, reason = radcom_ready(radcom_cache)
    if wants_radcom and not ready:
        state["radcom"]["status"] = "running"
        cmd = [python, str(PHASE2_ROOT / "scripts" / "preprocess_all_tasks.py"), "--cache-ids", "radcom"]
        state["radcom"]["process_command"] = " ".join(cmd)
        append_event(state, f"Radcom cache not ready: {reason}")
        radcom_proc = launch_process(cmd, "radcom")
    elif wants_radcom:
        state["radcom"]["status"] = "completed"
        state["radcom"]["ready"] = True
        state["radcom"]["progress_percent"] = 100.0
        state["radcom"]["eta_seconds"] = 0.0
        state["radcom"]["completed_at_ts"] = now_ts()
        state["radcom"]["completed_at_utc"] = utc_now()
        state["radcom"]["duration_seconds"] = 0.0
        append_event(state, "Radcom cache already ready.")

    bench_proc = None
    if nonradcom_tasks:
        cmd = [
            python,
            str(PHASE2_ROOT / "scripts" / "run_all_tasks.py"),
            "--session-root",
            str(SESSION_ROOT),
            "--output-root",
            str(RESULTS_ROOT),
            "--tasks",
            *nonradcom_tasks,
            "--modes",
            *modes,
            "--seeds",
            *[str(seed) for seed in seeds],
            "--num-workers",
            str(int(DEFAULT_NUM_WORKERS if state.get("num_workers") is None else state.get("num_workers"))),
            "--save-every",
            str(int(state.get("save_every") or DEFAULT_SAVE_EVERY)),
            "--skip-existing",
        ]
        if state.get("train_subset_fraction") is not None:
            cmd += ["--train-subset-fraction", str(state["train_subset_fraction"])]
        if state.get("train_subset_size") is not None:
            cmd += ["--train-subset-size", str(state["train_subset_size"])]
        bench["process_command"] = " ".join(cmd)
        append_event(state, "Ready benchmark queue started.")
        bench_proc = launch_process(cmd, "bench-ready")
    else:
        append_event(state, "No ready-task benchmark queue to run before radcom.")

    if bench_proc is not None and radcom_proc is None:
        state["phase"] = PHASE_READY_BENCH

    write_state(state)
    last_write = now_ts()

    while radcom_proc or bench_proc:
        enforce_gpu_guard(state, bench_proc)
        fd_map = {}
        if radcom_proc:
            fd_map[radcom_proc["master_fd"]] = (radcom_proc, parse_radcom_output)
        if bench_proc:
            fd_map[bench_proc["master_fd"]] = (bench_proc, parse_benchmark_output)

        ready_fds: list[int] = []
        if fd_map:
            ready_fds, _, _ = select.select(list(fd_map.keys()), [], [], 1.0)

        for fd in ready_fds:
            proc_info, parser = fd_map[fd]
            try:
                data = os.read(fd, 4096)
            except OSError:
                data = b""
            if data:
                process_data(proc_info, data, state, parser)

        if bench_proc and bench_proc["proc"].poll() is not None:
            finalize_process_output(bench_proc, state, parse_benchmark_output)
            if bench_proc.get("stop_reason"):
                finish_current_run(state, "error")
                raise RuntimeError(str(bench_proc["stop_reason"]))
            if bench_proc["proc"].returncode != 0:
                raise subprocess.CalledProcessError(bench_proc["proc"].returncode, bench_proc["cmd"])
            finish_current_run(state, "completed")
            bench_proc = None
            bench["process_command"] = None
            append_event(state, "Ready benchmark queue completed.")
            if wants_radcom and not state["radcom"]["ready"]:
                state["phase"] = PHASE_WAIT_RADCOM

        if radcom_proc and radcom_proc["proc"].poll() is not None:
            finalize_process_output(radcom_proc, state, parse_radcom_output)
            if radcom_proc["proc"].returncode != 0:
                raise subprocess.CalledProcessError(radcom_proc["proc"].returncode, radcom_proc["cmd"])
            ready, reason = radcom_ready(radcom_cache)
            if not ready:
                raise RuntimeError(f"Radcom cache still not ready after preprocessing: {reason}")
            state["radcom"]["status"] = "completed"
            state["radcom"]["ready"] = True
            state["radcom"]["progress_percent"] = 100.0
            state["radcom"]["eta_seconds"] = 0.0
            state["radcom"]["process_command"] = None
            state["radcom"]["completed_at_ts"] = now_ts()
            state["radcom"]["completed_at_utc"] = utc_now()
            pass1_started = state["radcom"].get("pass1_started_at_ts")
            state["radcom"]["duration_seconds"] = (
                max(0.0, now_ts() - float(pass1_started)) if pass1_started is not None else 0.0
            )
            append_event(state, "Radcom cache verified and ready.")
            radcom_proc = None

        if bench_proc is None and radcom_proc is not None:
            state["phase"] = PHASE_WAIT_RADCOM
        elif bench_proc is not None:
            state["phase"] = PHASE_PARALLEL if radcom_proc is not None else PHASE_READY_BENCH

        if now_ts() - last_write >= 3.0:
            write_state(state)
            last_write = now_ts()

    bench["status"] = "running" if any(item["status"] == "pending" for item in bench["run_plan"]) else "completed"
    if bench["status"] == "completed":
        bench["completed_at_ts"] = now_ts()
        bench["completed_at_utc"] = utc_now()
        if bench.get("started_at_ts") is not None:
            bench["duration_seconds"] = max(0.0, now_ts() - float(bench["started_at_ts"]))
    write_state(state)


def run_radcom_benchmark_phase(state: dict, python: str) -> None:
    pending_radcom = [item for item in state["benchmarks"]["run_plan"] if item["task"] == "radcom" and item["status"] == "pending"]
    if not pending_radcom:
        state["benchmarks"]["status"] = "completed"
        state["benchmarks"]["completed_at_ts"] = now_ts()
        state["benchmarks"]["completed_at_utc"] = utc_now()
        if state["benchmarks"].get("started_at_ts") is not None:
            state["benchmarks"]["duration_seconds"] = max(
                0.0, now_ts() - float(state["benchmarks"]["started_at_ts"])
            )
        return

    state["phase"] = PHASE_RADCOM_BENCH
    state["benchmarks"]["status"] = "running"
    state["benchmarks"]["stage"] = "radcom"
    if state["benchmarks"].get("started_at_ts") is None:
        state["benchmarks"]["started_at_ts"] = now_ts()
        state["benchmarks"]["started_at_utc"] = utc_now()
    cmd = [
        python,
        str(PHASE2_ROOT / "scripts" / "run_all_tasks.py"),
        "--session-root",
        str(SESSION_ROOT),
        "--output-root",
        str(RESULTS_ROOT),
        "--tasks",
        "radcom",
        "--modes",
        *state["modes"],
        "--seeds",
        *[str(seed) for seed in state["seeds"]],
        "--num-workers",
        str(int(DEFAULT_NUM_WORKERS if state.get("num_workers") is None else state.get("num_workers"))),
        "--save-every",
        str(int(state.get("save_every") or DEFAULT_SAVE_EVERY)),
        "--skip-existing",
    ]
    if state.get("train_subset_fraction") is not None:
        cmd += ["--train-subset-fraction", str(state["train_subset_fraction"])]
    if state.get("train_subset_size") is not None:
        cmd += ["--train-subset-size", str(state["train_subset_size"])]
    state["benchmarks"]["process_command"] = " ".join(cmd)
    append_event(state, "Radcom benchmark queue started.")
    proc_info = launch_process(cmd, "bench-radcom")
    write_state(state)

    last_write = now_ts()
    while True:
        enforce_gpu_guard(state, proc_info)
        ready_fds, _, _ = select.select([proc_info["master_fd"]], [], [], 1.0)
        if proc_info["master_fd"] in ready_fds:
            try:
                data = os.read(proc_info["master_fd"], 4096)
            except OSError:
                data = b""
            if data:
                process_data(proc_info, data, state, parse_benchmark_output)

        if proc_info["proc"].poll() is not None:
            finalize_process_output(proc_info, state, parse_benchmark_output)
            if proc_info.get("stop_reason"):
                finish_current_run(state, "error")
                raise RuntimeError(str(proc_info["stop_reason"]))
            if proc_info["proc"].returncode != 0:
                raise subprocess.CalledProcessError(proc_info["proc"].returncode, proc_info["cmd"])
            finish_current_run(state, "completed")
            state["benchmarks"]["process_command"] = None
            state["benchmarks"]["stage"] = "completed"
            state["benchmarks"]["status"] = "completed"
            state["benchmarks"]["completed_at_ts"] = now_ts()
            state["benchmarks"]["completed_at_utc"] = utc_now()
            if state["benchmarks"].get("started_at_ts") is not None:
                state["benchmarks"]["duration_seconds"] = max(
                    0.0, now_ts() - float(state["benchmarks"]["started_at_ts"])
                )
            append_event(state, "Radcom benchmark queue completed.")
            write_state(state)
            return

        if now_ts() - last_write >= 3.0:
            write_state(state)
            last_write = now_ts()


def run_plain(cmd: list[str], state: dict, phase: str, label: str, source: str) -> None:
    section_name = (
        "plots"
        if phase == PHASE_PLOTS
        else "summary"
        if phase == PHASE_SUMMARY
        else "comparison"
        if phase == PHASE_COMPARE
        else None
    )
    state["phase"] = phase
    ensure_current(state, label, " ".join(cmd))
    state["current"]["progress_percent"] = 0.0
    state["current"]["progress_label"] = "running"
    state["current"]["eta_seconds"] = None
    if section_name is not None:
        section = state[section_name]
        section["status"] = "running"
        section["started_at_ts"] = now_ts()
        section["started_at_utc"] = utc_now()
        section["completed_at_ts"] = None
        section["completed_at_utc"] = None
        section["duration_seconds"] = None
        section["completed_units"] = 0
        section["progress_percent"] = 0.0
        section["current_task"] = None
        section["current_mode"] = None
        section["current_seed"] = None
        section["current_item_label"] = None
    append_event(state, f"Started {label.lower()}.")
    write_state(state)
    proc_info = launch_process(cmd, source)
    last_write = now_ts()

    while True:
        ready_fds, _, _ = select.select([proc_info["master_fd"]], [], [], 1.0)
        if proc_info["master_fd"] in ready_fds:
            try:
                data = os.read(proc_info["master_fd"], 4096)
            except OSError:
                data = b""
            if data:
                process_data(proc_info, data, state, parse_postprocess_output)

        if proc_info["proc"].poll() is not None:
            finalize_process_output(proc_info, state, parse_postprocess_output)
            if proc_info["proc"].returncode != 0:
                raise subprocess.CalledProcessError(proc_info["proc"].returncode, proc_info["cmd"])
            break

        if now_ts() - last_write >= 3.0:
            write_state(state)
            last_write = now_ts()

    state["current"]["progress_percent"] = 100.0
    state["current"]["progress_label"] = "completed"
    state["current"]["eta_seconds"] = 0.0
    if section_name is not None:
        section = state[section_name]
        section["status"] = "completed"
        section["completed_at_ts"] = now_ts()
        section["completed_at_utc"] = utc_now()
        section["completed_units"] = max(int(section.get("completed_units") or 0), int(section.get("total_units") or 0))
        section["progress_percent"] = 100.0
        if section.get("started_at_ts") is not None:
            section["duration_seconds"] = max(0.0, now_ts() - float(section["started_at_ts"]))
        section["eta_seconds"] = 0.0
    append_event(state, f"{label} completed.")
    write_state(state)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run ready WavesFM benchmarks in parallel with radcom preprocessing, then append radcom runs."
    )
    p.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    p.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--radcom-cache", type=Path, default=CACHE_ROOT / "radcom.h5")
    p.add_argument("--session-root", type=Path, default=None)
    p.add_argument("--reuse-current-session", action="store_true")
    p.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    p.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY)
    p.add_argument("--train-subset-fraction", type=float, default=None)
    p.add_argument("--train-subset-size", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_subset_fraction is not None and args.train_subset_size is not None:
        raise ValueError("Provide only one of --train-subset-fraction or --train-subset-size.")
    install_signal_handlers()
    python = sys.executable
    requested_tasks = list(dict.fromkeys(str(task) for task in args.tasks))
    runnable_tasks, blocked_tasks = classify_requested_tasks(
        requested_tasks,
        radcom_cache=args.radcom_cache.resolve(),
        allow_build_from_raw_tasks={"radcom"},
    )
    if not runnable_tasks:
        blocked_summary = "; ".join(f"{task}: {reason}" for task, reason in blocked_tasks.items()) or "no tasks selected"
        raise RuntimeError(f"No runnable tasks selected after cache filtering: {blocked_summary}")
    session_root = resolve_session_root(args.session_root, args.reuse_current_session)
    layout = configure_session_paths(session_root)
    for key in ("session_root", "imports_root", "results_root", "summary_root", "plots_root", "detailed_plots_root", "comparison_root"):
        Path(layout[key]).mkdir(parents=True, exist_ok=True)
    update_current_session_pointer(session_root)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    with RUN_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"\n[{utc_now()}] [tracker] SESSION START\n")

    previous_state = safe_read_json(STATUS_PATH)
    state = init_state(
        list(runnable_tasks),
        list(args.modes),
        list(args.seeds),
        args.radcom_cache.resolve(),
        num_workers=args.num_workers,
        save_every=args.save_every,
        train_subset_fraction=args.train_subset_fraction,
        train_subset_size=args.train_subset_size,
    )
    state["requested_tasks"] = requested_tasks
    state["blocked_tasks"] = dict(blocked_tasks)
    tracker_cmd = [
        sys.executable,
        "phase2_vivor4/scripts/wait_for_radcom_and_run_next.py",
        "--session-root",
        str(session_root),
        "--tasks",
        *[str(task) for task in runnable_tasks],
        "--modes",
        *[str(mode) for mode in args.modes],
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--radcom-cache",
        str(args.radcom_cache.resolve()),
        "--num-workers",
        str(args.num_workers),
        "--save-every",
        str(args.save_every),
    ]
    if args.train_subset_fraction is not None:
        tracker_cmd += ["--train-subset-fraction", str(args.train_subset_fraction)]
    if args.train_subset_size is not None:
        tracker_cmd += ["--train-subset-size", str(args.train_subset_size)]
    carry_forward_dashboard_context(state, previous_state)
    seed_benchmark_history_from_disk(state)
    if blocked_tasks:
        append_event(
            state,
            "Skipping blocked tasks: " + ", ".join(f"{task} ({reason})" for task, reason in blocked_tasks.items()),
        )
    append_event(state, "Parallel managed pipeline initialized.")
    write_state(state)
    preflight_report = run_preflight_check(state)
    write_session_manifest(state, tracker_cmd, preflight_report=preflight_report)

    run_parallel_ready_phase(state, python, args.radcom_cache.resolve())
    run_radcom_benchmark_phase(state, python)

    state["benchmarks"]["status"] = "completed"
    state["benchmarks"]["stage"] = "completed"
    write_state(state)

    run_plain(
        [
            python,
            str(PHASE2_ROOT / "scripts" / "plot_local_detailed_eval.py"),
            "--results-root",
            str(RESULTS_ROOT),
            "--output-root",
            str(layout["plots_root"]),
            "--device",
            "cpu",
            "--num-workers",
            "0",
            "--overwrite",
        ],
        state,
        PHASE_PLOTS,
        "Refreshing detailed-evaluation plots",
        "plots",
    )
    run_plain(
        [
            python,
            str(PHASE2_ROOT / "scripts" / "summarize_local_results.py"),
            "--results-root",
            str(RESULTS_ROOT),
            "--summary-root",
            str(layout["summary_root"]),
        ],
        state,
        PHASE_SUMMARY,
        "Summarizing local results",
        "summary",
    )
    run_plain(
        [
            python,
            str(PHASE2_ROOT / "scripts" / "compare_with_official.py"),
            "--local-json",
            str(layout["summary_agg_json"]),
            "--local-runs-json",
            str(layout["summary_runs_json"]),
            "--output-root",
            str(layout["comparison_root"]),
        ],
        state,
        PHASE_COMPARE,
        "Comparing with official results",
        "compare",
    )

    state["state"] = "completed"
    state["phase"] = PHASE_COMPLETED
    state["pipeline_eta_seconds"] = 0.0
    append_event(state, "Parallel managed pipeline completed.")
    write_session_manifest(state, tracker_cmd, preflight_report=preflight_report)
    write_state(state)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        try:
            terminate_active_children(f"tracker exception: {type(exc).__name__}: {exc}")
        except Exception:
            pass
        existing = safe_read_json(STATUS_PATH)
        failed = existing if isinstance(existing, dict) else {
            "started_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
            "state": "error",
            "phase": PHASE_ERROR,
            "parallel_mode": True,
            "pipeline_eta_seconds": None,
            "session_root": str(SESSION_ROOT),
            "session_manifest_path": str(session_layout(SESSION_ROOT)["manifest_path"]),
            "status_path": str(STATUS_PATH),
            "dashboard_path": str(DASHBOARD_PATH),
            "run_log_path": str(RUN_LOG_PATH),
            "last_output": "",
            "last_error": f"{type(exc).__name__}: {exc}",
            "recent_events": [{"time_utc": utc_now(), "message": "Automation failed."}],
            "preflight": {
                "status": "pending",
                "started_at_utc": None,
                "completed_at_utc": None,
                "report_path": str(session_layout(SESSION_ROOT)["preflight_report_path"]),
                "summary": None,
            },
            "plan": {
                "task_order": DEFAULT_TASKS,
                "launch_order": current_launch_order(build_run_plan(DEFAULT_TASKS, DEFAULT_MODES, DEFAULT_SEEDS)),
                "modes": DEFAULT_MODES,
                "seeds": DEFAULT_SEEDS,
                "change_messages": [],
            },
            "current": {
                "label": "Automation failed",
                "command": None,
                "started_at_ts": now_ts(),
                "started_at_utc": utc_now(),
                "progress_percent": 0.0,
                "progress_label": "error",
                "eta_seconds": None,
            },
            "radcom": {
                "cache_path": str(CACHE_ROOT / "radcom.h5"),
                "status": "unknown",
                "ready": False,
                "current_pass": None,
                "current": 0,
                "total": 0,
                "progress_percent": 0.0,
                "eta_seconds": None,
                "pass1_started_at_ts": None,
                "pass2_started_at_ts": None,
                "completed_at_ts": None,
                "completed_at_utc": None,
                "duration_seconds": None,
                "process_command": None,
            },
            "benchmarks": {
                "status": "unknown",
                "stage": "unknown",
                "started_at_ts": None,
                "started_at_utc": None,
                "completed_at_ts": None,
                "completed_at_utc": None,
                "duration_seconds": None,
                "total_runs": 0,
                "nonradcom_total_runs": 0,
                "radcom_total_runs": 0,
                "completed_runs": 0,
                "completed_nonradcom_runs": 0,
                "completed_radcom_runs": 0,
                "expected_skip_runs": 0,
                "avg_run_seconds": None,
                "duration_estimate_seconds": None,
                "benchmark_eta_seconds": None,
                "eta_nonradcom_seconds": None,
                "eta_radcom_runs_seconds": None,
                "progress_percent": 0.0,
                "current_run": None,
                "next_run": None,
                "next_runs": [],
                "process_command": None,
                "run_plan": [],
            },
            "summary": {
                "status": "pending",
                "started_at_ts": None,
                "started_at_utc": None,
                "completed_at_ts": None,
                "completed_at_utc": None,
                "duration_seconds": None,
                "eta_seconds": None,
                "runs_json_path": str(SUMMARY_RUNS_JSON),
                "aggregated_json_path": str(SUMMARY_AGG_JSON),
            },
            "plots": {
                "status": "pending",
                "started_at_ts": None,
                "started_at_utc": None,
                "completed_at_ts": None,
                "completed_at_utc": None,
                "duration_seconds": None,
                "eta_seconds": None,
                "manifest_path": str(PLOT_MANIFEST_PATH),
            },
            "comparison": {
                "status": "pending",
                "started_at_ts": None,
                "started_at_utc": None,
                "completed_at_ts": None,
                "completed_at_utc": None,
                "duration_seconds": None,
                "eta_seconds": None,
                "json_path": str(COMPARE_JSON),
                "official_json_path": str(OFFICIAL_JSON),
            },
            "overview": {},
            "tasks": DEFAULT_TASKS,
            "modes": DEFAULT_MODES,
            "seeds": DEFAULT_SEEDS,
        }
        failed["updated_at_utc"] = utc_now()
        failed["state"] = "error"
        failed["phase"] = PHASE_ERROR
        failed["last_error"] = f"{type(exc).__name__}: {exc}"
        recent_events = list(failed.get("recent_events") or [])
        recent_events.append({"time_utc": utc_now(), "message": "Automation failed."})
        failed["recent_events"] = recent_events[-50:]
        atomic_write(STATUS_PATH, json.dumps(failed, indent=2) + "\n")
        atomic_write(DASHBOARD_PATH, render_dashboard(failed))
        raise
