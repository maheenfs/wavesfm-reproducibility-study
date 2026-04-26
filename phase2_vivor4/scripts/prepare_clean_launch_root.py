from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import shlex
import shutil
from pathlib import Path

from benchmark_config import (
    DEFAULT_SAVE_EVERY,
    PHASE2_ROOT,
    PROJECT_ROOT,
    TASK_SPECS,
    cache_path_for_task,
)


DEFAULT_BLOCKED_TASKS = ("rfs",)
DEFAULT_READY_TASK_ORDER = (
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
    "lwm-beam-challenge",
)
ROOT_FLATTENED_ITEMS = (
    "README.md",
    "__init__.py",
    "__pycache__",
    "automation_logs",
    "caches",
    "cleanup_checkpoints.py",
    "comparisons",
    "data.py",
    "dataset_classes",
    "device_speed_tests",
    "discarded_runs",
    "engine.py",
    "hub.py",
    "local_results",
    "lora.py",
    "main_finetune.py",
    "models_vit.py",
    "official_results",
    "plots",
    "pos_embed.py",
    "preprocessing",
    "requirements.txt",
    "run_all_tasks.py",
    "runs",
    "scripts",
    "utils.py",
    "wavesfm.png",
    "wavesfm_demo.ipynb",
)
PARTIAL_CACHE_GLOBS = ("*_partial_restart_*.h5", "*.tmp", "*.partial")
PHASE2_GENERATED_DIRS = (
    "local_results",
    "discarded_runs",
    "automation_logs",
    "comparisons",
    "runs",
)
PHASE2_GENERATED_PATHS = (
    ("plots", "detailed_eval"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Prepare a clean remote WavesFM launch root by quarantining bad transfer artifacts, "
            "optionally archiving generated phase2 state, and emitting a ready-to-run launch manifest."
        )
    )
    p.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    p.add_argument("--phase2-root", type=Path, default=PHASE2_ROOT)
    p.add_argument("--device", default="cuda", choices=("cpu", "mps", "cuda"))
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--modes", nargs="+", default=["lp", "ft2", "lora"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_READY_TASK_ORDER),
        choices=list(TASK_SPECS.keys()),
        help="Subset of benchmark tasks to include in the prepared session.",
    )
    p.add_argument("--blocked-tasks", nargs="+", default=list(DEFAULT_BLOCKED_TASKS))
    p.add_argument(
        "--session-name",
        default=None,
        help=(
            "Optional explicit session directory name under phase2_vivor4/runs/. "
            "Use this to label a second-pass rerun so it stays distinct in reports."
        ),
    )
    p.add_argument(
        "--archive-phase2-state",
        action="store_true",
        help=(
            "Archive generated phase2 artifacts like local_results, discarded_runs, "
            "automation_logs, comparisons, runs, and plots/detailed_eval before relaunch."
        ),
    )
    return p.parse_args()


def timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def move_if_exists(src: Path, dst_root: Path) -> Path | None:
    if not src.exists():
        return None
    dst_root.mkdir(parents=True, exist_ok=True)
    dst = dst_root / src.name
    idx = 1
    while dst.exists():
        dst = dst_root / f"{src.name}_{idx}"
        idx += 1
    shutil.move(str(src), str(dst))
    return dst


def ready_and_blocked_tasks(
    project_root: Path,
    candidate_tasks: list[str],
    blocked_tasks: set[str],
) -> tuple[list[str], dict[str, str]]:
    blocked: dict[str, str] = {}
    ready: list[str] = []
    for task in candidate_tasks:
        if task in blocked_tasks:
            blocked[task] = "explicitly blocked"
            continue
        cache_path = cache_path_for_task(task, project_root / "datasets_h5")
        if cache_path.exists():
            ready.append(task)
        else:
            blocked[task] = f"missing cache: {cache_path.name}"
    for task in sorted(blocked_tasks):
        if task not in blocked and task in candidate_tasks:
            blocked[task] = "explicitly blocked"
    return ready, blocked


def build_launch_command(args: argparse.Namespace, ready_tasks: list[str]) -> dict[str, object]:
    if args.session_name:
        session_name = args.session_name
    else:
        session_name = f"{timestamp()}-{(platform.node().split('.', 1)[0] or 'host')}"
    session_root = args.phase2_root.resolve() / "runs" / session_name
    idx = 1
    while session_root.exists():
        session_root = args.phase2_root.resolve() / "runs" / f"{session_name}_{idx}"
        idx += 1
    session_name = session_root.name
    radcom_cache = args.project_root.resolve() / "datasets_h5" / "radcom.h5"

    preflight_cmd = [
        "python3",
        "phase2_vivor4/scripts/preflight_check.py",
        "--session-root",
        str(session_root),
        "--tasks",
        *ready_tasks,
        "--modes",
        *args.modes,
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--radcom-cache",
        str(radcom_cache),
    ]

    tracker_cmd = [
        "python3",
        "phase2_vivor4/scripts/wait_for_radcom_and_run_next.py",
        "--session-root",
        str(session_root),
        "--tasks",
        *ready_tasks,
        "--modes",
        *args.modes,
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--radcom-cache",
        str(radcom_cache),
        "--num-workers",
        str(args.num_workers),
        "--save-every",
        str(args.save_every),
    ]
    return {
        "session_name": session_name,
        "session_root": session_root,
        "radcom_cache": radcom_cache,
        "preflight_command": preflight_cmd,
        "tracker_command": tracker_cmd,
    }


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    phase2_root = args.phase2_root.resolve()
    quarantine_root = project_root / "_transfer_quarantine"
    stamp = timestamp()

    moved_root_items: dict[str, str] = {}
    for name in ROOT_FLATTENED_ITEMS:
        src = project_root / name
        moved = move_if_exists(src, quarantine_root / "root_flattened")
        if moved is not None:
            moved_root_items[str(src.relative_to(project_root))] = str(moved.relative_to(project_root))

    moved_partial_caches: dict[str, str] = {}
    cache_root = project_root / "datasets_h5"
    for pattern in PARTIAL_CACHE_GLOBS:
        for path in sorted(cache_root.glob(pattern)):
            moved = move_if_exists(path, quarantine_root / "datasets_h5")
            if moved is not None:
                moved_partial_caches[str(path.relative_to(project_root))] = str(moved.relative_to(project_root))

    archived_phase2: dict[str, str] = {}
    if args.archive_phase2_state:
        archive_root = phase2_root / "remote_archives" / stamp
        for name in PHASE2_GENERATED_DIRS:
            src = phase2_root / name
            moved = move_if_exists(src, archive_root)
            if moved is not None:
                archived_phase2[str(src.relative_to(project_root))] = str(moved.relative_to(project_root))
        for parts in PHASE2_GENERATED_PATHS:
            src = phase2_root.joinpath(*parts)
            moved = move_if_exists(src, archive_root / Path(*parts[:-1]))
            if moved is not None:
                archived_phase2[str(src.relative_to(project_root))] = str(moved.relative_to(project_root))

    (phase2_root / "local_results" / "by_task").mkdir(parents=True, exist_ok=True)
    (phase2_root / "local_results" / "summaries").mkdir(parents=True, exist_ok=True)
    (phase2_root / "discarded_runs" / "by_task").mkdir(parents=True, exist_ok=True)
    (phase2_root / "automation_logs").mkdir(parents=True, exist_ok=True)
    (phase2_root / "comparisons").mkdir(parents=True, exist_ok=True)
    (phase2_root / "plots" / "detailed_eval").mkdir(parents=True, exist_ok=True)
    (phase2_root / "runs").mkdir(parents=True, exist_ok=True)

    ready_tasks, blocked = ready_and_blocked_tasks(project_root, args.tasks, set(args.blocked_tasks))
    launch_plan = build_launch_command(args, ready_tasks)
    session_root = Path(str(launch_plan["session_root"]))
    manifest = {
        "generated_at": dt.datetime.now().astimezone().isoformat(),
        "project_root": str(project_root),
        "phase2_root": str(phase2_root),
        "session_name": str(launch_plan["session_name"]),
        "session_root": str(session_root),
        "device": args.device,
        "num_workers": args.num_workers,
        "save_every": args.save_every,
        "modes": args.modes,
        "seeds": args.seeds,
        "ready_tasks": ready_tasks,
        "blocked_tasks": blocked,
        "moved_root_items": moved_root_items,
        "moved_partial_caches": moved_partial_caches,
        "archived_phase2_state": archived_phase2,
        "launch_env": {"WAVESFM_FORCE_DEVICE": args.device},
        "radcom_cache": str(launch_plan["radcom_cache"]),
        "preflight_command": launch_plan["preflight_command"],
        "tracker_command": launch_plan["tracker_command"],
        "tmux_attach_train": "tmux attach -t train || tmux new -s train",
        "tmux_attach_codex": "tmux attach -t codex || tmux new -s codex",
    }

    manifest_path = phase2_root / "runs" / f"clean_launch_manifest_{stamp}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    shell_path = phase2_root / "runs" / f"launch_ready_non_rfs_{stamp}.sh"
    shell_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f'cd "{project_root}"',
        'source .venv/bin/activate',
        f'export WAVESFM_FORCE_DEVICE="{args.device}"',
        f'echo "Session root: {session_root}"',
        shlex.join(launch_plan["preflight_command"]),
        shlex.join(launch_plan["tracker_command"]),
        "",
    ]
    shell_path.write_text("\n".join(shell_lines), encoding="utf-8")
    shell_path.chmod(0o755)

    print(f"Manifest written: {manifest_path}")
    print(f"Launch script written: {shell_path}")
    if moved_root_items:
        print("Moved flattened root items:")
        for src, dst in moved_root_items.items():
            print(f"  {src} -> {dst}")
    if moved_partial_caches:
        print("Moved partial cache artifacts:")
        for src, dst in moved_partial_caches.items():
            print(f"  {src} -> {dst}")
    if archived_phase2:
        print("Archived generated phase2 state:")
        for src, dst in archived_phase2.items():
            print(f"  {src} -> {dst}")
    print("Ready tasks:")
    for task in ready_tasks:
        print(f"  - {task}")
    print("Blocked tasks:")
    for task, reason in blocked.items():
        print(f"  - {task}: {reason}")
    print("Suggested launch:")
    print(f"  source .venv/bin/activate")
    print(f"  export WAVESFM_FORCE_DEVICE={args.device}")
    print(f"  {shlex.join(launch_plan['preflight_command'])}")
    print(f"  {shlex.join(launch_plan['tracker_command'])}")


if __name__ == "__main__":
    main()
