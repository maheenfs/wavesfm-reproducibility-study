#!/usr/bin/env python3
"""Move replaceable project directories onto /local/data0 and keep home-path symlinks."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OffloadTarget:
    rel_path: str
    create_when_missing: bool = False


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_PROJECT_ROOT = Path(
    os.environ.get(
        "WAVESFM_LOCAL_PROJECT_ROOT",
        f"/local/data0/{Path.home().name}/{REPO_ROOT.name}",
    )
)
OFFLOAD_TARGETS = (
    OffloadTarget(".venv"),
    OffloadTarget("datasets_raw", create_when_missing=True),
    OffloadTarget("datasets_h5", create_when_missing=True),
    OffloadTarget("_transfer_quarantine", create_when_missing=True),
    OffloadTarget("phase2_vivor4/automation_logs", create_when_missing=True),
    OffloadTarget("phase2_vivor4/caches", create_when_missing=True),
    OffloadTarget("phase2_vivor4/comparisons", create_when_missing=True),
    OffloadTarget("phase2_vivor4/device_speed_tests", create_when_missing=True),
    OffloadTarget("phase2_vivor4/discarded_runs", create_when_missing=True),
    OffloadTarget("phase2_vivor4/local_results", create_when_missing=True),
    OffloadTarget("phase2_vivor4/plots", create_when_missing=True),
    OffloadTarget("phase2_vivor4/remote_archives", create_when_missing=True),
    OffloadTarget("phase2_vivor4/runs", create_when_missing=True),
)
CANONICAL_HOME_PATHS = (
    Path("notes"),
    Path("README.md"),
    Path("checkpoints"),
    Path("phase2_vivor4/README.md"),
    Path("phase2_vivor4/scripts"),
    Path("phase2_vivor4/systemd"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ensure replaceable WavesFM runtime directories live on /local/data0 "
            "while the home-path project tree keeps stable symlinks."
        )
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="status",
        choices=("status", "apply"),
        help="Show storage status or reconcile the configured offload targets.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Home-path project root to reconcile.",
    )
    parser.add_argument(
        "--local-project-root",
        type=Path,
        default=DEFAULT_LOCAL_PROJECT_ROOT,
        help="Mirror root on the fast local disk.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print warnings and errors.",
    )
    return parser.parse_args()


def log(message: str, *, quiet: bool = False) -> None:
    if not quiet:
        print(message)


def run_rsync(source: Path, destination: Path) -> None:
    source_arg = f"{source}/" if source.is_dir() else str(source)
    destination_arg = f"{destination}/" if destination.is_dir() else str(destination)
    subprocess.run(
        ["rsync", "-a", "--partial", source_arg, destination_arg],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def same_target(home_path: Path, local_path: Path) -> bool:
    if not home_path.is_symlink():
        return False
    try:
        return home_path.resolve(strict=False) == local_path.resolve(strict=False)
    except OSError:
        return False


def ensure_symlink(home_path: Path, local_path: Path) -> str:
    if same_target(home_path, local_path):
        return "symlink-ok"
    if home_path.exists() or home_path.is_symlink():
        raise RuntimeError(f"refusing to replace non-target path: {home_path}")
    home_path.parent.mkdir(parents=True, exist_ok=True)
    home_path.symlink_to(local_path, target_is_directory=True)
    return "symlink-created"


def make_backup_path(home_path: Path) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return home_path.with_name(
        f"{home_path.name}.__storage_offload_backup__.{timestamp}.{os.getpid()}"
    )


def offload_directory(home_path: Path, local_path: Path) -> str:
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if home_path.is_symlink():
        return ensure_symlink(home_path, local_path)

    if not home_path.exists():
        if local_path.exists():
            return ensure_symlink(home_path, local_path)
        return "missing"

    if not home_path.is_dir():
        raise RuntimeError(f"expected directory target, got file: {home_path}")

    if not local_path.exists():
        shutil.move(str(home_path), str(local_path))
        home_path.symlink_to(local_path, target_is_directory=True)
        return "moved-whole-dir"

    if not local_path.is_dir():
        raise RuntimeError(f"local target is not a directory: {local_path}")

    run_rsync(home_path, local_path)
    backup_path = make_backup_path(home_path)
    home_path.rename(backup_path)

    try:
        home_path.symlink_to(local_path, target_is_directory=True)
        run_rsync(backup_path, local_path)
        shutil.rmtree(backup_path)
        return "merged-and-symlinked"
    except Exception:
        if home_path.is_symlink():
            home_path.unlink()
        if backup_path.exists():
            backup_path.rename(home_path)
        raise


def describe_status(home_path: Path, local_path: Path) -> str:
    if same_target(home_path, local_path):
        if local_path.exists():
            return "symlinked"
        return "broken-symlink"
    if home_path.exists() and local_path.exists():
        return "duplicated"
    if home_path.exists():
        return "home-only"
    if local_path.exists():
        return "local-only"
    return "missing"


def describe_home_status(home_path: Path) -> str:
    if not home_path.exists() and not home_path.is_symlink():
        return "missing"
    if home_path.is_symlink():
        return "symlinked"
    return "present"


def find_unexpected_local_entries(local_project_root: Path) -> list[Path]:
    allowed_roots = {Path(target.rel_path).parts[0] for target in OFFLOAD_TARGETS}
    unexpected: list[Path] = []
    if not local_project_root.exists():
        return unexpected

    for child in sorted(local_project_root.iterdir(), key=lambda path: path.name):
        if child.name in allowed_roots:
            continue
        unexpected.append(child)
    return unexpected


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    local_project_root = args.local_project_root
    overall_ok = True

    for target in OFFLOAD_TARGETS:
        home_path = repo_root / target.rel_path
        local_path = local_project_root / target.rel_path
        if args.command == "status":
            status = describe_status(home_path, local_path)
            log(f"{target.rel_path}: {status}", quiet=args.quiet)
            if status in {"duplicated", "home-only", "local-only", "broken-symlink"}:
                overall_ok = False
            continue

        if not home_path.exists() and not home_path.is_symlink():
            if local_path.exists():
                action = ensure_symlink(home_path, local_path)
                log(f"{target.rel_path}: {action}", quiet=args.quiet)
                continue
            if target.create_when_missing:
                local_path.mkdir(parents=True, exist_ok=True)
                action = ensure_symlink(home_path, local_path)
                log(f"{target.rel_path}: {action}", quiet=args.quiet)
                continue
            log(f"{target.rel_path}: skipped-missing", quiet=args.quiet)
            continue

        try:
            action = offload_directory(home_path, local_path)
            log(f"{target.rel_path}: {action}", quiet=args.quiet)
        except Exception as exc:  # pragma: no cover - operational error path
            overall_ok = False
            print(f"{target.rel_path}: ERROR: {exc}", file=sys.stderr)

    if args.command == "status":
        for rel_path in CANONICAL_HOME_PATHS:
            status = describe_home_status(repo_root / rel_path)
            log(f"home:{rel_path}: {status}", quiet=args.quiet)
            if status != "present":
                overall_ok = False

        unexpected = find_unexpected_local_entries(local_project_root)
        if unexpected:
            log("warning: unexpected local-data entries outside the offload targets:", quiet=args.quiet)
            for path in unexpected:
                log(f"  {path}", quiet=args.quiet)
        else:
            log("local-data-root: expected-targets-only", quiet=args.quiet)

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
