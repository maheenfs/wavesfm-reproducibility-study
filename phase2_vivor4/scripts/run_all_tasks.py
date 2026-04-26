from __future__ import annotations

import argparse
import datetime
import json
import re
import shlex
import subprocess
from pathlib import Path

from benchmark_config import (
    ALL_MODES,
    CACHE_ROOT,
    CHECKPOINT_PATH,
    DEFAULT_SAVE_EVERY,
    LOCAL_RESULTS_ROOT,
    PHASE2_ROOT,
    TASK_SPECS,
    build_train_command,
    cache_path_for_task,
)


CHECKPOINT_RE = re.compile(r"checkpoint_(\d+)\.pth$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run WavesFM benchmarks into a clearly labeled local_results tree.")
    p.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    p.add_argument("--output-root", type=Path, default=LOCAL_RESULTS_ROOT)
    p.add_argument(
        "--session-root",
        type=Path,
        default=None,
        help="Accepted for tracker compatibility; run output is controlled by --output-root.",
    )
    p.add_argument("--ckpt-path", type=Path, default=CHECKPOINT_PATH)
    p.add_argument("--tasks", nargs="+", default=list(TASK_SPECS.keys()), choices=list(TASK_SPECS.keys()))
    p.add_argument("--modes", nargs="+", default=["lp", "ft2", "lora"], choices=list(ALL_MODES))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument(
        "--train-subset-fraction",
        type=float,
        default=None,
        help="Use only this fraction of each selected task's training split.",
    )
    p.add_argument(
        "--train-subset-size",
        type=int,
        default=None,
        help="Use only this many samples from each selected task's training split.",
    )
    p.add_argument(
        "--discarded-root",
        type=Path,
        default=PHASE2_ROOT / "discarded_runs" / "by_task",
        help="Where to move contaminated/inconsistent run directories before restarting cleanly.",
    )
    p.add_argument("--skip-existing", action="store_true", help="Skip runs that are already complete.")
    p.add_argument(
        "--save-every",
        type=int,
        default=DEFAULT_SAVE_EVERY,
        help="Checkpoint frequency in epochs for launched runs.",
    )
    p.add_argument(
        "--resume-partial",
        dest="resume_partial",
        action="store_true",
        default=True,
        help="Resume incomplete runs from the latest checkpoint when available.",
    )
    p.add_argument(
        "--no-resume-partial",
        dest="resume_partial",
        action="store_false",
        help="Do not resume incomplete runs from checkpoints.",
    )
    p.add_argument(
        "--force-restart",
        action="store_true",
        help="Archive any existing selected run directory and restart it from scratch.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    p.add_argument(
        "--path-override",
        action="append",
        default=[],
        help="Override cache path, format task=/abs/path (repeatable).",
    )
    return p.parse_args()


def parse_overrides(entries: list[str]) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid override: {entry}")
        task, path = entry.split("=", 1)
        overrides[task.strip()] = Path(path).expanduser().resolve()
    return overrides


def _latest_logged_epoch(log_path: Path) -> int | None:
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


def _latest_epoch_checkpoint(run_dir: Path) -> Path | None:
    checkpoints: list[tuple[int, Path]] = []
    for path in run_dir.glob("checkpoint_*.pth"):
        match = CHECKPOINT_RE.fullmatch(path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda item: item[0])
    return checkpoints[-1][1]


def _checkpoint_epoch(path: Path | None) -> int | None:
    if path is None:
        return None
    match = CHECKPOINT_RE.fullmatch(path.name)
    if not match:
        return None
    return int(match.group(1))


def _resume_checkpoint(run_dir: Path) -> Path | None:
    latest_epoch = _latest_epoch_checkpoint(run_dir)
    if latest_epoch is not None:
        return latest_epoch
    best = run_dir / "best.pth"
    if best.exists():
        return best
    return None


def _run_completed(run_dir: Path, expected_epochs: int) -> bool:
    latest_epoch = _latest_logged_epoch(run_dir / "log.txt")
    if latest_epoch is not None and latest_epoch + 1 >= expected_epochs:
        return True
    final_ckpt = run_dir / f"checkpoint_{expected_epochs - 1:03d}.pth"
    return final_ckpt.exists()


def _archive_run_dir(run_dir: Path) -> Path:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    target = run_dir.with_name(f"{run_dir.name}_rerun_{stamp}")
    idx = 1
    while target.exists():
        target = run_dir.with_name(f"{run_dir.name}_rerun_{stamp}_{idx}")
        idx += 1
    run_dir.rename(target)
    return target


def _discard_run_dir(discarded_root: Path, task: str, mode: str, seed: int, run_dir: Path, reason: str) -> Path:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = discarded_root / task / mode
    target = base / f"s{seed}_{reason}_{stamp}"
    idx = 1
    while target.exists():
        target = base / f"s{seed}_{reason}_{stamp}_{idx}"
        idx += 1
    target.parent.mkdir(parents=True, exist_ok=True)
    run_dir.rename(target)
    return target


def main() -> None:
    args = parse_args()
    if args.train_subset_fraction is not None and args.train_subset_size is not None:
        raise ValueError("Provide only one of --train-subset-fraction or --train-subset-size.")
    overrides = parse_overrides(args.path_override)
    args.output_root.mkdir(parents=True, exist_ok=True)

    for task in args.tasks:
        cache_path = overrides.get(task, cache_path_for_task(task, args.cache_root))
        if not cache_path.exists():
            raise FileNotFoundError(f"Missing cache for {task}: {cache_path}")

        for mode in args.modes:
            for seed in args.seeds:
                run_dir = args.output_root / task / mode / f"s{seed}"
                metadata_path = run_dir / "metadata.json"
                run_completed = _run_completed(run_dir, TASK_SPECS[task]["epochs"])
                resume_ckpt = _resume_checkpoint(run_dir)
                latest_logged_epoch = _latest_logged_epoch(run_dir / "log.txt")
                latest_ckpt_epoch = _checkpoint_epoch(_latest_epoch_checkpoint(run_dir))

                dirty_resume = (
                    run_dir.exists()
                    and not run_completed
                    and latest_logged_epoch is not None
                    and (
                        latest_ckpt_epoch is None
                        or latest_logged_epoch > latest_ckpt_epoch
                    )
                )

                if dirty_resume and not args.force_restart:
                    reason = (
                        f"dirty_resume_log{latest_logged_epoch}_ckpt{latest_ckpt_epoch}"
                        if latest_ckpt_epoch is not None
                        else f"dirty_resume_log{latest_logged_epoch}_nockpt"
                    )
                    discarded_to = (
                        _discard_run_dir(args.discarded_root, task, mode, seed, run_dir, reason)
                        if not args.dry_run
                        else args.discarded_root / task / mode / f"s{seed}_{reason}_DRYRUN"
                    )
                    print(f"  DISCARD inconsistent run dir -> {discarded_to}")
                    run_completed = False
                    resume_ckpt = None
                    latest_logged_epoch = None
                    latest_ckpt_epoch = None

                if args.force_restart and run_dir.exists():
                    archived_to = _archive_run_dir(run_dir) if not args.dry_run else run_dir.with_name(f"{run_dir.name}_rerun_DRYRUN")
                    print(f"  ARCHIVE existing run dir -> {archived_to}")
                    run_completed = False
                    resume_ckpt = None

                cmd = build_train_command(
                    task=task,
                    mode=mode,
                    seed=seed,
                    cache_path=cache_path,
                    output_dir=run_dir,
                    ckpt_path=args.ckpt_path,
                    num_workers=args.num_workers,
                    val_split=args.val_split,
                    save_every=args.save_every,
                    resume_path=resume_ckpt if args.resume_partial and not run_completed and resume_ckpt is not None else None,
                    train_subset_fraction=args.train_subset_fraction,
                    train_subset_size=args.train_subset_size,
                )

                metadata = {
                    "task": task,
                    "task_name": TASK_SPECS[task]["display_name"],
                    "mode": mode,
                    "seed": seed,
                    "cache_path": str(cache_path),
                    "output_dir": str(run_dir),
                    "command": cmd,
                    "expected_epochs": TASK_SPECS[task]["epochs"],
                    "save_every": args.save_every,
                    "train_subset_fraction": args.train_subset_fraction,
                    "train_subset_size": args.train_subset_size,
                    "run_completed_before_launch": run_completed,
                    "resume_partial_enabled": args.resume_partial,
                    "resume_checkpoint": (
                        str(resume_ckpt)
                        if args.resume_partial and not run_completed and resume_ckpt is not None
                        else None
                    ),
                }
                print(f"[run] task={task} mode={mode} seed={seed}")
                print("  out:", run_dir)
                print("  cmd:", " ".join(shlex.quote(part) for part in cmd), "\n")

                if run_completed:
                    print("  state: completed")
                elif resume_ckpt is not None and args.resume_partial:
                    print(f"  state: resume from {resume_ckpt}")
                else:
                    print("  state: start from scratch")

                if args.skip_existing and run_completed:
                    print("  SKIP (run already complete)\n")
                    continue

                if not args.dry_run:
                    run_dir.mkdir(parents=True, exist_ok=True)
                    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
                    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
