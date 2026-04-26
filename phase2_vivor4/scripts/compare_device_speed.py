from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


TRAIN_STEP_RE = re.compile(r"\[train\] epoch=(?P<epoch>[0-9.]+) step=(?P<step>\d+)/(?P<steps>\d+) ")
DONE_RE = re.compile(r"\[done\] training time ")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a one-checkpoint CPU vs device timing comparison for WavesFM fine-tuning.")
    p.add_argument("--metadata", type=Path, required=True, help="Path to the saved run metadata.json to clone.")
    p.add_argument("--output-root", type=Path, required=True, help="Root directory for temporary benchmark outputs and logs.")
    p.add_argument("--devices", nargs="+", default=["cpu", "mps"], help="Devices to benchmark in order.")
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Total epochs to pass to main_finetune.py. If omitted, runs exactly one resumed epoch.",
    )
    p.add_argument(
        "--label",
        default="",
        help="Optional label for the benchmark directory. Defaults to task_mode_seed_checkpoint timestamp.",
    )
    return p.parse_args()


def load_metadata(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("command"), list):
        raise ValueError(f"metadata command is missing or not a list: {path}")
    return payload


def replace_flag(cmd: list[str], flag: str, value: str) -> list[str]:
    updated: list[str] = []
    i = 0
    replaced = False
    while i < len(cmd):
        item = cmd[i]
        if item == flag:
            updated.extend([flag, value])
            i += 2
            replaced = True
            continue
        updated.append(item)
        i += 1
    if not replaced:
        updated.extend([flag, value])
    return updated


def value_after(cmd: list[str], flag: str) -> str | None:
    for i, item in enumerate(cmd[:-1]):
        if item == flag:
            return cmd[i + 1]
    return None


def checkpoint_epoch(resume_path: str | None) -> int | None:
    if not resume_path:
        return None
    match = re.search(r"checkpoint_(\d+)\.pth$", resume_path)
    if not match:
        return None
    return int(match.group(1))


def build_benchmark_command(
    base_cmd: list[str],
    *,
    device: str,
    output_dir: Path,
    epochs: int,
) -> list[str]:
    cmd = list(base_cmd)
    cmd = replace_flag(cmd, "--device", device)
    cmd = replace_flag(cmd, "--output-dir", str(output_dir))
    cmd = replace_flag(cmd, "--epochs", str(epochs))
    return cmd


def benchmark_label(metadata: dict) -> str:
    resume_path = metadata.get("resume_checkpoint") or value_after(metadata["command"], "--resume") or "scratch"
    ckpt_epoch = checkpoint_epoch(resume_path)
    ckpt_label = f"ckpt{ckpt_epoch:03d}" if ckpt_epoch is not None else "scratch"
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{metadata['task']}_{metadata['mode']}_s{metadata['seed']}_{ckpt_label}_{stamp}"


def run_and_time(cmd: list[str], *, cwd: Path, log_path: Path) -> dict:
    start = time.monotonic()
    milestone_seconds: dict[str, float] = {}
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"# command: {' '.join(shlex.quote(part) for part in cmd)}\n")
        log_file.write(f"# started_at: {dt.datetime.now(dt.timezone.utc).isoformat()}\n")
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=os.environ.copy(),
        )
        assert process.stdout is not None
        for raw_line in process.stdout:
            now = time.monotonic()
            elapsed = now - start
            line = raw_line.rstrip("\n")
            stamped = f"[+{elapsed:8.3f}s] {line}"
            print(stamped)
            log_file.write(stamped + "\n")
            log_file.flush()

            step_match = TRAIN_STEP_RE.search(line)
            if step_match:
                step = int(step_match.group("step"))
                if "first_train_step" not in milestone_seconds:
                    milestone_seconds["first_train_step"] = elapsed
                milestone_seconds[f"step_{step}"] = elapsed
            if "[ckpt] saved to" in line and "checkpoint" in line:
                milestone_seconds.setdefault("checkpoint_saved", elapsed)
            if DONE_RE.search(line):
                milestone_seconds["done"] = elapsed

        return_code = process.wait()
        total = time.monotonic() - start
        log_file.write(f"# finished_at: {dt.datetime.now(dt.timezone.utc).isoformat()}\n")
        log_file.write(f"# total_wall_seconds: {total:.6f}\n")
        log_file.write(f"# return_code: {return_code}\n")

    return {
        "return_code": return_code,
        "total_wall_seconds": total,
        "milestones": milestone_seconds,
        "log_path": str(log_path),
        "command": cmd,
    }


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.metadata)
    base_cmd = metadata["command"]
    resume_path = metadata.get("resume_checkpoint") or value_after(base_cmd, "--resume")
    resume_epoch = checkpoint_epoch(resume_path)
    if args.epochs is not None:
        epochs = args.epochs
    else:
        if resume_epoch is None:
            raise ValueError("Could not infer checkpoint epoch from metadata; pass --epochs explicitly.")
        epochs = resume_epoch + 2

    label = args.label or benchmark_label(metadata)
    bench_root = args.output_root / label
    bench_root.mkdir(parents=True, exist_ok=False)

    summary = {
        "label": label,
        "metadata_path": str(args.metadata.resolve()),
        "source_output_dir": metadata.get("output_dir"),
        "task": metadata.get("task"),
        "mode": metadata.get("mode"),
        "seed": metadata.get("seed"),
        "resume_checkpoint": resume_path,
        "resume_epoch": resume_epoch + 1 if resume_epoch is not None else None,
        "epochs_argument": epochs,
        "devices": {},
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }

    repo_root = Path(__file__).resolve().parents[2] / "wavesfm"

    for device in args.devices:
        device_root = bench_root / device
        device_out = device_root / "output"
        device_root.mkdir(parents=True, exist_ok=False)
        cmd = build_benchmark_command(base_cmd, device=device, output_dir=device_out, epochs=epochs)
        result = run_and_time(cmd, cwd=repo_root, log_path=device_root / "run.log")
        summary["devices"][device] = result
        if result["return_code"] != 0:
            (bench_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            raise SystemExit(result["return_code"])

    device_keys = list(summary["devices"].keys())
    if len(device_keys) >= 2:
        baseline = summary["devices"][device_keys[0]]
        comparison = summary["devices"][device_keys[1]]
        base_total = baseline["total_wall_seconds"]
        comp_total = comparison["total_wall_seconds"]
        summary["comparison"] = {
            "baseline_device": device_keys[0],
            "comparison_device": device_keys[1],
            "baseline_total_wall_seconds": base_total,
            "comparison_total_wall_seconds": comp_total,
            "speedup_factor": (base_total / comp_total) if comp_total else None,
            "time_delta_seconds": comp_total - base_total,
        }

    (bench_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[done] summary written to {bench_root / 'summary.json'}")


if __name__ == "__main__":
    main()
