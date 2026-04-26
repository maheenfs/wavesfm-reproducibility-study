from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from benchmark_config import LOCAL_RESULTS_ROOT, LOCAL_SUMMARY_ROOT, TASK_SPECS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize local WavesFM benchmark runs.")
    p.add_argument("--results-root", type=Path, default=LOCAL_RESULTS_ROOT)
    p.add_argument("--summary-root", type=Path, default=LOCAL_SUMMARY_ROOT)
    return p.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def pick_best_entry(entries: list[dict]) -> dict | None:
    best_entry = None
    last_best = None
    for entry in entries:
        cur = entry.get("best_metric")
        if cur is None:
            continue
        if last_best is None or cur != last_best:
            last_best = cur
            best_entry = entry
    return best_entry


def flatten_metrics(metrics: dict | None, prefix: str) -> dict:
    if not metrics:
        return {}
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.pstdev(values)


def format_percent(value: float | None) -> str | None:
    if value is None:
        return None
    text = f"{float(value):.2f}".rstrip("0").rstrip(".")
    return f"{text}%"


def build_experiment_fields(meta: dict) -> dict:
    subset_fraction = meta.get("train_subset_fraction")
    subset_fraction = float(subset_fraction) if subset_fraction is not None else None
    subset_size = meta.get("train_subset_size")
    subset_size = int(subset_size) if subset_size is not None else None
    subset_percent = subset_fraction * 100.0 if subset_fraction is not None else None
    if subset_fraction is not None or subset_size is not None:
        label = (
            f"{format_percent(subset_percent)} subset"
            if subset_percent is not None
            else f"{subset_size:,}-sample subset"
        )
        kind = "train_subset_study"
    else:
        label = "full train split"
        kind = "full_data_session"
    return {
        "experiment_kind": kind,
        "experiment_label": label,
        "train_subset_fraction": subset_fraction,
        "train_subset_percent": subset_percent,
        "train_subset_size": subset_size,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        if not fieldnames:
            f.write("")
            return
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.summary_root.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict] = []
    for meta_path in sorted(args.results_root.glob("*/*/s*/metadata.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        log_path = meta_path.parent / "log.txt"
        entries = load_jsonl(log_path)
        best_entry = pick_best_entry(entries)
        row = {
            "task": meta["task"],
            "mode": meta["mode"],
            "seed": meta["seed"],
            "run_dir": str(meta_path.parent),
            "status": "ok" if best_entry else "missing_log_or_metrics",
            "best_epoch": best_entry.get("epoch") if best_entry else None,
            "best_key": best_entry.get("best_key") if best_entry else None,
            "best_metric": best_entry.get("best_metric") if best_entry else None,
            "primary_metric": TASK_SPECS[meta["task"]]["primary_metric"],
        }
        row.update(build_experiment_fields(meta))
        row.update(flatten_metrics(best_entry.get("val") if best_entry else None, "val."))
        row.update(flatten_metrics(best_entry.get("train") if best_entry else None, "train."))
        run_rows.append(row)

    (args.summary_root / "local_results_runs.json").write_text(
        json.dumps(run_rows, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(args.summary_root / "local_results_runs.csv", run_rows)

    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in run_rows:
        if row["status"] != "ok":
            continue
        grouped.setdefault((row["task"], row["mode"]), []).append(row)

    agg_rows: list[dict] = []
    for (task, mode), rows in sorted(grouped.items()):
        agg = {
            "task": task,
            "mode": mode,
            "n_seeds": len(rows),
            "primary_metric": TASK_SPECS[task]["primary_metric"],
        }
        for key in (
            "experiment_kind",
            "experiment_label",
            "train_subset_fraction",
            "train_subset_percent",
            "train_subset_size",
        ):
            agg[key] = rows[0].get(key)
        numeric_keys = sorted(
            key
            for key in rows[0].keys()
            if key.startswith("val.") or key in {"best_metric"}
            if all(isinstance(r.get(key), (int, float)) for r in rows)
        )
        for key in numeric_keys:
            vals = [float(r[key]) for r in rows]
            mean_val, std_val = mean_std(vals)
            agg[f"{key}.mean"] = mean_val
            agg[f"{key}.std"] = std_val
        agg_rows.append(agg)

    (args.summary_root / "local_results_aggregated.json").write_text(
        json.dumps(agg_rows, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(args.summary_root / "local_results_aggregated.csv", agg_rows)

    manifest = {
        "status": "completed",
        "generated_at_utc": utc_now(),
        "results_root": str(args.results_root),
        "summary_root": str(args.summary_root),
        "runs_json": str(args.summary_root / "local_results_runs.json"),
        "aggregated_json": str(args.summary_root / "local_results_aggregated.json"),
        "completed_task_modes": len(agg_rows),
        "total_task_modes": len({(row["task"], row["mode"]) for row in run_rows}),
        "run_entries": len(run_rows),
        "entries": [
            {
                "task": row["task"],
                "mode": row["mode"],
                "n_seeds": row["n_seeds"],
                "primary_metric": row["primary_metric"],
                "experiment_kind": row.get("experiment_kind"),
                "experiment_label": row.get("experiment_label"),
                "train_subset_fraction": row.get("train_subset_fraction"),
                "train_subset_percent": row.get("train_subset_percent"),
                "train_subset_size": row.get("train_subset_size"),
            }
            for row in agg_rows
        ],
    }
    (args.summary_root / "summary_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
