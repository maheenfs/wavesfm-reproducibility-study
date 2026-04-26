from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from benchmark_config import COMPARISON_ROOT, LOCAL_SUMMARY_ROOT, OFFICIAL_RESULTS_ROOT, TASK_SPECS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare local summarized results against official WavesFM references.")
    p.add_argument(
        "--official-json",
        type=Path,
        default=OFFICIAL_RESULTS_ROOT / "official_results_all.json",
    )
    p.add_argument(
        "--local-json",
        type=Path,
        default=LOCAL_SUMMARY_ROOT / "local_results_aggregated.json",
    )
    p.add_argument(
        "--local-runs-json",
        type=Path,
        default=None,
        help="Accepted for tracker compatibility; not required for aggregate comparison.",
    )
    p.add_argument("--output-root", type=Path, default=COMPARISON_ROOT)
    return p.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    args.output_root.mkdir(parents=True, exist_ok=True)

    official = json.loads(args.official_json.read_text(encoding="utf-8"))["tasks"]
    local_rows = json.loads(args.local_json.read_text(encoding="utf-8"))
    local_index = {(row["task"], row["mode"]): row for row in local_rows}
    experiment_keys = (
        "experiment_kind",
        "experiment_label",
        "train_subset_fraction",
        "train_subset_percent",
        "train_subset_size",
    )

    comparison_rows: list[dict] = []
    compared_keys: set[tuple[str, str]] = set()
    for task, payload in sorted(official.items()):
        primary_metric = payload["primary_metric"]
        for mode, metrics in payload["modes"].items():
            compared_keys.add((task, mode))
            row = {
                "task": task,
                "task_name": payload["task_name"],
                "mode": mode,
                "primary_metric": primary_metric,
                "official_split": payload["reported_split"],
                "official_source_url": payload["source_url"],
                "official_source_updated": payload["source_updated"],
            }
            local = local_index.get((task, mode))
            for key in experiment_keys:
                row[key] = local.get(key) if local is not None else None
            official_primary = metrics.get(primary_metric)
            row["official_primary"] = official_primary
            if local is None:
                row["local_primary_mean"] = None
                row["local_primary_std"] = None
                row["delta_local_minus_official"] = None
                row["status"] = "missing_local_result"
            else:
                local_key = f"val.{primary_metric}.mean"
                local_std_key = f"val.{primary_metric}.std"
                local_mean = local.get(local_key)
                local_std = local.get(local_std_key)
                row["local_primary_mean"] = local_mean
                row["local_primary_std"] = local_std
                row["delta_local_minus_official"] = (
                    None if local_mean is None or official_primary is None else float(local_mean) - float(official_primary)
                )
                row["status"] = "ok"
            comparison_rows.append(row)

    for (task, mode), local in sorted(local_index.items()):
        if (task, mode) in compared_keys:
            continue
        spec = TASK_SPECS.get(task, {})
        primary_metric = spec.get("primary_metric", local.get("primary_metric"))
        reference_task = spec.get("official_comparison_task")
        reference_payload = official.get(reference_task) if reference_task else None
        reference_metrics = (reference_payload or {}).get("modes", {}).get(mode, {})
        official_primary = reference_metrics.get(primary_metric) if reference_metrics else None
        local_key = f"val.{primary_metric}.mean"
        local_std_key = f"val.{primary_metric}.std"
        local_mean = local.get(local_key)
        row = {
            "task": task,
            "task_name": spec.get("display_name", task),
            "mode": mode,
            "primary_metric": primary_metric,
            "official_split": (reference_payload or {}).get("reported_split"),
            "official_source_url": (reference_payload or {}).get("source_url"),
            "official_source_updated": (reference_payload or {}).get("source_updated"),
            "official_reference_task": reference_task,
            "official_primary": official_primary,
            "local_primary_mean": local_mean,
            "local_primary_std": local.get(local_std_key),
            "delta_local_minus_official": (
                None if local_mean is None or official_primary is None else float(local_mean) - float(official_primary)
            ),
            "status": "hypothesis_reference" if official_primary is not None else "no_official_reference",
        }
        for key in experiment_keys:
            row[key] = local.get(key)
        comparison_rows.append(row)

    (args.output_root / "local_vs_official.json").write_text(
        json.dumps(comparison_rows, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_root / "local_vs_official.csv", comparison_rows)

    manifest = {
        "status": "completed",
        "generated_at_utc": utc_now(),
        "official_json": str(args.official_json),
        "local_json": str(args.local_json),
        "local_runs_json": str(args.local_runs_json) if args.local_runs_json is not None else None,
        "output_root": str(args.output_root),
        "comparison_json": str(args.output_root / "local_vs_official.json"),
        "completed_task_modes": len(comparison_rows),
        "total_task_modes": len(comparison_rows),
        "entries": [
            {
                "task": row["task"],
                "mode": row["mode"],
                "status": row["status"],
                "primary_metric": row["primary_metric"],
                "official_primary": row.get("official_primary"),
                "local_primary_mean": row.get("local_primary_mean"),
                "local_primary_std": row.get("local_primary_std"),
                "delta_local_minus_official": row.get("delta_local_minus_official"),
                "experiment_kind": row.get("experiment_kind"),
                "experiment_label": row.get("experiment_label"),
                "train_subset_fraction": row.get("train_subset_fraction"),
                "train_subset_percent": row.get("train_subset_percent"),
                "train_subset_size": row.get("train_subset_size"),
            }
            for row in comparison_rows
        ],
    }
    (args.output_root / "comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
