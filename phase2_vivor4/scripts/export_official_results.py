from __future__ import annotations

import csv
import json

from benchmark_config import (
    OFFICIAL_RESULTS,
    OFFICIAL_RESULTS_ROOT,
    OFFICIAL_RESULTS_UPDATED,
    OFFICIAL_RESULTS_URL,
    OFFICIAL_TASK_ROOT,
    TASK_SPECS,
)


def main() -> None:
    OFFICIAL_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    OFFICIAL_TASK_ROOT.mkdir(parents=True, exist_ok=True)

    summary = {
        "source_url": OFFICIAL_RESULTS_URL,
        "source_updated": OFFICIAL_RESULTS_UPDATED,
        "tasks": {},
    }
    rows: list[dict] = []

    for task, task_spec in TASK_SPECS.items():
        if task not in OFFICIAL_RESULTS:
            continue
        official = OFFICIAL_RESULTS[task]
        payload = {
            "task_id": task,
            "task_name": task_spec["display_name"],
            "dataset_name": task_spec["dataset_name"],
            "task_type": task_spec["task_type"],
            "primary_metric": task_spec["primary_metric"],
            "dataset_protocol_url": official["dataset_protocol_url"],
            "reported_split": official["reported_split"],
            "source_url": OFFICIAL_RESULTS_URL,
            "source_updated": OFFICIAL_RESULTS_UPDATED,
            "modes": official["modes"],
        }
        summary["tasks"][task] = payload
        (OFFICIAL_TASK_ROOT / f"{task}.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

        for mode, metrics in official["modes"].items():
            row = {
                "task_id": task,
                "task_name": task_spec["display_name"],
                "dataset_name": task_spec["dataset_name"],
                "task_type": task_spec["task_type"],
                "mode": mode,
                "primary_metric": task_spec["primary_metric"],
                "dataset_protocol_url": official["dataset_protocol_url"],
                "reported_split": official["reported_split"],
                "source_url": OFFICIAL_RESULTS_URL,
                "source_updated": OFFICIAL_RESULTS_UPDATED,
            }
            row.update(metrics)
            rows.append(row)

    (OFFICIAL_RESULTS_ROOT / "official_results_all.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with (OFFICIAL_RESULTS_ROOT / "official_results_all.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
