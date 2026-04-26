from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

from benchmark_config import CACHE_ROOT, CACHE_SPECS, RAW_ROOT, build_preprocess_command, discover_raw_input


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess all WavesFM datasets into h5 caches.")
    p.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    p.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    p.add_argument("--cache-ids", nargs="+", default=list(CACHE_SPECS.keys()), choices=list(CACHE_SPECS.keys()))
    p.add_argument(
        "--path-override",
        action="append",
        default=[],
        help="Override raw input path, format cache_id=/abs/path (repeatable).",
    )
    p.add_argument("--skip-missing", action="store_true", help="Skip missing raw inputs instead of failing.")
    p.add_argument("--skip-existing", action="store_true", help="Skip caches that already exist.")
    p.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return p.parse_args()


def parse_overrides(entries: list[str]) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid override: {entry}")
        cache_id, path = entry.split("=", 1)
        overrides[cache_id.strip()] = Path(path).expanduser().resolve()
    return overrides


def main() -> None:
    args = parse_args()
    overrides = parse_overrides(args.path_override)
    args.cache_root.mkdir(parents=True, exist_ok=True)

    for cache_id in args.cache_ids:
        output_path = args.cache_root / CACHE_SPECS[cache_id]["cache_name"]
        raw_input = overrides.get(cache_id, discover_raw_input(cache_id, args.raw_root))

        print(f"[cache] {cache_id}")
        print(f"  raw: {raw_input}")
        print(f"  out: {output_path}")

        if args.skip_existing and output_path.exists():
            print("  SKIP (cache exists)\n")
            continue

        if not raw_input.exists():
            if args.skip_missing:
                print("  SKIP (raw input missing)\n")
                continue
            raise FileNotFoundError(f"Missing raw input for {cache_id}: {raw_input}")

        cmd = build_preprocess_command(cache_id, raw_input, output_path)
        print("  CMD:", " ".join(shlex.quote(part) for part in cmd), "\n")
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

