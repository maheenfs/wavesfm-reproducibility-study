"""Repack RadCom OTA keyed by tuple strings into normalized, column-style datasets."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import h5py
import numpy as np
from tqdm import tqdm


def _parse_key(key: str) -> Tuple[str, str, str, str]:
    parsed = ast.literal_eval(key)
    modulation, signal_type, snr, sample_idx = parsed
    return str(modulation), str(signal_type), str(snr), str(sample_idx)


def _to_2ct(sample: np.ndarray) -> np.ndarray:
    arr = np.asarray(sample, dtype=np.float32)
    i_part = arr[:128]
    q_part = arr[128:]
    return np.stack([i_part, q_part], axis=0)[:, None, :]


def _maybe_parse_snr(values: Sequence[str]) -> Tuple[bool, List[float] | List[str]]:
    parsed: List[float] = []
    for val in values:
        try:
            parsed.append(float(val))
        except (TypeError, ValueError):
            return False, list(values)
    return True, parsed


def preprocess_radcom(
    input_path: Path,
    output: Path,
    batch_size: int = 256,
    compression: str | None = None,
    sort_keys: bool = True,
    overwrite: bool = False,
) -> Path:
    input_path = Path(input_path)
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as src, h5py.File(output, "w") as dst:
        raw_keys = sorted(src.keys()) if sort_keys else list(src.keys())
        parsed_keys = [_parse_key(k) for k in raw_keys]
        label_pairs = sorted({(m, s) for m, s, _, _ in parsed_keys})
        label_map = {pair: i for i, pair in enumerate(label_pairs)}
        sample_shape_raw = src[raw_keys[0]].shape
        snr_tokens = [snr for _, _, snr, _ in parsed_keys]
        snr_is_float, _snr_values = _maybe_parse_snr(snr_tokens)

        n = len(raw_keys)
        sample_example = _to_2ct(np.zeros(sample_shape_raw, dtype=np.float32))
        sample_shape = sample_example.shape
        chunk = min(batch_size, n)
        str_dtype = h5py.string_dtype(encoding="utf-8")

        dst.create_dataset(
            "sample",
            shape=(n, *sample_shape),
            dtype="float32",
            chunks=(chunk, *sample_shape),
            compression=compression,
        )
        dst.create_dataset("label", shape=(n,), dtype="int64", chunks=(chunk,), compression=compression)
        dst.create_dataset("modulation", shape=(n,), dtype=str_dtype, chunks=(chunk,), compression=compression)
        dst.create_dataset("signal_type", shape=(n,), dtype=str_dtype, chunks=(chunk,), compression=compression)
        snr_dtype = "float32" if snr_is_float else str_dtype
        dst.create_dataset("snr", shape=(n,), dtype=snr_dtype, chunks=(chunk,), compression=compression)

        dst.attrs["source"] = str(input_path)
        dst.attrs["version"] = "v2"
        dst.attrs["sorted_keys"] = bool(sort_keys)
        dst.attrs["sample_shape_raw"] = json.dumps(list(sample_shape_raw))
        dst.attrs["sample_shape"] = json.dumps(list(sample_shape))
        dst.attrs["sample_dtype"] = "float32"
        dst.attrs["label_pairs"] = json.dumps(label_pairs)
        counts = np.zeros(len(label_pairs), dtype=np.int64)
        sum_ch = np.zeros(2, dtype=np.float64)
        sumsq_ch = np.zeros(2, dtype=np.float64)
        total = 0
        idx = 0
        for start in tqdm(range(0, n, batch_size), desc="Pass 1: read+write", unit="batch"):
            end = min(start + batch_size, n)
            keys_batch = raw_keys[start:end]
            parsed_batch = parsed_keys[start:end]
            batch_samples = np.empty((len(keys_batch), *sample_shape), dtype=np.float32)
            labels = np.empty((len(keys_batch),), dtype=np.int64)
            mods = np.empty((len(keys_batch),), dtype=object)
            sigs = np.empty((len(keys_batch),), dtype=object)
            if snr_is_float:
                snrs = np.empty((len(keys_batch),), dtype=np.float32)
            else:
                snrs = np.empty((len(keys_batch),), dtype=object)

            for j, key in enumerate(keys_batch):
                mod, sig, snr, _ = parsed_batch[j]
                sample = _to_2ct(src[key][:])
                batch_samples[j] = sample
                lbl = label_map[(mod, sig)]
                labels[j] = lbl
                counts[lbl] += 1
                mods[j] = mod
                sigs[j] = sig
                if snr_is_float:
                    snrs[j] = float(snr)
                else:
                    snrs[j] = snr

            sum_ch += batch_samples.sum(axis=(0, 2, 3))
            sumsq_ch += np.square(batch_samples, dtype=np.float64).sum(axis=(0, 2, 3))
            total += batch_samples.shape[0] * batch_samples.shape[2] * batch_samples.shape[3]
            dst["sample"][idx:idx + len(batch_samples)] = batch_samples
            dst["label"][idx:idx + len(batch_samples)] = labels
            dst["modulation"][idx:idx + len(batch_samples)] = mods
            dst["snr"][idx:idx + len(batch_samples)] = snrs
            dst["signal_type"][idx:idx + len(batch_samples)] = sigs
            idx += len(batch_samples)

        mean = (sum_ch / float(total)).astype(np.float32)
        var = sumsq_ch / float(total) - np.square(mean, dtype=np.float64)
        std = np.sqrt(np.clip(var, 1e-12, None)).astype(np.float32)
        dst.attrs["mean"] = mean
        dst.attrs["std"] = std

        for start in tqdm(range(0, n, batch_size), desc="Pass 2: normalize", unit="batch"):
            end = min(start + batch_size, n)
            batch_samples = dst["sample"][start:end]
            batch_samples = (batch_samples - mean[None, :, None, None]) / std[None, :, None, None]
            dst["sample"][start:end] = batch_samples

        freq = counts.astype(np.float64) / max(1, counts.sum())
        weights = np.where(freq > 0, 1.0 / freq, 0.0)
        weights = weights / weights.sum().clip(min=1e-8)
        dst.attrs["class_weights"] = weights.astype(np.float32)

    return output


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repack and normalize RadCom OTA keyed by tuple labels.")
    p.add_argument("--input", required=True, help="Source file with tuple keys (modulation, signal_type, snr, sample_idx).")
    p.add_argument("--output", required=True, help="Destination path for reorganized arrays.")
    p.add_argument("--batch-size", type=int, default=256, help="Chunk size for output datasets (default: 256).")
    p.add_argument(
        "--compression",
        default="none",
        choices=["gzip", "lzf", "none"],
        help="Dataset compression (default: none).",
    )
    p.add_argument("--no-sort", action="store_true", help="Keep original key order instead of sorting.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output file.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    comp = None if args.compression == "none" else args.compression
    out = preprocess_radcom(
        input_path=Path(args.input),
        output=Path(args.output),
        batch_size=args.batch_size,
        compression=comp,
        sort_keys=not args.no_sort,
        overwrite=args.overwrite,
    )
    print(f"Wrote reorganized RadCom cache to {out}")


if __name__ == "__main__":
    main()
