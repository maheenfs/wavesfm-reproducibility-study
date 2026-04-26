from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import h5py
import numpy as np


DEFAULT_TRAIN_DATA = (
    Path(__file__).resolve().parents[2]
    / "datasets_raw"
    / "deepmimo"
    / "lwm_beam_labels"
    / "beam_prediction_challenge"
    / "train"
    / "bp_data_train.p"
)
DEFAULT_TRAIN_LABELS = DEFAULT_TRAIN_DATA.with_name("bp_label_train.p")
DEFAULT_TEST_DATA = DEFAULT_TRAIN_DATA.parents[1] / "test" / "bp_data_test.p"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "datasets_h5" / "lwm-beam-challenge.h5"


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _class_weights(labels: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    freq = counts / max(1, counts.sum())
    weights = np.zeros_like(freq)
    nonzero = freq > 0
    weights[nonzero] = 1.0 / freq[nonzero]
    weights = weights / weights.sum().clip(min=1e-8)
    return weights.astype(np.float32)


def build_cache(
    train_data_path: Path,
    train_label_path: Path,
    output_path: Path,
    *,
    test_data_path: Path | None = None,
    overwrite: bool = False,
) -> Path:
    train_data_path = train_data_path.expanduser().resolve()
    train_label_path = train_label_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    test_data_path = test_data_path.expanduser().resolve() if test_data_path else None

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_data = _as_numpy(_load_pickle(train_data_path)).astype(np.float32)
    labels = _as_numpy(_load_pickle(train_label_path)).astype(np.int64).reshape(-1)

    if train_data.ndim != 3:
        raise ValueError(f"Expected train data shape (N,H,W), got {train_data.shape}.")
    if labels.ndim != 1 or labels.shape[0] != train_data.shape[0]:
        raise ValueError(f"Label shape {labels.shape} does not match train data shape {train_data.shape}.")
    if labels.min(initial=0) < 0 or labels.max(initial=0) >= 64:
        raise ValueError("LWM beam labels must be in [0, 63].")

    # Keep the official challenge feature values unchanged; only add the channel axis.
    train_sample = train_data[:, None, :, :]

    test_sample = None
    if test_data_path and test_data_path.exists():
        test_data = _as_numpy(_load_pickle(test_data_path)).astype(np.float32)
        if test_data.ndim != 3:
            raise ValueError(f"Expected test data shape (N,H,W), got {test_data.shape}.")
        test_sample = test_data[:, None, :, :]

    counts = np.bincount(labels, minlength=64).astype(np.int64)
    missing = np.flatnonzero(counts == 0).astype(np.int64)
    labels_json = json.dumps([f"beam_{idx}" for idx in range(64)])

    with h5py.File(output_path, "w") as h5:
        h5.create_dataset("sample", data=train_sample, chunks=(min(128, len(train_sample)), 1, 32, 32), compression="gzip")
        h5.create_dataset("label", data=labels, chunks=(min(256, len(labels)),), compression="gzip")
        if test_sample is not None:
            h5.create_dataset(
                "public_test_sample",
                data=test_sample,
                chunks=(min(128, len(test_sample)), 1, 32, 32),
                compression="gzip",
            )

        h5.attrs["version"] = "v1"
        h5.attrs["input_source"] = "official_lwm_beam_challenge_pickles"
        h5.attrs["train_data_pickle"] = str(train_data_path)
        h5.attrs["train_label_pickle"] = str(train_label_path)
        if test_data_path:
            h5.attrs["test_data_pickle"] = str(test_data_path)
        h5.attrs["sample_shape"] = json.dumps(list(train_sample.shape[1:]))
        h5.attrs["sample_format"] = "official_lwm_beam_challenge_single_channel"
        h5.attrs["labels"] = labels_json
        h5.attrs["n_classes"] = 64
        h5.attrs["label_min"] = int(labels.min())
        h5.attrs["label_max"] = int(labels.max())
        h5.attrs["observed_classes"] = int(np.count_nonzero(counts))
        h5.attrs["missing_classes"] = json.dumps(missing.tolist())
        h5.attrs["class_counts"] = json.dumps(counts.tolist())
        h5.attrs["class_weights"] = _class_weights(labels, 64)
        h5.attrs["note"] = (
            "Separate third-run hypothesis cache from official LWM beam challenge train labels; "
            "not the WavesFM DeepMIMO detailed-eval cache."
        )

    return output_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build an H5 cache from the official LWM beam challenge pickles.")
    p.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_DATA)
    p.add_argument("--train-labels", type=Path, default=DEFAULT_TRAIN_LABELS)
    p.add_argument("--test-data", type=Path, default=DEFAULT_TEST_DATA)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = build_cache(
        args.train_data,
        args.train_labels,
        args.output,
        test_data_path=args.test_data,
        overwrite=args.overwrite,
    )
    print(f"[done] wrote {path}")


if __name__ == "__main__":
    main()
