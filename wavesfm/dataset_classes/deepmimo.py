from __future__ import annotations

from pathlib import Path
import json
import warnings

import h5py
import torch

from dataset_classes.base import ImageDataset


class DeepMIMO(ImageDataset):
    """Dataset wrapper for DeepMIMO caches with LoS and beam labels."""

    def __init__(
        self,
        h5_path: str | Path,
        *,
        label_key: str = "label_los",
        n_beams: int | None = None,
        label_dtype: torch.dtype | None = torch.long,
    ):
        self.selected_n_beams = int(n_beams) if n_beams is not None else None
        if label_key.startswith("label_beam") and n_beams is None:
            raise ValueError("DeepMIMO beam labels require n_beams to select label_beam_{n}.")
        if n_beams is not None:
            label_key = f"label_beam_{int(n_beams)}"
        super().__init__(
            h5_path,
            sample_key="sample",
            label_key=label_key,
            label_dtype=label_dtype,
            meta_keys=("scenario",),
        )
        self.n_beams = None
        self.beam_options = None
        self.effective_n_beams = None
        self.input_source = None
        self.missing_beams = ()
        with h5py.File(self.h5_path, "r") as h5:
            n_beams = h5.attrs.get("n_beams", None)
            if n_beams is not None:
                self.n_beams = int(n_beams)
            input_source = h5.attrs.get("input_source", None)
            if input_source is not None:
                self.input_source = str(input_source)
            beam_options = h5.attrs.get("beam_options", None)
            if beam_options:
                try:
                    self.beam_options = tuple(json.loads(beam_options))
                except Exception:
                    self.beam_options = None
            if self.selected_n_beams is not None:
                eff_key = f"effective_n_beams_{self.selected_n_beams}"
                eff = h5.attrs.get(eff_key, None)
                if eff is not None:
                    self.effective_n_beams = int(eff)
                missing_key = f"missing_beams_{self.selected_n_beams}"
                raw_missing = h5.attrs.get(missing_key, None)
                if raw_missing:
                    try:
                        self.missing_beams = tuple(int(x) for x in json.loads(raw_missing))
                    except Exception:
                        self.missing_beams = ()

        if self.selected_n_beams is not None:
            if self.beam_options is not None and self.selected_n_beams not in self.beam_options:
                raise ValueError(
                    f"DeepMIMO cache {self.h5_path} does not contain label_beam_{self.selected_n_beams}. "
                    f"Available beam options: {self.beam_options}."
                )
            issues = []
            if self.effective_n_beams is not None and self.effective_n_beams != self.selected_n_beams:
                issues.append(
                    f"effective_n_beams_{self.selected_n_beams}={self.effective_n_beams}"
                )
            if self.missing_beams:
                issues.append(f"missing_beams_{self.selected_n_beams}={list(self.missing_beams)}")
            if issues:
                source = f" input_source={self.input_source}" if self.input_source else ""
                message = (
                    f"DeepMIMO cache {self.h5_path} has beam-label coverage issues for "
                    f"label_beam_{self.selected_n_beams}: {', '.join(issues)}.{source}"
                )
                if self.input_source == "pickle":
                    raise ValueError(
                        f"{message} Rebuild the cache from the official scenario folders before rerunning "
                        "deepmimo-beam."
                    )
                warnings.warn(
                    f"{message} Proceeding with the selected codebook size for training; audit these results "
                    "as provisional until the beam-label generation protocol is reconciled.",
                    RuntimeWarning,
                    stacklevel=2,
                )
