"""Utilities for parsing KataGo analysis JSON dumps.

This module provides a simple helper that converts the verbose JSON
produced by KataGo's analysis engine into a compact pair of files:
  * a JSONL file containing per-position metadata and pointers to arrays
  * a set of NPZ files storing board features and activations

The expected input JSON has, for each position, the keys described in the
project specification (pi_T, Q, visits, ownership, value, and the feature
planes X and activations A).  Because the raw JSON can be extremely large,
arrays are extracted and stored separately in ``npz_dir`` while the JSONL
record only keeps the path to the saved arrays.

The parser does not attempt to apply board symmetries or other heavy
processing; that should be performed upstream when generating the KataGo
analysis.  Here we only rearrange the data for easier consumption by the
training pipeline.
"""

from __future__ import annotations

import json
import pathlib
from typing import Iterable, Dict, Any

import numpy as np


def _save_npz(npz_dir: pathlib.Path, base_name: str, arrays: Dict[str, np.ndarray]) -> str:
    """Save a dictionary of arrays to an NPZ file and return the relative path."""
    npz_dir.mkdir(parents=True, exist_ok=True)
    npz_path = npz_dir / f"{base_name}.npz"
    np.savez_compressed(npz_path, **arrays)
    return str(npz_path)


def parse_katago_json(json_path: str | pathlib.Path,
                      jsonl_path: str | pathlib.Path,
                      npz_dir: str | pathlib.Path) -> None:
    """Parse a KataGo JSON dump into JSONL and NPZ artefacts.

    Parameters
    ----------
    json_path:
        Path to the input JSON file.  The file is expected to contain a
        list of position dictionaries or one dictionary per line.
    jsonl_path:
        Destination for the compact JSONL metadata file.
    npz_dir:
        Directory where extracted NumPy arrays will be stored.
    """

    json_path = pathlib.Path(json_path)
    jsonl_path = pathlib.Path(jsonl_path)
    npz_dir = pathlib.Path(npz_dir)

    records: Iterable[Dict[str, Any]]
    with json_path.open("r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            records = json.load(f)
        else:
            records = (json.loads(line) for line in f)

        with jsonl_path.open("w", encoding="utf-8") as out_f:
            for idx, rec in enumerate(records):
                arrays = {}
                if "X" in rec and isinstance(rec["X"], dict):
                    for k, v in rec["X"].items():
                        arrays[f"X_{k}"] = np.array(v, dtype=np.float32)
                    rec["X"] = _save_npz(npz_dir, f"{idx:08d}_X", arrays)
                    arrays = {}
                if "A" in rec and isinstance(rec["A"], dict):
                    for k, v in rec["A"].items():
                        arrays[f"A_{k}"] = np.array(v, dtype=np.float32)
                    rec["A"] = _save_npz(npz_dir, f"{idx:08d}_A", arrays)
                out_f.write(json.dumps(rec) + "\n")


__all__ = ["parse_katago_json"]
