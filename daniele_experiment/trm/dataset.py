from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import json
import numpy as np
import torch
from torch.utils.data import Dataset


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


@dataclass
class TRMRecord:
    slate_id: str
    trunkfinal_path: Optional[Path]
    # Optional supervision targets
    policy: Optional[np.ndarray] = None  # [P], distribution over 361+1
    value: Optional[np.ndarray] = None   # [3], win/loss/noresult


def _load_trm_records(policy_jsonl: Path) -> List[TRMRecord]:
    records: List[TRMRecord] = []
    # Get the project root (parent of daniele_experiment)
    project_root = policy_jsonl.parent.parent.parent
    for obj in _read_jsonl(policy_jsonl):
        sid = obj.get("slate_id") or obj.get("position_id") or obj.get("id")
        trunk_path = obj.get("trunkfinal_path")
        # Resolve relative path from project root
        if trunk_path:
            # trunk_path is like "games/trunkfinal/..." relative to project root
            # But we need to go to daniele_experiment/games/trunkfinal/
            trunk_path = policy_jsonl.parent / "trunkfinal" / Path(trunk_path).name
        # Policy distribution over candidates if present; otherwise try raw policy0
        # If only candidate list with probs is present, aggregate into a (361+1) vector
        policy_vec: Optional[np.ndarray] = None
        if "policy0" in obj:
            # Full tensor order from python/gamestate.py is flat (pos_len*pos_len + 1)
            policy_vec = np.asarray(obj["policy0"], dtype=np.float32)
        elif "candidates" in obj:
            # Build a dense vector from sparse candidate list fields: idx361 and policy_slate
            max_moves = 19 * 19 + 1
            dense = np.zeros((max_moves,), dtype=np.float32)
            for c in obj["candidates"]:
                idx = int(c.get("idx361"))
                p = float(c.get("policy_slate", 0.0))
                dense[idx] = p
            s = float(dense.sum())
            if s > 0:
                dense /= s
            policy_vec = dense
        # Value if present
        value_vec: Optional[np.ndarray] = None
        val = obj.get("value")
        if val is not None:
            value_vec = np.asarray(val, dtype=np.float32)
        records.append(
            TRMRecord(
                slate_id=str(sid) if sid is not None else "",
                trunkfinal_path=Path(trunk_path) if trunk_path else None,
                policy=policy_vec,
                value=value_vec,
            )
        )
    return records


class TRMDataset(Dataset):
    """Dataset for TRM that preserves CxHxW trunkfinal as evidence.

    Each item provides:
      - x_chw: torch.FloatTensor [C, H, W]
      - policy (optional): torch.FloatTensor [361+1]
      - value (optional): torch.FloatTensor [3]
      - meta: dict with slate_id and path
    """

    def __init__(self, *, slates_jsonl: Path) -> None:
        self.slates_jsonl = Path(slates_jsonl)
        self.records = _load_trm_records(self.slates_jsonl)
        if len(self.records) == 0:
            raise ValueError(f"No records found in {self.slates_jsonl}")

        # Infer shape from first available trunkfinal file
        self._shape: Optional[Tuple[int, int, int]] = None
        for r in self.records:
            if r.trunkfinal_path and r.trunkfinal_path.exists():
                arr = np.load(r.trunkfinal_path)
                if arr.ndim == 3:
                    self._shape = (int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2]))
                elif arr.ndim == 2:
                    # Treat as [C=channels, H*W] malformed; try square
                    side = int(np.sqrt(arr.shape[1]))
                    self._shape = (int(arr.shape[0]), side, side)
                else:
                    raise ValueError(f"Unexpected trunkfinal shape {arr.shape} in {r.trunkfinal_path}")
                break
        if self._shape is None:
            raise FileNotFoundError("Could not infer trunkfinal shape; missing files")

    @property
    def chw_shape(self) -> Tuple[int, int, int]:
        assert self._shape is not None
        return self._shape

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        if r.trunkfinal_path is None or not r.trunkfinal_path.exists():
            raise FileNotFoundError(f"Missing trunkfinal for slate {r.slate_id}: {r.trunkfinal_path}")
        arr = np.load(r.trunkfinal_path).astype(np.float32)
        if arr.ndim != 3:
            raise ValueError(f"Expected CHW array, got shape {arr.shape} in {r.trunkfinal_path}")
        x = torch.from_numpy(arr)  # [C,H,W]

        sample = {
            "x_chw": x,
            "meta": {
                "slate_id": r.slate_id,
                "trunkfinal_path": str(r.trunkfinal_path),
            },
        }
        if r.policy is not None:
            sample["policy"] = torch.from_numpy(r.policy)
        if r.value is not None:
            sample["value"] = torch.from_numpy(r.value)
        return sample


