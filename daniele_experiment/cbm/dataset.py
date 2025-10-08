from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SlateRecord:
    slate_id: str
    trunkfinal_path: Optional[Path]
    candidates_idx361: List[int]
    candidates_policy_slate: List[float]
    played_idx361: Optional[int]


def _read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_slates(slates_path: Path) -> List[SlateRecord]:
    records: List[SlateRecord] = []
    for obj in _read_jsonl(slates_path):
        slate_id = obj.get("slate_id")
        trunkfinal_path = obj.get("trunkfinal_path")
        candidates = obj.get("candidates", [])
        played = obj.get("played")

        candidates_idx = [int(c.get("idx361")) for c in candidates if "idx361" in c]
        candidates_prob = [float(c.get("policy_slate", 0.0)) for c in candidates]
        played_idx = int(played["idx361"]) if (played and played.get("idx361") is not None) else None

        records.append(
            SlateRecord(
                slate_id=slate_id,
                trunkfinal_path=Path(trunkfinal_path) if trunkfinal_path else None,
                candidates_idx361=candidates_idx,
                candidates_policy_slate=candidates_prob,
                played_idx361=played_idx,
            )
        )
    return records


def _load_concept_labels(labels_jsonl: Optional[Path]) -> Dict[str, List[float]]:
    if labels_jsonl is None:
        return {}
    mapping: Dict[str, List[float]] = {}
    for obj in _read_jsonl(labels_jsonl):
        sid = obj.get("slate_id")
        labels = obj.get("concept_labels")
        if sid is None or labels is None:
            continue
        mapping[str(sid)] = [float(v) for v in labels]
    return mapping


class SlateMovesDataset(Dataset):
    """Dataset over slates.jsonl and trunkfinal .npy activations.

    Each item contains:
    - trunkfinal: FloatTensor [input_dim]
    - concept_labels: FloatTensor [num_concepts] (optional, zero if missing)
    - target_idx: LongTensor [] (selected move index in 0..total_moves-1)
    - candidate_mask: FloatTensor [total_moves] with 1 for candidate moves
    """

    def __init__(
        self,
        slates_path: Path,
        *,
        labels_path: Optional[Path] = None,
        total_moves: int = 19 * 19 + 1,
        target_from: str = "played",
        require_trunkfinal: bool = True,
        num_concepts: Optional[int] = None,
    ) -> None:
        self.slates_path = Path(slates_path)
        self.records = _load_slates(self.slates_path)
        self.labels_map = _load_concept_labels(Path(labels_path) if labels_path else None)
        self.total_moves = total_moves
        self.target_from = target_from
        self.require_trunkfinal = require_trunkfinal
        self._num_concepts = num_concepts

        # Infer input_dim and num_concepts if possible from first sample
        self._input_dim = None
        self._infer_shapes()

    @property
    def input_dim(self) -> int:
        assert self._input_dim is not None
        return self._input_dim

    @property
    def num_concepts(self) -> int:
        assert self._num_concepts is not None
        return self._num_concepts

    def _infer_shapes(self) -> None:
        # Infer input_dim from first available trunkfinal file
        for rec in self.records:
            if rec.trunkfinal_path and rec.trunkfinal_path.exists():
                arr = np.load(rec.trunkfinal_path)
                self._input_dim = int(arr.size)
                break
        if self._input_dim is None:
            if self.require_trunkfinal:
                raise FileNotFoundError("Could not infer input_dim; no trunkfinal files found")
            # Fallback arbitrary
            self._input_dim = 1

        # Infer num_concepts from labels if not provided
        if self._num_concepts is None:
            for sid, labels in self.labels_map.items():
                self._num_concepts = len(labels)
                break
        if self._num_concepts is None:
            # Default to zero concepts if none provided
            self._num_concepts = 0

    def __len__(self) -> int:
        return len(self.records)

    def _choose_target(self, rec: SlateRecord) -> Optional[int]:
        if self.target_from == "played":
            return rec.played_idx361
        # Fallback: use top candidate by rank (first in list)
        return rec.candidates_idx361[0] if rec.candidates_idx361 else None

    def __getitem__(self, idx: int):
        rec = self.records[idx]

        # Load trunkfinal activation
        if rec.trunkfinal_path is None:
            if self.require_trunkfinal:
                raise FileNotFoundError(f"Missing trunkfinal path for {rec.slate_id}")
            x = np.zeros((self._input_dim,), dtype=np.float32)
        else:
            arr = np.load(rec.trunkfinal_path)
            x = arr.astype(np.float32).reshape(-1)

        # Concept labels
        labels = self.labels_map.get(rec.slate_id)
        if labels is None:
            concept = np.zeros((self.num_concepts,), dtype=np.float32)
        else:
            concept = np.asarray(labels, dtype=np.float32)

        # Target index
        tgt = self._choose_target(rec)
        if tgt is None:
            # No target; set to ignore index beyond range, caller must handle
            target_idx = -1
        else:
            target_idx = int(tgt)

        # Candidate mask
        mask = np.zeros((self.total_moves,), dtype=np.float32)
        for m in rec.candidates_idx361:
            if 0 <= m < self.total_moves:
                mask[m] = 1.0

        # Soft policy target (if available): place policy_slate on candidate indices
        soft = np.zeros((self.total_moves,), dtype=np.float32)
        if rec.candidates_policy_slate:
            for m, p in zip(rec.candidates_idx361, rec.candidates_policy_slate):
                if 0 <= m < self.total_moves:
                    soft[m] = float(p)
            s = float(soft.sum())
            if s > 0:
                soft /= s
        else:
            # If no slate probs, fall back to one-hot on target if valid
            if 0 <= target_idx < self.total_moves:
                soft[target_idx] = 1.0

        return {
            "x": torch.from_numpy(x),
            "concept_labels": torch.from_numpy(concept),
            "target_idx": torch.tensor(target_idx, dtype=torch.long),
            "candidate_mask": torch.from_numpy(mask),
            "soft_policy": torch.from_numpy(soft),
            "slate_id": rec.slate_id,
        }


def slate_collate_fn(batch: List[Dict]):
    # Filter items with valid targets (>= 0)
    batch = [b for b in batch if b["target_idx"].item() >= 0]
    if not batch:
        return None
    x = torch.stack([b["x"] for b in batch], dim=0)
    concept = torch.stack([b["concept_labels"] for b in batch], dim=0) if batch[0]["concept_labels"].numel() > 0 else None
    target = torch.stack([b["target_idx"] for b in batch], dim=0)
    mask = torch.stack([b["candidate_mask"] for b in batch], dim=0)
    soft = torch.stack([b["soft_policy"] for b in batch], dim=0)
    slate_ids = [b["slate_id"] for b in batch]
    return {
        "x": x,
        "concept_labels": concept,
        "target_idx": target,
        "candidate_mask": mask,
        "soft_policy": soft,
        "slate_id": slate_ids,
    }


