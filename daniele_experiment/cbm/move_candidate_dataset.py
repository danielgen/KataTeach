from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
class CandidateRecord:
    slate_id: str
    trunkfinal_path: Optional[Path]
    move_idx361: int
    policy_slate: float
    is_actual: bool


def _load_candidates_from_slates(slates_path: Path) -> Dict[str, List[CandidateRecord]]:
    grouped: Dict[str, List[CandidateRecord]] = {}
    for obj in _read_jsonl(slates_path):
        sid = obj.get("slate_id")
        trunkfinal_path = obj.get("trunkfinal_path")
        cands = obj.get("candidates", [])
        for c in cands:
            rec = CandidateRecord(
                slate_id=sid,
                trunkfinal_path=Path(trunkfinal_path) if trunkfinal_path else None,
                move_idx361=int(c.get("idx361")),
                policy_slate=float(c.get("policy_slate", 0.0)),
                is_actual=bool(c.get("is_actual", False)),
            )
            grouped.setdefault(sid, []).append(rec)
    return grouped


def _load_move_concept_labels(labels_jsonl: Optional[Path]) -> Dict[Tuple[str, int], List[float]]:
    # Expect records like: {"slate_id": ..., "move_idx361": 123, "concept_labels": [...]}
    mapping: Dict[Tuple[str, int], List[float]] = {}
    if labels_jsonl is None:
        return mapping
    for obj in _read_jsonl(labels_jsonl):
        sid = obj.get("slate_id")
        mv = obj.get("move_idx361")
        labels = obj.get("concept_labels")
        if sid is None or mv is None or labels is None:
            continue
        mapping[(str(sid), int(mv))] = [float(v) for v in labels]
    return mapping


class MoveCandidateDataset(Dataset):
    """Per-candidate dataset grouped by slate for move-conditioned CBM.

    Each item returns a dictionary with slate info and all candidates for that slate.
    Collate function should not stack across slates; rather, it should create a batch
    of slates where each slate contains a variable number of candidates.
    """

    def __init__(
        self,
        slates_path: Path,
        *,
        labels_path: Optional[Path] = None,
        total_moves: int = 19 * 19 + 1,
        require_trunkfinal: bool = True,
        num_labeled_concepts: Optional[int] = None,
        num_latent_concepts: int = 0,
    ) -> None:
        self.slates_path = Path(slates_path)
        self.grouped = _load_candidates_from_slates(self.slates_path)
        self.labels_map = _load_move_concept_labels(Path(labels_path) if labels_path else None)
        self.total_moves = total_moves
        self.require_trunkfinal = require_trunkfinal
        self._num_labeled_concepts = num_labeled_concepts
        self._num_latent_concepts = num_latent_concepts
        self._input_dim = None
        self._slate_ids = list(self.grouped.keys())
        self._infer_shapes()

    @property
    def input_dim(self) -> int:
        assert self._input_dim is not None
        return self._input_dim

    @property
    def num_labeled_concepts(self) -> int:
        assert self._num_labeled_concepts is not None
        return self._num_labeled_concepts

    @property
    def num_latent_concepts(self) -> int:
        return self._num_latent_concepts

    @property
    def num_concepts(self) -> int:
        return self.num_labeled_concepts + self.num_latent_concepts

    def _infer_shapes(self) -> None:
        for sid, recs in self.grouped.items():
            for rec in recs:
                if rec.trunkfinal_path and rec.trunkfinal_path.exists():
                    arr = np.load(rec.trunkfinal_path)
                    self._input_dim = int(arr.size)
                    break
            if self._input_dim is not None:
                break
        if self._input_dim is None:
            if self.require_trunkfinal:
                raise FileNotFoundError("Could not infer input_dim; no trunkfinal files found")
            self._input_dim = 1

        if self._num_labeled_concepts is None:
            # Try infer from first label entry
            for (_sid, _mv), labels in self.labels_map.items():
                self._num_labeled_concepts = len(labels)
                break
        if self._num_labeled_concepts is None:
            self._num_labeled_concepts = 0

    def __len__(self) -> int:
        return len(self._slate_ids)

    def __getitem__(self, idx: int):
        sid = self._slate_ids[idx]
        recs = self.grouped[sid]
        # Use the first rec to find x
        trunk_path = recs[0].trunkfinal_path if recs else None
        if trunk_path is None:
            if self.require_trunkfinal:
                raise FileNotFoundError(f"Missing trunkfinal path for {sid}")
            x = np.zeros((self._input_dim,), dtype=np.float32)
        else:
            arr = np.load(trunk_path)
            x = arr.astype(np.float32).reshape(-1)

        moves = np.array([r.move_idx361 for r in recs], dtype=np.int64)
        slate_probs = np.array([r.policy_slate for r in recs], dtype=np.float32)
        # Normalize to sum to 1
        s = float(slate_probs.sum())
        if s > 0:
            slate_probs = slate_probs / s
        else:
            # uniform fallback
            slate_probs = np.ones_like(slate_probs) / max(1, slate_probs.size)

        # Optional concept labels per move (only for labeled concepts)
        if self.num_labeled_concepts > 0:
            labels = np.zeros((len(recs), self.num_labeled_concepts), dtype=np.float32)
            concept_mask = np.zeros((len(recs), self.num_concepts), dtype=np.float32)
            
            for i, r in enumerate(recs):
                lab = self.labels_map.get((sid, r.move_idx361))
                if lab is not None:
                    labels[i, :len(lab)] = np.asarray(lab, dtype=np.float32)
                    # Mark labeled concepts as having labels
                    concept_mask[i, :self.num_labeled_concepts] = 1.0
        else:
            labels = None
            concept_mask = None

        return {
            "slate_id": sid,
            "x": torch.from_numpy(x),                 # [D]
            "moves": torch.from_numpy(moves),         # [K]
            "slate_probs": torch.from_numpy(slate_probs),  # [K]
            "concept_labels": torch.from_numpy(labels) if labels is not None else None,  # [K, num_labeled_concepts]
            "concept_mask": torch.from_numpy(concept_mask) if concept_mask is not None else None,  # [K, num_concepts]
        }


def slate_group_collate(batch: List[Dict]):
    # Batch of variable-length slates; do not stack candidates across slates
    return batch



