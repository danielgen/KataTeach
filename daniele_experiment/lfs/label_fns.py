"""Weak labeling functions for Go concept tags.

These heuristics operate on the teacher outputs produced by KataGo and
return probabilistic labels for a small ontology of human-interpretable
concepts.  The goal is not to be perfectly accurate but to provide noisy
supervision that can later be refined by a label model.

Each labeling function takes a record dictionary (as emitted by
``parse_katago_json``) and returns a float confidence in ``[0,1]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any

import numpy as np


LabelFn = Callable[[Dict[str, Any]], float]


@dataclass
class LabeledTag:
    name: str
    fn: LabelFn


# ---------------------------------------------------------------------------
# Individual labeling functions
# ---------------------------------------------------------------------------

def tenuki_ok(rec: Dict[str, Any], eps: float = 0.02) -> float:
    """Tenuki is acceptable if the best two moves have similar Q."""
    q = np.asarray(rec.get("Q", []), dtype=np.float32)
    if q.size < 2:
        return 0.0
    best = np.max(q)
    second = np.partition(q, -2)[-2]
    return float(best - second < eps)


def invasion_viable(rec: Dict[str, Any], threshold: float = -0.6) -> float:
    """Label when there exists a strongly opponent-owned point."""
    o_t = np.asarray(rec.get("O_T"))
    if o_t.size == 0:
        return 0.0
    return float(np.min(o_t) < threshold)


def cut_available(rec: Dict[str, Any], delta: float = 0.03) -> float:
    """Very rough proxy: multiple moves within ``delta`` Q of best."""
    q = np.asarray(rec.get("Q", []), dtype=np.float32)
    if q.size == 0:
        return 0.0
    best = np.max(q)
    near_best = np.sum(best - q < delta)
    return float(near_best >= 2)


def ladder_works(rec: Dict[str, Any]) -> float:
    """Placeholder heuristic using provided 'ladder' flag if present."""
    return float(rec.get("ladder_works", 0.0))


def sente_line(rec: Dict[str, Any], gamma: float = 0.05) -> float:
    """Sente if best move's follow-up punishes tenuki (approximated)."""
    q = np.asarray(rec.get("Q", []), dtype=np.float32)
    visits = np.asarray(rec.get("visits", []), dtype=np.float32)
    if q.size == 0 or visits.size == 0:
        return 0.0
    best_idx = int(np.argmax(q))
    if visits[best_idx] <= 0:
        return 0.0
    avg_q = np.average(q, weights=visits)
    return float(q[best_idx] - avg_q > gamma)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def apply_label_fns(rec: Dict[str, Any]) -> Dict[str, float]:
    """Run all labeling functions on ``rec`` and return tag→prob mapping."""
    fns = [
        LabeledTag("tenuki_ok", tenuki_ok),
        LabeledTag("invasion_viable", invasion_viable),
        LabeledTag("cut_available", cut_available),
        LabeledTag("ladder_works", ladder_works),
        LabeledTag("sente_line", sente_line),
    ]
    return {t.name: t.fn(rec) for t in fns}


def simple_confidence_combine(tag_probs: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Wrap tag probabilities with a ``p`` field for downstream consumption."""
    return {tag: {"p": prob} for tag, prob in tag_probs.items()}


__all__ = [
    "tenuki_ok",
    "invasion_viable",
    "cut_available",
    "ladder_works",
    "sente_line",
    "apply_label_fns",
    "simple_confidence_combine",
]
