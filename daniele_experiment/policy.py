from __future__ import annotations

"""Utilities for computing and storing top policy moves.

This module offers two separate helpers:

``extract_top_policy_moves``
    Read a JSONL file produced by ``katago analysis`` and record, for
    each position, the set of moves that are within a winrate threshold
    of the best move.

``compute_policy_suggestions``
    Run a 1-visit neural-network evaluation on one or more SGF files and
    record the top policy moves for each position using the raw network
    policy.  The resulting mappings can be saved to JSON and later
    displayed on the labeling web page.

The analysis JSON is expected to contain for each position at least the
fields ``moves`` (list of move strings in SGF coordinate form) and
``Q`` (corresponding winrates expressed as probabilities from the
current player's perspective).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Iterable

from sgfmill import sgf, sgf_moves, common as sgf_common

from load_model import load_model
from gamestate import GameState, Board


MoveInfo = Dict[str, float]
PolicyMap = Dict[int, List[MoveInfo]]


def extract_top_policy_moves(
    analysis_jsonl: Path | str,
    *,
    threshold: float = -0.005,
) -> PolicyMap:
    """Return mapping of move index to strong policy moves.

    Parameters
    ----------
    analysis_jsonl:
        Path to a JSONL file emitted by ``katago analysis``.
    threshold:
        Inclusive drop in winrate (absolute) from the best move that we
        still consider a top move.  The default of ``-0.005`` collects
        all moves within a 0.5% drop of the best move's winrate.
    """

    policy: PolicyMap = {}
    path = Path(analysis_jsonl)
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            moves: Sequence[str] = rec.get("moves", [])
            winrates: Sequence[float] = rec.get("Q", [])
            if not moves or not winrates:
                continue
            best = max(winrates)
            top: List[MoveInfo] = []
            for mv, q in zip(moves, winrates):
                if q >= best + threshold:
                    top.append({"move": mv, "winrate": float(q)})
            if top:
                policy[idx] = top
    return policy


def save_policy_map(policy: PolicyMap, path: Path | str) -> None:
    """Save ``policy`` to ``path`` in JSON format."""
    Path(path).write_text(json.dumps(policy), encoding="utf-8")

def _loc_to_sgf(loc: int, board: Board) -> str:
    """Convert an internal ``loc`` to an SGF coordinate string."""
    if loc == Board.PASS_LOC:
        return "pass"
    x = board.loc_x(loc)
    y = board.loc_y(loc)
    return sgf_common.format_vertex((y, x))


def compute_policy_suggestions(
    sgf_paths: Iterable[Path | str],
    model_path: Path | str,
    *,
    threshold: float = -0.005,
    device: str = "cpu",
) -> None:
    """Run 1-visit policy evaluation for each SGF in ``sgf_paths``.

    For every SGF, a JSON file is written next to it containing a mapping
    of move index to the list of strong policy moves, using the same
    format expected by ``label_page.py``.

    Parameters
    ----------
    sgf_paths:
        Iterable of SGF file paths to analyse.
    model_path:
        Path to the KataGo model checkpoint.
    threshold:
        Inclusive drop in probability from the best move to still be
        considered a top policy move.  The default of ``-0.005`` keeps
        moves within a 0.5% drop of the best move's probability.
    device:
        Torch device on which to load the model (e.g. ``"cpu"`` or
        ``"cuda"``).
    """

    model, _, _ = load_model(model_path, use_swa=False, device=device, pos_len=19, verbose=False)
    for sgf_path in sgf_paths:
        path = Path(sgf_path)
        with path.open("rb") as f:
            game = sgf.Sgf_game.from_bytes(f.read())
        board_size = game.get_size()
        gs = GameState(board_size, GameState.RULES_TT)
        _, plays = sgf_moves.get_main_moves(game)

        policy: PolicyMap = {}
        for idx in range(len(plays) + 1):
            outputs = gs.get_model_outputs(model)
            moves_probs = outputs["moves_and_probs0"]
            if not moves_probs:
                break
            best = max(p for _m, p in moves_probs)
            top: List[MoveInfo] = []
            for mv, prob in moves_probs:
                if prob >= best + threshold:
                    top.append({"move": _loc_to_sgf(mv, gs.board), "winrate": float(prob)})
            if top:
                policy[idx] = top

            if idx < len(plays):
                color, move = plays[idx]
                pla = Board.BLACK if color == "b" else Board.WHITE
                if move is None:
                    gs.play(pla, Board.PASS_LOC)
                else:
                    row, col = move
                    loc = gs.board.loc(col, row)
                    gs.play(pla, loc)

        out_path = path.with_suffix(path.suffix + ".policy.json")
        save_policy_map(policy, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute 1-visit policy suggestions for SGFs")
    parser.add_argument("model", type=Path, help="KataGo model checkpoint")
    parser.add_argument("sgfs", nargs="+", type=Path, help="SGF files to analyse")
    parser.add_argument("--threshold", type=float, default=-0.005, help="Probability drop from best move")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for model")
    args = parser.parse_args()

    compute_policy_suggestions(args.sgfs, args.model, threshold=args.threshold, device=args.device)


__all__ = ["extract_top_policy_moves", "save_policy_map", "compute_policy_suggestions"]


if __name__ == "__main__":
    main()
