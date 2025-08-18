from __future__ import annotations

"""Utilities for computing and storing top policy moves.

This module provides functionality to run 1-visit neural-network evaluation 
on SGF files and record the top policy moves for each position using the raw 
network policy. The resulting mappings can be saved to JSON and later 
displayed on the labeling web page.
"""

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Dict, List, Iterable

from sgfmill import sgf, sgf_moves, common as sgf_common

# Add python directory to path for KataGo modules
sys.path.append(str(Path(__file__).parent.parent / "python"))

from load_model import load_model
from gamestate import GameState, Board


MoveInfo = Dict[str, float]
PolicyMap = Dict[int, List[MoveInfo]]


def save_combined_data(sgf_data: str, policy: PolicyMap, path: Path | str) -> None:
    """Save combined SGF and policy data to ``path`` in JSON format."""
    combined_data = {
        "sgf": sgf_data,
        "policy": policy
    }
    Path(path).write_text(json.dumps(combined_data, indent=2), encoding="utf-8")


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
    device: str = "cuda",
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
            sgf_bytes = f.read()
            game = sgf.Sgf_game.from_bytes(sgf_bytes)
        
        # Store original SGF content as string
        sgf_content = sgf_bytes.decode('utf-8')
        board_size = game.get_size()
        gs = GameState(board_size, GameState.RULES_TT)
        
        # Get the main sequence of moves
        sequence = game.get_main_sequence()
        plays = []
        for node in sequence[1:]:  # Skip root node
            if node.has_property("B"):
                try:
                    color, move = node.get_move()
                    if color == "b":
                        plays.append(("b", move))
                except ValueError:
                    # Handle pass moves or other cases
                    plays.append(("b", None))
            elif node.has_property("W"):
                try:
                    color, move = node.get_move()
                    if color == "w":
                        plays.append(("w", move))
                except ValueError:
                    # Handle pass moves or other cases
                    plays.append(("w", None))

        policy: PolicyMap = {}
        for idx in range(len(plays) + 1):
            outputs = gs.get_model_outputs(model)
            moves_probs = outputs["moves_and_probs0"]
            if not moves_probs:
                break
            
            # First filter moves by policy probability - only evaluate top moves
            # Sort moves by probability (descending)
            sorted_moves = sorted(moves_probs, key=lambda x: x[1], reverse=True)
            
            # Take moves that together make up 95% of probability mass
            cumulative_prob = 0.0
            candidate_moves = []
            for mv, prob in sorted_moves:
                candidate_moves.append((mv, prob))
                cumulative_prob += prob
                if cumulative_prob >= 0.95:
                    break
            
            # Also ensure we always include at least the top 3 moves
            if len(candidate_moves) < 3 and len(sorted_moves) >= 3:
                candidate_moves = sorted_moves[:3]
            
            print(f"Position {idx}: Evaluating {len(candidate_moves)} out of {len(moves_probs)} legal moves")
            
            # Evaluate each candidate move by playing it and getting the resulting winrate
            move_winrates = []
            current_player = gs.board.pla
            
            for mv, prob in candidate_moves:
                # Play the candidate move
                gs.play(current_player, mv)
                
                # Evaluate the resulting position
                try:
                    next_outputs = gs.get_model_outputs(model)
                    next_value = next_outputs["value"]
                    # Since we played a move, the perspective flipped - take 1 - opponent_winrate
                    opponent_winrate = float(next_value[0])
                    our_winrate = 1.0 - opponent_winrate
                    move_winrates.append((mv, our_winrate))
                except:
                    # Fallback to policy probability if evaluation fails
                    move_winrates.append((mv, prob))
                
                # Undo the move to restore the original position
                gs.undo()
            
            # Find best winrate and collect moves within threshold
            if move_winrates:
                best_winrate = max(winrate for _mv, winrate in move_winrates)
                top: List[MoveInfo] = []
                for mv, winrate in move_winrates:
                    if winrate >= best_winrate + threshold:
                        top.append({"move": _loc_to_sgf(mv, gs.board), "winrate": float(winrate)})
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

        # Create policy directory if it doesn't exist
        policy_dir = path.parent / "policy"
        policy_dir.mkdir(exist_ok=True)
        
        # Save combined data with cleaner filename
        out_path = policy_dir / f"{path.stem}.json"
        save_combined_data(sgf_content, policy, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute 1-visit policy suggestions for SGFs")
    parser.add_argument("model", type=Path, help="KataGo model checkpoint")
    parser.add_argument("sgfs", nargs="+", help="SGF files to analyse (supports wildcards)")
    parser.add_argument("--threshold", type=float, default=-0.005, help="Probability drop from best move")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device for model")
    args = parser.parse_args()

    # Expand wildcards in SGF paths
    sgf_paths = []
    for pattern in args.sgfs:
        matches = glob.glob(pattern)
        if matches:
            sgf_paths.extend(matches)
        else:
            # If no matches, treat as literal path
            sgf_paths.append(pattern)
    
    # Convert to Path objects
    sgf_paths = [Path(p) for p in sgf_paths]

    compute_policy_suggestions(sgf_paths, args.model, threshold=args.threshold, device=args.device)


__all__ = ["save_combined_data", "compute_policy_suggestions"]


if __name__ == "__main__":
    main()
    # python policy.py D:\KataGo\kata1-b28c512nbt-s9584861952-d4960414494\model.ckpt D:\KataGo\daniele_experiment\games\*.sgf


