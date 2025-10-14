#!/usr/bin/env python3
"""
Generate a dataset of self-play games with per-move model outputs and trunkfinal activations.

Outputs per game (UUID folder under --output-dir):
  - game.sgf           Full game record
  - meta.json          Metadata about the game generation
  - moves.jsonl        One JSON per line with per-move model outputs
  - trunkfinal/        move_###.npy per move
  - viz.html           Optional (for the first --save-html games)

Example:
  python daniele_experiment/generate_games_dataset.py \
    --model daniele_experiment/model.ckpt \
    --num-games 1000 \
    --output-dir games \
    --save-html 5 \
    --run-snorkel
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np

# Add KataGo python directory to path
sys.path.append(str(Path(__file__).parent.parent / "python"))

from gamestate import GameState, Board  # noqa: E402
from load_model import load_model       # noqa: E402

# Import common utilities
from common_utils import (
    get_device, select_move_with_sampling, create_sgf, convert_numpy_to_python,
    _idx361_from_loc, calculate_dynamic_threshold
)

# Import HTML renderer directly (must be importable)
from visualize_katago_outputs_custom import generate_html_visualization  # type: ignore












def generate_games(
    model_path: Path,
    num_games: int,
    output_dir: Path,
    board_size: int = 19,
    device: str | None = None,
    initial_prob_threshold: float = 0.05,
    final_prob_threshold: float = 0.01,
    transition_moves: int = 50,
    resign_threshold: float = 0.10,
    resign_consec: int = 3,
    save_html: int = 0,
    html_max_moves: int = 200,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    dev = device or get_device()
    print(f"Loading model: {model_path} on device {dev}")
    model, swa_model, _ = load_model(str(model_path), use_swa=False, device=dev, pos_len=board_size, verbose=False)
    if swa_model is not None:
        model = swa_model
    model.eval()

    for game_index in range(1, num_games + 1):
        game_uuid = str(uuid.uuid4())
        game_dir = output_dir / game_uuid
        trunk_dir = game_dir / "trunkfinal"
        trunk_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Game {game_index}/{num_games} | uuid={game_uuid} ===")

        gs = GameState(board_size, GameState.RULES_TT)
        moves: List[Tuple[int, int]] = []
        lows_by_player: Dict[int, int] = {Board.BLACK: 0, Board.WHITE: 0}
        start_ts = time.time()

        # Open moves.jsonl for appending
        moves_jsonl_path = game_dir / "moves.jsonl"
        game_dir.mkdir(exist_ok=True)
        with moves_jsonl_path.open("w", encoding="utf-8") as f_jsonl:
            # Initial position snapshot (move 0) for HTML if needed
            html_positions: List[Dict[str, Any]] = []
            if save_html > 0 and game_index <= save_html:
                init_outputs = gs.get_model_outputs(model, extra_output_names=["trunkfinal"])  # not saved for move 0
                html_positions.append({
                    "move_number": 0,
                    "player": "Initial",
                    "last_move": None,
                    "board_state": [0] * gs.board.arrsize,
                    **convert_numpy_to_python(init_outputs),
                })

            max_moves_safety = 400
            for move_number in range(1, max_moves_safety + 1):
                pla = gs.board.pla
                pla_str = "Black" if pla == Board.BLACK else "White"

                outputs = gs.get_model_outputs(model, extra_output_names=["trunkfinal"])  # before playing
                moves_and_probs = outputs.get("moves_and_probs0", [])
                if not moves_and_probs:
                    print(f"No legal moves for {pla_str} at move {move_number}")
                    break

                # Dynamic threshold
                thr = calculate_dynamic_threshold(move_number - 1, initial_prob_threshold, final_prob_threshold, transition_moves)

                # Sample move
                move_loc, move_prob, _ = select_move_with_sampling(moves_and_probs, thr)
                gs.play(pla, move_loc)
                moves.append((pla, move_loc))

                # Get post-move outputs for winrate and dump trunkfinal of pre-move
                post = gs.get_model_outputs(model)
                try:
                    opp_win = float(post["value"][0])
                    our_win = 1.0 - opp_win
                except Exception:
                    our_win = float(outputs["value"][0])

                if our_win < resign_threshold:
                    lows_by_player[pla] += 1
                else:
                    lows_by_player[pla] = 0

                # Save trunkfinal from pre-move evaluation if present
                if "trunkfinal" in outputs:
                    np.save(trunk_dir / f"move_{move_number:03d}.npy", outputs["trunkfinal"])  # type: ignore[arg-type]

                # Write per-move outputs JSONL (post state, but include selected move info)
                record = {
                    "move_number": move_number,
                    "player": "b" if pla == Board.BLACK else "w",
                    "move_loc": int(move_loc),
                    "idx361": int(_idx361_from_loc(move_loc, gs.board)),
                    "selected_prob": float(move_prob),
                    **convert_numpy_to_python(post),
                }
                f_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")

                # For optional HTML, collect a compact snapshot (bounded by html_max_moves)
                if save_html > 0 and game_index <= save_html and move_number <= html_max_moves:
                    board_state = [0] * gs.board.arrsize
                    for i in range(gs.board.arrsize):
                        if gs.board.board[i] == Board.BLACK:
                            board_state[i] = 1
                        elif gs.board.board[i] == Board.WHITE:
                            board_state[i] = -1
                    html_positions.append({
                        "move_number": move_number,
                        "player": pla_str,
                        "last_move": (pla, move_loc),
                        "board_state": board_state,
                        **convert_numpy_to_python(post),
                    })

                print(f"Move {move_number}: {pla_str} plays {move_loc} (prob={move_prob:.3f}, winrate={our_win:.1%}, thr={thr:.3f})")

                if lows_by_player[pla] >= resign_consec:
                    print(f"{pla_str} resigns after {resign_consec} consecutive moves with winrate < {resign_threshold:.0%}")
                    break

                # Two consecutive passes ends the game (simple rule)
                if len(moves) >= 2 and moves[-1][1] == Board.PASS_LOC and moves[-2][1] == Board.PASS_LOC:
                    print("Two consecutive passes - game end")
                    break

        # Write SGF and metadata
        sgf_content = create_sgf(moves, board_size, game_index)
        (game_dir / "game.sgf").write_text(sgf_content, encoding="utf-8")
        meta = {
            "uuid": game_uuid,
            "board_size": board_size,
            "device": dev,
            "start_time": start_ts,
            "end_time": time.time(),
            "num_moves": len(moves),
            "initial_prob_threshold": initial_prob_threshold,
            "final_prob_threshold": final_prob_threshold,
            "transition_moves": transition_moves,
            "resign_threshold": resign_threshold,
            "resign_consec": resign_consec,
        }
        (game_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # Optional HTML rendering
        if save_html > 0 and game_index <= save_html and generate_html_visualization is not None:
            try:
                viz_path = game_dir / "viz.html"
                # Build SGF content again for the helper
                sgf_content = create_sgf(moves, board_size, game_index)
                
                # Load snorkel data if available
                snorkel_path = game_dir / "snorkel.jsonl"
                snorkel_data = {}
                if snorkel_path.exists():
                    with snorkel_path.open("r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                rec = json.loads(line)
                                snorkel_data[rec["move_number"]] = rec["analysis"]
                            except Exception:
                                continue
                
                # Merge snorkel data with HTML positions
                for pos in html_positions:
                    move_num = pos["move_number"]
                    if move_num in snorkel_data:
                        pos["analysis"] = snorkel_data[move_num]
                
                generate_html_visualization(html_positions, sgf_content, str(viz_path))
                print(f"  HTML saved: {viz_path}")
            except Exception as e:
                print(f"  HTML generation failed: {e}")


def run_snorkel(games_dir: Path, model_path: Path) -> None:
    """Comprehensive snorkel analysis that computes all 28 concepts per move.
    Writes games/<uuid>/snorkel.jsonl with per-move comprehensive analysis.
    """
    from snorkel_board_positions import analyze_position_comprehensive
    from gamestate import GameState, Board
    from load_model import load_model
    
    print("Running comprehensive snorkel analysis...")
    
    # Load model for ownership analysis
    print(f"Loading model: {model_path}")
    model, swa_model, _ = load_model(str(model_path), use_swa=False, device=get_device(), pos_len=19, verbose=False)
    if swa_model is not None:
        model = swa_model
    model.eval()
    
    for game_dir in sorted(games_dir.iterdir()):
        if not game_dir.is_dir():
            continue
        moves_path = game_dir / "moves.jsonl"
        if not moves_path.exists():
            continue
        
        print(f"Analyzing game: {game_dir.name}")
        out_path = game_dir / "snorkel.jsonl"
        
        # Load the game moves
        moves_data = []
        with moves_path.open("r", encoding="utf-8") as f_in:
            for line in f_in:
                try:
                    rec = json.loads(line)
                    moves_data.append(rec)
                except Exception:
                    continue
        
        if not moves_data:
            continue
        
        # Reconstruct the game state for analysis
        board_size = 19
        gs = GameState(board_size, GameState.RULES_TT)
        
        # Track ownership before each move for territory analysis
        ownership_before = None
        
        with out_path.open("w", encoding="utf-8") as f_out:
            for i, rec in enumerate(moves_data):
                try:
                    move_number = rec.get("move_number", i + 1)
                    player = rec.get("player", "b")
                    move_loc = rec.get("move_loc")
                    policy = rec.get("policy0", [])
                    
                    # Convert player string to Board constant
                    current_player = Board.BLACK if player == "b" else Board.WHITE
                    
                    # Get last move location
                    last_move_loc = None
                    if i > 0:
                        last_move_loc = moves_data[i-1].get("move_loc")
                    
                    # Get current ownership from model evaluation
                    outputs = gs.get_model_outputs(model)
                    ownership = outputs.get("ownership", np.zeros((board_size, board_size), dtype=np.float32))
                    
                    # Convert ownership from [1, 19, 19] to [19, 19] if needed
                    if ownership.ndim == 3:
                        ownership = ownership[0]  # Take first (and only) batch
                    
                    # Perform comprehensive analysis
                    analysis = analyze_position_comprehensive(
                        board=gs.board,
                        ownership=ownership,
                        policy=np.array(policy) if policy else np.zeros(361),
                        player=current_player,
                        move_loc=move_loc,
                        last_move_loc=last_move_loc,
                        before_ownership=ownership_before
                    )
                    
                    # Convert numpy arrays to lists for JSON serialization
                    analysis_serializable = convert_numpy_to_python(analysis)
                    
                    # Create output record matching the specification format
                    out = {
                        "move_number": move_number,
                        "player": player,
                        "move_loc": move_loc,
                        "analysis": analysis_serializable,
                    }
                    
                    # Use a more robust JSON serialization
                    try:
                        f_out.write(json.dumps(out, ensure_ascii=False, default=str) + "\n")
                    except (TypeError, ValueError) as e:
                        # Fallback: convert everything to strings if JSON serialization fails
                        out_safe = convert_numpy_to_python(out)
                        f_out.write(json.dumps(out_safe, ensure_ascii=False, default=str) + "\n")
                    
                    # Store current ownership as "before" for next move
                    ownership_before = ownership.copy()
                    
                    # Advance game state for next move
                    if move_loc is not None and move_loc != Board.PASS_LOC:
                        try:
                            gs.play(current_player, move_loc)
                        except Exception:
                            # Skip invalid moves
                            pass
                            
                except Exception as e:
                    print(f"Error analyzing move {i}: {e}")
                    continue
        
        print(f"  Completed analysis for {game_dir.name}")
    
    print("Snorkel analysis completed!")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate self-play games with model outputs and trunkfinal dumps")
    p.add_argument("--model", type=Path, required=True, help="Path to model checkpoint")
    p.add_argument("--num-games", type=int, required=True, help="Number of games to generate")
    p.add_argument("--output-dir", type=Path, default=Path("games"), help="Output directory (default: games)")
    p.add_argument("--board-size", type=int, default=19, choices=[9, 13, 19], help="Board size")
    p.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/mps/cpu)")
    p.add_argument("--initial-prob-threshold", type=float, default=0.05, help="Early-game sampling threshold")
    p.add_argument("--final-prob-threshold", type=float, default=0.01, help="Late-game sampling threshold")
    p.add_argument("--transition-moves", type=int, default=50, help="Moves to decay threshold")
    p.add_argument("--resign-threshold", type=float, default=0.10, help="Resign winrate threshold (0-1)")
    p.add_argument("--resign-consec", type=int, default=3, help="Consecutive low-win moves to resign")
    p.add_argument("--save-html", type=int, default=0, help="Render HTML for the first N games (0=off)")
    p.add_argument("--html-max-moves", type=int, default=200, help="Max moves to include in HTML")
    p.add_argument("--run-snorkel", action="store_true", help="Run snorkel_board_positions.py over games after generation")
    args = p.parse_args()

    if not args.model.exists():
        print(f"Model not found: {args.model}")
        sys.exit(1)

    dev = None if args.device == "auto" else args.device
    generate_games(
        model_path=args.model,
        num_games=args.num_games,
        output_dir=args.output_dir,
        board_size=args.board_size,
        device=dev,
        initial_prob_threshold=args.initial_prob_threshold,
        final_prob_threshold=args.final_prob_threshold,
        transition_moves=args.transition_moves,
        resign_threshold=args.resign_threshold,
        resign_consec=args.resign_consec,
        save_html=args.save_html,
        html_max_moves=args.html_max_moves,
    )

    if args.run_snorkel:
        run_snorkel(args.output_dir, args.model)


if __name__ == "__main__":
    main()


