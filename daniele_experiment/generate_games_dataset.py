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
    get_device, select_move_with_sampling, select_move_with_temperature,
    create_sgf, convert_numpy_to_python,
    _idx361_from_loc, calculate_dynamic_threshold, calculate_temperature
)

# Import snorkel analysis
from snorkel_board_positions import analyze_position_comprehensive

# Import HTML renderer directly (must be importable)
from visualize_katago_outputs_custom import generate_html_visualization  # type: ignore












def generate_games(
    model_path: Path,
    num_games: int,
    output_dir: Path,
    board_size: int = 19,
    device: str | None = None,
    initial_temperature: float = 1.2,
    final_temperature: float = 0.8,
    transition_moves: int = 60,
    min_prob: float = 0.01,
    top_k: int = 10,
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

                # Dynamic temperature (decays from initial to final over transition_moves)
                temp = calculate_temperature(move_number - 1, initial_temperature, final_temperature, transition_moves)

                # Sample move with temperature-based sampling
                move_loc, move_prob, _ = select_move_with_temperature(moves_and_probs, temp, min_prob, top_k)
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

                # Normalize ownership to current player's perspective before storing
                # KataGo outputs ownership from White's perspective (positive = White territory)
                post_normalized = post.copy()
                if "ownership" in post_normalized:
                    ownership_raw = post_normalized["ownership"]
                    # Convert from [1, 19, 19] to [19, 19] if needed
                    if ownership_raw.ndim == 3:
                        ownership_raw = ownership_raw[0]
                    # Flip ownership only for Black (to convert from White's perspective to Black's)
                    if pla == Board.BLACK:
                        post_normalized["ownership"] = -ownership_raw
                    else:
                        post_normalized["ownership"] = ownership_raw

                # Write per-move outputs JSONL (post state, but include selected move info)
                record = {
                    "move_number": move_number,
                    "player": "b" if pla == Board.BLACK else "w",
                    "move_loc": int(move_loc),
                    "idx361": int(_idx361_from_loc(move_loc, gs.board)),
                    "selected_prob": float(move_prob),
                    **convert_numpy_to_python(post_normalized),
                }
                f_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")

                # For optional HTML, collect a compact snapshot (bounded by html_max_moves)
                if save_html > 0 and game_index <= save_html and move_number <= html_max_moves:
                    board_state = [0] * gs.board.arrsize
                    for i in range(gs.board.arrsize):
                        if gs.board.board[i] == 1:  # Board.BLACK
                            board_state[i] = 1
                        elif gs.board.board[i] == 2:  # Board.WHITE
                            board_state[i] = -1
                    html_positions.append({
                        "move_number": move_number,
                        "player": pla_str,
                        "last_move": (pla, move_loc),
                        "board_state": board_state,
                        **convert_numpy_to_python(post),
                    })
                    print(f"    Added HTML position {move_number}, total: {len(html_positions)}")

                print(f"Move {move_number}: {pla_str} plays {move_loc} (prob={move_prob:.3f}, winrate={our_win:.1%}, temp={temp:.2f})")

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
            "initial_temperature": initial_temperature,
            "final_temperature": final_temperature,
            "transition_moves": transition_moves,
            "min_prob": min_prob,
            "top_k": top_k,
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
                    print(f"  Loading snorkel data from {snorkel_path}")
                    with snorkel_path.open("r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                rec = json.loads(line)
                                # Store both analysis and pass_counterfactual
                                snorkel_data[rec["move_number"]] = {
                                    "analysis": rec.get("analysis"),
                                    "pass_counterfactual": rec.get("pass_counterfactual"),
                                }
                            except Exception:
                                continue
                    print(f"  Loaded snorkel data for {len(snorkel_data)} moves")
                else:
                    print(f"  No snorkel data found at {snorkel_path}")
                
                # Merge snorkel data with HTML positions
                positions_with_snorkel = 0
                for pos in html_positions:
                    move_num = pos["move_number"]
                    if move_num in snorkel_data:
                        pos["analysis"] = snorkel_data[move_num]["analysis"]
                        pos["pass_counterfactual"] = snorkel_data[move_num]["pass_counterfactual"]
                        positions_with_snorkel += 1
                
                print(f"  Merged snorkel data into {positions_with_snorkel} out of {len(html_positions)} positions")
                
                print(f"  Generating HTML with {len(html_positions)} positions")
                generate_html_visualization(html_positions, sgf_content, str(viz_path))
                print(f"  HTML saved: {viz_path}")
            except Exception as e:
                print(f"  HTML generation failed: {e}")


def generate_html_with_snorkel(games_dir: Path, save_html: int, html_max_moves: int, model_path: Path, device: str | None = None) -> None:
    """Generate HTML files with snorkel analysis data included.
    
    If save_html > 0, only processes the first save_html games.
    If save_html == 0, processes all games that have snorkel data.
    """
    from gamestate import GameState, Board
    from load_model import load_model
    import json
    
    print("Generating HTML files with snorkel analysis...")
    
    # Load global stats if available (for percentile display)
    global_stats = None
    stats_path = games_dir / "global_stats.json"
    if stats_path.exists():
        try:
            with stats_path.open("r", encoding="utf-8") as f:
                global_stats = json.load(f)
            print(f"  Loaded global stats from {stats_path}")
        except Exception as e:
            print(f"  Warning: Could not load global stats: {e}")
    
    # Find all game directories
    game_dirs = [d for d in games_dir.iterdir() if d.is_dir()]
    game_dirs.sort(key=lambda x: x.name)  # Sort by UUID for consistent ordering
    
    # Filter to only games with snorkel data
    games_with_snorkel = []
    for game_dir in game_dirs:
        snorkel_path = game_dir / "snorkel.jsonl"
        if snorkel_path.exists():
            games_with_snorkel.append(game_dir)
    
    # Limit to save_html games if specified, otherwise process all
    if save_html > 0:
        games_to_process = games_with_snorkel[:save_html]
    else:
        games_to_process = games_with_snorkel
    
    print(f"  Found {len(games_with_snorkel)} games with snorkel data, processing {len(games_to_process)}")
    
    for game_index, game_dir in enumerate(games_to_process, 1):
        try:
            # Load game metadata
            meta_path = game_dir / "meta.json"
            if not meta_path.exists():
                continue
                
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            board_size = meta["board_size"]
            
            # Load moves from moves.jsonl
            moves_path = game_dir / "moves.jsonl"
            if not moves_path.exists():
                print(f"  Skipping {game_dir.name}: no moves.jsonl found")
                continue
                
            moves_data = []
            with moves_path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        move_data = json.loads(line)
                        moves_data.append(move_data)
                    except Exception:
                        continue
            
            # Extract moves from the data
            moves = []
            for move_data in moves_data:
                if "move_loc" in move_data and "player" in move_data:
                    player_str = move_data["player"]
                    loc = move_data["move_loc"]
                    # Convert player string to Board constant
                    player = Board.BLACK if player_str == "b" else Board.WHITE
                    moves.append((player, loc))
            
            print(f"  Extracted {len(moves)} moves from moves.jsonl")
            
            # Load snorkel data (we already filtered for this, but double-check)
            snorkel_path = game_dir / "snorkel.jsonl"
            if not snorkel_path.exists():
                print(f"  Skipping {game_dir.name}: no snorkel data found")
                continue
                
            snorkel_data = {}
            with snorkel_path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        # Store both analysis and pass_counterfactual
                        snorkel_data[rec["move_number"]] = {
                            "analysis": rec.get("analysis"),
                            "pass_counterfactual": rec.get("pass_counterfactual"),
                        }
                    except Exception:
                        continue
            
            # Reconstruct HTML positions with board states
            html_positions = []
            game_state = GameState(board_size, GameState.RULES_TT)
            
            # Load model for this game
            dev = device or get_device()
            model, swa_model, _ = load_model(str(model_path), use_swa=False, device=dev, pos_len=board_size, verbose=False)
            if swa_model is not None:
                model = swa_model
            model.eval()
            
            # Add initial position
            initial_outputs = game_state.get_model_outputs(model)
            converted_initial_outputs = convert_numpy_to_python(initial_outputs)
            html_positions.append({
                "move_number": 0,
                "player": "Initial",
                "last_move": None,
                "board_state": [0] * game_state.board.arrsize,
                "analysis": snorkel_data.get(0, {}).get("analysis", {}),
                "pass_counterfactual": snorkel_data.get(0, {}).get("pass_counterfactual"),
                **converted_initial_outputs
            })
            
            # Add move positions
            for move_num, (player, loc) in enumerate(moves[:html_max_moves], 1):
                game_state.play(player, loc)
                outputs = game_state.get_model_outputs(model)
                converted_outputs = convert_numpy_to_python(outputs)
                
                # Create board state array
                board_state = [0] * game_state.board.arrsize
                for i in range(game_state.board.arrsize):
                    if game_state.board.board[i] == 1:  # Board.BLACK
                        board_state[i] = 1
                    elif game_state.board.board[i] == 2:  # Board.WHITE
                        board_state[i] = -1
                
                snorkel_rec = snorkel_data.get(move_num, {})
                html_positions.append({
                    "move_number": move_num,
                    "player": "Black" if player == Board.BLACK else "White",
                    "last_move": (player, loc),
                    "board_state": board_state,
                    "analysis": snorkel_rec.get("analysis", {}),
                    "pass_counterfactual": snorkel_rec.get("pass_counterfactual"),
                    **converted_outputs
                })
            
            # Generate SGF content
            sgf_content = create_sgf(moves, board_size, game_index)
            
            # Generate HTML (with optional global stats for percentile display)
            viz_path = game_dir / "viz.html"
            generate_html_visualization(html_positions, sgf_content, str(viz_path), global_stats=global_stats)
            print(f"  HTML with snorkel data saved: {viz_path}")
            
        except Exception as e:
            print(f"  HTML generation failed for {game_dir.name}: {e}")


def run_snorkel(games_dir: Path, model_path: Path, device: str | None = None) -> None:
    """Comprehensive snorkel analysis that computes all 28 concepts per move.
    Writes games/<uuid>/snorkel.jsonl with per-move comprehensive analysis.
    
    Ownership convention:
        - KataGo always outputs ownership from White's perspective (positive = White)
        - We pass raw ownership to analyze_position_comprehensive which handles normalization
        - Both before and after ownership should be raw KataGo outputs (White's perspective)
    
    Pass Counterfactual:
        - For territory-related concepts (building, solidification, reduction, invasion),
          we compute a "pass counterfactual" ownership to avoid anticipatory ownership issues.
        - Before playing the actual move, we compute what ownership would be if the player passed.
        - This counterfactual is used as the baseline for territory concepts instead of the
          previous move's ownership, which may already "anticipate" the expected move.
    """
    from snorkel_board_positions import analyze_position_comprehensive
    from gamestate import GameState, Board
    from load_model import load_model
    
    print("Running comprehensive snorkel analysis...")
    
    # Load model for ownership analysis
    dev = device or get_device()
    print(f"Loading model: {model_path} on device {dev}")
    model, swa_model, _ = load_model(str(model_path), use_swa=False, device=dev, pos_len=19, verbose=False)
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
        
        # Track ownership and board state before each move for territory and group analysis
        # ownership_before is always raw KataGo output (White's perspective)
        # Initialize to zeros for the first move (empty board has neutral ownership)
        ownership_before_raw = np.zeros((board_size, board_size), dtype=np.float32)
        board_before = None
        
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
                    
                    # Capture board state BEFORE the move (for group comparisons)
                    board_before = gs.board.copy()
                    ownership_before_for_analysis = ownership_before_raw.copy()
                    
                    # Compute pass counterfactual ownership BEFORE playing the actual move
                    # This gives us "what would ownership be if the player passed?"
                    # Used to avoid anticipatory ownership issues in territory concepts.
                    pass_counterfactual_raw = None
                    pass_counterfactual_frame_player = None
                    if move_loc is not None and move_loc != Board.PASS_LOC:
                        try:
                            # Play a pass move temporarily to compute counterfactual
                            gs.play(current_player, Board.PASS_LOC)
                            
                            # Get ownership after the pass
                            # After pass, gs.board.pla is the opponent (who plays next)
                            counterfactual_outputs = gs.get_model_outputs(model)
                            pass_counterfactual_raw = counterfactual_outputs.get(
                                "ownership", np.zeros((board_size, board_size), dtype=np.float32)
                            )
                            if pass_counterfactual_raw.ndim == 3:
                                pass_counterfactual_raw = pass_counterfactual_raw[0]
                            
                            # After pass, ownership is from the opponent's perspective (who is to play)
                            pass_counterfactual_frame_player = gs.board.pla
                            
                            # Undo the pass to restore original state
                            gs.undo()
                        except Exception as e:
                            # If counterfactual computation fails, try to recover state
                            if gs.can_undo() and len(gs.moves) > 0 and gs.moves[-1][1] == Board.PASS_LOC:
                                gs.undo()
                            print(f"    Warning: Pass counterfactual failed for move {i}: {e}")
                            pass_counterfactual_raw = None
                            pass_counterfactual_frame_player = None
                    
                    # Play the actual move to get post-move state
                    if move_loc is not None and move_loc != Board.PASS_LOC:
                        try:
                            gs.play(current_player, move_loc)
                        except Exception:
                            # Skip invalid moves
                            pass
                    
                    # Get post-move ownership
                    # Raw ownership from get_model_outputs is from current player to move's perspective
                    # After the move, gs.board.pla is the opponent (next to play)
                    ownership_after = rec.get("ownership_after")
                    if ownership_after is not None:
                        ownership_after_raw = np.array(ownership_after).reshape(board_size, board_size)
                    else:
                        # Get from model evaluation after the move
                        outputs_after = gs.get_model_outputs(model)
                        ownership_after_raw = outputs_after.get("ownership", np.zeros((board_size, board_size), dtype=np.float32))
                        if ownership_after_raw.ndim == 3:
                            ownership_after_raw = ownership_after_raw[0]
                    
                    # Who is to play after the move (opponent)
                    post_move_player = gs.board.pla
                    
                    # Perform comprehensive analysis AFTER the move
                    # ownership_after_raw is from post_move_player's perspective (who is to play now)
                    # ownership_before_for_analysis is from current_player's perspective (was to play before move)
                    # before_board.pla == current_player, so it will normalize correctly
                    # pass_counterfactual_raw provides the "what if I passed" ownership for territory concepts
                    analysis = analyze_position_comprehensive(
                        board=gs.board,  # Post-move board
                        ownership=ownership_after_raw,  # From post_move_player's perspective
                        policy=np.array(policy) if policy else np.zeros(361),
                        player=current_player,
                        move_loc=move_loc,
                        last_move_loc=last_move_loc,
                        before_ownership=ownership_before_for_analysis,  # From current_player's perspective
                        before_board=board_before,  # Pre-move board state (board_before.pla == current_player)
                        ownership_frame_player=post_move_player,  # Frame of ownership_after_raw
                        pass_counterfactual_ownership=pass_counterfactual_raw,  # "What if I passed?"
                        pass_counterfactual_frame_player=pass_counterfactual_frame_player
                    )
                    
                    # Convert numpy arrays to lists for JSON serialization
                    analysis_serializable = convert_numpy_to_python(analysis)
                    
                    # Create output record matching the specification format
                    out = {
                        "move_number": move_number,
                        "player": player,
                        "move_loc": move_loc,
                        "analysis": analysis_serializable,
                        # Store raw pass_counterfactual for visualization
                        "pass_counterfactual": pass_counterfactual_raw.tolist() if pass_counterfactual_raw is not None else None,
                    }
                    
                    # Use a more robust JSON serialization
                    try:
                        f_out.write(json.dumps(out, ensure_ascii=False, default=str) + "\n")
                    except (TypeError, ValueError) as e:
                        # Fallback: convert everything to strings if JSON serialization fails
                        out_safe = convert_numpy_to_python(out)
                        f_out.write(json.dumps(out_safe, ensure_ascii=False, default=str) + "\n")
                    
                    # Store raw ownership_after for next move's "before" analysis
                    # Keep it in raw KataGo format (White's perspective)
                    ownership_before_raw = ownership_after_raw.copy()
                            
                except Exception as e:
                    print(f"Error analyzing move {i}: {e}")
                    continue
        
        print(f"  Completed analysis for {game_dir.name}")
    
    print("Snorkel analysis completed!")


def compute_global_stats(games_dir: Path) -> Dict[str, Any]:
    """
    Compute global statistics from all snorkel.jsonl files for percentile computation.
    
    Returns a dict with percentile arrays for key features:
    - For each feature, stores [p10, p25, p50, p75, p90] values
    """
    print("Computing global statistics from all games...")
    
    # Features to track for percentile computation
    features_to_track = [
        # Territory features
        "potential_territory", "solid_territory", "building_count", "building_intensity",
        "solidification_count", "solidification_intensity", "reduction_count", "reduction_intensity",
        "invasion_intensity",
        # Group features
        "current_group_strength", "current_group_strength_delta",
        "current_group_connectivity", "current_group_connectivity_delta",
        "current_group_influence_count", "current_group_influence_count_delta",
        "current_group_influence_strength", "current_group_influence_strength_delta",
        "liberties",
        # All groups average
        "group_strength_delta", "group_connectivity_delta",
        "influence_count_delta", "influence_strength_delta",
        "max_group_strength_delta", "max_group_connectivity_delta",
        # Attack features
        "avg_attack_intensity", "max_attack_intensity", "aji_reduction_intensity",
        # Sacrifice features
        "direct_sacrifice_intensity", "indirect_sacrifice", "indirect_sacrifice_intensity",
    ]
    
    # Collect all values for each feature
    feature_values: Dict[str, List[float]] = {f: [] for f in features_to_track}
    
    game_count = 0
    move_count = 0
    
    for game_dir in sorted(games_dir.iterdir()):
        if not game_dir.is_dir():
            continue
        snorkel_path = game_dir / "snorkel.jsonl"
        if not snorkel_path.exists():
            continue
        
        game_count += 1
        
        with snorkel_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    analysis = rec.get("analysis", {})
                    move_count += 1
                    
                    for feature in features_to_track:
                        value = analysis.get(feature)
                        if value is not None and isinstance(value, (int, float)):
                            feature_values[feature].append(float(value))
                except Exception:
                    continue
    
    print(f"  Processed {game_count} games, {move_count} moves")
    
    # Compute percentiles for each feature
    percentiles = [10, 25, 50, 75, 90]
    stats: Dict[str, Any] = {
        "percentiles": percentiles,
        "features": {},
        "game_count": game_count,
        "move_count": move_count,
    }
    
    for feature, values in feature_values.items():
        if len(values) >= 10:  # Need at least 10 values for meaningful percentiles
            arr = np.array(values)
            stats["features"][feature] = {
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "p10": float(np.percentile(arr, 10)),
                "p25": float(np.percentile(arr, 25)),
                "p50": float(np.percentile(arr, 50)),
                "p75": float(np.percentile(arr, 75)),
                "p90": float(np.percentile(arr, 90)),
                "count": len(values),
            }
            print(f"  {feature}: n={len(values)}, p50={stats['features'][feature]['p50']:.3f}")
    
    # Save stats to JSON
    stats_path = games_dir / "global_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved global stats to {stats_path}")
    
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="Generate self-play games with model outputs and trunkfinal dumps")
    p.add_argument("--model", type=Path, required=True, help="Path to model checkpoint")
    p.add_argument("--num-games", type=int, required=True, help="Number of games to generate")
    p.add_argument("--output-dir", type=Path, default=Path("games"), help="Output directory (default: games)")
    p.add_argument("--board-size", type=int, default=19, choices=[9, 13, 19], help="Board size")
    p.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/mps/cpu)")
    p.add_argument("--initial-temperature", type=float, default=1.2, help="Early-game sampling temperature (>1 = more diverse)")
    p.add_argument("--final-temperature", type=float, default=0.8, help="Late-game sampling temperature (<1 = more deterministic)")
    p.add_argument("--transition-moves", type=int, default=60, help="Moves over which temperature decays")
    p.add_argument("--min-prob", type=float, default=0.01, help="Minimum policy probability to consider a move (safety filter)")
    p.add_argument("--top-k", type=int, default=10, help="Maximum number of top moves to sample from (safety filter)")
    p.add_argument("--resign-threshold", type=float, default=0.10, help="Resign winrate threshold (0-1)")
    p.add_argument("--resign-consec", type=int, default=3, help="Consecutive low-win moves to resign")
    p.add_argument("--save-html", type=int, default=0, help="Render HTML for the first N games (0=off)")
    p.add_argument("--html-max-moves", type=int, default=200, help="Max moves to include in HTML")
    p.add_argument("--run-snorkel", action="store_true", help="Run snorkel_board_positions.py over games after generation")
    p.add_argument("--compute-stats", action="store_true", help="Compute global stats from all snorkel data (for percentiles)")
    p.add_argument("--stats-only", action="store_true", help="Only compute stats, skip game generation")
    args = p.parse_args()

    # Handle stats-only mode (doesn't require model)
    if args.stats_only:
        if not args.output_dir.exists():
            print(f"Output directory not found: {args.output_dir}")
            sys.exit(1)
        compute_global_stats(args.output_dir)
        return
    
    if not args.model.exists():
        print(f"Model not found: {args.model}")
        sys.exit(1)

    dev = None if args.device == "auto" else args.device
    # If running snorkel analysis, don't generate HTML during game generation
    # We'll generate HTML after snorkel analysis is complete
    save_html_during_generation = args.save_html if not args.run_snorkel else 0
    
    generate_games(
        model_path=args.model,
        num_games=args.num_games,
        output_dir=args.output_dir,
        board_size=args.board_size,
        device=dev,
        initial_temperature=args.initial_temperature,
        final_temperature=args.final_temperature,
        transition_moves=args.transition_moves,
        min_prob=args.min_prob,
        top_k=args.top_k,
        resign_threshold=args.resign_threshold,
        resign_consec=args.resign_consec,
        save_html=save_html_during_generation,
        html_max_moves=args.html_max_moves,
    )

    if args.run_snorkel:
        dev = None if args.device == "auto" else args.device
        run_snorkel(args.output_dir, args.model, dev)
        
        # Compute global stats if requested (or always after snorkel)
        if args.compute_stats:
            compute_global_stats(args.output_dir)
        
        # Now generate HTML with snorkel data included
        # When --run-snorkel is used, generate HTML for games that have snorkel data
        # If save_html > 0, limit to first N games; if 0, generate for all games with snorkel data
        print("Generating HTML with snorkel analysis data...")
        generate_html_with_snorkel(args.output_dir, args.save_html, args.html_max_moves, args.model, dev)
    elif args.compute_stats:
        # Just compute stats without running snorkel
        compute_global_stats(args.output_dir)


if __name__ == "__main__":
    main()


