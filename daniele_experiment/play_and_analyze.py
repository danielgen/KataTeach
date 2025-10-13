#!/usr/bin/env python3
"""Script to play games and analyze them using 1-visit KataGo neural network.

This script plays N games using the KataGo neural network and automatically
analyzes each game, saving both SGF files and policy analysis.

Usage:
    python play_and_analyze.py <model_path> <num_games> [options]
"""

# TODO change this to use cpp scripts and python wrappers in python/ to:
# 1. play games with n_visits = 1
# 2. obtain value output for policy candidate moves 

## once this works properly, delete play_games.py and policy.py

import argparse
import json
import sys
import time
import uuid
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from sgfmill import sgf, sgf_moves, common as sgf_common

# Add python directory to path for KataGo modules
sys.path.append(str(Path(__file__).parent.parent / "python"))
sys.path.append(str(Path(__file__).parent))

from load_model import load_model
from gamestate import GameState, Board
import torch
# Import get_device function directly
def get_device(device: str = "auto") -> str:
    """Auto-detect the best available PyTorch device with fallback logic."""
    if device != "auto":
        return device

    # Check for MPS (Apple Silicon Macs)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"

    # Check for CUDA (NVIDIA GPUs)
    if torch.cuda.is_available():
        return "cuda"

    # Fallback to CPU
    return "cpu"


MoveInfo = Dict[str, float]
PositionInfo = Dict[str, object]  # Contains policy moves and actual move info
PolicyMap = Dict[int, PositionInfo]


def save_combined_data(sgf_data: str, policy: PolicyMap, path: Path | str) -> None:
    """Save combined SGF and policy data to ``path`` in JSON format."""
    combined_data = {
        "sgf": sgf_data,
        "policy": policy
    }
    Path(path).write_text(json.dumps(combined_data, indent=2), encoding="utf-8")


def save_trunkfinal(trunkfinal_data: np.ndarray, game_uuid: str, move_number: int, trunkfinal_dir: Path) -> Path:
    """Save trunkfinal data to a numpy file.
    
    Args:
        trunkfinal_data: The trunkfinal tensor as numpy array
        game_uuid: UUID of the game
        move_number: Move number (1-based)
        trunkfinal_dir: Directory to save trunkfinal files
        
    Returns:
        Path to the saved file
    """
    filename = f"{game_uuid}_move_{move_number:03d}.npy"
    filepath = trunkfinal_dir / filename
    np.save(filepath, trunkfinal_data)
    return filepath


def _idx361_from_loc(loc: int, board: Board) -> int:
    """Convert KataGo loc to 361-style index (0-360 for board positions, 361 for pass)."""
    if loc == Board.PASS_LOC:
        return 361
    x, y = board.loc_x(loc), board.loc_y(loc)
    return y * board.size + x


def _xy_from_loc(loc: int, board: Board) -> List[int]:
    """Convert KataGo loc to [x, y] coordinates."""
    if loc == Board.PASS_LOC:
        return [-1, -1]
    return [board.loc_x(loc), board.loc_y(loc)]


def _append_jsonl(path: Path, obj: dict) -> None:
    """Append a JSON object to a JSONL file."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _loc_to_sgf(loc: int, board: Board) -> str:
    """Convert an internal ``loc`` to an SGF coordinate string."""
    if loc == Board.PASS_LOC:
        return "pass"
    x = board.loc_x(loc)
    y = board.loc_y(loc)
    return sgf_common.format_vertex((y, x))


def loc_to_sgf_coords(loc: int, board: Board) -> str:
    """Convert an internal location to SGF coordinate string."""
    if loc == Board.PASS_LOC:
        return ""  # Pass move in SGF
    x = board.loc_x(loc)
    y = board.loc_y(loc)
    # SGF uses lowercase letters for coordinates
    sgf_x = chr(ord('a') + x)
    sgf_y = chr(ord('a') + y)
    return sgf_x + sgf_y


def _loc_to_human_coord(loc: int, board: Board) -> str:
    """Convert an internal location to human-readable coordinate string (e.g., 'Q16', 'D4')."""
    if loc == Board.PASS_LOC:
        return "pass"
    x = board.loc_x(loc)
    y = board.loc_y(loc)
    
    # Convert to human-readable format: A-T (skip I), 1-19
    # x: 0->A, 1->B, ..., 7->H, 8->J, ..., 18->T
    if x < 8:
        letter = chr(ord('A') + x)
    else:
        letter = chr(ord('A') + x + 1)  # Skip 'I'
    
    # y: 0->19, 1->18, ..., 18->1
    number = board.size - y
    
    return f"{letter}{number}"


def evaluate_moves_batched(gs: GameState, moves: List[int], model) -> List[float]:
    """Evaluate multiple moves in a batched fashion for better performance.
    
    Args:
        gs: Current game state
        moves: List of moves to evaluate
        model: KataGo model
    
    Returns:
        List of winrates for each move from current player's perspective
    """
    if not moves:
        return []
    
    # Store original state
    current_player = gs.board.pla
    
    # Prepare batch data
    batch_states = []
    valid_moves = []
    
    for move in moves:
        try:
            # Play the move
            gs.play(current_player, move)
            
            # Get features for this position
            from features import Features
            features = Features(model.config, model.pos_len)
            bin_input_data, global_input_data = gs.get_input_features(features)
            
            batch_states.append((bin_input_data, global_input_data))
            valid_moves.append(move)
            
            # Undo to restore original position
            gs.undo()
        except:
            # Skip invalid moves
            continue
    
    if not batch_states:
        return []
    
    # Create batch tensors - squeeze out the batch dimension from individual features
    batch_bin = torch.stack([torch.tensor(bin_data, dtype=torch.float32, device=model.device).squeeze(0) 
                            for bin_data, _ in batch_states])
    batch_global = torch.stack([torch.tensor(global_data, dtype=torch.float32, device=model.device).squeeze(0) 
                               for _, global_data in batch_states])
    
    # Batch evaluation
    with torch.no_grad():
        model.eval()
        model_outputs = model(batch_bin, batch_global)
        postprocessed = model.postprocess_output(model_outputs)
        
        # Extract values (perspective is flipped after playing move)
        value_logits = postprocessed[0][1]  # [batch_size, 3] for win/loss/draw
        batch_winrates = []
        
        for i in range(len(valid_moves)):
            # Convert logits to probabilities and get winrate
            # Since we played a move, perspective flipped - take 1 - opponent_winrate  
            opponent_winrate = float(torch.softmax(value_logits[i], dim=0)[0])  # Win probability for current player (now opponent)
            our_winrate = 1.0 - opponent_winrate
            batch_winrates.append(our_winrate)
    
    return batch_winrates


def calculate_dynamic_threshold(move_number: int, initial_threshold: float = 0.05, final_threshold: float = 0.01, transition_moves: int = 50) -> float:
    """Calculate probability threshold that decreases as game progresses.
    
    Args:
        move_number: Current move number (0-based)
        initial_threshold: Starting threshold for early game (default 5% = 0.05)
        final_threshold: Final threshold for late game (default 1% = 0.01)
        transition_moves: Number of moves over which to transition (default 50)
    
    Returns:
        Current probability threshold
    """
    if move_number >= transition_moves:
        return final_threshold
    
    # Linear interpolation from initial to final threshold
    progress = move_number / transition_moves
    return initial_threshold - (initial_threshold - final_threshold) * progress


def select_move_with_sampling(moves_and_probs: List[Tuple[int, float]], prob_threshold: float = 0.01, move_number: int = 0) -> Tuple[int, float, bool]:
    """Select a move by sampling from moves within prob_threshold of the best move.
    
    Args:
        moves_and_probs: List of (move, probability) tuples
        prob_threshold: Probability threshold (default 1% = 0.01)
        move_number: Current move number for dynamic threshold calculation (optional)
    
    Returns:
        Tuple of (selected_move, its_probability, was_sampled)
    """
    if not moves_and_probs:
        raise ValueError("No moves available")
    
    # Find the best probability
    best_prob = max(prob for _, prob in moves_and_probs)
    
    # Find all moves within the threshold
    candidate_moves = []
    for move, prob in moves_and_probs:
        if prob >= best_prob - prob_threshold:
            candidate_moves.append((move, prob))
    
    # If only one candidate, return it (no sampling)
    if len(candidate_moves) == 1:
        return candidate_moves[0][0], candidate_moves[0][1], False
    
    # Sample from candidates using their probabilities as weights
    moves, probs = zip(*candidate_moves)
    
    # Normalize probabilities within the candidate set
    total_prob = sum(probs)
    normalized_probs = [p / total_prob for p in probs]
    
    # Sample based on normalized probabilities
    selected_idx = random.choices(range(len(moves)), weights=normalized_probs)[0]
    selected_move = moves[selected_idx]
    original_prob = probs[selected_idx]
    
    # Check if we selected the best move or an alternative
    best_move = max(candidate_moves, key=lambda x: x[1])[0]
    was_sampled = selected_move != best_move
    
    return selected_move, original_prob, was_sampled


def compute_policy_analysis(
    sgf_content: str,
    model,
    *,
    threshold: float = -0.005,
    verbose: bool = True,
    max_moves_per_position: int = 10,
    trunkfinal_dir: Path = None,
    game_uuid: str = None,
    output_dir: Path = None
) -> PolicyMap:
    """Analyze an SGF game and compute policy suggestions and actual move values.
    
    Parameters
    ----------
    sgf_content:
        SGF content as string
    model:
        Loaded KataGo model
    threshold:
        Inclusive drop in probability from the best move to still be
        considered a top policy move.
    verbose:
        Whether to print progress information
        
    Returns
    -------
    PolicyMap containing suggestions and actual move values
    """
    game = sgf.Sgf_game.from_string(sgf_content)
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
                plays.append(("b", None))
        elif node.has_property("W"):
            try:
                color, move = node.get_move()
                if color == "w":
                    plays.append(("w", move))
            except ValueError:
                plays.append(("w", None))

    policy: PolicyMap = {}
    total_positions = len(plays) + 1
    
    # Initialize JSONL files if output_dir is provided
    slates_path = None
    moves_path = None
    if output_dir is not None:
        slates_path = output_dir / "slates.jsonl"
        moves_path = output_dir / "moves.jsonl"
    
    for idx in range(total_positions):
        if verbose:
            print(f"Analyzing position {idx + 1}/{total_positions}...")
        
        start_time = time.time()
        outputs = gs.get_model_outputs(model, extra_output_names=["trunkfinal"])
        moves_probs = outputs["moves_and_probs0"]
        
        # Save trunkfinal data if directory is provided
        if trunkfinal_dir is not None and game_uuid is not None and "trunkfinal" in outputs:
            trunkfinal_data = outputs["trunkfinal"]
            save_trunkfinal(trunkfinal_data, game_uuid, idx + 1, trunkfinal_dir)
        if not moves_probs:
            break
        
        # First filter moves by policy probability - only evaluate top moves
        sorted_moves = sorted(moves_probs, key=lambda x: x[1], reverse=True)
        
        # Take moves that together make up 95% of probability mass, but cap at max_moves_per_position
        cumulative_prob = 0.0
        candidate_moves = []
        for mv, prob in sorted_moves:
            candidate_moves.append((mv, prob))
            cumulative_prob += prob
            if cumulative_prob >= 0.95 or len(candidate_moves) >= max_moves_per_position:
                break
        
        # Also ensure we always include at least the top 3 moves
        if len(candidate_moves) < 3 and len(sorted_moves) >= 3:
            candidate_moves = sorted_moves[:3]
        
        if verbose:
            print(f"Position {idx}: Evaluating {len(candidate_moves)} out of {len(moves_probs)} legal moves")
        
        # Evaluate candidate moves using batched evaluation for better performance
        candidate_move_locs = [mv for mv, _ in candidate_moves]
        candidate_probs = {mv: prob for mv, prob in candidate_moves}
        
        try:
            # Use batched evaluation
            winrates = evaluate_moves_batched(gs, candidate_move_locs, model)
            move_winrates = [(candidate_move_locs[i], winrates[i]) for i in range(len(winrates))]
        except Exception as e:
            if verbose:
                print(f"Batched evaluation failed, falling back to individual evaluation: {e}")
            # Fallback to individual evaluation
            move_winrates = []
            current_player = gs.board.pla
            
            for mv, prob in candidate_moves:
                try:
                    gs.play(current_player, mv)
                    next_outputs = gs.get_model_outputs(model)
                    next_value = next_outputs["value"]
                    opponent_winrate = float(next_value[0])
                    our_winrate = 1.0 - opponent_winrate
                    move_winrates.append((mv, our_winrate))
                    gs.undo()
                except:
                    move_winrates.append((mv, prob))  # Fallback to policy probability
        
        # Generate JSONL records if paths are provided
        if slates_path is not None and moves_path is not None and game_uuid is not None:
            # Prepare candidate data for JSONL
            raw_probs = [prob for _, prob in candidate_moves]
            denom = sum(raw_probs) if raw_probs else 1.0
            slate_probs = [p / denom for p in raw_probs]
            
            # Create winrate map
            winrate_map = {candidate_move_locs[i]: float(winrates[i]) for i in range(len(winrates))} if move_winrates else {}
            
            # Determine played move for this position
            played_loc = None
            if idx < len(plays) and plays[idx][1] is not None:
                row, col = plays[idx][1]
                played_loc = gs.board.loc(col, row)
            
            # Build candidates array
            cands_json = []
            for rank, (mv, p_raw) in enumerate(candidate_moves):
                p_slate = slate_probs[rank] if rank < len(slate_probs) else 0.0
                cands_json.append({
                    "move_loc": int(mv),
                    "coord_sgf": _loc_to_sgf(mv, gs.board),
                    "coord_human": _loc_to_human_coord(mv, gs.board),
                    "xy": _xy_from_loc(mv, gs.board),
                    "idx361": _idx361_from_loc(mv, gs.board),
                    "policy_raw": float(p_raw),
                    "policy_slate": float(p_slate),
                    "winrate": float(winrate_map.get(mv, float("nan"))),
                    "rank": rank,
                    "is_actual": bool(played_loc == mv),
                })
            
            # Build played move info
            played_info = None
            if played_loc is not None:
                played_policy_raw = next((prob for (mv, prob) in moves_probs if mv == played_loc), None)
                played_info = {
                    "move_loc": int(played_loc),
                    "coord_sgf": _loc_to_sgf(played_loc, gs.board),
                    "coord_human": _loc_to_human_coord(played_loc, gs.board),
                    "xy": _xy_from_loc(played_loc, gs.board),
                    "idx361": _idx361_from_loc(played_loc, gs.board),
                    "policy_raw": float(played_policy_raw) if played_policy_raw is not None else None,
                    "policy_slate": float(next((p["policy_slate"] for p in cands_json if p["move_loc"] == played_loc), 0.0)),
                    "winrate": float(winrate_map.get(played_loc, float("nan"))),
                }
            
            # Create slate record
            slate_id = f"{game_uuid}:{idx}"
            slate_json = {
                "game_uuid": game_uuid,
                "pos_idx": idx,
                "player": "b" if gs.board.pla == Board.BLACK else "w",
                "board_size": gs.board.size,
                "trunkfinal_path": str(trunkfinal_dir / f"{game_uuid}_move_{idx+1:03d}.npy") if trunkfinal_dir else None,
                "slate_id": slate_id,
                "candidates": cands_json,
                "played": played_info
            }
            
            # Write slate record
            _append_jsonl(slates_path, slate_json)
            
            # Write flattened move records
            for c in cands_json:
                flat = {
                    "slate_id": slate_id,
                    "game_uuid": game_uuid,
                    "pos_idx": idx,
                    "player": slate_json["player"],
                    "trunkfinal_path": slate_json["trunkfinal_path"],
                    "move_loc": c["move_loc"],
                    "coord_sgf": c["coord_sgf"],
                    "coord_human": c["coord_human"],
                    "xy": c["xy"],
                    "idx361": c["idx361"],
                    "rank": c["rank"],
                    "policy_raw": c["policy_raw"],
                    "policy_slate": c["policy_slate"],
                    "winrate": c["winrate"],
                    "is_actual": c["is_actual"]
                }
                _append_jsonl(moves_path, flat)

        # Find best winrate and collect moves within threshold
        position_data = {}
        if move_winrates:
            best_winrate = max(winrate for _mv, winrate in move_winrates)
            top: List[MoveInfo] = []
            for mv, winrate in move_winrates:
                if winrate >= best_winrate + threshold:
                    policy_prob = candidate_probs.get(mv, 0.0)
                    top.append({
                        "move": _loc_to_sgf(mv, gs.board), 
                        "winrate": float(winrate),
                        "policy_prob": float(policy_prob)
                    })
            
            # If we have an actual move to be played, ensure it's included in suggestions
            if idx < len(plays):
                color, move = plays[idx]
                if move is not None:
                    row, col = move
                    actual_move_loc = gs.board.loc(col, row)
                    actual_move_sgf = _loc_to_sgf(actual_move_loc, gs.board)
                    
                    # Check if actual move is already in suggestions
                    actual_move_in_suggestions = any(s["move"] == actual_move_sgf for s in top)
                    
                    if not actual_move_in_suggestions:
                        # Find the actual move's winrate and policy probability
                        actual_winrate = None
                        actual_policy_prob = None
                        
                        for mv, winrate in move_winrates:
                            if mv == actual_move_loc:
                                actual_winrate = winrate
                                break
                        
                        actual_policy_prob = candidate_probs.get(actual_move_loc, None)
                        if actual_policy_prob is None:
                            # Find from original moves_probs if not in candidates
                            for mv_loc, prob in moves_probs:
                                if mv_loc == actual_move_loc:
                                    actual_policy_prob = prob
                                    break
                        
                        # Add the actual move to suggestions even if it doesn't meet threshold
                        if actual_winrate is not None:
                            top.append({
                                "move": actual_move_sgf,
                                "winrate": float(actual_winrate),
                                "policy_prob": float(actual_policy_prob) if actual_policy_prob is not None else 0.0,
                                "is_actual_move": True  # Mark this as the actual move played
                            })
            
            if top:
                position_data["suggestions"] = top

        # If this is not the last position, evaluate the actual move played
        if idx < len(plays):
            color, move = plays[idx]
            pla = Board.BLACK if color == "b" else Board.WHITE
            
            actual_move_loc = None
            actual_move_sgf = None
            actual_move_winrate = None
            
            if move is None:
                actual_move_loc = Board.PASS_LOC
                actual_move_sgf = "pass"
            else:
                row, col = move
                actual_move_loc = gs.board.loc(col, row)
                actual_move_sgf = _loc_to_sgf(actual_move_loc, gs.board)
            
            # Evaluate the actual move played
            actual_move_policy_prob = None
            if actual_move_loc is not None:
                # Get the policy probability for the actual move
                actual_move_policy_prob = candidate_probs.get(actual_move_loc, None)
                
                # If the actual move wasn't in the candidate moves, get its probability from the original policy
                if actual_move_policy_prob is None:
                    # Find the policy probability from the original moves_probs
                    for mv_loc, prob in moves_probs:
                        if mv_loc == actual_move_loc:
                            actual_move_policy_prob = prob
                            break
                
                try:
                    gs.play(pla, actual_move_loc)
                    actual_outputs = gs.get_model_outputs(model)
                    actual_value = actual_outputs["value"]
                    # Since we played a move, the perspective flipped
                    opponent_winrate = float(actual_value[0])
                    actual_move_winrate = 1.0 - opponent_winrate
                    gs.undo()  # Undo to restore position
                except:
                    actual_move_winrate = None
            
            # Store actual move information with both winrate and policy probability
            position_data["actual_move"] = {
                "move": actual_move_sgf,
                "winrate": actual_move_winrate,
                "policy_prob": float(actual_move_policy_prob) if actual_move_policy_prob is not None else None,
                "player": color
            }
            
            # Now play the actual move to advance the game state
            gs.play(pla, actual_move_loc)
        
        # Only store position data if we have either suggestions or actual move info
        if position_data:
            policy[idx] = position_data
        
        if verbose:
            elapsed = time.time() - start_time
            print(f"  Position {idx + 1} completed in {elapsed:.2f}s")

    return policy


def play_single_game(model, game_id: int, board_size: int = 19, prob_threshold: float = 0.01, 
                     resignation_threshold: float = 0.10, consecutive_low_moves: int = 3,
                     initial_prob_threshold: float = 0.05, transition_moves: int = 50,
                     trunkfinal_dir: Path = None, game_uuid: str = None) -> Tuple[str, str]:
    """Play a single game using 1-visit neural network evaluation.
    
    Returns:
        Tuple of (SGF content, game result description)
    """
    # Initialize game state
    gs = GameState(board_size, GameState.RULES_TT)
    moves = []
    
    # Track consecutive passes for game termination
    consecutive_passes = 0
    max_moves = 400  # Safety limit to prevent infinite games
    
    # Track winrates for resignation logic
    winrate_history = []  # List of (player, winrate) tuples
    
    print(f"Starting game {game_id}...")
    
    for move_number in range(max_moves):
        current_player = gs.board.pla
        player_str = "Black" if current_player == Board.BLACK else "White"
        
        # Get model outputs for current position (including trunkfinal)
        outputs = gs.get_model_outputs(model, extra_output_names=["trunkfinal"])
        moves_and_probs = outputs["moves_and_probs0"]
        
        # Save trunkfinal data if directory is provided
        if trunkfinal_dir is not None and game_uuid is not None and "trunkfinal" in outputs:
            trunkfinal_data = outputs["trunkfinal"]
            save_trunkfinal(trunkfinal_data, game_uuid, move_number + 1, trunkfinal_dir)
        
        if not moves_and_probs:
            print(f"No legal moves available for {player_str} at move {move_number}")
            break
        
        # Calculate dynamic probability threshold based on game progress
        current_threshold = calculate_dynamic_threshold(move_number, initial_prob_threshold, prob_threshold, transition_moves)
        
        # For 1-visit play, sample from moves within current_threshold of the best move
        best_move, best_prob, was_sampled = select_move_with_sampling(moves_and_probs, current_threshold, move_number)
        
        # Play the move
        gs.play(current_player, best_move)
        moves.append((current_player, best_move))
        
        # Calculate the winrate after making the selected move (to match analysis output)
        try:
            move_outputs = gs.get_model_outputs(model)
            move_value = move_outputs["value"]
            # Since we played a move, perspective flipped - take 1 - opponent_winrate
            opponent_winrate = float(move_value[0])
            move_winrate = 1.0 - opponent_winrate
        except:
            # Fallback: use position evaluation before the move
            position_winrate = float(outputs["value"][0])
            move_winrate = position_winrate
        
        # Store move winrate for resignation logic (same as what we display)
        winrate_history.append((current_player, move_winrate))
        
        # Check for resignation condition (winrate < 10% for 3 consecutive moves by current player)
        if len(winrate_history) >= consecutive_low_moves:
            # Get the last few moves by the current player
            current_player_recent_winrates = [
                winrate for player, winrate in winrate_history[-consecutive_low_moves*2:]  # Look at last 6 moves
                if player == current_player
            ]
            
            # If we have at least 3 moves by current player and all are below threshold
            if (len(current_player_recent_winrates) >= consecutive_low_moves and
                all(wr < resignation_threshold for wr in current_player_recent_winrates[-consecutive_low_moves:])):
                print(f"Game {game_id}: {player_str} resigns (winrate {move_winrate:.1%} < {resignation_threshold:.1%} for {consecutive_low_moves} consecutive moves)")
                break
        
        # Check for pass
        if best_move == Board.PASS_LOC:
            consecutive_passes += 1
            sampling_info = " [sampled]" if was_sampled else ""
            threshold_info = f" (threshold: {current_threshold:.1%})" if move_number < transition_moves else ""
            print(f"Move {move_number + 1}: {player_str} passes (prob: {best_prob:.3f}, winrate: {move_winrate:.1%}){sampling_info}{threshold_info}")
        else:
            consecutive_passes = 0
            move_str = loc_to_sgf_coords(best_move, gs.board)
            sampling_info = " [sampled]" if was_sampled else ""
            threshold_info = f" (threshold: {current_threshold:.1%})" if move_number < transition_moves else ""
            print(f"Move {move_number + 1}: {player_str} plays {move_str} (prob: {best_prob:.3f}, winrate: {move_winrate:.1%}){sampling_info}{threshold_info}")
        
        # Game ends after two consecutive passes
        if consecutive_passes >= 2:
            print(f"Game {game_id} ended after {move_number + 1} moves (two consecutive passes)")
            break
    
    # Create SGF content
    sgf_content = create_sgf(moves, board_size, game_id)
    
    # Determine result (simplified - just count moves)
    black_moves = sum(1 for pla, _ in moves if pla == Board.BLACK)
    white_moves = sum(1 for pla, _ in moves if pla == Board.WHITE)
    result = f"Game {game_id}: {len(moves)} moves ({black_moves} Black, {white_moves} White)"
    
    return sgf_content, result


def create_sgf(moves: List[Tuple[int, int]], board_size: int, game_id: int) -> str:
    """Create SGF content from a list of moves."""
    # Create SGF game
    game = sgf.Sgf_game(size=board_size)
    
    # Set game info
    root = game.get_root()
    root.set("FF", 4)
    root.set("GM", 1) 
    root.set("SZ", board_size)
    root.set("KM", 7.5)  # Standard komi
    root.set("RU", "Tromp-Taylor")
    root.set("PB", f"KataGo-1visit")
    root.set("PW", f"KataGo-1visit")
    root.set("GN", f"1-visit-game-{game_id}")
    root.set("DT", time.strftime("%Y-%m-%d"))
    
    # Create a temporary board to use the coordinate conversion methods
    temp_board = Board(board_size)
    
    # Add moves
    for pla, loc in moves:
        color = "b" if pla == Board.BLACK else "w"
        
        if loc == Board.PASS_LOC:
            # Pass move
            node = game.extend_main_sequence()
            node.set_move(color, None)
        else:
            # Convert loc to board coordinates using Board methods
            x = temp_board.loc_x(loc)
            y = temp_board.loc_y(loc)
            
            if 0 <= x < board_size and 0 <= y < board_size:  # Validate coordinates
                node = game.extend_main_sequence() 
                node.set_move(color, (y, x))
    
    return game.serialise().decode('utf-8')


def play_and_analyze_games(
    model,
    num_games: int,
    output_dir: Path,
    board_size: int = 19,
    prob_threshold: float = 0.01,
    analysis_threshold: float = -0.005,
    max_moves_per_position: int = 10,
    resignation_threshold: float = 0.10,
    consecutive_low_moves: int = 3,
    initial_prob_threshold: float = 0.05,
    transition_moves: int = 50,
) -> None:
    """Play N games, save them as SGF files, and analyze them."""
    
    output_dir.mkdir(exist_ok=True)
    policy_dir = output_dir / "policy"
    policy_dir.mkdir(exist_ok=True)
    trunkfinal_dir = output_dir / "trunkfinal"
    trunkfinal_dir.mkdir(exist_ok=True)
    
    print(f"Playing and analyzing {num_games} games...")
    
    for game_id in range(1, num_games + 1):
        try:
            # Generate UUID for this game
            game_uuid = str(uuid.uuid4())
            
            # Play the game
            sgf_content, result = play_single_game(model, game_id, board_size, prob_threshold, 
                                                 resignation_threshold, consecutive_low_moves,
                                                 initial_prob_threshold, transition_moves,
                                                 trunkfinal_dir, game_uuid)
            
            # Save SGF file
            sgf_file = output_dir / f"{game_uuid}.sgf"
            sgf_file.write_text(sgf_content, encoding='utf-8')
            
            print(f"✓ {result}")
            print(f"  SGF saved: {sgf_file}")
            
            # Analyze the game
            print(f"  Analyzing game {game_id}...")
            policy = compute_policy_analysis(sgf_content, model, threshold=analysis_threshold, verbose=True, 
                                           max_moves_per_position=max_moves_per_position,
                                           trunkfinal_dir=trunkfinal_dir, game_uuid=game_uuid,
                                           output_dir=output_dir)
            
            # Save policy analysis
            policy_file = policy_dir / f"{sgf_file.stem}.json"
            save_combined_data(sgf_content, policy, policy_file)
            print(f"  Analysis saved: {policy_file}")
            
        except Exception as e:
            print(f"✗ Error in game {game_id}: {e}")
            continue
    
    print(f"\nCompleted! Games saved to {output_dir}")
    print(f"  - SGF files: {output_dir}")
    print(f"  - Policy analysis: {policy_dir}")
    print(f"  - Trunkfinal data: {trunkfinal_dir}")
    print(f"  - Slates dataset: {output_dir}/slates.jsonl")
    print(f"  - Moves dataset: {output_dir}/moves.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play games and analyze them using 1-visit KataGo neural network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Play and analyze 5 new games with dynamic thresholds (5% -> 1% over 50 moves)
  python play_and_analyze.py model.ckpt 5
  
  # Play games with custom dynamic threshold settings
  python play_and_analyze.py model.ckpt 3 --initial-prob-threshold 0.08 --transition-moves 30 --max-moves-per-position 3
  
  # Use fixed threshold (no dynamics) by setting initial = final
  python play_and_analyze.py model.ckpt 2 --initial-prob-threshold 0.02 --prob-threshold 0.02
  
  # Use specific device (auto-detection is default)
  python play_and_analyze.py model.ckpt 2 --device cpu
        """
    )
    
    parser.add_argument("model", type=Path, help="Path to KataGo model checkpoint")
    parser.add_argument("num_games", type=int, help="Number of games to play and analyze")
    parser.add_argument("--output-dir", type=Path, default="games", 
                       help="Directory to save SGF files (default: games)")
    parser.add_argument("--board-size", type=int, default=19, choices=[9, 13, 19],
                       help="Board size (default: 19)")
    parser.add_argument("--device", type=str, default="auto", 
                       help="PyTorch device (default: auto - will use mps/cuda/cpu as available)")
    parser.add_argument("--prob-threshold", type=float, default=0.05,
                       help="Probability threshold for move sampling (default: 0.05 = 5%%)")
    parser.add_argument("--analysis-threshold", type=float, default=-0.005,
                       help="Winrate drop threshold for policy analysis (default: -0.005)")
    parser.add_argument("--max-moves-per-position", type=int, default=10,
                       help="Maximum number of moves to analyze per position (default: 10)")
    parser.add_argument("--resignation-threshold", type=float, default=0.10,
                       help="Winrate threshold for resignation (default: 0.10 = 10%%)")
    parser.add_argument("--consecutive-low-moves", type=int, default=3,
                       help="Number of consecutive low winrate moves before resignation (default: 3)")
    parser.add_argument("--initial-prob-threshold", type=float, default=0.05,
                       help="Initial probability threshold for early game diversity (default: 0.05 = 5%%)")
    parser.add_argument("--transition-moves", type=int, default=50,
                       help="Number of moves over which to transition from initial to final threshold (default: 50)")
    
    args = parser.parse_args()
    
    if not args.model.exists():
        print(f"Error: Model file {args.model} does not exist")
        sys.exit(1)
    
    if args.num_games <= 0:
        print("Error: Number of games must be positive")
        sys.exit(1)
    
    try:
        # Auto-detect device if needed
        actual_device = get_device(args.device)
        print(f"Loading model from {args.model}...")
        print(f"Using device: {actual_device}")
        model, _, _ = load_model(args.model, use_swa=False, device=actual_device, pos_len=19, verbose=False)
        
        play_and_analyze_games(
            model=model,
            num_games=args.num_games,
            output_dir=args.output_dir,
            board_size=args.board_size,
            prob_threshold=args.prob_threshold,
            analysis_threshold=args.analysis_threshold,
            max_moves_per_position=args.max_moves_per_position,
            resignation_threshold=args.resignation_threshold,
            consecutive_low_moves=args.consecutive_low_moves,
            initial_prob_threshold=args.initial_prob_threshold,
            transition_moves=args.transition_moves
        )
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 