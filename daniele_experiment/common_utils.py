#!/usr/bin/env python3
"""
Common utilities for game generation and analysis scripts.

This module contains shared functions used across multiple scripts to avoid code duplication.
"""

import random
import time
from pathlib import Path
from typing import List, Tuple

import torch
from sgfmill import sgf

# Add python directory to path for KataGo modules
import sys
sys.path.append(str(Path(__file__).parent.parent / "python"))

from board import Board


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


def select_move_with_sampling(moves_and_probs: List[Tuple[int, float]], prob_threshold: float = 0.01) -> Tuple[int, float, bool]:
    """Select a move by sampling from moves within prob_threshold of the best move.
    
    Args:
        moves_and_probs: List of (move, probability) tuples
        prob_threshold: Probability threshold (default 1% = 0.01)
    
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


def select_move_with_temperature(
    moves_and_probs: List[Tuple[int, float]], 
    temperature: float = 1.0,
    min_prob: float = 0.01,
    top_k: int = 10
) -> Tuple[int, float, bool]:
    """Select a move using temperature-based sampling with safety filters.
    
    Conservative approach: applies top-k filtering and minimum probability 
    threshold BEFORE temperature scaling to avoid selecting bad moves.
    
    Args:
        moves_and_probs: List of (move, probability) tuples
        temperature: Sampling temperature (1.0 = policy distribution, 
                     <1.0 = more deterministic, >1.0 = more uniform)
        min_prob: Minimum probability to consider a move (default 1%)
        top_k: Maximum number of top moves to consider (default 10)
    
    Returns:
        Tuple of (selected_move, its_probability, was_sampled)
    """
    import numpy as np
    
    if not moves_and_probs:
        raise ValueError("No moves available")
    
    # Sort by probability descending
    sorted_moves = sorted(moves_and_probs, key=lambda x: x[1], reverse=True)
    
    # Filter: keep only moves with prob >= min_prob AND limit to top_k
    candidates = []
    for move, prob in sorted_moves[:top_k]:
        if prob >= min_prob:
            candidates.append((move, prob))
    
    # Fallback: if no candidates pass threshold, take the best move
    if not candidates:
        return sorted_moves[0][0], sorted_moves[0][1], False
    
    # If only one candidate or temperature is 0, return best
    if len(candidates) == 1 or temperature == 0:
        return candidates[0][0], candidates[0][1], False
    
    moves, probs = zip(*candidates)
    probs = np.array(probs, dtype=np.float64)
    
    # Apply temperature scaling
    # temp < 1: sharpen (more deterministic), temp > 1: flatten (more diverse)
    scaled = np.power(probs, 1.0 / temperature)
    scaled /= scaled.sum()
    
    # Sample
    selected_idx = np.random.choice(len(moves), p=scaled)
    selected_move = moves[selected_idx]
    original_prob = probs[selected_idx]
    
    # Was it the best move?
    was_sampled = selected_idx != 0
    
    return selected_move, float(original_prob), was_sampled


def calculate_temperature(
    move_number: int, 
    initial_temp: float = 1.2, 
    final_temp: float = 0.8, 
    transition_moves: int = 60
) -> float:
    """Calculate temperature that decays as game progresses.
    
    Conservative defaults:
        - Start at 1.2 (modest diversity boost)
        - End at 0.8 (slightly favor best move in endgame)
        - Transition over 60 moves
    
    Args:
        move_number: Current move number (0-based)
        initial_temp: Starting temperature (default 1.2)
        final_temp: Final temperature (default 0.8)
        transition_moves: Number of moves to transition (default 60)
    
    Returns:
        Current temperature
    """
    if move_number >= transition_moves:
        return final_temp
    
    # Linear interpolation
    progress = move_number / transition_moves
    return initial_temp - (initial_temp - final_temp) * progress


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


def _loc_to_sgf(loc: int, board: Board) -> str:
    """Convert an internal ``loc`` to an SGF coordinate string."""
    if loc == Board.PASS_LOC:
        return "pass"
    x = board.loc_x(loc)
    y = board.loc_y(loc)
    from sgfmill import common as sgf_common
    return sgf_common.format_vertex((y, x))


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


def convert_numpy_to_python(obj):
    """Convert numpy types and other non-JSON-serializable types to Python native types."""
    import numpy as np
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, )):
        return int(obj)
    elif isinstance(obj, (np.floating, )):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_python(v) for v in obj]
    elif isinstance(obj, set):
        return list(obj)
    elif hasattr(obj, '__dict__'):
        # Handle custom objects by converting to dict
        return {k: convert_numpy_to_python(v) for k, v in obj.__dict__.items()}
    return obj
