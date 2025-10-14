#!/usr/bin/env python3
"""Script to play N games using 1-visit KataGo neural network.

This script creates complete games by having the KataGo neural network play
against itself using only single-visit evaluations (no search tree). Games
are saved as SGF files.

Usage:
    python play_games.py <model_path> <num_games> [options]
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple
import random
import time
import uuid

# Add python directory to path for KataGo modules
sys.path.append(str(Path(__file__).parent.parent / "python"))

from load_model import load_model
from gamestate import GameState, Board
from sgfmill import sgf

# Import common utilities
from common_utils import (
    get_device, select_move_with_sampling, loc_to_sgf_coords, create_sgf
)






def play_single_game(model, game_id: int, board_size: int = 19, device: str = "auto", prob_threshold: float = 0.01) -> Tuple[str, str]:
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
    
    print(f"Starting game {game_id}...")
    
    for move_number in range(max_moves):
        current_player = gs.board.pla
        player_str = "Black" if current_player == Board.BLACK else "White"
        
        # Get model outputs for current position
        outputs = gs.get_model_outputs(model)
        moves_and_probs = outputs["moves_and_probs0"]
        
        if not moves_and_probs:
            print(f"No legal moves available for {player_str} at move {move_number}")
            break
        
        # For 1-visit play, sample from moves within prob_threshold of the best move
        best_move, best_prob, was_sampled = select_move_with_sampling(moves_and_probs, prob_threshold)
        
        # Play the move
        gs.play(current_player, best_move)
        moves.append((current_player, best_move))
        
        # Check for pass
        if best_move == Board.PASS_LOC:
            consecutive_passes += 1
            sampling_info = " [sampled]" if was_sampled else ""
            print(f"Move {move_number + 1}: {player_str} passes (prob: {best_prob:.3f}){sampling_info}")
        else:
            consecutive_passes = 0
            move_str = loc_to_sgf_coords(best_move, gs.board)
            sampling_info = " [sampled]" if was_sampled else ""
            print(f"Move {move_number + 1}: {player_str} plays {move_str} (prob: {best_prob:.3f}){sampling_info}")
        
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




def play_n_games(
    model_path: Path,
    num_games: int,
    output_dir: Path,
    board_size: int = 19,
    device: str = "auto",
    prob_threshold: float = 0.01
) -> None:
    """Play N games and save them as SGF files."""
    
    # Auto-detect device if needed
    actual_device = get_device(device)
    print(f"Loading model from {model_path}...")
    print(f"Using device: {actual_device}")
    model, _, _ = load_model(model_path, use_swa=False, device=actual_device, pos_len=board_size, verbose=False)
    
    output_dir.mkdir(exist_ok=True)
    
    print(f"Playing {num_games} games...")
    
    for game_id in range(1, num_games + 1):
        try:
            sgf_content, result = play_single_game(model, game_id, board_size, device, prob_threshold)
            
            # Save SGF file
            sgf_file = output_dir / f"{uuid.uuid4()}.sgf"
            sgf_file.write_text(sgf_content, encoding='utf-8')
            
            print(f"✓ {result}")
            print(f"  Saved: {sgf_file}")
            
        except Exception as e:
            print(f"✗ Error in game {game_id}: {e}")
            continue
    
    print(f"\nCompleted! Games saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play N games using 1-visit KataGo neural network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python play_games.py model.ckpt 10
  python play_games.py model.ckpt 5 --output-dir games --board-size 13
  python play_games.py model.ckpt 1 --device cpu
  python play_games.py model.ckpt 3 --prob-threshold 0.02  # 2% threshold for more variety
        """
    )
    
    parser.add_argument("model", type=Path, help="Path to KataGo model checkpoint")
    parser.add_argument("num_games", type=int, help="Number of games to play")
    parser.add_argument("--output-dir", type=Path, default="games", 
                       help="Directory to save SGF files (default: games)")
    parser.add_argument("--board-size", type=int, default=19, choices=[9, 13, 19],
                       help="Board size (default: 19)")
    parser.add_argument("--device", type=str, default="auto", 
                       help="PyTorch device (default: auto - will use mps/cuda/cpu as available)")
    parser.add_argument("--prob-threshold", type=float, default=0.01,
                       help="Probability threshold for move sampling (default: 0.01 = 1%%)")
    
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
        play_n_games(args.model, args.num_games, args.output_dir, args.board_size, actual_device, args.prob_threshold)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
    
# Example usage:
# python play_games.py model.ckpt 5
# python play_games.py model.ckpt 3 --prob-threshold 0.02 