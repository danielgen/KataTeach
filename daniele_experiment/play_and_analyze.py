#!/usr/bin/env python3
"""Script to play games and analyze them using 1-visit KataGo neural network.

This script plays N games using the KataGo neural network and automatically
analyzes each game, saving both SGF files and policy analysis.

Usage:
    python play_and_analyze.py <model_path> <num_games> [options]
"""

import argparse
import json
import sys
import time
import uuid
import random
from pathlib import Path
from typing import Dict, List, Tuple

from sgfmill import sgf, sgf_moves, common as sgf_common

# Add python directory to path for KataGo modules
sys.path.append(str(Path(__file__).parent.parent / "python"))

from load_model import load_model
from gamestate import GameState, Board
from query_analysis_engine_example import KataGo, sgfmill_to_str


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


def board_to_initial_stones(board: Board) -> List[Tuple[str, str]]:
    """Convert a Board to KataGo initialStones format."""
    stones: List[Tuple[str, str]] = []
    for y in range(board.size):
        for x in range(board.size):
            loc = board.loc(x, y)
            pla = board.board[loc]
            if pla == Board.BLACK:
                stones.append(("B", sgfmill_to_str((y, x))))
            elif pla == Board.WHITE:
                stones.append(("W", sgfmill_to_str((y, x))))
    return stones


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


def compute_policy_analysis(
    sgf_content: str,
    model,
    katago: KataGo,
    *,
    threshold: float = -0.005,
    verbose: bool = True
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
    for idx in range(len(plays) + 1):
        outputs = gs.get_model_outputs(model)
        moves_probs = outputs["moves_and_probs0"]
        if not moves_probs:
            break
        
        # First filter moves by policy probability - only evaluate top moves
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
        
        if verbose:
            print(f"Position {idx}: Evaluating {len(candidate_moves)} out of {len(moves_probs)} legal moves")
        
        # Evaluate each candidate move by querying KataGo on the resulting position
        move_winrates: List[Tuple[int, float]] = []
        current_player = gs.board.pla
        next_player = Board.get_opp(current_player)
        komi = gs.rules.get("whiteKomi", 7.5)

        derived_positions: List[Tuple[int, float, Dict[str, object]]] = []
        for mv, prob in candidate_moves:
            board_copy = gs.board.copy()
            board_copy.play(current_player, mv)
            stones = board_to_initial_stones(board_copy)
            query = {
                "id": str(uuid.uuid4()),
                "initialStones": stones,
                "initialPlayer": "B" if next_player == Board.BLACK else "W",
                "rules": "Chinese",
                "komi": komi,
                "boardXSize": board_size,
                "boardYSize": board_size,
                "includePolicy": True,
                "maxVisits": 1,
            }
            derived_positions.append((mv, prob, query))

        for mv, prob, query in derived_positions:
            try:
                result = katago.query_raw(query)
                our_winrate = float(result["rootInfo"]["rawWinrate"])
                move_winrates.append((mv, our_winrate))
            except Exception:
                # Fallback to policy probability if evaluation fails
                move_winrates.append((mv, prob))
        
        # Find best winrate and collect moves within threshold
        position_data = {}
        if move_winrates:
            best_winrate = max(winrate for _mv, winrate in move_winrates)
            top: List[MoveInfo] = []
            for mv, winrate in move_winrates:
                if winrate >= best_winrate + threshold:
                    top.append({"move": _loc_to_sgf(mv, gs.board), "winrate": float(winrate)})
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
            if actual_move_loc is not None:
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
            
            # Store actual move information
            position_data["actual_move"] = {
                "move": actual_move_sgf,
                "winrate": actual_move_winrate,
                "player": color
            }
            
            # Now play the actual move to advance the game state
            gs.play(pla, actual_move_loc)
        
        # Only store position data if we have either suggestions or actual move info
        if position_data:
            policy[idx] = position_data

    return policy


def play_single_game(model, game_id: int, board_size: int = 19, prob_threshold: float = 0.01) -> Tuple[str, str]:
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
    katago: KataGo,
    num_games: int,
    output_dir: Path,
    board_size: int = 19,
    prob_threshold: float = 0.01,
    analysis_threshold: float = -0.005,
) -> None:
    """Play N games, save them as SGF files, and analyze them."""
    
    output_dir.mkdir(exist_ok=True)
    policy_dir = output_dir / "policy"
    policy_dir.mkdir(exist_ok=True)
    
    print(f"Playing and analyzing {num_games} games...")
    
    for game_id in range(1, num_games + 1):
        try:
            # Play the game
            sgf_content, result = play_single_game(model, game_id, board_size, prob_threshold)
            
            # Save SGF file
            sgf_file = output_dir / f"{uuid.uuid4()}.sgf"
            sgf_file.write_text(sgf_content, encoding='utf-8')
            
            print(f"✓ {result}")
            print(f"  SGF saved: {sgf_file}")
            
            # Analyze the game
            print(f"  Analyzing game {game_id}...")
            policy = compute_policy_analysis(sgf_content, model, katago, threshold=analysis_threshold, verbose=False)
            
            # Save policy analysis
            policy_file = policy_dir / f"{sgf_file.stem}.json"
            save_combined_data(sgf_content, policy, policy_file)
            print(f"  Analysis saved: {policy_file}")
            
        except Exception as e:
            print(f"✗ Error in game {game_id}: {e}")
            continue
    
    print(f"\nCompleted! Games saved to {output_dir}, analysis in {policy_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play games and analyze them using 1-visit KataGo neural network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Play and analyze 5 new games
  python play_and_analyze.py model.ckpt 5
  
  # Play games with custom settings
  python play_and_analyze.py model.ckpt 3 --output-dir my_games --board-size 13 --prob-threshold 0.02
  
  # Use CPU instead of GPU
  python play_and_analyze.py model.ckpt 2 --device cpu
        """
    )
    
    parser.add_argument("model", type=Path, help="Path to KataGo model checkpoint")
    parser.add_argument("num_games", type=int, help="Number of games to play and analyze")
    parser.add_argument("--output-dir", type=Path, default="games", 
                       help="Directory to save SGF files (default: games)")
    parser.add_argument("--board-size", type=int, default=19, choices=[9, 13, 19],
                       help="Board size (default: 19)")
    parser.add_argument("--device", type=str, default="cuda", 
                       help="PyTorch device (default: cuda)")
    parser.add_argument("--prob-threshold", type=float, default=0.01,
                       help="Probability threshold for move sampling (default: 0.01 = 1%%)")
    parser.add_argument("--analysis-threshold", type=float, default=-0.005,
                       help="Winrate drop threshold for policy analysis (default: -0.005)")
    parser.add_argument("--katago-binary", type=Path, default="katago",
                       help="Path to KataGo analysis binary (default: katago)")
    parser.add_argument("--katago-config", type=Path,
                       default=Path("cpp/configs/analysis_example.cfg"),
                       help="Path to KataGo analysis config")
    parser.add_argument("--katago-model", type=Path, default=None,
                       help="Path to KataGo model for analysis (default: --model)")
    
    args = parser.parse_args()
    
    if not args.model.exists():
        print(f"Error: Model file {args.model} does not exist")
        sys.exit(1)
    
    if args.num_games <= 0:
        print("Error: Number of games must be positive")
        sys.exit(1)
    
    katago = None
    try:
        print(f"Loading model from {args.model}...")
        model, _, _ = load_model(args.model, use_swa=False, device=args.device, pos_len=19, verbose=False)

        katago_model_path = args.katago_model if args.katago_model is not None else args.model
        katago = KataGo(str(args.katago_binary), str(args.katago_config), str(katago_model_path))

        play_and_analyze_games(
            model=model,
            katago=katago,
            num_games=args.num_games,
            output_dir=args.output_dir,
            board_size=args.board_size,
            prob_threshold=args.prob_threshold,
            analysis_threshold=args.analysis_threshold
        )

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        if katago is not None:
            katago.close()


if __name__ == "__main__":
    main() 