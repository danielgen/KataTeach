#!/usr/bin/env python3
"""
KataGo Outputs Visualization with Custom Board Rendering

This script plays a short game and generates an HTML visualization
showing all KataGo model outputs at each move using custom board rendering.
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np

# Add the python directory to the path so we can import KataGo modules
sys.path.append(str(Path(__file__).parent.parent / "python"))

from board import Board
from gamestate import GameState
from model_pytorch import Model
from load_model import load_model
import torch

# Import common utilities
from common_utils import get_device as get_device_str, convert_numpy_to_python

# Import snorkel analysis
from snorkel_board_positions import analyze_position_comprehensive, urgency_by_region, urgency_intensity_by_region

def get_device():
    """Get the appropriate device for model inference."""
    device_str = get_device_str()
    return torch.device(device_str)


def play_short_game(model, max_moves=10):
    """Play a short game and capture KataGo outputs at each step."""
    print(f"Playing a short game with max {max_moves} moves...")
    
    # Initialize game state
    board_size = 19
    game_state = GameState(board_size, GameState.RULES_TT)
    
    # Diverse play and resignation settings
    initial_prob_threshold = 0.05  # early game more diversity
    final_prob_threshold = 0.01    # late game more greedy
    transition_moves = 50          # moves over which to transition
    resignation_threshold = 0.10   # 10% winrate
    consecutive_low_moves = 3      # consecutive low-winrate moves per player before resigning
    consecutive_lows_by_player = {Board.BLACK: 0, Board.WHITE: 0}

    game_data = []
    moves = []
    ownership_before = None
    pass_counterfactual = None  # Track pass counterfactual ownership
    
    # Add initial position
    initial_outputs = game_state.get_model_outputs(model)
    converted_initial_outputs = convert_numpy_to_python(initial_outputs)
    
    # Add snorkel analysis for initial position
    try:
        # Convert ownership to numpy array (19x19)
        ownership = np.array(initial_outputs.get("ownership", [0.0] * 361)).reshape(19, 19)
        policy = np.array(initial_outputs.get("policy0", [0.0] * 361))
        
        analysis = analyze_position_comprehensive(
            board=game_state.board,
            ownership=ownership,
            policy=policy,
            player=game_state.board.pla,  # Use board.pla for consistency
            move_loc=None,
            last_move_loc=None,
            before_ownership=None,
            before_board=None
        )
        analysis_serializable = convert_numpy_to_python(analysis)
    except Exception as e:
        print(f"Warning: Snorkel analysis failed for initial position: {e}")
        analysis_serializable = {}
    
    game_data.append({
        "move_number": 0,
        "player": "Initial",
        "last_move": None,
        "board_state": [0] * game_state.board.arrsize,  # Empty board with full size
        "analysis": analysis_serializable,
        "pass_counterfactual": None,  # No counterfactual for initial position
        **converted_initial_outputs
    })
    
    # Play moves
    for move_num in range(1, max_moves + 1):
        # Get current player
        current_player = game_state.board.pla
        player_str = "Black" if current_player == Board.BLACK else "White"
        
        # Get model outputs for current position
        outputs = game_state.get_model_outputs(model)
        
        # Store ownership before the move for analysis
        ownership_before = np.array(outputs.get("ownership", [0.0] * 361)).reshape(19, 19)
        
        # BEFORE you play, capture the board state
        board_before = game_state.board.copy()
        
        # Compute pass counterfactual ownership BEFORE playing the actual move
        # This gives us "what would ownership be if the player passed?"
        # Used to avoid anticipatory ownership issues in territory concepts.
        pass_counterfactual = None
        pass_counterfactual_frame_player = None
        try:
            # Play a pass move temporarily to compute counterfactual
            game_state.play(current_player, Board.PASS_LOC)
            
            # Get ownership after the pass
            counterfactual_outputs = game_state.get_model_outputs(model)
            pass_counterfactual = np.array(counterfactual_outputs.get("ownership", [0.0] * 361)).reshape(19, 19)
            
            # After pass, ownership is from the opponent's perspective (who is to play)
            pass_counterfactual_frame_player = game_state.board.pla
            
            # Undo the pass to restore original state
            game_state.undo()
            print(f"  Computed pass counterfactual: shape={pass_counterfactual.shape}, min={pass_counterfactual.min():.3f}, max={pass_counterfactual.max():.3f}")
        except Exception as e:
            # If counterfactual computation fails, try to recover state
            if game_state.can_undo() and len(game_state.moves) > 0 and game_state.moves[-1][1] == Board.PASS_LOC:
                game_state.undo()
            print(f"    Warning: Pass counterfactual failed for move {move_num}: {e}")
            import traceback
            traceback.print_exc()
            pass_counterfactual = None
            pass_counterfactual_frame_player = None
        
        # Always use the main policy (policy0) for move selection
        # policy0 = current player's move distribution
        # policy1 = opponent's predicted next move distribution (not used for selection)
        moves_and_probs = outputs["moves_and_probs0"]
        
        if not moves_and_probs:
            print(f"No legal moves available for {player_str} at move {move_num}")
            break
        
        # Use sampling to get more realistic moves (like in play_games.py)
        import random
        
        # Dynamic probability threshold that decreases over time
        if move_num - 1 >= transition_moves:
            current_threshold = final_prob_threshold
        else:
            progress = (move_num - 1) / transition_moves
            current_threshold = initial_prob_threshold - (initial_prob_threshold - final_prob_threshold) * progress

        # Find the best probability
        best_prob = max(prob for _, prob in moves_and_probs)
        
        # Find all moves within current_threshold of the best move
        candidate_moves = []
        for move, prob in moves_and_probs:
            if prob >= best_prob - current_threshold:
                candidate_moves.append((move, prob))
        
        # If only one candidate, return it (no sampling)
        if len(candidate_moves) == 1:
            best_move, best_prob = candidate_moves[0]
        else:
            # Sample from candidates using their probabilities as weights
            candidate_moves_list, candidate_probs = zip(*candidate_moves)
            
            # Normalize probabilities within the candidate set
            total_prob = sum(candidate_probs)
            normalized_probs = [p / total_prob for p in candidate_probs]
            
            # Sample based on normalized probabilities
            selected_idx = random.choices(range(len(candidate_moves_list)), weights=normalized_probs)[0]
            best_move = candidate_moves_list[selected_idx]
            best_prob = candidate_probs[selected_idx]
        
        # Debug: print top moves
        print(f"  Top 5 moves: {moves_and_probs[:5]}")
        print(f"  Selected move: {best_move} (prob: {best_prob:.6f}, threshold: {current_threshold:.3f})")
        
        # Play the move
        game_state.play(current_player, best_move)
        moves.append((current_player, best_move))
        
        # Get board state AFTER playing the move
        # Use the full board size including walls and padding (arrsize = (19+1)*(19+2)+1 = 420)
        board_state = [0] * game_state.board.arrsize
        for i in range(game_state.board.arrsize):
            if game_state.board.board[i] == 1:
                board_state[i] = 1  # Black
            elif game_state.board.board[i] == 2:
                board_state[i] = -1  # White
        
        # Get outputs after the move
        post_move_outputs = game_state.get_model_outputs(model)
        # Compute winrate for the move just played (perspective flipped)
        try:
            opponent_winrate = float(post_move_outputs["value"][0])
            our_winrate = 1.0 - opponent_winrate
        except Exception:
            # Fallback: use position evaluation before the move
            our_winrate = float(outputs["value"][0])
        
        # Resignation logic: track consecutive low winrates per player
        if our_winrate < resignation_threshold:
            consecutive_lows_by_player[current_player] += 1
        else:
            consecutive_lows_by_player[current_player] = 0
        
        if consecutive_lows_by_player[current_player] >= consecutive_low_moves:
            print(f"{player_str} resigns: winrate < {resignation_threshold:.0%} for {consecutive_low_moves} consecutive moves")
            # Store the post-move data before breaking
            converted_outputs = convert_numpy_to_python(post_move_outputs)
            
            # Add snorkel analysis
            try:
                ownership_after = np.array(post_move_outputs.get("ownership", [0.0] * 361)).reshape(19, 19)
                policy = np.array(post_move_outputs.get("policy0", [0.0] * 361))
                last_move_loc = moves[-1][1] if moves else None
                
                # Analyze from the perspective of the player who MADE the move
                # Raw ownership from get_model_outputs is from current player to move's perspective
                # After move: game_state.board.pla is opponent (who is to play next)
                # ownership_after is from game_state.board.pla's perspective
                # ownership_before was captured when current_player was to play, so it's from current_player's perspective
                post_move_player = game_state.board.pla  # Who is to play next (opponent)
                
                # Sanity check logging
                print(f"Move {move_num} (resignation): pre_player={board_before.pla}, post_player={post_move_player}")
                
                analysis = analyze_position_comprehensive(
                    board=game_state.board,            # post-move board
                    ownership=ownership_after,         # post-move ownership (from post_move_player's perspective)
                    policy=policy,                     # post-move policy0
                    player=current_player,             # The player who MADE the move
                    move_loc=best_move,
                    last_move_loc=last_move_loc,
                    before_ownership=ownership_before, # pre-move ownership (from current_player's perspective)
                    before_board=board_before,         # pre-move board (for deltas/attack)
                    ownership_frame_player=post_move_player,  # Frame of ownership_after
                    pass_counterfactual_ownership=pass_counterfactual,  # "What if I passed?"
                    pass_counterfactual_frame_player=pass_counterfactual_frame_player
                )
                analysis_serializable = convert_numpy_to_python(analysis)
                
                # Additional sanity checks
                if 'building_count' in analysis and 'reduction_count' in analysis:
                    print(f"  Territory analysis: building={analysis['building_count']}, reduction={analysis['reduction_count']}")
                if 'invasion' in analysis:
                    print(f"  Invasion: {analysis['invasion']}, intensity={analysis.get('invasion_intensity', 0):.3f}")
                if analysis.get('used_pass_counterfactual'):
                    print(f"  Using pass counterfactual for territory concepts")
            except Exception as e:
                print(f"Warning: Snorkel analysis failed for move {move_num}: {e}")
                analysis_serializable = {}
            
            game_data.append({
                "move_number": move_num,
                "player": player_str,
                "last_move": (current_player, best_move),
                "board_state": board_state,
                "analysis": analysis_serializable,
                "pass_counterfactual": pass_counterfactual.tolist() if pass_counterfactual is not None else None,
                **converted_outputs
            })
            
            # Store urgency for the next move (since urgency is computed before the move)
            # Get policy before the move for urgency calculation
            pre_move_policy = np.array(outputs.get("policy0", [0.0] * 361))
            urgency_data = {
                "urgency": convert_numpy_to_python(urgency_by_region(pre_move_policy)),
                "urgency_intensity": convert_numpy_to_python(urgency_intensity_by_region(pre_move_policy))
            }
            # Store this for the next move's analysis
            if len(game_data) > 0:
                game_data[-1]["next_move_urgency"] = urgency_data
            break
        
        # Convert numpy arrays to Python objects for JSON serialization
        converted_outputs = convert_numpy_to_python(post_move_outputs)
        
        # Add snorkel analysis
        try:
            ownership_after = np.array(post_move_outputs.get("ownership", [0.0] * 361)).reshape(19, 19)
            policy = np.array(post_move_outputs.get("policy0", [0.0] * 361))
            last_move_loc = moves[-2][1] if len(moves) > 1 else None  # Previous move
            
            # Analyze from the perspective of the player who MADE the move
            # Raw ownership from get_model_outputs is from current player to move's perspective
            # After move: game_state.board.pla is opponent (who is to play next)
            # ownership_after is from game_state.board.pla's perspective
            # ownership_before was captured when current_player was to play, so it's from current_player's perspective
            post_move_player = game_state.board.pla  # Who is to play next (opponent)
            
            # Sanity check logging
            print(f"Move {move_num}: pre_player={board_before.pla}, post_player={post_move_player}")
            
            analysis = analyze_position_comprehensive(
                board=game_state.board,            # post-move board
                ownership=ownership_after,         # post-move ownership (from post_move_player's perspective)
                policy=policy,                     # post-move policy0
                player=current_player,             # The player who MADE the move
                move_loc=best_move,
                last_move_loc=last_move_loc,
                before_ownership=ownership_before, # pre-move ownership (from current_player's perspective)
                before_board=board_before,         # pre-move board (for deltas/attack)
                ownership_frame_player=post_move_player,  # Frame of ownership_after
                pass_counterfactual_ownership=pass_counterfactual,  # "What if I passed?"
                pass_counterfactual_frame_player=pass_counterfactual_frame_player
            )
            analysis_serializable = convert_numpy_to_python(analysis)
            
            # Additional sanity checks
            if 'building_count' in analysis and 'reduction_count' in analysis:
                print(f"  Territory analysis: building={analysis['building_count']}, reduction={analysis['reduction_count']}")
            if 'invasion' in analysis:
                print(f"  Invasion: {analysis['invasion']}, intensity={analysis.get('invasion_intensity', 0):.3f}")
            if analysis.get('used_pass_counterfactual'):
                print(f"  Using pass counterfactual for territory concepts")
        except Exception as e:
            print(f"Warning: Snorkel analysis failed for move {move_num}: {e}")
            analysis_serializable = {}
        
        # Store game data
        game_data.append({
            "move_number": move_num,
            "player": player_str,
            "last_move": (current_player, best_move),
            "board_state": board_state,
            "analysis": analysis_serializable,
            "pass_counterfactual": pass_counterfactual.tolist() if pass_counterfactual is not None else None,
            **converted_outputs
        })
        
        # Store urgency for the next move (since urgency is computed before the move)
        if move_num < max_moves:  # Don't store for the last move
            # Get policy before the move for urgency calculation
            pre_move_policy = np.array(outputs.get("policy0", [0.0] * 361))
            urgency_data = {
                "urgency": convert_numpy_to_python(urgency_by_region(pre_move_policy)),
                "urgency_intensity": convert_numpy_to_python(urgency_intensity_by_region(pre_move_policy))
            }
            # Store this for the next move's analysis
            if len(game_data) > 0:
                game_data[-1]["next_move_urgency"] = urgency_data
        
        # Print move information
        if best_move == Board.PASS_LOC:
            print(f"Move {move_num}: {player_str} passes (prob: {best_prob:.3f})")
        else:
            # Convert to SGF coordinates for display
            x = game_state.board.loc_x(best_move)
            y = game_state.board.loc_y(best_move)
            sgf_coord = f"{chr(ord('a') + x)}{chr(ord('a') + y)}"
            print(f"Move {move_num}: {player_str} plays at {best_move} ({sgf_coord}) (prob: {best_prob:.3f}, winrate: {our_winrate:.1%})")
        
        # Debug: print board state
        black_count = sum(1 for x in board_state if x == 1)
        white_count = sum(1 for x in board_state if x == -1)
        print(f"  Board state: {black_count} black, {white_count} white stones")
        
        # Debug: check what values are actually in the board
        unique_values = set(game_state.board.board)
        print(f"  Board values: {unique_values}")
        
        # Debug: check the specific location that was just played
        if best_move != Board.PASS_LOC:
            print(f"  Location {best_move} has value: {game_state.board.board[best_move]}")
        
        # Debug: print all non-zero board positions
        non_zero_positions = []
        for i in range(game_state.board.arrsize):
            if game_state.board.board[i] != 0:
                x = game_state.board.loc_x(i)
                y = game_state.board.loc_y(i)
                if 0 <= x < 19 and 0 <= y < 19:  # Only show valid board positions
                    sgf_coord = f"{chr(ord('a') + x)}{chr(ord('a') + y)}"
                    non_zero_positions.append(f"{i}({sgf_coord})={game_state.board.board[i]}")
        print(f"  Non-zero positions: {non_zero_positions}")
        
        # Debug: check bottom row specifically
        bottom_row_stones = []
        for x in range(19):
            loc = game_state.board.loc(x, 0)  # Bottom row (y=0)
            if game_state.board.board[loc] != 0:
                bottom_row_stones.append(f"x={x}, loc={loc}, value={game_state.board.board[loc]}")
        if bottom_row_stones:
            print(f"  Bottom row stones: {bottom_row_stones}")
        else:
            print("  No stones on bottom row")
    
    # Generate SGF content
    sgf_content = generate_sgf(moves)
    
    return game_data, sgf_content

def generate_sgf(moves):
    """Generate SGF content from moves."""
    sgf_lines = [
        "(;FF[4]CA[UTF-8]AP[KataGo:1.0]ST[2]",
        "RU[Chinese]SZ[19]KM[7.5]",
        "PW[White]PB[Black]"
    ]
    
    # Create a temporary board for coordinate conversion
    temp_board = Board(19)
    
    for player, move in moves:
        if move == Board.PASS_LOC:
            move_str = "[]"
        else:
            # Convert to SGF format
            x = temp_board.loc_x(move)
            y = temp_board.loc_y(move)
            move_str = f"[{chr(ord('a') + x)}{chr(ord('a') + y)}]"
        
        color = "B" if player == Board.BLACK else "W"
        sgf_lines.append(f";{color}{move_str}")
    
    sgf_lines.append(")")
    return "\n".join(sgf_lines)


def generate_html_visualization(game_data, sgf_content, output_file, global_stats=None):
    """Generate HTML visualization with custom board rendering.
    
    Args:
        game_data: List of position data dicts
        sgf_content: SGF string for the game
        output_file: Path to write HTML file
        global_stats: Optional dict with global statistics for percentile computation
    """
    
    # Reduce data size by keeping only essential fields for visualization
    print(f"Converting {len(game_data)} positions to JSON...")
    
    # Create a simplified version of game_data with only essential fields
    simplified_data = []
    positions_with_analysis = 0
    for pos in game_data:
        analysis_data = pos.get("analysis", {})
        if analysis_data:
            positions_with_analysis += 1
        
        # Determine if we need to flip ownership/scoring/futurepos to White's perspective
        # Raw model output is from current player's perspective (who is to play next)
        # After Black plays -> White to play -> ownership is from White's perspective (no flip)
        # After White plays -> Black to play -> ownership is from Black's perspective (flip needed)
        # player field indicates who just played
        player = pos.get("player", "Unknown")
        need_flip = (player == "White")  # After White plays, Black is to move, so flip to White's perspective
        
        def flip_nested_array(arr):
            """Flip sign of all values in a potentially nested array structure."""
            if not arr:
                return arr
            if isinstance(arr, np.ndarray):
                return (-arr).tolist()
            # Check depth of nesting
            if isinstance(arr[0], (list, np.ndarray)):
                # 2D or 3D array
                if isinstance(arr[0], np.ndarray) or (isinstance(arr[0], list) and arr[0] and isinstance(arr[0][0], (list, np.ndarray))):
                    # 3D array: [batch][height][width]
                    return [[[-v for v in row] for row in channel] for channel in arr]
                else:
                    # 2D array: [height][width]
                    return [[-v for v in row] for row in arr]
            else:
                # 1D array
                return [-v for v in arr]
        
        # Get ownership and normalize to White's perspective
        ownership = pos.get("ownership", [])
        if need_flip and ownership:
            ownership = flip_nested_array(ownership)
        
        # Get scoring and normalize to White's perspective
        scoring = pos.get("scoring", [])
        if need_flip and scoring:
            scoring = flip_nested_array(scoring)
        
        # Get futurepos and normalize to White's perspective
        futurepos0 = pos.get("futurepos0", [])
        if need_flip and futurepos0:
            futurepos0 = flip_nested_array(futurepos0)
        
        futurepos1 = pos.get("futurepos1", [])
        if need_flip and futurepos1:
            futurepos1 = flip_nested_array(futurepos1)
        
        # Get pass counterfactual ownership and normalize to White's perspective
        # Pass counterfactual is "what if the player passed?" ownership
        # After a pass by current player (who just played), it's opponent's turn
        # So ownership perspective is from opponent's view
        # If Black just played, pass leads to White's turn -> ownership from White's perspective (no flip)
        # If White just played, pass leads to Black's turn -> ownership from Black's perspective (need flip)
        pass_counterfactual = pos.get("pass_counterfactual", None)
        if pass_counterfactual is not None:
            # Pass counterfactual: after player's pass, it's opponent's turn
            # If Black just played (player="Black"), opponent is White, so ownership is from White's perspective (no flip)
            # If White just played (player="White"), opponent is Black, so ownership is from Black's perspective (flip to White)
            need_flip_counterfactual = (player == "White")
            if need_flip_counterfactual:
                pass_counterfactual = flip_nested_array(pass_counterfactual)
        
        simplified_pos = {
            "move_number": pos.get("move_number", 0),
            "player": player,
            "last_move": pos.get("last_move"),
            "board_state": pos.get("board_state", []),
            "analysis": analysis_data,
            # Essential model outputs only (normalized to White's perspective)
            "policy0": pos.get("policy0", []),
            "policy1": pos.get("policy1", []),
            "ownership": ownership,
            "pass_counterfactual": pass_counterfactual,
            "scoring": scoring,
            "futurepos0": futurepos0,
            "futurepos1": futurepos1,
            "seki": pos.get("seki", []),
            "value": pos.get("value", [0.5]),
            "scoremean": pos.get("scoremean", 0.0),
            "lead": pos.get("lead", 0.0),
            "scorestdev": pos.get("scorestdev", 0.0),
            "vtime": pos.get("vtime", 0.0),
            "estv": pos.get("estv", 0.0),
            "td_value": pos.get("td_value", [0.5]),
            "td_value2": pos.get("td_value2", [0.5]),
            "td_value3": pos.get("td_value3", [0.5]),
        }
        simplified_data.append(simplified_pos)
    
    print(f"Positions with analysis data: {positions_with_analysis} out of {len(game_data)}")
    
    try:
        game_data_js = json.dumps(simplified_data)
        print(f"JSON serialization successful, length: {len(game_data_js)}")
    except Exception as e:
        print(f"JSON serialization failed: {e}")
        # Fallback: try with default=str
        game_data_js = json.dumps(simplified_data, default=str)
        print(f"JSON serialization with default=str successful, length: {len(game_data_js)}")
    
    sgf_js = json.dumps(sgf_content)
    
    # Embed global stats for percentile computation (or null if not available)
    global_stats_js = json.dumps(global_stats) if global_stats else "null"
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KataGo Outputs Visualization</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            padding-top: 20px;
            background-color: #f5f5f5;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        h1 {{
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }}
        
        .controls {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            text-align: center;
        }}
        
        .move-info {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }}
        
        .move-slider {{
            width: 100%;
            max-width: 600px;
            margin: 10px 0;
        }}
        
        .control-buttons {{
            margin-top: 15px;
        }}
        
        .control-buttons button {{
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            margin: 0 5px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }}
        
        .control-buttons button:hover {{
            background: #0056b3;
        }}
        
        .control-buttons button:disabled {{
            background: #ccc;
            cursor: not-allowed;
        }}
        
        .boards-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .board-section {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 20px;
        }}
        
        .board-section h3 {{
            margin-top: 0;
            color: #333;
            text-align: center;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }}
        
        .go-board {{
            width: 380px;
            height: 380px;
            margin: 0 auto;
            border: 2px solid #8B4513;
            border-radius: 4px;
            background: #DEB887;
            position: relative;
        }}
        
        
        .board-cell {{
            position: absolute;
            width: 20px;
            height: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: bold;
        }}
        
        .stone {{
            width: 18px;
            height: 18px;
            border-radius: 50%;
            border: 1px solid #333;
        }}
        
        .stone.black {{
            background: #000;
        }}
        
        .stone.white {{
            background: #fff;
            border: 1px solid #333;
        }}
        
        .label {{
            position: absolute;
            font-size: 10px;
            font-weight: bold;
            text-shadow: 1px 1px 1px rgba(255,255,255,0.8);
            z-index: 10;
        }}
        
        .info-panel {{
            margin-top: 15px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 4px;
            font-size: 14px;
        }}
        
        .heatmap-legend {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            justify-content: center;
        }}
        
        .heatmap-legend span {{
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            color: white;
            font-weight: bold;
        }}
        
        .value-display {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }}
        
        .value-item {{
            background: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            text-align: center;
        }}
        
        .value-item strong {{
            display: block;
            margin-bottom: 5px;
            color: #333;
        }}
        
        .sticky-container {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 1000;
            background: white;
            border-radius: 0 0 8px 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .sticky-section {{
            padding: 15px 20px;
            border-bottom: 1px solid #eee;
        }}
        
        .sticky-section:last-child {{
            border-bottom: none;
        }}
        
        #stickyValueContainer {{
            position: relative;
            width: 100%;
        }}
        
        .sticky-value-section h3 {{
            margin: 0 0 10px 0;
            text-align: center;
            font-size: 16px;
            color: #333;
        }}
        
        .detail-value {{
            color: #868e96;
            font-style: italic;
        }}
        
        .percentile {{
            font-size: 9px;
            color: #6c757d;
            font-weight: normal;
            margin-left: 2px;
        }}
        
        .feature-category {{
            margin-bottom: 8px;
            background: #fafbfc;
            border-radius: 4px;
            padding: 6px 10px;
            border: 1px solid #e1e4e8;
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 4px 12px;
        }}
        
        .category-header {{
            font-weight: 600;
            font-size: 11px;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 0.3px;
            margin-right: 8px;
            min-width: 100px;
        }}
        
        .feature-grid {{
            display: flex;
            flex-wrap: wrap;
            gap: 4px 16px;
            flex: 1;
        }}
        
        .feature-grid .sticky-value-item {{
            padding: 2px 0;
            font-size: 12px;
            background: none;
            display: flex;
            align-items: center;
            gap: 4px;
        }}
        
        .feature-grid .sticky-value-item strong {{
            font-size: 11px;
            margin-bottom: 0;
            color: #6c757d;
            font-weight: 500;
        }}
        
        .feature-grid .sticky-value-item strong::after {{
            content: ':';
        }}
        
        .feature-grid .sticky-value-item span {{
            font-weight: 600;
            color: #212529;
        }}
        
        .feature-explanation {{
            font-size: 9px;
            color: #999;
            font-style: italic;
            margin-left: 4px;
            font-weight: normal;
        }}
        
        .sticky-value-display {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }}
        
        .sticky-value-item {{
            background: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            text-align: center;
        }}
        
        .sticky-value-item strong {{
            display: block;
            margin-bottom: 5px;
            color: #333;
        }}
        
    </style>
</head>
<body>
    <div class="container">
        <h1>KataGo Model Outputs Visualization</h1>
        <div class="controls">
            <div class="move-info" id="moveInfo">Move 0: Initial Position</div>
            <input type="range" class="move-slider" id="moveSlider" min="0" max="{len(game_data)-1}" value="0">
            <div class="control-buttons">
                <button onclick="previousMove()">← Previous</button>
                <button onclick="nextMove()">Next →</button>
                <button onclick="toggleAutoPlay()" id="autoPlayBtn">Auto Play</button>
                <button onclick="resetToStart()">Reset</button>
            </div>
        </div>
        <div class="sticky-container" id="stickyValueContainer">
            <!-- Sticky sections will be generated by JavaScript -->
        </div>
        <div class="boards-container" id="boardsContainer">
            <!-- Boards will be generated by JavaScript -->
        </div>
    </div>
    
    <script>
        const gameData = {game_data_js};
        const globalStats = {global_stats_js};
        let currentMove = 0;
        let autoPlayInterval = null;
        
        // Compute percentile for a value given feature stats
        function computePercentile(value, featureName) {{
            if (!globalStats || !globalStats.features || !globalStats.features[featureName]) {{
                return null;
            }}
            const stats = globalStats.features[featureName];
            if (value === null || value === undefined) return null;
            
            // Linear interpolation between known percentiles
            const percentiles = [
                {{p: 0, v: stats.min}},
                {{p: 10, v: stats.p10}},
                {{p: 25, v: stats.p25}},
                {{p: 50, v: stats.p50}},
                {{p: 75, v: stats.p75}},
                {{p: 90, v: stats.p90}},
                {{p: 100, v: stats.max}}
            ];
            
            // Find where value falls
            for (let i = 0; i < percentiles.length - 1; i++) {{
                const lower = percentiles[i];
                const upper = percentiles[i + 1];
                if (value >= lower.v && value <= upper.v) {{
                    // Linear interpolation
                    if (upper.v === lower.v) return lower.p;
                    const fraction = (value - lower.v) / (upper.v - lower.v);
                    return Math.round(lower.p + fraction * (upper.p - lower.p));
                }}
            }}
            
            // Handle edge cases
            if (value < stats.min) return 0;
            if (value > stats.max) return 100;
            return 50;  // Default fallback
        }}
        
        // Format value with percentile
        function formatWithPercentile(value, featureName, decimals = 3) {{
            if (value === null || value === undefined) return '-';
            const formatted = typeof value === 'number' ? value.toFixed(decimals) : value;
            const percentile = computePercentile(value, featureName);
            if (percentile !== null) {{
                return `${{formatted}} <span class="percentile">(p${{percentile}})</span>`;
            }}
            return formatted;
        }}
        
        function updateDisplay() {{
            const data = gameData[currentMove];
            document.getElementById('moveInfo').textContent = 
                `Move ${{data.move_number}}: ${{data.player}} ${{data.last_move ? getMoveString(data.last_move) : 'Initial Position'}}`;
            document.getElementById('moveSlider').value = currentMove;
            
            updateBoards(data);
        }}
        
        function getMoveString(move) {{
            if (!move) return '';
            if (move[1] === 361) return 'Pass';
            // Convert KataGo location to x,y coordinates using the same logic as KataGo's loc_x/loc_y
            const dy = 20; // size + 1 = 19 + 1 = 20
            const x = (move[1] % dy) - 1;
            const y = Math.floor(move[1] / dy) - 1;
            return String.fromCharCode(97 + x) + String.fromCharCode(97 + y);
        }}
        
        function getMoveLocation(move) {{
            if (!move) return null;
            if (move[1] === 361) return null;
            // Convert KataGo location to x,y coordinates using the same logic as KataGo's loc_x/loc_y
            const dy = 20; // size + 1 = 19 + 1 = 20
            const x = (move[1] % dy) - 1;
            const y = Math.floor(move[1] / dy) - 1;
            return {{x: x, y: y}};
        }}
        
        function initializeBoards() {{
            const container = document.getElementById('boardsContainer');
            container.innerHTML = '';
            
            // Board state
            addBoardSection('Board State', 'board_state');
            
            // Policy outputs
            addBoardSection('Policy 0 (Current Player)', 'policy0');
            addBoardSection('Policy 1 (Next Player)', 'policy1');
            
            // Ownership
            addBoardSection('Ownership', 'ownership');
            
            // Pass Counterfactual Ownership (what if player passed?)
            addBoardSection('Pass Counterfactual (What If Pass?)', 'pass_counterfactual');
            
            // Scoring
            addBoardSection('Scoring', 'scoring');
            
            // Future position
            addBoardSection('Future Position 0', 'futurepos0');
            addBoardSection('Future Position 1', 'futurepos1');
            
            // Seki
            addBoardSection('Seki', 'seki');
            
        }}
        
        function addBoardSection(title, type) {{
            const container = document.getElementById('boardsContainer');
            const section = document.createElement('div');
            section.className = 'board-section';
            
            section.innerHTML = `
                <h3>${{title}}</h3>
                <div class="go-board" id="board-${{type}}"></div>
                <div class="info-panel" id="info-${{type}}"></div>
            `;
            
            container.appendChild(section);
        }}
        
        function addValueSection() {{
            const container = document.getElementById('stickyValueContainer');
            const section = document.createElement('div');
            section.className = 'sticky-section';
            section.id = 'value-section';
            
            section.innerHTML = `
                <h3>Value & Score Predictions</h3>
                <div class="sticky-value-display">
                    <div class="sticky-value-item">
                        <strong>Current Player</strong>
                        <span id="current-player">-</span>
                    </div>
                    <div class="sticky-value-item">
                        <strong>Win Rate</strong>
                        <span id="value-winrate">-</span>%
                    </div>
                    <div class="sticky-value-item">
                        <strong>Score Mean</strong>
                        <span id="score-mean">-</span>
                    </div>
                    <div class="sticky-value-item">
                        <strong>Lead</strong>
                        <span id="lead">-</span>
                    </div>
                    <div class="sticky-value-item">
                        <strong>Score StdDev</strong>
                        <span id="score-stdev">-</span>
                    </div>
                    <div class="sticky-value-item">
                        <strong>Variance Time</strong>
                        <span id="vtime">-</span>
                    </div>
                    <div class="sticky-value-item">
                        <strong>Value Error</strong>
                        <span id="estv">-</span>
                    </div>
                    <div class="sticky-value-item">
                        <strong>TD Value (Long)</strong>
                        <span id="value-td-long">-</span>%
                    </div>
                    <div class="sticky-value-item">
                        <strong>TD Value (Mid)</strong>
                        <span id="value-td-mid">-</span>%
                    </div>
                    <div class="sticky-value-item">
                        <strong>TD Value (Short)</strong>
                        <span id="value-td-short">-</span>%
                    </div>
                </div>
            `;
            
            container.appendChild(section);
        }}
        
        function addSnorkelSection() {{
            const container = document.getElementById('stickyValueContainer');
            const section = document.createElement('div');
            section.className = 'sticky-section';
            section.id = 'snorkel-section';
            
            section.innerHTML = `
                <h3>Snorkel Analysis (Current Move Effects)</h3>
                
                    <!-- Territory Section -->
                <div class="feature-category">
                    <div class="category-header">Territory</div>
                    <div class="feature-grid">
                        <div class="sticky-value-item" title="Change in weakly-owned empty points (ownership > 0.10)">
                            <strong>Potential Δ</strong>
                            <span class="feature-explanation">weak territory</span>
                        <span id="potential-territory">-</span>
                    </div>
                        <div class="sticky-value-item" title="Change in strongly-owned empty points (ownership >= 0.70)">
                            <strong>Solid Δ</strong>
                            <span class="feature-explanation">strong territory</span>
                        <span id="solid-territory">-</span>
                    </div>
                        <div class="sticky-value-item" title="Empty points that became owned (neutral → owned)">
                        <strong>Building</strong>
                        <span class="feature-explanation">new territory</span>
                        <span id="building-count">-</span>
                    </div>
                        <div class="sticky-value-item" title="Points already owned that became stronger">
                        <strong>Solidification</strong>
                        <span class="feature-explanation">strengthened</span>
                        <span id="solidification-count">-</span>
                    </div>
                        <div class="sticky-value-item" title="Opponent territory reduced">
                            <strong>Reduction</strong>
                            <span class="feature-explanation">opponent lost</span>
                            <span id="reduction-count">-</span>
                        </div>
                        <div class="sticky-value-item" title="Flipped opponent territory to own">
                            <strong>Invasion</strong>
                            <span class="feature-explanation">flipped to own</span>
                            <span id="invasion">-</span>
                        </div>
                        <div class="sticky-value-item detail-value" title="Intensities: building @ solidify @ reduce">
                            <strong>Intensities</strong>
                            <span class="feature-explanation">B:S:R</span>
                            <span id="territory-intensities">-</span>
                        </div>
                    </div>
                </div>
                
                <!-- Current Group Section (the group containing this move) -->
                <div class="feature-category">
                    <div class="category-header">This Move's Group</div>
                    <div class="feature-grid">
                        <div class="sticky-value-item" title="Current strength of the group containing this stone">
                            <strong>Strength</strong>
                            <span class="feature-explanation">ownership</span>
                            <span id="current-group-strength">-</span>
                        </div>
                        <div class="sticky-value-item" title="Change in strength of this group">
                            <strong>Strength Δ</strong>
                            <span class="feature-explanation">change</span>
                            <span id="current-group-strength-delta">-</span>
                        </div>
                        <div class="sticky-value-item" title="Current connectivity of the group containing this stone">
                            <strong>Connectivity</strong>
                            <span class="feature-explanation">nearby</span>
                            <span id="current-group-connectivity">-</span>
                        </div>
                        <div class="sticky-value-item" title="Change in connectivity of this group">
                            <strong>Connectivity Δ</strong>
                            <span class="feature-explanation">change</span>
                            <span id="current-group-connectivity-delta">-</span>
                        </div>
                        <div class="sticky-value-item" title="Influence area of this group (unique empty points controlled)">
                            <strong>Influence</strong>
                            <span class="feature-explanation">area</span>
                            <span id="current-group-influence-count">-</span>
                        </div>
                        <div class="sticky-value-item" title="Change in influence area of this group">
                            <strong>Influence Δ</strong>
                            <span class="feature-explanation">change</span>
                            <span id="current-group-influence-count-delta">-</span>
                        </div>
                        <div class="sticky-value-item" title="Influence strength of this group (mean ownership of influenced points)">
                            <strong>Influence Str</strong>
                            <span class="feature-explanation">strength</span>
                            <span id="current-group-influence-strength">-</span>
                        </div>
                        <div class="sticky-value-item" title="Change in influence strength of this group">
                            <strong>Influence Str Δ</strong>
                            <span class="feature-explanation">change</span>
                            <span id="current-group-influence-strength-delta">-</span>
                        </div>
                        <div class="sticky-value-item" title="Liberty count of the group containing the played stone">
                            <strong>Liberties</strong>
                            <span class="feature-explanation">empty adj</span>
                            <span id="liberties">-</span>
                        </div>
                        <div class="sticky-value-item" title="Move created a new separate group">
                            <strong>New Group</strong>
                            <span class="feature-explanation">isolated</span>
                            <span id="creates-new-group">-</span>
                        </div>
                    </div>
                </div>
                
                <!-- All Groups Section -->
                <div class="feature-category">
                    <div class="category-header">All Groups (Average)</div>
                    <div class="feature-grid">
                        <div class="sticky-value-item" title="Average change in ownership over all own groups">
                            <strong>Strength Δ</strong>
                            <span class="feature-explanation">avg change</span>
                        <span id="group-strength-delta">-</span>
                    </div>
                        <div class="sticky-value-item" title="Average change in connectivity of all own groups">
                            <strong>Connectivity Δ</strong>
                            <span class="feature-explanation">avg change</span>
                        <span id="group-connectivity-delta">-</span>
                    </div>
                        <div class="sticky-value-item" title="Change in count of influenced empty points">
                        <strong>Influence Count Δ</strong>
                        <span class="feature-explanation">points</span>
                        <span id="influence-count-delta">-</span>
                    </div>
                        <div class="sticky-value-item" title="Change in average influence strength">
                            <strong>Influence Str Δ</strong>
                            <span class="feature-explanation">avg</span>
                            <span id="influence-strength-delta">-</span>
                        </div>
                        <div class="sticky-value-item detail-value" title="Max strength delta among all groups">
                            <strong>Max Str Δ</strong>
                            <span class="feature-explanation">max</span>
                            <span id="max-group-strength-delta">-</span>
                        </div>
                        <div class="sticky-value-item detail-value" title="Max connectivity delta among all groups">
                            <strong>Max Conn Δ</strong>
                            <span class="feature-explanation">max</span>
                            <span id="max-group-connectivity-delta">-</span>
                        </div>
                    </div>
                </div>
                
                <!-- Tactics Section -->
                <div class="feature-category">
                    <div class="category-header">Tactics</div>
                    <div class="feature-grid">
                        <div class="sticky-value-item" title="Move separates 2+ opponent groups">
                        <strong>Cut</strong>
                        <span class="feature-explanation">separates</span>
                        <span id="is-cut">-</span>
                    </div>
                        <div class="sticky-value-item" title="Move connects 2+ own groups">
                        <strong>Connection</strong>
                        <span class="feature-explanation">joins</span>
                        <span id="is-connection">-</span>
                    </div>
                        <div class="sticky-value-item" title="Number of groups connected minus 1">
                            <strong>Conn. Gain</strong>
                            <span class="feature-explanation">groups</span>
                        <span id="connection-strength-gain">-</span>
                    </div>
                        <div class="sticky-value-item detail-value" id="merged-groups-regions-item" style="display: none;" title="Regions where merged groups are located">
                            <strong>Merged Regions</strong>
                            <span class="feature-explanation">locations</span>
                            <span id="merged-groups-regions">-</span>
                        </div>
                        <div class="sticky-value-item detail-value" id="merged-groups-heads-item" style="display: none;" title="Head stone locations of merged groups">
                            <strong>Merged Heads</strong>
                            <span class="feature-explanation">locations</span>
                            <span id="merged-groups-heads">-</span>
                        </div>
                        <div class="sticky-value-item" title="Move is adjacent to own stone">
                        <strong>Extension</strong>
                        <span class="feature-explanation">adjacent</span>
                        <span id="is-extension">-</span>
                    </div>
                        <div class="sticky-value-item" title="Move puts opponent group into atari">
                        <strong>Atari</strong>
                        <span class="feature-explanation">2 liberties</span>
                        <span id="atari">-</span>
                    </div>
                    </div>
                </div>
                
                    <!-- Attack Section -->
                <div class="feature-category">
                    <div class="category-header">Attack</div>
                    <div class="feature-grid">
                        <div class="sticky-value-item" title="Opponent group strength decreased">
                        <strong>Attack</strong>
                        <span class="feature-explanation">weakened</span>
                        <span id="attack">-</span>
                    </div>
                        <div class="sticky-value-item" title="Opponent group dropped to likely-dead level">
                            <strong>Killing</strong>
                            <span class="feature-explanation">likely dead</span>
                        <span id="killing-attack">-</span>
                    </div>
                        <div class="sticky-value-item" title="Reduced opponent's potential (aji)">
                            <strong>Reduce Aji</strong>
                            <span class="feature-explanation">potential</span>
                            <span id="reduce-aji">-</span>
                        </div>
                        <div class="sticky-value-item detail-value" title="Attack intensities: avg / max">
                            <strong>Intensity</strong>
                            <span class="feature-explanation">avg/max</span>
                            <span id="attack-intensity">-</span>
                        </div>
                        <div class="sticky-value-item detail-value" id="attacked-groups-count-item" style="display: none;" title="Number of groups under attack">
                            <strong>Groups Attacked</strong>
                            <span class="feature-explanation">count</span>
                            <span id="attacked-groups-count">-</span>
                        </div>
                        <div class="sticky-value-item detail-value" id="attacked-groups-regions-item" style="display: none;" title="Regions where attacked groups are located">
                            <strong>Attacked Regions</strong>
                            <span class="feature-explanation">locations</span>
                            <span id="attacked-groups-regions">-</span>
                        </div>
                        <div class="sticky-value-item detail-value" id="attacked-groups-heads-item" style="display: none;" title="Head stone locations of attacked groups">
                            <strong>Attacked Heads</strong>
                            <span class="feature-explanation">locations</span>
                            <span id="attacked-groups-heads">-</span>
                        </div>
                        <div class="sticky-value-item detail-value" id="attacked-groups-intensities-item" style="display: none;" title="Strength deltas for each attacked group (how much each group was weakened)">
                            <strong>Group Intensities</strong>
                            <span class="feature-explanation">deltas</span>
                            <span id="attacked-groups-intensities">-</span>
                        </div>
                    </div>
                </div>
                
                <!-- Sacrifice Section -->
                <div class="feature-category">
                    <div class="category-header">Sacrifice</div>
                    <div class="feature-grid">
                        <div class="sticky-value-item" title="Stone is in opponent territory after move">
                            <strong>Direct</strong>
                            <span class="feature-explanation">stone lost</span>
                        <span id="direct-sacrifice">-</span>
                    </div>
                        <div class="sticky-value-item" title="Intensity of direct sacrifice (ownership value)">
                            <strong>Direct Intensity</strong>
                            <span class="feature-explanation">ownership</span>
                        <span id="direct-sacrifice-intensity">-</span>
                    </div>
                        <div class="sticky-value-item" title="Own stones that flipped to opponent territory">
                            <strong>Indirect</strong>
                            <span class="feature-explanation">stones lost</span>
                        <span id="indirect-sacrifice">-</span>
                    </div>
                        <div class="sticky-value-item" title="Intensity of indirect sacrifice (average ownership swing)">
                            <strong>Indirect Intensity</strong>
                            <span class="feature-explanation">avg swing</span>
                        <span id="indirect-sacrifice-intensity">-</span>
                    </div>
                    </div>
                </div>
                
                    <!-- Policy Section -->
                <div class="feature-category">
                    <div class="category-header">Policy</div>
                    <div class="feature-grid">
                        <div class="sticky-value-item" title="Top move has >95% probability">
                            <strong>Only Move</strong>
                            <span class="feature-explanation">>95% prob</span>
                            <span id="is-only-move">-</span>
                        </div>
                        <div class="sticky-value-item" title="Move is far from last move, ignoring local follow-up">
                            <strong>Tenuki</strong>
                            <span class="feature-explanation">distant</span>
                            <span id="is-tenuki">-</span>
                        </div>
                        <div class="sticky-value-item" title="Top urgency regions by policy mass">
                            <strong>Urgency</strong>
                            <span class="feature-explanation">regions</span>
                            <span id="urgency-summary">-</span>
                        </div>
                    </div>
                </div>
                
                <!-- Regional Summary -->
                <div class="feature-category" id="regional-category" style="display: none;">
                    <div class="category-header">Regional</div>
                    <div class="feature-grid" id="regional-grid">
                    </div>
                </div>
            `;
            
            container.appendChild(section);
        }}
        
        
        function updateBoards(data) {{
            // Update each board
            updateBoardState(data);
            updatePolicy(data, 'policy0');
            updatePolicy(data, 'policy1');
            updateOwnership(data);
            updatePassCounterfactual(data);
            updateScoring(data);
            updateFuturePos(data, 'futurepos0', 0);
            updateFuturePos(data, 'futurepos1', 1);
            updateSeki(data);
            
            // Update value info
            updateValueInfo(data);
            
            // Update snorkel info
            updateSnorkelInfo(data);
        }}
        
        function drawGridLines(board) {{
            // Draw horizontal lines
            for (let i = 0; i < 19; i++) {{
                const line = document.createElement('div');
                line.style.position = 'absolute';
                line.style.left = '10px';
                line.style.right = '10px';
                line.style.height = '1px';
                line.style.top = `${{10 + i * 20}}px`;
                line.style.backgroundColor = '#8B4513';
                line.style.zIndex = '1';
                board.appendChild(line);
            }}
            
            // Draw vertical lines
            for (let i = 0; i < 19; i++) {{
                const line = document.createElement('div');
                line.style.position = 'absolute';
                line.style.top = '10px';
                line.style.bottom = '10px';
                line.style.width = '1px';
                line.style.left = `${{10 + i * 20}}px`;
                line.style.backgroundColor = '#8B4513';
                line.style.zIndex = '1';
                board.appendChild(line);
            }}
        }}

        function updateBoardState(data) {{
            const board = document.getElementById('board-board_state');
            if (!board) return;
            
            
            board.innerHTML = '';
            drawGridLines(board);
            
            // Create stones and labels at proper Go board coordinates
            for (let y = 0; y < 19; y++) {{
                for (let x = 0; x < 19; x++) {{
                    // Convert x,y to KataGo location using the same logic as KataGo's loc() function
                    const dy = 20; // size + 1 = 19 + 1 = 20
                    const loc = (x + 1) + dy * (y + 1);
                    const stone = data.board_state[loc] || 0; // Default to 0 if out of bounds
                    
                    // Debug: log bottom row stones
                    if (y === 0 && stone !== 0) {{
                        console.log(`Bottom row stone at (x=${{x}}, y=${{y}}), loc=${{loc}}, stone=${{stone}}`);
                    }}
                    
                    // Calculate pixel position (Go board coordinates)
                    const pixelX = 10 + x * 20;  // 10px margin + 20px spacing
                    const pixelY = 10 + y * 20;  // 10px margin + 20px spacing
                    
                    if (stone === 1 || stone === -1) {{
                        const stoneEl = document.createElement('div');
                        stoneEl.className = `stone ${{stone === 1 ? 'black' : 'white'}}`;
                        stoneEl.style.position = 'absolute';
                        stoneEl.style.left = `${{pixelX - 9}}px`;  // Center the stone (18px wide)
                        stoneEl.style.top = `${{pixelY - 9}}px`;   // Center the stone (18px high)
                        stoneEl.style.zIndex = '10';
                        board.appendChild(stoneEl);
                    }}
                    
                    // Add move number for last move
                    if (data.last_move) {{
                        const moveLoc = getMoveLocation(data.last_move);
                        if (moveLoc && moveLoc.x === x && moveLoc.y === y) {{
                            const label = document.createElement('div');
                            label.className = 'label';
                            label.textContent = data.move_number;
                            label.style.position = 'absolute';
                            label.style.left = `${{pixelX - 5}}px`;
                            label.style.top = `${{pixelY - 5}}px`;
                            label.style.color = 'red';
                            label.style.zIndex = '15';
                            board.appendChild(label);
                        }}
                    }}
                }}
            }}
        }}
        
        function updatePolicy(data, type) {{
            const board = document.getElementById(`board-${{type}}`);
            if (!board || !data[type]) return;
            
            board.innerHTML = '';
            drawGridLines(board);
            
            const policy = data[type];
            const maxProb = Math.max(...policy);
            const threshold = maxProb * 0.05;
            
            // Create stones and policy values at proper Go board coordinates
            for (let y = 0; y < 19; y++) {{
                for (let x = 0; x < 19; x++) {{
                    // Policy arrays are 1D with simple row-major indexing: y * 19 + x
                    const policyIndex = y * 19 + x;
                    const prob = policy[policyIndex];
                    
                    // Convert x,y to KataGo location for board state lookup
                    const dy = 20; // size + 1 = 19 + 1 = 20
                    const loc = (x + 1) + dy * (y + 1);
                    
                    // Calculate pixel position (Go board coordinates)
                    const pixelX = 10 + x * 20;  // 10px margin + 20px spacing
                    const pixelY = 10 + y * 20;  // 10px margin + 20px spacing
                    
                    // Add stones
                    const stone = data.board_state[loc];
                    if (stone === 1 || stone === -1) {{
                        const stoneEl = document.createElement('div');
                        stoneEl.className = `stone ${{stone === 1 ? 'black' : 'white'}}`;
                        stoneEl.style.position = 'absolute';
                        stoneEl.style.left = `${{pixelX - 9}}px`;  // Center the stone (18px wide)
                        stoneEl.style.top = `${{pixelY - 9}}px`;   // Center the stone (18px high)
                        stoneEl.style.zIndex = '10';
                        board.appendChild(stoneEl);
                    }}
                    
                    // Add policy probability
                    if (prob > threshold) {{
                        const label = document.createElement('div');
                        label.className = 'label';
                        label.textContent = Math.round(prob * 100);
                        label.style.position = 'absolute';
                        label.style.left = `${{pixelX - 5}}px`;
                        label.style.top = `${{pixelY - 5}}px`;
                        label.style.color = 'red';
                        label.style.zIndex = '15';
                        board.appendChild(label);
                    }}
                }}
            }}
            
            // Show pass probability
            const passProb = policy[361];
            if (passProb > threshold) {{
                const info = document.getElementById(`info-${{type}}`);
                if (info) {{
                    info.innerHTML = `Pass: ${{(passProb * 100).toFixed(1)}}%`;
                }}
            }}
        }}
        
        function updateOwnership(data) {{
            const board = document.getElementById('board-ownership');
            if (!board || !data.ownership) return;
            
            board.innerHTML = '';
            drawGridLines(board);
            
            // Create stones and ownership values at proper Go board coordinates
            for (let y = 0; y < 19; y++) {{
                for (let x = 0; x < 19; x++) {{
                    // Ownership is 2D array with [y][x] indexing
                    const ownership = data.ownership[0][y][x];
                    
                    // Calculate pixel position (Go board coordinates)
                    const pixelX = 10 + x * 20;  // 10px margin + 20px spacing
                    const pixelY = 10 + y * 20;  // 10px margin + 20px spacing
                    
                    // Add stones
                    // Convert x,y to KataGo location for board state lookup
                    const dy = 20; // size + 1 = 19 + 1 = 20
                    const loc = (x + 1) + dy * (y + 1);
                    const stone = data.board_state[loc] || 0; // Default to 0 if out of bounds
                    if (stone === 1 || stone === -1) {{
                        const stoneEl = document.createElement('div');
                        stoneEl.className = `stone ${{stone === 1 ? 'black' : 'white'}}`;
                        stoneEl.style.position = 'absolute';
                        stoneEl.style.left = `${{pixelX - 9}}px`;  // Center the stone (18px wide)
                        stoneEl.style.top = `${{pixelY - 9}}px`;   // Center the stone (18px high)
                        stoneEl.style.zIndex = '10';
                        board.appendChild(stoneEl);
                    }}
                    
                    // Add ownership
                    if (Math.abs(ownership) > 0.1) {{
                        const label = document.createElement('div');
                        label.className = 'label';
                        label.textContent = Math.round(Math.abs(ownership) * 10);
                        label.style.position = 'absolute';
                        label.style.left = `${{pixelX - 5}}px`;
                        label.style.top = `${{pixelY - 5}}px`;
                    // Color based on who the value is good for:
                    // Red text = good for White (positive ownership from White's perspective)
                    // Grey text = good for Black (negative ownership from White's perspective)
                    // KataGo always outputs ownership from White's perspective (positive = White)
                    // data.player is who just played, so the current player to move is the opponent
                    const isGoodForWhite = ownership > 0;
                    const shouldUseBlackText = isGoodForWhite;  // Red for White, Grey for Black
                        
                        label.style.color = shouldUseBlackText ? 'red' : 'grey';
                        label.style.textShadow = shouldUseBlackText ? '1px 1px 1px rgba(255,255,255,0.8)' : '1px 1px 1px rgba(0,0,0,0.8)';
                        label.style.zIndex = '15';
                        board.appendChild(label);
                    }}
                }}
            }}
            
            const info = document.getElementById('info-ownership');
            if (info) {{
                info.innerHTML = `
                    <div class="heatmap-legend">
                        <span style="color: red; background: rgba(255,255,255,0.8);">Red Text: White</span>
                        <span style="color: grey; background: rgba(0,0,0,0.8);">Grey Text: Black</span>
                    </div>
                `;
            }}
        }}
        
        function updatePassCounterfactual(data) {{
            const board = document.getElementById('board-pass_counterfactual');
            if (!board) return;
            
            board.innerHTML = '';
            drawGridLines(board);
            
            // Show "N/A" if no counterfactual data (initial position or pass moves)
            if (!data.pass_counterfactual || !Array.isArray(data.pass_counterfactual)) {{
                const info = document.getElementById('info-pass_counterfactual');
                if (info) {{
                    info.innerHTML = `
                        <div style="text-align: center; color: #666; padding: 10px;">
                            No counterfactual available (initial position or pass move)
                        </div>
                    `;
                }}
                return;
            }}
            
            // Debug: log data structure
            console.log('pass_counterfactual data structure:', {{
                isArray: Array.isArray(data.pass_counterfactual),
                length: data.pass_counterfactual.length,
                firstRowLength: data.pass_counterfactual[0] ? data.pass_counterfactual[0].length : 'N/A',
                sample: data.pass_counterfactual[0] ? data.pass_counterfactual[0][0] : 'N/A'
            }});
            
            // Create stones and counterfactual ownership values at proper Go board coordinates
            for (let y = 0; y < 19; y++) {{
                for (let x = 0; x < 19; x++) {{
                    // Pass counterfactual is 2D array with [y][x] indexing (no batch dimension)
                    // Handle both 2D [y][x] and 3D [0][y][x] formats for robustness
                    let ownership;
                    if (data.pass_counterfactual[0] && Array.isArray(data.pass_counterfactual[0]) && 
                        data.pass_counterfactual[0][0] && Array.isArray(data.pass_counterfactual[0][0])) {{
                        // 3D format: [batch][y][x]
                        ownership = data.pass_counterfactual[0][y][x];
                    }} else {{
                        // 2D format: [y][x]
                        ownership = data.pass_counterfactual[y] ? data.pass_counterfactual[y][x] : 0;
                    }}
                    
                    // Calculate pixel position (Go board coordinates)
                    const pixelX = 10 + x * 20;  // 10px margin + 20px spacing
                    const pixelY = 10 + y * 20;  // 10px margin + 20px spacing
                    
                    // Add stones (same as ownership board)
                    const dy = 20; // size + 1 = 19 + 1 = 20
                    const loc = (x + 1) + dy * (y + 1);
                    const stone = data.board_state[loc] || 0;
                    if (stone === 1 || stone === -1) {{
                        const stoneEl = document.createElement('div');
                        stoneEl.className = `stone ${{stone === 1 ? 'black' : 'white'}}`;
                        stoneEl.style.position = 'absolute';
                        stoneEl.style.left = `${{pixelX - 9}}px`;
                        stoneEl.style.top = `${{pixelY - 9}}px`;
                        stoneEl.style.zIndex = '10';
                        board.appendChild(stoneEl);
                    }}
                    
                    // Add counterfactual ownership value
                    if (ownership !== undefined && ownership !== null && Math.abs(ownership) > 0.1) {{
                        const label = document.createElement('div');
                        label.className = 'label';
                        label.textContent = Math.round(Math.abs(ownership) * 10);
                        label.style.position = 'absolute';
                        label.style.left = `${{pixelX - 5}}px`;
                        label.style.top = `${{pixelY - 5}}px`;
                        // Color based on who the value is good for (same as ownership)
                        const isGoodForWhite = ownership > 0;
                        label.style.color = isGoodForWhite ? 'red' : 'grey';
                        label.style.textShadow = isGoodForWhite ? '1px 1px 1px rgba(255,255,255,0.8)' : '1px 1px 1px rgba(0,0,0,0.8)';
                        label.style.zIndex = '15';
                        board.appendChild(label);
                    }}
                }}
            }}
            
            const info = document.getElementById('info-pass_counterfactual');
            if (info) {{
                info.innerHTML = `
                    <div class="heatmap-legend">
                        <span style="color: red; background: rgba(255,255,255,0.8);">Red Text: White</span>
                        <span style="color: grey; background: rgba(0,0,0,0.8);">Grey Text: Black</span>
                    </div>
                    <div style="font-size: 11px; color: #666; margin-top: 5px; text-align: center;">
                        Shows ownership if the player had passed instead of playing. Used to detect actual move impact vs. anticipated outcome.
                    </div>
                `;
            }}
        }}
        
        function updateScoring(data) {{
            const board = document.getElementById('board-scoring');
            if (!board || !data.scoring) return;
            
            board.innerHTML = '';
            drawGridLines(board);
            
            // Create stones and scoring values at proper Go board coordinates
            for (let y = 0; y < 19; y++) {{
                for (let x = 0; x < 19; x++) {{
                    // Scoring is 2D array with [y][x] indexing
                    const scoring = data.scoring[0][y][x];
                    
                    // Calculate pixel position (Go board coordinates)
                    const pixelX = 10 + x * 20;  // 10px margin + 20px spacing
                    const pixelY = 10 + y * 20;  // 10px margin + 20px spacing
                    
                    // Add stones
                    // Convert x,y to KataGo location for board state lookup
                    const dy = 20; // size + 1 = 19 + 1 = 20
                    const loc = (x + 1) + dy * (y + 1);
                    const stone = data.board_state[loc] || 0; // Default to 0 if out of bounds
                    if (stone === 1 || stone === -1) {{
                        const stoneEl = document.createElement('div');
                        stoneEl.className = `stone ${{stone === 1 ? 'black' : 'white'}}`;
                        stoneEl.style.position = 'absolute';
                        stoneEl.style.left = `${{pixelX - 9}}px`;  // Center the stone (18px wide)
                        stoneEl.style.top = `${{pixelY - 9}}px`;   // Center the stone (18px high)
                        stoneEl.style.zIndex = '10';
                        board.appendChild(stoneEl);
                    }}
                    
                    // Add scoring
                    if (Math.abs(scoring) > 0.1) {{
                        const label = document.createElement('div');
                        label.className = 'label';
                        label.textContent = Math.round(Math.abs(scoring) * 10);
                        label.style.position = 'absolute';
                        label.style.left = `${{pixelX - 5}}px`;
                        label.style.top = `${{pixelY - 5}}px`;
                    // Color based on who the value is good for:
                    // Red text = good for White (positive scoring from White's perspective)
                    // Grey text = good for Black (negative scoring from White's perspective)
                    // KataGo always outputs scoring from White's perspective (positive = White)
                    const isGoodForWhite = scoring > 0;
                    const shouldUseBlackText = isGoodForWhite;  // Red for White, Grey for Black
                        label.style.color = shouldUseBlackText ? 'red' : 'grey';
                        label.style.textShadow = shouldUseBlackText ? '1px 1px 1px rgba(255,255,255,0.8)' : '1px 1px 1px rgba(0,0,0,0.8)';
                        label.style.zIndex = '15';
                        board.appendChild(label);
                    }}
                }}
            }}
            
            const info = document.getElementById('info-scoring');
            if (info) {{
                info.innerHTML = `
                    <div class="heatmap-legend">
                        <span style="color: red; background: rgba(255,255,255,0.8);">Red Text: White</span>
                        <span style="color: grey; background: rgba(0,0,0,0.8);">Grey Text: Black</span>
                    </div>
                `;
            }}
        }}
        
        function updateFuturePos(data, type, channel) {{
            const board = document.getElementById(`board-${{type}}`);
            if (!board || !data.futurepos) return;
            
            board.innerHTML = '';
            drawGridLines(board);
            
            // Create stones and future position values at proper Go board coordinates
            for (let y = 0; y < 19; y++) {{
                for (let x = 0; x < 19; x++) {{
                    // Future position is 2D array with [y][x] indexing
                    const futurepos = data.futurepos[channel][y][x];
                    
                    // Calculate pixel position (Go board coordinates)
                    const pixelX = 10 + x * 20;  // 10px margin + 20px spacing
                    const pixelY = 10 + y * 20;  // 10px margin + 20px spacing
                    
                    // Add stones
                    // Convert x,y to KataGo location for board state lookup
                    const dy = 20; // size + 1 = 19 + 1 = 20
                    const loc = (x + 1) + dy * (y + 1);
                    const stone = data.board_state[loc] || 0; // Default to 0 if out of bounds
                    if (stone === 1 || stone === -1) {{
                        const stoneEl = document.createElement('div');
                        stoneEl.className = `stone ${{stone === 1 ? 'black' : 'white'}}`;
                        stoneEl.style.position = 'absolute';
                        stoneEl.style.left = `${{pixelX - 9}}px`;  // Center the stone (18px wide)
                        stoneEl.style.top = `${{pixelY - 9}}px`;   // Center the stone (18px high)
                        stoneEl.style.zIndex = '10';
                        board.appendChild(stoneEl);
                    }}
                    
                    // Add future position
                    if (Math.abs(futurepos) > 0.1) {{
                        const label = document.createElement('div');
                        label.className = 'label';
                        label.textContent = Math.round(Math.abs(futurepos) * 10);
                        label.style.position = 'absolute';
                        label.style.left = `${{pixelX - 5}}px`;
                        label.style.top = `${{pixelY - 5}}px`;
                    // Color based on who the value is good for:
                    // Red text = good for White (positive futurepos from White's perspective)
                    // Grey text = good for Black (negative futurepos from White's perspective)
                    // KataGo always outputs futurepos from White's perspective (positive = White)
                    const isGoodForWhite = futurepos > 0;
                    const shouldUseBlackText = isGoodForWhite;  // Red for White, Grey for Black
                        
                        label.style.color = shouldUseBlackText ? 'red' : 'grey';
                        label.style.textShadow = shouldUseBlackText ? '1px 1px 1px rgba(255,255,255,0.8)' : '1px 1px 1px rgba(0,0,0,0.8)';
                        label.style.zIndex = '15';
                        board.appendChild(label);
                    }}
                }}
            }}
            
            const info = document.getElementById(`info-${{type}}`);
            if (info) {{
                info.innerHTML = `
                    <div class="heatmap-legend">
                        <span style="color: red; background: rgba(255,255,255,0.8);">Red Text: White</span>
                        <span style="color: grey; background: rgba(0,0,0,0.8);">Grey Text: Black</span>
                    </div>
                `;
            }}
        }}
        
        function updateSeki(data) {{
            const board = document.getElementById('board-seki');
            if (!board || !data.seki) return;
            
            board.innerHTML = '';
            drawGridLines(board);
            
            // Create stones and seki values at proper Go board coordinates
            for (let y = 0; y < 19; y++) {{
                for (let x = 0; x < 19; x++) {{
                    // Seki is 2D array with [y][x] indexing
                    const seki = data.seki[y][x];
                    
                    // Calculate pixel position (Go board coordinates)
                    const pixelX = 10 + x * 20;  // 10px margin + 20px spacing
                    const pixelY = 10 + y * 20;  // 10px margin + 20px spacing
                    
                    // Add stones
                    // Convert x,y to KataGo location for board state lookup
                    const dy = 20; // size + 1 = 19 + 1 = 20
                    const loc = (x + 1) + dy * (y + 1);
                    const stone = data.board_state[loc] || 0; // Default to 0 if out of bounds
                    if (stone === 1 || stone === -1) {{
                        const stoneEl = document.createElement('div');
                        stoneEl.className = `stone ${{stone === 1 ? 'black' : 'white'}}`;
                        stoneEl.style.position = 'absolute';
                        stoneEl.style.left = `${{pixelX - 9}}px`;  // Center the stone (18px wide)
                        stoneEl.style.top = `${{pixelY - 9}}px`;   // Center the stone (18px high)
                        stoneEl.style.zIndex = '10';
                        board.appendChild(stoneEl);
                    }}
                    
                    // Add seki
                    if (Math.abs(seki) > 0.1) {{
                        const label = document.createElement('div');
                        label.className = 'label';
                        label.textContent = 'S';
                        label.style.position = 'absolute';
                        label.style.left = `${{pixelX - 5}}px`;
                        label.style.top = `${{pixelY - 5}}px`;
                        label.style.color = 'purple';
                        label.style.zIndex = '15';
                        board.appendChild(label);
                    }}
                }}
            }}
            
            const info = document.getElementById('info-seki');
            if (info) {{
                info.innerHTML = `
                    <div class="heatmap-legend">
                        <span style="background: rgba(128,0,128,0.3);">Purple: Seki Probability</span>
                    </div>
                `;
            }}
        }}
        
        function updateValueInfo(data) {{
            if (!data) return;
            
            // Determine current player (opposite of who just played)
            let currentPlayer = 'Black';
            if (data.player === 'Black') {{
                currentPlayer = 'White';
            }} else if (data.player === 'White') {{
                currentPlayer = 'Black';
            }} else if (data.player === 'Initial') {{
                currentPlayer = 'Black'; // Black starts first
            }}
            
            const elements = {{
                'value-winrate': (data.value[0] * 100).toFixed(1),
                'score-mean': data.scoremean.toFixed(1),
                'lead': data.lead.toFixed(1),
                'score-stdev': data.scorestdev.toFixed(1),
                'vtime': data.vtime.toFixed(1),
                'estv': data.estv.toFixed(3),
                'value-td-long': (data.td_value[0] * 100).toFixed(1),
                'value-td-mid': (data.td_value2[0] * 100).toFixed(1),
                'value-td-short': (data.td_value3[0] * 100).toFixed(1),
                'current-player': currentPlayer
            }};
            
            for (const [id, value] of Object.entries(elements)) {{
                const element = document.getElementById(id);
                if (element) {{
                    element.textContent = value;
                }}
            }}
        }}
        
        function updateSnorkelInfo(data) {{
            // Use analysis attached to the same entry you're viewing
            const analysisData = data && data.analysis ? data.analysis : null;
            
            // For urgency, use the next_move_urgency if available (computed before the move)
            const urgencyData = data && data.next_move_urgency ? data.next_move_urgency : null;
            
            if (!analysisData) {{
                // Clear all snorkel fields if no analysis data
                const elements = {{
                    'group-strength-delta': '-',
                    'group-connectivity-delta': '-',
                    'influence-count-delta': '-',
                    'influence-strength-delta': '-',
                    'max-group-strength-delta': '-',
                    'max-group-connectivity-delta': '-',
                    'potential-territory': '-',
                    'solid-territory': '-',
                    'building-count': '-',
                    'solidification-count': '-',
                    'reduction-count': '-',
                    'invasion': '-',
                    'territory-intensities': '-',
                    'direct-sacrifice': '-',
                    'direct-sacrifice-intensity': '-',
                    'indirect-sacrifice': '-',
                    'indirect-sacrifice-intensity': '-',
                    'is-cut': '-',
                    'is-connection': '-',
                    'connection-strength-gain': '-',
                    'merged-groups-regions': '-',
                    'merged-groups-heads': '-',
                    'is-extension': '-',
                    'liberties': '-',
                    'atari': '-',
                    'attack': '-',
                    'killing-attack': '-',
                    'reduce-aji': '-',
                    'attack-intensity': '-',
                    'attacked-groups-count': '-',
                    'attacked-groups-regions': '-',
                    'attacked-groups-heads': '-',
                    'attacked-groups-intensities': '-',
                    'creates-new-group': '-',
                    'is-only-move': '-',
                    'is-tenuki': '-',
                    'urgency-summary': '-',
                    'current-group-strength': '-',
                    'current-group-strength-delta': '-',
                    'current-group-connectivity': '-',
                    'current-group-connectivity-delta': '-',
                    'current-group-influence-count': '-',
                    'current-group-influence-count-delta': '-',
                    'current-group-influence-strength': '-',
                    'current-group-influence-strength-delta': '-'
                }};
                
                for (const [id, value] of Object.entries(elements)) {{
                    const element = document.getElementById(id);
                    if (element) {{
                        element.textContent = value;
                    }}
                }}
                
                // Hide regional category
                const regionalCategory = document.getElementById('regional-category');
                if (regionalCategory) {{
                    regionalCategory.style.display = 'none';
                }}
                
                // Hide attacked groups info
                const attackedGroupsCountItem = document.getElementById('attacked-groups-count-item');
                const attackedGroupsRegionsItem = document.getElementById('attacked-groups-regions-item');
                const attackedGroupsHeadsItem = document.getElementById('attacked-groups-heads-item');
                const attackedGroupsIntensitiesItem = document.getElementById('attacked-groups-intensities-item');
                if (attackedGroupsCountItem) attackedGroupsCountItem.style.display = 'none';
                if (attackedGroupsRegionsItem) attackedGroupsRegionsItem.style.display = 'none';
                if (attackedGroupsHeadsItem) attackedGroupsHeadsItem.style.display = 'none';
                if (attackedGroupsIntensitiesItem) attackedGroupsIntensitiesItem.style.display = 'none';
                
                // Hide merged groups info
                const mergedGroupsRegionsItem = document.getElementById('merged-groups-regions-item');
                const mergedGroupsHeadsItem = document.getElementById('merged-groups-heads-item');
                if (mergedGroupsRegionsItem) mergedGroupsRegionsItem.style.display = 'none';
                if (mergedGroupsHeadsItem) mergedGroupsHeadsItem.style.display = 'none';
                
                return;
            }}
            
            const analysis = analysisData;
            
            // Helper to format value with delta
            function formatWithDelta(value, delta) {{
                if (value === undefined) return '-';
                let result = String(value);
                if (delta !== undefined && delta !== 0) {{
                    const sign = delta > 0 ? '+' : '';
                    result += ` (${{sign}}${{delta}})`;
                }}
                return result;
            }}
            
            // Build territory intensities string
            const intensityParts = [];
            if (analysis.building_intensity) intensityParts.push(`B:${{(analysis.building_intensity).toFixed(2)}}`);
            if (analysis.solidification_intensity) intensityParts.push(`S:${{(analysis.solidification_intensity).toFixed(2)}}`);
            if (analysis.reduction_intensity) intensityParts.push(`R:${{(analysis.reduction_intensity).toFixed(2)}}`);
            const territoryIntensities = intensityParts.length > 0 ? intensityParts.join(' ') : '-';
            
            // Build attack intensity string
            const attackIntensity = analysis.attack ? 
                `${{(analysis.avg_attack_intensity || 0).toFixed(2)}} / ${{(analysis.max_attack_intensity || 0).toFixed(2)}}` : '-';
            
            // Build merged groups info (only if connection = yes)
            let mergedGroupsRegions = '-';
            let mergedGroupsHeads = '-';
            const showMergedGroups = analysis.connection && analysis.merged_groups_regions && analysis.merged_groups_regions.length > 0;
            
            if (showMergedGroups) {{
                // Format regions
                if (analysis.merged_groups_regions && analysis.merged_groups_regions.length > 0) {{
                    const regionNames = analysis.merged_groups_regions.map(r => r.replace('_', ' '));
                    mergedGroupsRegions = regionNames.join(', ');
                }}
                
                // Format head locations (convert to coordinates)
                if (analysis.merged_groups_head_locs && analysis.merged_groups_head_locs.length > 0) {{
                    const headCoords = analysis.merged_groups_head_locs.map(loc => {{
                        // Convert KataGo location to (x, y) coordinates
                        const dy = 20; // size + 1 = 19 + 1 = 20
                        const x = (loc % dy) - 1;
                        const y = Math.floor(loc / dy) - 1;
                        // Convert to Go notation (A-T for rows, 1-19 for columns)
                        const letter = String.fromCharCode(65 + y); // A=0, B=1, ..., T=18
                        return `${{letter}}${{x + 1}}`;
                    }});
                    mergedGroupsHeads = headCoords.join(', ');
                }}
            }}
            
            // Show/hide merged groups info
            const mergedGroupsRegionsItem = document.getElementById('merged-groups-regions-item');
            const mergedGroupsHeadsItem = document.getElementById('merged-groups-heads-item');
            if (mergedGroupsRegionsItem) mergedGroupsRegionsItem.style.display = showMergedGroups ? 'flex' : 'none';
            if (mergedGroupsHeadsItem) mergedGroupsHeadsItem.style.display = showMergedGroups ? 'flex' : 'none';
            
            // Build attacked groups info (only if attack = yes)
            let attackedGroupsCount = '-';
            let attackedGroupsRegions = '-';
            let attackedGroupsHeads = '-';
            let attackedGroupsIntensities = '-';
            const showAttackedGroups = analysis.attack && analysis.attacked_groups_count > 0;
            
            if (showAttackedGroups) {{
                attackedGroupsCount = analysis.attacked_groups_count || 0;
                
                // Format regions
                if (analysis.attacked_groups_regions && analysis.attacked_groups_regions.length > 0) {{
                    const regionNames = analysis.attacked_groups_regions.map(r => r.replace('_', ' '));
                    attackedGroupsRegions = regionNames.join(', ');
                }}
                
                // Format head locations (convert to coordinates)
                if (analysis.attacked_groups_head_locs && analysis.attacked_groups_head_locs.length > 0) {{
                    const headCoords = analysis.attacked_groups_head_locs.map(loc => {{
                        // Convert KataGo location to (x, y) coordinates
                        // KataGo uses: loc = (x + 1) + (size + 1) * (y + 1)
                        const dy = 20; // size + 1 = 19 + 1 = 20
                        const x = (loc % dy) - 1;
                        const y = Math.floor(loc / dy) - 1;
                        // Convert to Go notation (A-T for rows, 1-19 for columns)
                        // Note: Standard Go notation skips 'I', but we'll use A-T for simplicity
                        const letter = String.fromCharCode(65 + y); // A=0, B=1, ..., T=18
                        return `${{letter}}${{x + 1}}`;
                    }});
                    attackedGroupsHeads = headCoords.join(', ');
                }}
                
                // Format attack intensities by group (strength deltas)
                if (analysis.attacked_groups_strength_deltas && analysis.attacked_groups_strength_deltas.length > 0) {{
                    // Deltas are negative (strength decreased), format as positive values
                    const intensities = analysis.attacked_groups_strength_deltas.map(delta => {{
                        // Show as positive value (how much strength decreased)
                        return Math.abs(delta).toFixed(3);
                    }});
                    attackedGroupsIntensities = intensities.join(', ');
                }}
            }}
            
            // Show/hide attacked groups info
            const attackedGroupsCountItem = document.getElementById('attacked-groups-count-item');
            const attackedGroupsRegionsItem = document.getElementById('attacked-groups-regions-item');
            const attackedGroupsHeadsItem = document.getElementById('attacked-groups-heads-item');
            const attackedGroupsIntensitiesItem = document.getElementById('attacked-groups-intensities-item');
            if (attackedGroupsCountItem) attackedGroupsCountItem.style.display = showAttackedGroups ? 'flex' : 'none';
            if (attackedGroupsRegionsItem) attackedGroupsRegionsItem.style.display = showAttackedGroups ? 'flex' : 'none';
            if (attackedGroupsHeadsItem) attackedGroupsHeadsItem.style.display = showAttackedGroups ? 'flex' : 'none';
            if (attackedGroupsIntensitiesItem) attackedGroupsIntensitiesItem.style.display = showAttackedGroups ? 'flex' : 'none';
            
            // Build urgency summary (top 2 regions)
            const urgencyToShow = urgencyData || analysis.urgency;
            let urgencySummary = '-';
            if (urgencyToShow) {{
                const topUrgency = Object.entries(urgencyToShow)
                    .filter(([_, u]) => u > 0.01)
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 2)
                    .map(([r, u]) => `${{r.replace('_', ' ')}}: ${{(u * 100).toFixed(0)}}%`)
                    .join(', ');
                if (topUrgency) urgencySummary = topUrgency;
            }}
            
            // Update basic snorkel metrics (with percentiles where stats available)
            const elements = {{
                // Territory (with percentiles)
                'potential-territory': analysis.potential_territory !== undefined ? 
                    formatWithPercentile(analysis.potential_territory, 'potential_territory', 0) : '-',
                'solid-territory': analysis.solid_territory !== undefined ? 
                    formatWithPercentile(analysis.solid_territory, 'solid_territory', 0) : '-',
                'building-count': analysis.building_count !== undefined ? 
                    formatWithPercentile(analysis.building_count, 'building_count', 0) : '-',
                'solidification-count': analysis.solidification_count !== undefined ? 
                    formatWithPercentile(analysis.solidification_count, 'solidification_count', 0) : '-',
                'reduction-count': analysis.reduction_count !== undefined ? 
                    formatWithPercentile(analysis.reduction_count, 'reduction_count', 0) : '-',
                'invasion': analysis.invasion !== undefined ? (analysis.invasion ? 'Yes' : 'No') : '-',
                'territory-intensities': territoryIntensities,
                
                // Current Group (this move's group) - with percentiles
                'current-group-strength': analysis.current_group_strength !== undefined ? 
                    formatWithPercentile(analysis.current_group_strength, 'current_group_strength') : '-',
                'current-group-strength-delta': analysis.current_group_strength_delta !== undefined ? 
                    formatWithPercentile(analysis.current_group_strength_delta, 'current_group_strength_delta') : '-',
                'current-group-connectivity': analysis.current_group_connectivity !== undefined ? 
                    formatWithPercentile(analysis.current_group_connectivity, 'current_group_connectivity') : '-',
                'current-group-connectivity-delta': analysis.current_group_connectivity_delta !== undefined ? 
                    formatWithPercentile(analysis.current_group_connectivity_delta, 'current_group_connectivity_delta') : '-',
                'current-group-influence-count': analysis.current_group_influence_count !== undefined ? 
                    formatWithPercentile(analysis.current_group_influence_count, 'current_group_influence_count', 0) : '-',
                'current-group-influence-count-delta': analysis.current_group_influence_count_delta !== undefined ? 
                    formatWithPercentile(analysis.current_group_influence_count_delta, 'current_group_influence_count_delta', 0) : '-',
                'current-group-influence-strength': analysis.current_group_influence_strength !== undefined ? 
                    formatWithPercentile(analysis.current_group_influence_strength, 'current_group_influence_strength') : '-',
                'current-group-influence-strength-delta': analysis.current_group_influence_strength_delta !== undefined ? 
                    formatWithPercentile(analysis.current_group_influence_strength_delta, 'current_group_influence_strength_delta') : '-',
                'liberties': analysis.liberties !== undefined ? 
                    formatWithPercentile(analysis.liberties, 'liberties', 0) : '-',
                'creates-new-group': analysis.creates_new_group !== undefined ? (analysis.creates_new_group ? 'Yes' : 'No') : '-',
                
                // All Groups (average) - with percentiles
                'group-strength-delta': analysis.group_strength_delta !== undefined ? 
                    formatWithPercentile(analysis.group_strength_delta, 'group_strength_delta') : '-',
                'group-connectivity-delta': analysis.group_connectivity_delta !== undefined ? 
                    formatWithPercentile(analysis.group_connectivity_delta, 'group_connectivity_delta') : '-',
                'influence-count-delta': analysis.influence_count_delta !== undefined ? 
                    formatWithPercentile(analysis.influence_count_delta, 'influence_count_delta', 0) : '-',
                'influence-strength-delta': analysis.influence_strength_delta !== undefined ? 
                    formatWithPercentile(analysis.influence_strength_delta, 'influence_strength_delta') : '-',
                'max-group-strength-delta': analysis.max_group_strength_delta !== undefined ? 
                    formatWithPercentile(analysis.max_group_strength_delta, 'max_group_strength_delta') : '-',
                'max-group-connectivity-delta': analysis.max_group_connectivity_delta !== undefined ? 
                    formatWithPercentile(analysis.max_group_connectivity_delta, 'max_group_connectivity_delta') : '-',
                
                // Tactics
                'is-cut': analysis.cut !== undefined ? (analysis.cut ? 'Yes' : 'No') : '-',
                'is-connection': analysis.connection !== undefined ? (analysis.connection ? 'Yes' : 'No') : '-',
                'connection-strength-gain': analysis.connection_strength_gain !== undefined ? (analysis.connection_strength_gain || 0).toFixed(1) : '-',
                'merged-groups-regions': mergedGroupsRegions,
                'merged-groups-heads': mergedGroupsHeads,
                'is-extension': analysis.extension !== undefined ? (analysis.extension ? 'Yes' : 'No') : '-',
                'atari': analysis.atari !== undefined ? (analysis.atari ? 'Yes' : 'No') : '-',
                
                // Attack - with percentiles for intensities
                'attack': analysis.attack !== undefined ? (analysis.attack ? 'Yes' : 'No') : '-',
                'killing-attack': analysis.killing_attack !== undefined ? (analysis.killing_attack ? 'Yes' : 'No') : '-',
                'reduce-aji': analysis.reduce_aji !== undefined ? (analysis.reduce_aji ? 'Yes' : 'No') : '-',
                'attack-intensity': analysis.attack ? 
                    formatWithPercentile(analysis.avg_attack_intensity, 'avg_attack_intensity') + ' / ' + 
                    formatWithPercentile(analysis.max_attack_intensity, 'max_attack_intensity') : '-',
                'attacked-groups-count': attackedGroupsCount,
                'attacked-groups-regions': attackedGroupsRegions,
                'attacked-groups-heads': attackedGroupsHeads,
                'attacked-groups-intensities': attackedGroupsIntensities,
                
                // Sacrifice - with percentiles for intensities
                'direct-sacrifice': analysis.direct_sacrifice !== undefined ? (analysis.direct_sacrifice ? 'Yes' : 'No') : '-',
                'direct-sacrifice-intensity': analysis.direct_sacrifice_intensity !== undefined ? 
                    formatWithPercentile(analysis.direct_sacrifice_intensity, 'direct_sacrifice_intensity') : '-',
                'indirect-sacrifice': analysis.indirect_sacrifice !== undefined ? 
                    formatWithPercentile(analysis.indirect_sacrifice, 'indirect_sacrifice', 0) : '-',
                'indirect-sacrifice-intensity': analysis.indirect_sacrifice_intensity !== undefined ? 
                    formatWithPercentile(analysis.indirect_sacrifice_intensity, 'indirect_sacrifice_intensity') : '-',
                
                // Policy
                'is-only-move': analysis.forcing !== undefined ? (analysis.forcing ? 'Yes' : 'No') : '-',
                'is-tenuki': analysis.tenuki !== undefined ? (analysis.tenuki ? 'Yes' : 'No') : '-',
                'urgency-summary': urgencySummary
            }};
            
            for (const [id, value] of Object.entries(elements)) {{
                const element = document.getElementById(id);
                if (element) {{
                    // Use innerHTML since percentiles include HTML spans
                    element.innerHTML = value;
                }}
            }}
            
            // Update regional breakdown (only if there's meaningful data)
            const regionalCategory = document.getElementById('regional-category');
            const regionalGrid = document.getElementById('regional-grid');
            if (regionalCategory && regionalGrid && (analysis.building_count_by_region || analysis.solidification_count_by_region || analysis.reduction_count_by_region)) {{
                const regions = ['corner_tl', 'corner_tr', 'corner_bl', 'corner_br', 'side_left', 'side_right', 'side_top', 'side_bottom', 'center'];
                
                // Calculate totals for percentage computation
                let totalB = 0, totalS = 0, totalR = 0;
                for (const r of regions) {{
                    totalB += (analysis.building_count_by_region || {{}})[r] || 0;
                    totalS += (analysis.solidification_count_by_region || {{}})[r] || 0;
                    totalR += (analysis.reduction_count_by_region || {{}})[r] || 0;
                }}
                
                let hasData = false;
                let gridHtml = '';
                for (const r of regions) {{
                    const b = (analysis.building_count_by_region || {{}})[r] || 0;
                    const s = (analysis.solidification_count_by_region || {{}})[r] || 0;
                    const d = (analysis.reduction_count_by_region || {{}})[r] || 0;
                    if (b > 0 || s > 0 || d > 0) {{
                        hasData = true;
                        // Calculate percentages
                        const bPct = totalB > 0 ? Math.round((b / totalB) * 100) : 0;
                        const sPct = totalS > 0 ? Math.round((s / totalS) * 100) : 0;
                        const rPct = totalR > 0 ? Math.round((d / totalR) * 100) : 0;
                        
                        // Format: count (pct%) for each non-zero value
                        let parts = [];
                        if (b > 0) parts.push(`B:${{b}}(${{bPct}}%)`);
                        if (s > 0) parts.push(`S:${{s}}(${{sPct}}%)`);
                        if (d > 0) parts.push(`R:${{d}}(${{rPct}}%)`);
                        
                        gridHtml += `<div class="sticky-value-item"><strong>${{r.replace('_', ' ')}}</strong><span>${{parts.join(' ')}}</span></div>`;
                    }}
                }}
                if (hasData) {{
                    regionalCategory.style.display = 'flex';
                    regionalGrid.innerHTML = gridHtml;
                }} else {{
                    regionalCategory.style.display = 'none';
                }}
            }} else if (regionalCategory) {{
                regionalCategory.style.display = 'none';
            }}
        }}
        
        
        // Control functions
        function previousMove() {{
            if (currentMove > 0) {{
                currentMove--;
                updateDisplay();
            }}
        }}
        
        function nextMove() {{
            if (currentMove < gameData.length - 1) {{
                currentMove++;
                updateDisplay();
            }}
        }}
        
        function resetToStart() {{
            currentMove = 0;
            updateDisplay();
        }}
        
        function toggleAutoPlay() {{
            const btn = document.getElementById('autoPlayBtn');
            if (autoPlayInterval) {{
                clearInterval(autoPlayInterval);
                autoPlayInterval = null;
                btn.textContent = 'Auto Play';
                btn.style.background = '#007bff';
            }} else {{
                autoPlayInterval = setInterval(() => {{
                    if (currentMove < gameData.length - 1) {{
                        currentMove++;
                        updateDisplay();
                    }} else {{
                        clearInterval(autoPlayInterval);
                        autoPlayInterval = null;
                        btn.textContent = 'Auto Play';
                        btn.style.background = '#007bff';
                    }}
                }}, 1000);
                btn.textContent = 'Stop';
                btn.style.background = '#dc3545';
            }}
        }}
        
        // Event listeners
        document.getElementById('moveSlider').addEventListener('input', (e) => {{
            currentMove = parseInt(e.target.value);
            updateDisplay();
        }});
        
        // Keyboard controls
        document.addEventListener('keydown', (e) => {{
            switch(e.key) {{
                case 'ArrowLeft':
                    previousMove();
                    break;
                case 'ArrowRight':
                    nextMove();
                    break;
                case ' ':
                    e.preventDefault();
                    toggleAutoPlay();
                    break;
                case 'Home':
                    resetToStart();
                    break;
            }}
        }});
        
        // Initialize display
        addValueSection();
        addSnorkelSection();
        initializeBoards();
        updateDisplay();
        
        // Adjust body padding based on sticky header height
        function adjustBodyPadding() {{
            const stickyContainer = document.getElementById('stickyValueContainer');
            if (stickyContainer) {{
                const height = stickyContainer.offsetHeight;
                document.body.style.paddingTop = (height + 10) + 'px';
            }}
        }}
        
        // Adjust padding after content loads
        setTimeout(adjustBodyPadding, 100);
    </script>
</body>
</html>
"""
    
    # Write the HTML file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"Visualization saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Visualize KataGo model outputs')
    parser.add_argument('model_path', help='Path to the KataGo model file')
    parser.add_argument('--max-moves', type=int, default=300, help='Maximum number of moves to play')
    parser.add_argument('--output', default='katago_visualization.html', help='Output HTML file')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    model, swa_model, _ = load_model(args.model_path, use_swa=False, device=get_device(), pos_len=19, verbose=True)
    if swa_model is not None:
        model = swa_model
    model.eval()
    
    # Play game and generate visualization
    game_data, sgf_content = play_short_game(model, args.max_moves)
    
    # Generate HTML visualization
    generate_html_visualization(game_data, sgf_content, args.output)
    
    print(f"Generated visualization with {len(game_data)} positions")

if __name__ == "__main__":
    main()
