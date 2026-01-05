#!/usr/bin/env python3
"""
Script to extract features from board positions and ownership maps.

Usage examples:
    # Basic usage with board and ownership
    python extract_features.py --board empty --ownership ownership.npy
    
    # With policy and move information
    python extract_features.py --board moves.jsonl --ownership ownership.npy --policy policy.npy --move-loc 100
    
    # With before/after comparison
    python extract_features.py --board moves.jsonl --ownership after.npy --before-ownership before.npy --before-board moves_before.jsonl
    
    # Load from game directory
    python extract_features.py --game-dir games/1dae1d1e-4455-4570-82a2-2b3bf0b8f147 --move-number 10
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

# Add the python directory to the path
sys.path.append(str(Path(__file__).parent.parent / "python"))

from board import Board
from snorkel_board_positions import (
    analyze_position_comprehensive,
    territory_sizes,
    count_building_territory,
    urgency_by_region,
    urgency_intensity_by_region,
)


def load_board_from_moves(moves_data: list, up_to_move: Optional[int] = None) -> Board:
    """Create a Board object by playing moves from moves.jsonl data."""
    board = Board(19)
    
    for move_data in moves_data:
        move_num = move_data.get('move_number', 0)
        if up_to_move is not None and move_num > up_to_move:
            break
            
        player_str = move_data.get('player', '')
        move_loc = move_data.get('move_loc', 0)
        
        if player_str == 'b':
            player = Board.BLACK
        elif player_str == 'w':
            player = Board.WHITE
        else:
            continue
            
        if move_loc != 0:  # Skip pass moves for board reconstruction
            try:
                board.play(player, move_loc)
            except Exception as e:
                print(f"Warning: Could not play move {move_num}: {e}")
                continue
    
    return board


def load_ownership(file_path: Path) -> np.ndarray:
    """Load ownership map from file (supports .npy, .json, or jsonl format)."""
    if file_path.suffix == '.npy':
        ownership = np.load(file_path)
    elif file_path.suffix == '.json':
        with open(file_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                ownership = np.array(data).reshape(19, 19)
            else:
                ownership = np.array(data.get('ownership', data)).reshape(19, 19)
    elif file_path.suffix == '.jsonl':
        with open(file_path, 'r') as f:
            line = f.readline()
            data = json.loads(line)
            ownership = np.array(data.get('ownership', data)).reshape(19, 19)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    # Ensure it's 19x19
    if ownership.shape != (19, 19):
        ownership = ownership.reshape(19, 19)
    
    return ownership


def load_policy(file_path: Path) -> np.ndarray:
    """Load policy distribution from file."""
    if file_path.suffix == '.npy':
        policy = np.load(file_path)
    elif file_path.suffix in ['.json', '.jsonl']:
        with open(file_path, 'r') as f:
            if file_path.suffix == '.jsonl':
                line = f.readline()
                data = json.loads(line)
            else:
                data = json.load(f)
            
            # Try different possible keys
            if 'policy0' in data:
                policy = np.array(data['policy0'][:361])
            elif 'policy' in data:
                policy = np.array(data['policy'][:361])
            else:
                policy = np.array(data)[:361]
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    # Ensure it's 361 elements
    if len(policy) > 361:
        policy = policy[:361]
    elif len(policy) < 361:
        # Pad with zeros if needed
        padded = np.zeros(361)
        padded[:len(policy)] = policy
        policy = padded
    
    return policy


def load_from_game_dir(game_dir: Path, move_number: int) -> Dict[str, Any]:
    """Load board, ownership, and policy from a game directory."""
    moves_path = game_dir / "moves.jsonl"
    
    if not moves_path.exists():
        raise FileNotFoundError(f"moves.jsonl not found in {game_dir}")
    
    # Load moves up to move_number
    moves_data = []
    with open(moves_path, 'r') as f:
        for line in f:
            move_data = json.loads(line)
            moves_data.append(move_data)
            if move_data.get('move_number', 0) >= move_number:
                break
    
    if not moves_data:
        raise ValueError(f"No moves found in {moves_path}")
    
    # Get the specific move
    target_move = None
    for move_data in moves_data:
        if move_data.get('move_number', 0) == move_number:
            target_move = move_data
            break
    
    if target_move is None:
        raise ValueError(f"Move {move_number} not found in game data")
    
    # Create board state
    board = load_board_from_moves(moves_data, up_to_move=move_number)
    
    # Extract ownership and policy
    ownership = np.array(target_move['ownership']).reshape(19, 19)
    policy = np.array(target_move.get('policy0', [0] * 361)[:361])
    
    # Get move location and player
    move_loc = target_move.get('move_loc', None)
    player_str = target_move.get('player', 'b')
    player = Board.BLACK if player_str == 'b' else Board.WHITE
    
    # Get before state if available
    before_board = None
    before_ownership = None
    if move_number > 0:
        before_move = None
        for move_data in moves_data:
            if move_data.get('move_number', 0) == move_number - 1:
                before_move = move_data
                break
        
        if before_move:
            before_board = load_board_from_moves(moves_data, up_to_move=move_number - 1)
            before_ownership = np.array(before_move['ownership']).reshape(19, 19)
    
    return {
        'board': board,
        'ownership': ownership,
        'policy': policy,
        'player': player,
        'move_loc': move_loc,
        'before_board': before_board,
        'before_ownership': before_ownership,
    }


def extract_features(
    board: Board,
    ownership: np.ndarray,
    policy: Optional[np.ndarray] = None,
    player: Optional[int] = None,
    move_loc: Optional[int] = None,
    last_move_loc: Optional[int] = None,
    before_ownership: Optional[np.ndarray] = None,
    before_board: Optional[Board] = None,
    output_format: str = 'pretty'
) -> Dict[str, Any]:
    """
    Extract features from board position.
    
    Args:
        board: Board state
        ownership: Ownership map (19x19) from current player's perspective
        policy: Policy distribution (361), optional
        player: Current player (Board.BLACK or Board.WHITE), optional (uses board.pla)
        move_loc: Location of current move, optional
        last_move_loc: Location of last move, optional
        before_ownership: Ownership before move, optional
        before_board: Board state before move, optional
        output_format: 'pretty', 'json', or 'dict'
    
    Returns:
        Dictionary of features
    """
    # Use default policy if not provided
    if policy is None:
        policy = np.ones(361) / 361  # Uniform policy
    
    # Extract comprehensive features
    features = analyze_position_comprehensive(
        board=board,
        ownership=ownership,
        policy=policy,
        player=player,
        move_loc=move_loc,
        last_move_loc=last_move_loc,
        before_ownership=before_ownership,
        before_board=before_board,
    )
    
    return features


def print_features_pretty(features: Dict[str, Any]):
    """Print features in a human-readable format."""
    print("=" * 80)
    print("EXTRACTED FEATURES")
    print("=" * 80)
    
    # Group features by category
    categories = {
        'Urgency': ['urgency', 'urgency_intensity'],
        'Territory': [
            'potential_territory', 'solid_territory',
            'building_count', 'building_intensity', 'building_sum',
            'solidification_count', 'solidification_intensity', 'solidification_sum',
            'reduction_count', 'reduction_intensity', 'reduction_sum',
        ],
        'Territory (Local)': [
            'building_count_local', 'building_intensity_local', 'building_sum_local',
            'solidification_count_local', 'solidification_intensity_local', 'solidification_sum_local',
            'reduction_count_local', 'reduction_intensity_local', 'reduction_sum_local',
        ],
        'Territory (Global)': [
            'building_count_global', 'building_intensity_global', 'building_sum_global',
            'solidification_count_global', 'solidification_intensity_global', 'solidification_sum_global',
            'reduction_count_global', 'reduction_intensity_global', 'reduction_sum_global',
        ],
        'Tactical': [
            'cut', 'connection', 'connection_strength_gain',
            'extension', 'liberties', 'atari',
            'only_move', 'tenuki',
        ],
        'Attack': [
            'attack', 'avg_attack_intensity', 'max_attack_intensity',
            'killing_attack', 'kill_intensity',
            'reduce_aji', 'aji_reduction_intensity',
        ],
        'Sacrifice': [
            'direct_sacrifice', 'sacrifice_intensity',
            'indirect_sacrifice', 'indirect_sacrifice_intensity',
        ],
        'Groups': [
            'group_strength_delta', 'group_connectivity_delta',
            'influence_count_delta', 'influence_strength_delta',
            'creates_new_group',
        ],
        'Other': [
            'invasion', 'invasion_intensity',
            'leaves_weakness',
        ],
    }
    
    for category, keys in categories.items():
        print(f"\n{category}:")
        print("-" * 80)
        for key in keys:
            if key in features:
                value = features[key]
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")
    
    # Print regional features
    if 'building_count_by_region' in features:
        print("\nRegional Features:")
        print("-" * 80)
        regions = ['corner_tl', 'corner_tr', 'corner_bl', 'corner_br',
                   'side_left', 'side_right', 'side_top', 'side_bottom', 'center']
        
        for region in regions:
            print(f"\n  {region}:")
            if 'building_count_by_region' in features:
                print(f"    Building: {features['building_count_by_region'].get(region, 0)} "
                      f"(intensity: {features['building_intensity_by_region'].get(region, 0.0):.3f})")
            if 'reduction_count_by_region' in features:
                print(f"    Reduction: {features['reduction_count_by_region'].get(region, 0)} "
                      f"(intensity: {features['reduction_intensity_by_region'].get(region, 0.0):.3f})")


def main():
    parser = argparse.ArgumentParser(
        description='Extract features from board positions and ownership maps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input options
    parser.add_argument('--board', type=str, help='Board state: "empty", path to moves.jsonl, or path to board file')
    parser.add_argument('--ownership', type=str, help='Path to ownership map file (.npy, .json, or .jsonl)')
    parser.add_argument('--policy', type=str, help='Path to policy distribution file (optional)')
    parser.add_argument('--player', type=str, choices=['b', 'w', 'black', 'white'], help='Current player (b/w)')
    parser.add_argument('--move-loc', type=int, help='Location of current move')
    parser.add_argument('--last-move-loc', type=int, help='Location of last move')
    
    # Before/after comparison
    parser.add_argument('--before-ownership', type=str, help='Path to ownership map before move')
    parser.add_argument('--before-board', type=str, help='Path to board state before move (moves.jsonl)')
    
    # Game directory loading
    parser.add_argument('--game-dir', type=str, help='Path to game directory (loads from moves.jsonl)')
    parser.add_argument('--move-number', type=int, help='Move number to analyze (when using --game-dir)')
    
    # Output options
    parser.add_argument('--output', type=str, choices=['pretty', 'json', 'dict'], default='pretty',
                       help='Output format')
    parser.add_argument('--output-file', type=str, help='Save output to file (JSON format)')
    
    args = parser.parse_args()
    
    # Load data
    if args.game_dir:
        # Load from game directory
        game_dir = Path(args.game_dir)
        if not game_dir.exists():
            raise FileNotFoundError(f"Game directory not found: {game_dir}")
        
        if args.move_number is None:
            raise ValueError("--move-number is required when using --game-dir")
        
        data = load_from_game_dir(game_dir, args.move_number)
        board = data['board']
        ownership = data['ownership']
        policy = data.get('policy')
        player = data.get('player')
        move_loc = data.get('move_loc')
        before_board = data.get('before_board')
        before_ownership = data.get('before_ownership')
        
    else:
        # Load from individual files
        if args.ownership is None:
            raise ValueError("--ownership is required (or use --game-dir)")
        
        ownership = load_ownership(Path(args.ownership))
        
        # Load board
        if args.board is None or args.board.lower() == 'empty':
            board = Board(19)
        elif Path(args.board).exists():
            if Path(args.board).suffix == '.jsonl':
                with open(args.board, 'r') as f:
                    moves_data = [json.loads(line) for line in f]
                board = load_board_from_moves(moves_data)
            else:
                raise ValueError(f"Unsupported board file format: {args.board}")
        else:
            raise ValueError(f"Board file not found: {args.board}")
        
        # Load policy
        policy = None
        if args.policy:
            policy = load_policy(Path(args.policy))
        
        # Load before state
        before_board = None
        before_ownership = None
        if args.before_ownership:
            before_ownership = load_ownership(Path(args.before_ownership))
        if args.before_board:
            if Path(args.before_board).suffix == '.jsonl':
                with open(args.before_board, 'r') as f:
                    moves_data = [json.loads(line) for line in f]
                before_board = load_board_from_moves(moves_data)
        
        # Parse player
        player = None
        if args.player:
            player = Board.BLACK if args.player.lower() in ['b', 'black'] else Board.WHITE
        
        move_loc = args.move_loc
        last_move_loc = args.last_move_loc
    
    # Extract features
    features = extract_features(
        board=board,
        ownership=ownership,
        policy=policy,
        player=player,
        move_loc=move_loc,
        last_move_loc=last_move_loc,
        before_ownership=before_ownership,
        before_board=before_board,
        output_format=args.output
    )
    
    # Output results
    if args.output == 'json' or args.output_file:
        output_json = json.dumps(features, indent=2, default=str)
        if args.output_file:
            with open(args.output_file, 'w') as f:
                f.write(output_json)
            print(f"Features saved to {args.output_file}")
        else:
            print(output_json)
    elif args.output == 'pretty':
        print_features_pretty(features)
    else:
        print(features)


if __name__ == '__main__':
    main()

