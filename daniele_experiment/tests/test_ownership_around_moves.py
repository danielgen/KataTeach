#!/usr/bin/env python3
"""
Test ownership convention by checking ownership around actual move locations.
"""

import json
import numpy as np
import sys
from pathlib import Path

# Add the python directory to the path
sys.path.append(str(Path(__file__).parent.parent / "python"))

from board import Board

def loc_to_xy(loc, size=19):
    """Convert board location to (x,y) coordinates."""
    if loc == 0:  # Pass
        return None, None
    return (loc % (size + 1)) - 1, (loc // (size + 1)) - 1

def check_ownership_around_moves():
    """Check ownership around actual move locations to determine perspective."""
    
    # Load real game data
    game_dir = Path(__file__).parent.parent / "games" / "a64ca662-9218-4888-b9f3-ca9e93631ab4"
    
    with open(game_dir / "moves.jsonl", 'r') as f:
        move1 = json.loads(f.readline())
        move2 = json.loads(f.readline())
        move3 = json.loads(f.readline())
    
    ownership1 = np.array(move1['ownership']).reshape(19, 19)
    ownership2 = np.array(move2['ownership']).reshape(19, 19)
    ownership3 = np.array(move3['ownership']).reshape(19, 19)
    
    print("="*70)
    print("OWNERSHIP AROUND ACTUAL MOVE LOCATIONS")
    print("="*70)
    
    # Check first few moves
    moves = [move1, move2, move3]
    ownerships = [ownership1, ownership2, ownership3]
    
    for i, (move, ownership) in enumerate(zip(moves, ownerships)):
        player = move['player']
        move_loc = move['move_loc']
        move_num = move['move_number']
        
        print(f"\nMove {move_num} - Player: {player}, Location: {move_loc}")
        
        if move_loc == 0:  # Pass
            print("  Pass move - no location to check")
            continue
            
        x, y = loc_to_xy(move_loc)
        if x is None or y is None:
            print(f"  Invalid location: {move_loc}")
            continue
            
        print(f"  Board coordinates: ({x}, {y})")
        
        # Check ownership at the move location and surrounding area
        print(f"  Ownership at move location: {ownership[y, x]:.4f}")
        
        # Check 3x3 area around the move
        print("  Ownership in 3x3 area around move:")
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < 19 and 0 <= ny < 19:
                    own_val = ownership[ny, nx]
                    marker = " ← MOVE" if dx == 0 and dy == 0 else ""
                    print(f"    ({nx:2d},{ny:2d}): {own_val:7.4f}{marker}")
        
        # Analyze the pattern
        center_own = ownership[y, x]
        if center_own > 0:
            print(f"  ✅ Move location has POSITIVE ownership ({center_own:.4f})")
            if player == 'b':
                print("     → If this is Black's move, ownership is from Black's perspective")
            else:
                print("     → If this is White's move, ownership is from White's perspective")
        elif center_own < 0:
            print(f"  ❌ Move location has NEGATIVE ownership ({center_own:.4f})")
            if player == 'b':
                print("     → If this is Black's move, ownership is from White's perspective")
            else:
                print("     → If this is Black's move, ownership is from Black's perspective")
        else:
            print(f"  ⚠️  Move location has ZERO ownership ({center_own:.4f})")
    
    # Check consistency across moves
    print("\n" + "="*70)
    print("CONSISTENCY CHECK")
    print("="*70)
    
    # Check if the pattern is consistent
    black_moves = []
    white_moves = []
    
    for i, (move, ownership) in enumerate(zip(moves, ownerships)):
        if move['move_loc'] == 0:  # Skip pass
            continue
        x, y = loc_to_xy(move['move_loc'])
        if x is None or y is None:
            continue
            
        own_val = ownership[y, x]
        if move['player'] == 'b':
            black_moves.append(own_val)
        else:
            white_moves.append(own_val)
    
    print(f"Black moves ownership values: {[f'{v:.4f}' for v in black_moves]}")
    print(f"White moves ownership values: {[f'{v:.4f}' for v in white_moves]}")
    
    if black_moves and white_moves:
        black_avg = np.mean(black_moves)
        white_avg = np.mean(white_moves)
        
        print(f"\nAverage ownership for Black moves: {black_avg:.4f}")
        print(f"Average ownership for White moves: {white_avg:.4f}")
        
        if black_avg > 0 and white_avg > 0:
            print("✅ Both players have positive ownership at their moves")
            print("   → Ownership is from CURRENT PLAYER perspective")
        elif black_avg < 0 and white_avg < 0:
            print("❌ Both players have negative ownership at their moves")
            print("   → Ownership is from OPPONENT perspective")
        elif black_avg > 0 and white_avg < 0:
            print("🔄 Mixed pattern - ownership perspective unclear")
        else:
            print("🔄 Mixed pattern - ownership perspective unclear")
    
    return black_moves, white_moves

if __name__ == "__main__":
    check_ownership_around_moves()
