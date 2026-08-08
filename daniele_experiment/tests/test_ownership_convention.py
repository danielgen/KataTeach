#!/usr/bin/env python3
"""
Test to verify ownership convention and check if it breaks snorkel logic.
"""

import json
import numpy as np
import sys
from pathlib import Path

# Add the python directory to the path
sys.path.append(str(Path(__file__).parent.parent / "python"))

from board import Board
from daniele_experiment.snorkel_board_positions import (
    territory_sizes,
    count_building_territory,
)

def analyze_ownership_convention():
    """Analyze the ownership convention in real data."""
    
    # Load real game data
    game_dir = Path(__file__).parent.parent / "games" / "a64ca662-9218-4888-b9f3-ca9e93631ab4"
    
    with open(game_dir / "moves.jsonl", 'r') as f:
        move1 = json.loads(f.readline())
        move2 = json.loads(f.readline())
    
    ownership1 = np.array(move1['ownership']).reshape(19, 19)
    ownership2 = np.array(move2['ownership']).reshape(19, 19)
    
    print("="*60)
    print("OWNERSHIP CONVENTION ANALYSIS")
    print("="*60)
    
    print(f"\nMove 1 - Player: {move1['player']}, Move: {move1['move_loc']}")
    print(f"Ownership range: [{ownership1.min():.3f}, {ownership1.max():.3f}]")
    print(f"Ownership mean: {ownership1.mean():.3f}")
    print(f"Positive count: {np.sum(ownership1 > 0)}")
    print(f"Negative count: {np.sum(ownership1 < 0)}")
    
    print(f"\nMove 2 - Player: {move2['player']}, Move: {move2['move_loc']}")
    print(f"Ownership range: [{ownership2.min():.3f}, {ownership2.max():.3f}]")
    print(f"Ownership mean: {ownership2.mean():.3f}")
    print(f"Positive count: {np.sum(ownership2 > 0)}")
    print(f"Negative count: {np.sum(ownership2 < 0)}")
    
    # Test territory analysis with different perspectives
    print("\n" + "="*60)
    print("TERRITORY ANALYSIS WITH DIFFERENT PERSPECTIVES")
    print("="*60)
    
    # Test with original ownership (assuming it's from KataGo's perspective)
    print("\nOriginal ownership (KataGo perspective):")
    pot1_orig, solid1_orig = territory_sizes(ownership1, Board.BLACK)
    pot2_orig, solid2_orig = territory_sizes(ownership2, Board.WHITE)
    print(f"Move 1 (Black): Potential={pot1_orig}, Solid={solid1_orig}")
    print(f"Move 2 (White): Potential={pot2_orig}, Solid={solid2_orig}")
    
    # Test with flipped ownership (assuming it should be from player perspective)
    print("\nFlipped ownership (Player perspective):")
    pot1_flip, solid1_flip = territory_sizes(-ownership1, Board.BLACK)
    pot2_flip, solid2_flip = territory_sizes(-ownership2, Board.WHITE)
    print(f"Move 1 (Black): Potential={pot1_flip}, Solid={solid1_flip}")
    print(f"Move 2 (White): Potential={pot2_flip}, Solid={solid2_flip}")
    
    # Test building territory between moves
    print("\n" + "="*60)
    print("BUILDING TERRITORY ANALYSIS")
    print("="*60)
    
    # Test with original ownership
    count_orig, intensity_orig = count_building_territory(ownership1, ownership2, Board.WHITE)
    print(f"Original ownership - Building count: {count_orig}, Intensity: {intensity_orig:.3f}")
    
    # Test with flipped ownership
    count_flip, intensity_flip = count_building_territory(-ownership1, -ownership2, Board.WHITE)
    print(f"Flipped ownership - Building count: {count_flip}, Intensity: {intensity_flip:.3f}")
    
    # Analyze the pattern
    print("\n" + "="*60)
    print("ANALYSIS AND CONCLUSION")
    print("="*60)
    
    print("\nKey observations:")
    print(f"1. Move 1 (Black): Mean ownership = {ownership1.mean():.3f} (negative)")
    print(f"2. Move 2 (White): Mean ownership = {ownership2.mean():.3f} (positive)")
    print(f"3. Original territory analysis shows very little solid territory")
    print(f"4. Flipped territory analysis shows more reasonable territory")
    
    # Check if the pattern makes sense
    if ownership1.mean() < 0 and ownership2.mean() > 0:
        print("\n🔍 PATTERN DETECTED:")
        print("   - Black move has negative mean ownership")
        print("   - White move has positive mean ownership")
        print("   - This suggests ownership is from KataGo's perspective, not player perspective")
        print("   - The snorkel code expects player perspective ownership")
        
        print("\n⚠️  POTENTIAL ISSUE:")
        print("   - Snorkel logic may be broken if ownership is not from player perspective")
        print("   - Territory analysis functions expect positive values for own territory")
        print("   - Need to flip ownership sign before passing to snorkel functions")
        
        return True
    else:
        print("\n✅ No clear pattern detected - ownership convention unclear")
        return False

if __name__ == "__main__":
    analyze_ownership_convention()
