#!/usr/bin/env python3
"""
Test script to inspect evidence packets before API calls.

Usage:
    python scripts/test_evidence_packet.py --game-id <game_id> --move-number <move_number>
    python scripts/test_evidence_packet.py --game-id 0a0e3711-54f8-4067-99ff-75eaab7cba4e --move-number 1
"""
import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from kata_teach.commentary.evidence import (
    load_snorkel_data,
    load_concepts_data,
    build_evidence_packet,
)


def main():
    parser = argparse.ArgumentParser(
        description="Test evidence packet building",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--games-dir",
        type=Path,
        default=Path("games"),
        help="Directory containing game data (default: games)",
    )
    
    parser.add_argument(
        "--html-data-dir",
        type=Path,
        default=Path("daniele_experiment/linear_probes/html_data"),
        help="Directory containing html data (default: daniele_experiment/linear_probes/html_data)",
    )
    
    parser.add_argument(
        "--game-id",
        type=str,
        required=True,
        help="Game ID to test",
    )
    
    parser.add_argument(
        "--move-number",
        type=int,
        default=1,
        help="Move number to test (default: 1)",
    )
    
    args = parser.parse_args()
    
    # Set up paths
    snorkel_path = args.games_dir / args.game_id / "snorkel.jsonl"
    concepts_path = args.html_data_dir / args.game_id / "concepts.json"
    
    if not snorkel_path.exists():
        print(f"ERROR: Snorkel file not found: {snorkel_path}")
        sys.exit(1)
    
    if not concepts_path.exists():
        print(f"ERROR: Concepts file not found: {concepts_path}")
        sys.exit(1)
    
    # Load data
    print(f"Loading data for game {args.game_id}, move {args.move_number}...")
    snorkel_data = load_snorkel_data(snorkel_path)
    concepts_data = load_concepts_data(concepts_path)
    
    # Find the move
    moves = concepts_data.get('moves', [])
    move_data = None
    for m in moves:
        if m['move_number'] == args.move_number:
            move_data = m
            break
    
    if not move_data:
        print(f"ERROR: Move {args.move_number} not found in concepts.json")
        print(f"Available moves: {[m['move_number'] for m in moves[:10]]}...")
        sys.exit(1)
    
    if args.move_number not in snorkel_data:
        print(f"ERROR: Move {args.move_number} not found in snorkel.jsonl")
        sys.exit(1)
    
    snorkel_entry = snorkel_data[args.move_number]
    
    # Build evidence packet
    print("\n" + "="*80)
    print("BUILDING EVIDENCE PACKET")
    print("="*80 + "\n")
    
    # Load global stats
    global_stats = None
    if (args.games_dir / "global_stats.json").exists():
        print("Loading global_stats.json...")
        from kata_teach.commentary.evidence import load_global_stats
        global_stats = load_global_stats(args.games_dir)
        if global_stats:
            print(f"  Loaded stats for {len(global_stats.get('features', {}))} features")
        else:
            print("  Failed to load global_stats.json")
    else:
        print("  global_stats.json not found - percentiles will not be computed")
    
    packet = build_evidence_packet(
        game_id=args.game_id,
        move_data=move_data,
        snorkel_entry=snorkel_entry,
        global_stats=global_stats,
        games_dir=args.games_dir,
    )
    
    # Print the evidence packet as JSON
    print("EVIDENCE PACKET JSON:")
    print("-" * 80)
    print(packet.to_json())
    print("-" * 80)
    
    # Also print a human-readable summary
    print("\n" + "="*80)
    print("HUMAN-READABLE SUMMARY")
    print("="*80 + "\n")
    
    print(f"Game ID: {packet.game_id}")
    print(f"Player: {packet.player}")
    print(f"Move Number: {packet.move_number}")
    print(f"\nSelected Concepts ({len(packet.selected_concepts)}):")
    for concept in packet.selected_concepts:
        delta = packet.concept_deltas.get(concept, 0)
        print(f"  - {concept}: Δ = {delta:+.3f}")
    
    print(f"\nEvidence Highlights ({len(packet.evidence_highlights)}):")
    for highlight in packet.evidence_highlights:
        print(f"  - {highlight}")
    
    print(f"\nSnorkel Summary:")
    snorkel_dict = packet.snorkel
    print(f"  Territory:")
    print(f"    - building_count: {snorkel_dict.get('building_count', 0)} "
          f"({snorkel_dict.get('building_count_magnitude', 'N/A')})")
    print(f"    - solidification_count: {snorkel_dict.get('solidification_count', 0)} "
          f"({snorkel_dict.get('solidification_count_magnitude', 'N/A')})")
    print(f"    - reduction_count: {snorkel_dict.get('reduction_count', 0)} "
          f"({snorkel_dict.get('reduction_count_magnitude', 'N/A')})")
    
    print(f"  Tactics:")
    print(f"    - cut: {snorkel_dict.get('cut', False)}")
    print(f"    - connection: {snorkel_dict.get('connection', False)}")
    print(f"    - connection_strength_gain: {snorkel_dict.get('connection_strength_gain', 0.0)} "
          f"({snorkel_dict.get('connection_strength_gain_magnitude', 'N/A')})")
    print(f"    - extension: {snorkel_dict.get('extension', False)}")
    print(f"    - atari: {snorkel_dict.get('atari', False)}")
    print(f"    - invasion: {snorkel_dict.get('invasion', False)}")
    
    print(f"  Attack:")
    print(f"    - attack: {snorkel_dict.get('attack', False)}")
    print(f"    - max_attack_intensity: {snorkel_dict.get('max_attack_intensity')} "
          f"({snorkel_dict.get('max_attack_intensity_magnitude', 'N/A')})")
    print(f"    - attacked_groups_count: {snorkel_dict.get('attacked_groups_count', 0)}")
    
    print(f"  Group:")
    print(f"    - group_strength_delta: {snorkel_dict.get('group_strength_delta', 0.0)} "
          f"({snorkel_dict.get('group_strength_delta_magnitude', 'N/A')})")
    print(f"    - group_connectivity_delta: {snorkel_dict.get('group_connectivity_delta', 0.0)} "
          f"({snorkel_dict.get('group_connectivity_delta_magnitude', 'N/A')})")
    print(f"    - liberties: {snorkel_dict.get('liberties', 0)}")
    print(f"    - new_group: {snorkel_dict.get('new_group', False)}")
    
    print(f"\n  Other:")
    print(f"    - must_live: {snorkel_dict.get('must_live', False)}")
    print(f"    - influence_count_delta: {snorkel_dict.get('influence_count_delta', 0)} "
          f"({snorkel_dict.get('influence_count_delta_magnitude', 'N/A')})")
    
    print("\n" + "="*80)
    print("Raw snorkel entry percentiles (if available):")
    print("="*80 + "\n")
    analysis = snorkel_entry.get('analysis', {})
    percentiles = analysis.get('percentiles', {})
    if percentiles:
        print(json.dumps(percentiles, indent=2))
    else:
        print("No percentiles found in analysis.percentiles")
        print("\nChecking for individual percentile fields...")
        percentile_fields = {k: v for k, v in analysis.items() if 'percentile' in k.lower()}
        if percentile_fields:
            print(json.dumps(percentile_fields, indent=2))
        else:
            print("No percentile fields found.")


if __name__ == "__main__":
    main()

