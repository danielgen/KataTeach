#!/usr/bin/env python3
"""Convert between label_page.py export format and JSONL labels format.

This script provides bidirectional conversion between:
1. label_page.py export format (single JSON with nested structure)
2. labels.jsonl format (line-delimited JSON for training)

Usage:
    # Convert label_page.py export to labels.jsonl
    python convert_labels.py to-jsonl game_annotations.json slates.jsonl moves.jsonl labels.jsonl
    
    # Convert with concept filtering (exclude concepts with < 20 examples)
    python convert_labels.py to-jsonl game_annotations.json slates.jsonl moves.jsonl labels.jsonl --min-concept-examples 20
    
    # Convert labels.jsonl back to label_page.py import format
    python convert_labels.py from-jsonl labels.jsonl slates.jsonl game_annotations.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
import uuid


def human_to_sgf(human_coord: str, board_size: int = 19) -> str:
    """Convert human coordinate (like 'C16') to SGF coordinate (like 'cd').
    
    Args:
        human_coord: Human coordinate like 'C16', 'Q4', etc.
        board_size: Board size (default 19)
    
    Returns:
        SGF coordinate like 'cd', 'pd', etc.
    """
    if not human_coord or len(human_coord) < 2:
        return ""
    
    try:
        # Parse human coordinate: letter + number
        letter = human_coord[0].upper()
        number = int(human_coord[1:])
        
        # Convert letter to x coordinate (A=0, B=1, ..., T=18, skip I)
        if letter <= 'H':
            x = ord(letter) - ord('A')
        else:  # letter >= 'J' (skip I)
            x = ord(letter) - ord('A') - 1
        
        # Convert number to y coordinate (1=18, 2=17, ..., 19=0)
        y = board_size - number
        
        # Convert to SGF format (lowercase)
        sgf_x = chr(ord('a') + x)
        sgf_y = chr(ord('a') + y)
        
        return sgf_x + sgf_y
    except:
        return ""


def sgf_to_move_loc(sgf_coord: str, board_size: int = 19) -> int:
    """Convert SGF coordinate to KataGo move_loc integer.
    
    Args:
        sgf_coord: SGF coordinate like 'qd', 'pd', etc.
        board_size: Board size (default 19)
    
    Returns:
        KataGo move_loc integer, or -1 for pass/invalid
    """
    if not sgf_coord or sgf_coord == 'pass' or len(sgf_coord) != 2:
        return -1  # Use -1 for pass/invalid moves
    
    try:
        x = ord(sgf_coord[0]) - ord('a')
        y = ord(sgf_coord[1]) - ord('a')
        
        if 0 <= x < board_size and 0 <= y < board_size:
            return y * board_size + x
        else:
            return -1
    except:
        return -1


def move_loc_to_sgf(move_loc: int, board_size: int = 19) -> str:
    """Convert KataGo move_loc integer to SGF coordinate.
    
    Args:
        move_loc: KataGo move_loc integer
        board_size: Board size (default 19)
    
    Returns:
        SGF coordinate like 'qd', 'pd', or 'pass'
    """
    if move_loc < 0 or move_loc >= board_size * board_size:
        return 'pass'
    
    y = move_loc // board_size
    x = move_loc % board_size
    
    return chr(ord('a') + x) + chr(ord('a') + y)


def move_loc_to_human(move_loc: int, board_size: int = 19) -> str:
    """Convert KataGo move_loc integer to human-readable coordinate.
    
    Args:
        move_loc: KataGo move_loc integer
        board_size: Board size (default 19)
    
    Returns:
        Human coordinate like 'Q16', 'D4', or 'pass'
    """
    if move_loc < 0 or move_loc >= board_size * board_size:
        return 'pass'
    
    y = move_loc // board_size
    x = move_loc % board_size
    
    # Convert to human-readable format: A-T (skip I), 1-19
    if x < 8:
        letter = chr(ord('A') + x)
    else:
        letter = chr(ord('A') + x + 1)  # Skip 'I'
    
    # y: 0->19, 1->18, ..., 18->1
    number = board_size - y
    
    return f"{letter}{number}"


def convert_to_jsonl(
    label_page_export: Path,
    slates_jsonl: Path,
    moves_jsonl: Path,
    output_labels_jsonl: Path,
    game_uuid: str = None,
    min_concept_examples: int = 0
) -> None:
    """Convert label_page.py export to labels.jsonl format.
    
    Args:
        label_page_export: Input JSON file from label_page.py export
        slates_jsonl: Slates JSONL file to match against
        moves_jsonl: Moves JSONL file to match against  
        output_labels_jsonl: Output labels.jsonl file
        game_uuid: Game UUID (if None, will try to extract from slates)
        min_concept_examples: Minimum number of examples required to keep a concept (0 = no filtering)
    """
    
    # Load label page export
    with label_page_export.open('r', encoding='utf-8') as f:
        export_data = json.load(f)
    
    annotations = export_data.get('annotations', {})
    per_move_labels = annotations.get('per_move_labels', {})
    global_labels = annotations.get('global_labels', {})
    
    # Load slates to get slate_id mapping
    slate_mapping = {}  # pos_idx -> slate_id
    if slates_jsonl.exists():
        with slates_jsonl.open('r', encoding='utf-8') as f:
            for line in f:
                slate = json.loads(line.strip())
                slate_mapping[slate['pos_idx']] = slate['slate_id']
                if game_uuid is None:
                    game_uuid = slate['game_uuid']
    
    # Load moves to get move_loc mapping
    move_mapping = {}  # (slate_id, sgf_coord) -> move_loc
    if moves_jsonl.exists():
        with moves_jsonl.open('r', encoding='utf-8') as f:
            for line in f:
                move = json.loads(line.strip())
                key = (move['slate_id'], move['coord_sgf'])
                move_mapping[key] = move['move_loc']
    
    # Count concept frequencies if filtering is enabled
    concept_counts = {}
    if min_concept_examples > 0:
        print(f"Counting concept frequencies (min_examples={min_concept_examples})...")
        for pos_idx_str, move_labels in per_move_labels.items():
            for sgf_coord, tags in move_labels.items():
                for tag_name, is_set in tags.items():
                    if is_set:
                        concept_counts[tag_name] = concept_counts.get(tag_name, 0) + 1
        
        # Filter concepts
        filtered_concepts = {concept: count for concept, count in concept_counts.items() 
                           if count >= min_concept_examples}
        excluded_concepts = {concept: count for concept, count in concept_counts.items() 
                           if count < min_concept_examples}
        
        print(f"Total concepts: {len(concept_counts)}")
        print(f"Concepts kept (>= {min_concept_examples} examples): {len(filtered_concepts)}")
        print(f"Concepts excluded (< {min_concept_examples} examples): {len(excluded_concepts)}")
        
        if excluded_concepts:
            print("Excluded concepts:")
            for concept, count in sorted(excluded_concepts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {concept}: {count} examples")
    else:
        filtered_concepts = None
    
    # Convert to labels.jsonl format
    labels = []
    
    # Process per-move labels
    for pos_idx_str, move_labels in per_move_labels.items():
        pos_idx = int(pos_idx_str)
        slate_id = slate_mapping.get(pos_idx)
        
        if not slate_id:
            print(f"Warning: No slate_id found for position {pos_idx}")
            continue
            
        for human_coord, tags in move_labels.items():
            # Convert human coordinate to SGF format
            sgf_coord = human_to_sgf(human_coord)
            if not sgf_coord:
                print(f"Warning: Could not convert human coordinate {human_coord} to SGF at position {pos_idx}")
                continue
            
            move_loc = move_mapping.get((slate_id, sgf_coord))
            
            if move_loc is None:
                # Try to convert SGF coordinate directly
                move_loc = sgf_to_move_loc(sgf_coord)
                if move_loc == -1:
                    print(f"Warning: Could not convert move {human_coord} (SGF: {sgf_coord}) at position {pos_idx}")
                    continue
            
            # Group tags by category
            grouped_tags = {
                "global": [],
                "initiative": [],
                "strategic": [],
                "tactical": [],
                "stress_area": [],
                "reduce_area": [],
                "move_attributes": []
            }
            
            for tag_name, is_set in tags.items():
                if not is_set:
                    continue
                
                # Skip concept if it's filtered out
                if filtered_concepts is not None and tag_name not in filtered_concepts:
                    continue
                    
                # Parse category:tag format
                if ':' in tag_name:
                    category, tag = tag_name.split(':', 1)
                    if category in grouped_tags:
                        grouped_tags[category].append(tag)
                    else:
                        # Fallback to global if category not recognized
                        grouped_tags["global"].append(tag_name)
                else:
                    # Assume global tag if no category prefix
                    grouped_tags["global"].append(tag_name)
            
            # Create concept vector for the filtered concepts
            concept_vector = [0.0] * len(filtered_concepts) if filtered_concepts else []
            
            # Set active concepts based on filtered concept list
            if filtered_concepts:
                for i, (concept_name, _) in enumerate(sorted(filtered_concepts.items())):
                    # Check if this concept is active in the current move
                    concept_active = False
                    for tag_name, is_set in tags.items():
                        if is_set and tag_name == concept_name:
                            concept_active = True
                            break
                    
                    if concept_active:
                        concept_vector[i] = 1.0
            
            # Only create label record if there are actual concept activations
            if any(concept_vector):
                label_record = {
                    "slate_id": slate_id,
                    "move_idx361": move_loc,
                    "concept_labels": concept_vector
                }
                labels.append(label_record)
    
    # Write labels.jsonl
    with output_labels_jsonl.open('w', encoding='utf-8') as f:
        for label in labels:
            f.write(json.dumps(label, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(labels)} label records to {output_labels_jsonl}")


def convert_from_jsonl(
    labels_jsonl: Path,
    slates_jsonl: Path,
    output_label_page: Path,
    game_uuid: str = None
) -> None:
    """Convert labels.jsonl back to label_page.py import format.
    
    Args:
        labels_jsonl: Input labels.jsonl file
        slates_jsonl: Slates JSONL file for context
        output_label_page: Output JSON file compatible with label_page.py
        game_uuid: Game UUID (if None, will try to extract from slates)
    """
    
    # Load slates for context
    slate_context = {}  # slate_id -> slate_info
    if slates_jsonl.exists():
        with slates_jsonl.open('r', encoding='utf-8') as f:
            for line in f:
                slate = json.loads(line.strip())
                slate_context[slate['slate_id']] = slate
                if game_uuid is None:
                    game_uuid = slate['game_uuid']
    
    # Load labels
    labels_by_position = {}  # pos_idx -> {sgf_coord -> {tag -> bool}}
    global_labels_by_position = {}  # pos_idx -> {tag -> bool}
    
    if labels_jsonl.exists():
        with labels_jsonl.open('r', encoding='utf-8') as f:
            for line in f:
                label = json.loads(line.strip())
                slate_id = label['slate_id']
                move_loc = label['move_loc']
                tags = label['tags']
                
                # Get position info from slate context
                slate_info = slate_context.get(slate_id)
                if not slate_info:
                    print(f"Warning: No slate context for {slate_id}")
                    continue
                
                pos_idx = slate_info['pos_idx']
                sgf_coord = move_loc_to_sgf(move_loc)
                
                # Initialize position if needed
                if pos_idx not in labels_by_position:
                    labels_by_position[pos_idx] = {}
                if sgf_coord not in labels_by_position[pos_idx]:
                    labels_by_position[pos_idx][sgf_coord] = {}
                
                # Convert grouped tags back to flat format
                for category, tag_list in tags.items():
                    for tag in tag_list:
                        if category == "global":
                            # Global tags go to global_labels
                            if pos_idx not in global_labels_by_position:
                                global_labels_by_position[pos_idx] = {}
                            global_labels_by_position[pos_idx][tag] = True
                        else:
                            # Category tags get prefixed
                            full_tag = f"{category}:{tag}"
                            labels_by_position[pos_idx][sgf_coord][full_tag] = True
    
    # Create label_page.py compatible export
    export_data = {
        "format_version": "1.0",
        "exported_at": "converted_from_jsonl",
        "game_metadata": {
            "game_uuid": game_uuid or "unknown"
        },
        "annotation_statistics": {
            "total_positions_annotated": len(labels_by_position),
            "total_move_annotations": sum(len(moves) for moves in labels_by_position.values()),
            "total_global_annotations": len(global_labels_by_position),
            "annotation_timestamp": "converted_from_jsonl"
        },
        "annotations": {
            "per_move_labels": {str(k): v for k, v in labels_by_position.items()},
            "global_labels": {str(k): v for k, v in global_labels_by_position.items()}
        },
        "source_info": {
            "converted_from": str(labels_jsonl),
            "conversion_note": "Converted from JSONL format"
        }
    }
    
    # Write output
    with output_label_page.open('w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"Converted labels to label_page.py format: {output_label_page}")


def main():
    parser = argparse.ArgumentParser(description="Convert between label formats")
    subparsers = parser.add_subparsers(dest='command', help='Conversion direction')
    
    # to-jsonl command
    to_jsonl = subparsers.add_parser('to-jsonl', help='Convert label_page.py export to JSONL')
    to_jsonl.add_argument('label_export', type=Path, help='Input JSON from label_page.py export')
    to_jsonl.add_argument('slates_jsonl', type=Path, help='Slates JSONL file for reference')
    to_jsonl.add_argument('moves_jsonl', type=Path, help='Moves JSONL file for reference')
    to_jsonl.add_argument('output_labels', type=Path, help='Output labels.jsonl file')
    to_jsonl.add_argument('--game-uuid', help='Game UUID (auto-detected if not provided)')
    to_jsonl.add_argument('--min-concept-examples', type=int, default=0, 
                         help='Minimum number of examples required to keep a concept (default: 0 = no filtering)')
    
    # from-jsonl command
    from_jsonl = subparsers.add_parser('from-jsonl', help='Convert JSONL to label_page.py format')
    from_jsonl.add_argument('labels_jsonl', type=Path, help='Input labels.jsonl file')
    from_jsonl.add_argument('slates_jsonl', type=Path, help='Slates JSONL file for context')
    from_jsonl.add_argument('output_json', type=Path, help='Output JSON for label_page.py import')
    from_jsonl.add_argument('--game-uuid', help='Game UUID (auto-detected if not provided)')
    
    args = parser.parse_args()
    
    if args.command == 'to-jsonl':
        convert_to_jsonl(
            args.label_export,
            args.slates_jsonl,
            args.moves_jsonl,
            args.output_labels,
            args.game_uuid,
            args.min_concept_examples
        )
    elif args.command == 'from-jsonl':
        convert_from_jsonl(
            args.labels_jsonl,
            args.slates_jsonl,
            args.output_json,
            args.game_uuid
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
