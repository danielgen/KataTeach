#!/usr/bin/env python3
"""Enrich game data with annotation labels for better traceability.

This script takes the exported labels.json from the UI and merges them back
into the original combined game data, creating a comprehensive annotated
game record that includes:
- Original SGF data
- Policy analysis 
- Human annotations
- Game metadata
- Traceability information
"""

import argparse
import json
import uuid
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

def extract_game_metadata(sgf_content: str) -> Dict[str, str]:
    """Extract metadata from SGF content."""
    metadata = {}
    
    # Extract basic SGF properties
    properties = {
        'PB': 'black_player',
        'PW': 'white_player', 
        'DT': 'date',
        'RE': 'result',
        'KM': 'komi',
        'SZ': 'board_size',
        'RU': 'rules',
        'GN': 'game_name'
    }
    
    for sgf_prop, key in properties.items():
        import re
        pattern = f"{sgf_prop}\\[([^\\]]*)\\]"
        match = re.search(pattern, sgf_content)
        if match:
            metadata[key] = match.group(1)
    
    return metadata

def create_enriched_game_data(
    combined_data_path: Path,
    labels_path: Path,
    output_path: Path,
    annotation_session_id: str = None
) -> Dict[str, Any]:
    """Create enriched game data with annotations and metadata."""
    
    # Load original combined data
    with combined_data_path.open('r', encoding='utf-8') as f:
        combined_data = json.load(f)
    
    # Load annotation labels
    with labels_path.open('r', encoding='utf-8') as f:
        labels_data = json.load(f)
    
    # Extract game metadata from SGF
    sgf_content = combined_data.get('sgf', '')
    game_metadata = extract_game_metadata(sgf_content)
    
    # Create enriched structure
    enriched_data = {
        "format_version": "1.0",
        "created_at": datetime.now().isoformat(),
        "annotation_session_id": annotation_session_id or str(uuid.uuid4()),
        
        # Original data
        "sgf": sgf_content,
        "policy": combined_data.get('policy', {}),
        
        # Game metadata
        "game_metadata": {
            **game_metadata,
            "source_file": str(combined_data_path.name),
            "total_moves": len([k for k in combined_data.get('policy', {}).keys() if k.isdigit()]),
        },
        
        # Human annotations
        "annotations": {
            "per_move_labels": labels_data.get('perMoveLabels', {}),
            "global_labels": labels_data.get('globalLabels', {}),
            "annotation_metadata": {
                "annotated_positions": len(labels_data.get('perMoveLabels', {})),
                "total_move_annotations": sum(
                    len(moves) for moves in labels_data.get('perMoveLabels', {}).values()
                ),
                "global_annotations": len(labels_data.get('globalLabels', {}))
            }
        },
        
        # Enhanced position data (merge policy + annotations)
        "positions": {}
    }
    
    # Create enhanced position data that merges everything
    policy_data = combined_data.get('policy', {})
    per_move_labels = labels_data.get('perMoveLabels', {})
    global_labels = labels_data.get('globalLabels', {})
    
    # Get all positions (from policy or annotations)
    all_positions = set(policy_data.keys()) | set(per_move_labels.keys()) | set(global_labels.keys())
    
    for pos_key in all_positions:
        pos_num = int(pos_key) if pos_key.isdigit() else pos_key
        
        position_data = {
            "position_number": pos_num,
            "policy_analysis": policy_data.get(pos_key, {}),
            "move_annotations": per_move_labels.get(pos_key, {}),
            "global_annotations": global_labels.get(pos_key, {}),
        }
        
        # Add summary statistics
        if per_move_labels.get(pos_key):
            position_data["annotation_summary"] = {
                "annotated_moves": list(per_move_labels[pos_key].keys()),
                "total_tags_applied": sum(
                    sum(1 for v in move_tags.values() if v)
                    for move_tags in per_move_labels[pos_key].values()
                )
            }
        
        enriched_data["positions"][pos_key] = position_data
    
    # Save enriched data
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(enriched_data, f, indent=2, ensure_ascii=False)
    
    return enriched_data

def create_summary_report(enriched_data: Dict[str, Any]) -> str:
    """Create a human-readable summary report."""
    metadata = enriched_data.get('game_metadata', {})
    annotations = enriched_data.get('annotations', {})
    
    report = f"""
ANNOTATED GAME SUMMARY
=====================

Game Information:
- Players: {metadata.get('black_player', 'Unknown')} (Black) vs {metadata.get('white_player', 'Unknown')} (White)
- Date: {metadata.get('date', 'Unknown')}
- Result: {metadata.get('result', 'Unknown')}
- Rules: {metadata.get('rules', 'Unknown')}
- Komi: {metadata.get('komi', 'Unknown')}
- Total Moves: {metadata.get('total_moves', 'Unknown')}

Annotation Summary:
- Annotated Positions: {annotations.get('annotation_metadata', {}).get('annotated_positions', 0)}
- Total Move Annotations: {annotations.get('annotation_metadata', {}).get('total_move_annotations', 0)}
- Global Annotations: {annotations.get('annotation_metadata', {}).get('global_annotations', 0)}

Source Files:
- Original Game: {metadata.get('source_file', 'Unknown')}
- Session ID: {enriched_data.get('annotation_session_id', 'Unknown')}
- Created: {enriched_data.get('created_at', 'Unknown')}

Position Details:
"""
    
    # Add position-by-position breakdown
    positions = enriched_data.get('positions', {})
    for pos_key in sorted(positions.keys(), key=lambda x: int(x) if x.isdigit() else 0):
        pos = positions[pos_key]
        if pos.get('move_annotations') or pos.get('global_annotations'):
            report += f"\nPosition {pos_key}:\n"
            
            if pos.get('move_annotations'):
                for move, tags in pos['move_annotations'].items():
                    active_tags = [tag for tag, value in tags.items() if value]
                    if active_tags:
                        report += f"  {move}: {', '.join(active_tags)}\n"
            
            if pos.get('global_annotations'):
                active_global = [tag for tag, value in pos['global_annotations'].items() if value]
                if active_global:
                    report += f"  Global: {', '.join(active_global)}\n"
    
    return report

def main():
    parser = argparse.ArgumentParser(
        description="Enrich game data with annotation labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Enrich a single game with its annotations
  python enrich_labels.py game.json labels.json --output enriched_game.json
  
  # Create enriched data with summary report
  python enrich_labels.py game.json labels.json --output enriched_game.json --summary
  
  # Specify custom annotation session ID
  python enrich_labels.py game.json labels.json --output enriched_game.json --session-id "session_001"
        """
    )
    
    parser.add_argument("combined_data", type=Path, 
                       help="Original combined game data (JSON file with SGF + policy)")
    parser.add_argument("labels", type=Path,
                       help="Exported labels from the UI (labels.json)")
    parser.add_argument("--output", type=Path, required=True,
                       help="Output path for enriched game data")
    parser.add_argument("--summary", action="store_true",
                       help="Also create a human-readable summary report")
    parser.add_argument("--session-id", type=str,
                       help="Custom annotation session ID (default: auto-generated)")
    
    args = parser.parse_args()
    
    if not args.combined_data.exists():
        print(f"Error: Combined data file {args.combined_data} does not exist")
        return 1
    
    if not args.labels.exists():
        print(f"Error: Labels file {args.labels} does not exist")
        return 1
    
    try:
        print(f"Enriching game data...")
        print(f"  Combined data: {args.combined_data}")
        print(f"  Labels: {args.labels}")
        print(f"  Output: {args.output}")
        
        enriched_data = create_enriched_game_data(
            args.combined_data,
            args.labels, 
            args.output,
            args.session_id
        )
        
        print(f"✓ Enriched game data saved to {args.output}")
        
        if args.summary:
            summary_path = args.output.with_suffix('.summary.txt')
            summary = create_summary_report(enriched_data)
            summary_path.write_text(summary, encoding='utf-8')
            print(f"✓ Summary report saved to {summary_path}")
        
        # Print quick stats
        metadata = enriched_data.get('annotations', {}).get('annotation_metadata', {})
        print(f"\nAnnotation Summary:")
        print(f"  Positions annotated: {metadata.get('annotated_positions', 0)}")
        print(f"  Move annotations: {metadata.get('total_move_annotations', 0)}")
        print(f"  Global annotations: {metadata.get('global_annotations', 0)}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
