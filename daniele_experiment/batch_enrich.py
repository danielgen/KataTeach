#!/usr/bin/env python3
"""Batch process multiple annotation files to create enriched game data."""

import argparse
import json
from pathlib import Path
from typing import List, Tuple
from enrich_labels import create_enriched_game_data, create_summary_report

def find_matching_files(annotations_dir: Path, games_dir: Path) -> List[Tuple[Path, Path]]:
    """Find matching annotation and game files."""
    matches = []
    
    # Look for annotation files
    for annotation_file in annotations_dir.glob("*_annotations_*.json"):
        # Try to find matching game file
        # Extract game identifier from annotation filename
        name_parts = annotation_file.stem.split("_annotations_")
        if len(name_parts) >= 2:
            game_prefix = name_parts[0]
            
            # Look for matching game files
            for game_file in games_dir.glob("*.json"):
                if game_prefix in game_file.stem or game_file.stem in game_prefix:
                    matches.append((game_file, annotation_file))
                    break
    
    return matches

def batch_enrich(
    annotations_dir: Path,
    games_dir: Path, 
    output_dir: Path,
    create_summaries: bool = True
) -> None:
    """Batch process all matching annotation and game files."""
    
    output_dir.mkdir(exist_ok=True)
    
    matches = find_matching_files(annotations_dir, games_dir)
    
    if not matches:
        print("No matching annotation and game files found.")
        print(f"Annotation dir: {annotations_dir}")
        print(f"Games dir: {games_dir}")
        return
    
    print(f"Found {len(matches)} matching file pairs")
    
    for i, (game_file, annotation_file) in enumerate(matches, 1):
        print(f"\n[{i}/{len(matches)}] Processing:")
        print(f"  Game: {game_file}")
        print(f"  Annotations: {annotation_file}")
        
        try:
            # Create output filename
            output_name = f"enriched_{game_file.stem}.json"
            output_path = output_dir / output_name
            
            # Create enriched data
            enriched_data = create_enriched_game_data(
                game_file,
                annotation_file,
                output_path
            )
            
            print(f"  ✓ Enriched data: {output_path}")
            
            # Create summary if requested
            if create_summaries:
                summary_path = output_path.with_suffix('.summary.txt')
                summary = create_summary_report(enriched_data)
                summary_path.write_text(summary, encoding='utf-8')
                print(f"  ✓ Summary: {summary_path}")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    print(f"\nBatch processing complete! Output directory: {output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Batch process annotation files with their corresponding games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all annotations in a directory
  python batch_enrich.py annotations/ games/ --output enriched_games/
  
  # Process without creating summary reports
  python batch_enrich.py annotations/ games/ --output enriched_games/ --no-summaries
        """
    )
    
    parser.add_argument("annotations_dir", type=Path,
                       help="Directory containing annotation files")
    parser.add_argument("games_dir", type=Path, 
                       help="Directory containing original game files")
    parser.add_argument("--output", type=Path, required=True,
                       help="Output directory for enriched files")
    parser.add_argument("--no-summaries", action="store_true",
                       help="Don't create summary reports")
    
    args = parser.parse_args()
    
    if not args.annotations_dir.exists():
        print(f"Error: Annotations directory {args.annotations_dir} does not exist")
        return 1
    
    if not args.games_dir.exists():
        print(f"Error: Games directory {args.games_dir} does not exist")
        return 1
    
    try:
        batch_enrich(
            args.annotations_dir,
            args.games_dir,
            args.output,
            create_summaries=not args.no_summaries
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
