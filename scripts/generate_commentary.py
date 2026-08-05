#!/usr/bin/env python3
"""
CLI script for generating move commentary using OpenAI.

Processes games to generate natural-language commentary grounded in
snorkel analysis and probe concept data.

Usage:
    python scripts/generate_commentary.py --games-dir games --html-data-dir daniele_experiment/linear_probes/html_data
    python scripts/generate_commentary.py --game-id 00700686-05d0-4feb-bd80-0e9978faf6b2 --max-moves 10
    python scripts/generate_commentary.py --overwrite  # Regenerate all

Environment variables:
    OPENAI_API_KEY: Required. Your OpenAI API key.
    OPENAI_MODEL: Optional. Model to use (default: gpt-4.1-mini)
"""
import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from kata_teach.commentary import generate_game_commentary, CommentaryCache
from kata_teach.commentary.generate_commentary import discover_games, generate_all_commentary, OpenAIClient


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate AI commentary for Go game moves",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
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
        help="Directory containing html_data with concepts.json files (default: daniele_experiment/linear_probes/html_data)",
    )
    
    parser.add_argument(
        "--game-id",
        type=str,
        default=None,
        help="Process only this game ID (default: process all games)",
    )
    
    parser.add_argument(
        "--max-moves",
        type=int,
        default=None,
        help="Maximum number of moves to process per game (default: all)",
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing commentary instead of skipping cached moves",
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without calling OpenAI",
    )
    
    return parser.parse_args()


def print_progress(game_id: str, game_num: int, total_games: int) -> None:
    """Print progress for multi-game processing."""
    print(f"\n[{game_num}/{total_games}] Processing game: {game_id}")


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate paths
    if not args.games_dir.exists():
        logger.error(f"Games directory not found: {args.games_dir}")
        return 1
    
    if not args.html_data_dir.exists():
        logger.error(f"HTML data directory not found: {args.html_data_dir}")
        return 1
    
    # Discover games
    if args.game_id:
        game_ids = [args.game_id]
        # Validate the specific game exists
        snorkel_path = args.games_dir / args.game_id / "snorkel.jsonl"
        concepts_path = args.html_data_dir / args.game_id / "concepts.json"
        
        if not snorkel_path.exists():
            logger.error(f"Snorkel file not found: {snorkel_path}")
            return 1
        if not concepts_path.exists():
            logger.error(f"Concepts file not found: {concepts_path}")
            return 1
    else:
        game_ids = discover_games(args.games_dir, args.html_data_dir)
    
    if not game_ids:
        logger.warning("No games found to process")
        return 0
    
    print(f"Found {len(game_ids)} game(s) to process")
    
    # Dry run - just show what would be processed
    if args.dry_run:
        print("\nDry run - would process these games:")
        cache = CommentaryCache(args.html_data_dir)
        
        for game_id in game_ids:
            cached_moves = len(cache.get_cached_moves(game_id))
            print(f"  {game_id}: {cached_moves} moves already cached")
        
        print("\nSet OPENAI_API_KEY and remove --dry-run to generate commentary")
        return 0
    
    # Validate API key
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set")
        print("\nTo use this script, set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-key-here'")
        return 1
    
    # Generate commentary
    try:
        results = generate_all_commentary(
            games_dir=args.games_dir,
            html_data_dir=args.html_data_dir,
            game_ids=game_ids,
            max_moves=args.max_moves,
            overwrite=args.overwrite,
            progress_callback=print_progress,
        )
        
        # Print summary
        print("\n" + "=" * 50)
        print("Summary:")
        print("=" * 50)
        
        total_generated = 0
        for game_id, count in results.items():
            if count > 0:
                print(f"  {game_id}: {count} commentaries generated")
                total_generated += count
        
        print(f"\nTotal: {total_generated} commentaries generated across {len(results)} games")
        
        # Show output locations
        print("\nOutput files created:")
        for game_id in results.keys():
            commentary_path = args.html_data_dir / game_id / "commentary.jsonl"
            merged_path = args.html_data_dir / game_id / "concepts_with_commentary.json"
            if commentary_path.exists():
                print(f"  {commentary_path}")
            if merged_path.exists():
                print(f"  {merged_path}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

