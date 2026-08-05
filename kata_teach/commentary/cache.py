"""
Caching logic for commentary generation.

Stores generated commentary in JSONL format to avoid regenerating
on subsequent runs.
"""
import json
from pathlib import Path
from typing import Dict, Set, Optional, List, Any

from .schema import CommentaryOutput


class CommentaryCache:
    """
    Cache for generated commentary, stored as JSONL.
    
    Each game's commentary is stored in:
    linear_probes/html_data/<game_id>/commentary.jsonl
    """
    
    def __init__(self, html_data_dir: Path):
        """
        Initialize the cache.
        
        Args:
            html_data_dir: Base directory for html_data (linear_probes/html_data)
        """
        self.html_data_dir = Path(html_data_dir)
        self._cache: Dict[str, Dict[int, CommentaryOutput]] = {}
    
    def _get_cache_path(self, game_id: str) -> Path:
        """Get path to commentary cache file for a game."""
        return self.html_data_dir / game_id / "commentary.jsonl"
    
    def load_game_cache(self, game_id: str) -> Dict[int, CommentaryOutput]:
        """
        Load cached commentary for a game.
        
        Returns dict of move_number -> CommentaryOutput.
        """
        if game_id in self._cache:
            return self._cache[game_id]
        
        cache_path = self._get_cache_path(game_id)
        game_cache: Dict[int, CommentaryOutput] = {}
        
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            output = CommentaryOutput.from_dict(data)
                            game_cache[output.move_number] = output
                        except (json.JSONDecodeError, KeyError) as e:
                            # Skip malformed entries
                            continue
        
        self._cache[game_id] = game_cache
        return game_cache
    
    def get_cached_moves(self, game_id: str) -> Set[int]:
        """Get set of move numbers that have cached commentary."""
        cache = self.load_game_cache(game_id)
        return set(cache.keys())
    
    def get_commentary(self, game_id: str, move_number: int) -> Optional[CommentaryOutput]:
        """Get cached commentary for a specific move."""
        cache = self.load_game_cache(game_id)
        return cache.get(move_number)
    
    def save_commentary(self, game_id: str, commentary: CommentaryOutput) -> None:
        """
        Save a single commentary entry to cache.
        
        Appends to the JSONL file.
        """
        cache_path = self._get_cache_path(game_id)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Update in-memory cache
        if game_id not in self._cache:
            self._cache[game_id] = {}
        self._cache[game_id][commentary.move_number] = commentary
        
        # Append to file
        with open(cache_path, 'a') as f:
            f.write(json.dumps(commentary.to_dict()) + '\n')
    
    def save_batch(self, game_id: str, commentaries: List[CommentaryOutput]) -> None:
        """
        Save multiple commentary entries at once.
        
        More efficient for batch operations.
        """
        if not commentaries:
            return
            
        cache_path = self._get_cache_path(game_id)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Update in-memory cache
        if game_id not in self._cache:
            self._cache[game_id] = {}
        
        with open(cache_path, 'a') as f:
            for commentary in commentaries:
                self._cache[game_id][commentary.move_number] = commentary
                f.write(json.dumps(commentary.to_dict()) + '\n')

    def replace_game(self, game_id: str, commentaries: List[CommentaryOutput]) -> None:
        """Atomically replace a game's cache with one entry per move."""
        cache_path = self._get_cache_path(game_id)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = cache_path.with_suffix(".jsonl.tmp")
        ordered = sorted(commentaries, key=lambda item: item.move_number)
        with open(temp_path, "w") as f:
            for commentary in ordered:
                f.write(json.dumps(commentary.to_dict()) + "\n")
        temp_path.replace(cache_path)
        self._cache[game_id] = {item.move_number: item for item in ordered}
    
    def clear_game_cache(self, game_id: str) -> None:
        """Clear cache for a game (for overwrite mode)."""
        cache_path = self._get_cache_path(game_id)
        if cache_path.exists():
            cache_path.unlink()
        
        if game_id in self._cache:
            del self._cache[game_id]
    
    def get_all_commentary(self, game_id: str) -> List[CommentaryOutput]:
        """Get all cached commentary for a game, sorted by move number."""
        cache = self.load_game_cache(game_id)
        return sorted(cache.values(), key=lambda c: c.move_number)


def merge_concepts_with_commentary(
    concepts_path: Path,
    commentary_cache: CommentaryCache,
    game_id: str,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Merge concepts.json with generated commentary.
    
    Creates concepts_with_commentary.json with commentary field added to each move.
    
    Args:
        concepts_path: Path to concepts.json
        commentary_cache: CommentaryCache instance
        game_id: Game identifier
        output_path: Optional output path (defaults to same dir as concepts)
        
    Returns:
        Merged data dict
    """
    # Load concepts
    with open(concepts_path, 'r') as f:
        concepts_data = json.load(f)
    
    # Load all commentary
    all_commentary = {c.move_number: c for c in commentary_cache.get_all_commentary(game_id)}
    
    # Add commentary to each move
    for move in concepts_data.get('moves', []):
        move_num = move['move_number']
        if move_num in all_commentary:
            move['commentary'] = all_commentary[move_num].to_dict()
        else:
            move['commentary'] = None
    
    # Save merged file
    if output_path is None:
        output_path = concepts_path.parent / "concepts_with_commentary.json"
    
    with open(output_path, 'w') as f:
        json.dump(concepts_data, f, indent=2)
    
    return concepts_data
