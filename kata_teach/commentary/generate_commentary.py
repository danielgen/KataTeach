"""
Main commentary generation logic using OpenAI.
"""
import os
import json
import time
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from .schema import CommentaryOutput, EvidencePacket, COMMENTARY_JSON_SCHEMA
from .evidence import build_all_evidence_packets, load_concepts_data, load_snorkel_data, build_evidence_packet
from .prompts import SYSTEM_PROMPT, format_user_prompt, format_correction_prompt
from .cache import CommentaryCache, merge_concepts_with_commentary
from .grounding import validate_grounding

logger = logging.getLogger(__name__)

# Default model - can be overridden via OPENAI_MODEL env var
DEFAULT_MODEL = "gpt-5.6-terra"

# Rate limiting settings
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0  # seconds
MAX_BACKOFF = 60.0  # seconds


class OpenAIClient:
    """Wrapper for OpenAI API with rate limiting and retries."""
    
    def __init__(self, model: Optional[str] = None):
        """
        Initialize the OpenAI client.
        
        Args:
            model: Model name (defaults to OPENAI_MODEL env var or gpt-4.1-mini)
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
        logger.info(f"Using OpenAI model: {self.model}")
    
    def generate_commentary(
        self,
        evidence_packet: EvidencePacket,
        temperature: float = 0.2,
        max_tokens: int = 20000,  # Increased for reasoning models
    ) -> Tuple[Optional[CommentaryOutput], Optional[str]]:
        """
        Generate commentary for a single move.
        
        Args:
            evidence_packet: Evidence packet for the move
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            
        Returns:
            (CommentaryOutput or None, error_message or None)
        """
        evidence_json = evidence_packet.to_json()
        user_prompt = format_user_prompt(evidence_json)
        
        # Try to get response with retries
        response_text = self._call_api_with_retry(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        
        if response_text is None:
            return None, "API call failed after retries"
        
        # Parse JSON response
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            # Try to fix JSON with correction prompt
            return self._retry_with_correction(
                evidence_packet=evidence_packet,
                errors=[f"Invalid JSON: {str(e)}"],
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
        
        # Create output object
        try:
            output = CommentaryOutput.from_dict(data)
        except (KeyError, TypeError) as e:
            logger.warning(f"Failed to create CommentaryOutput: {e}")
            return self._retry_with_correction(
                evidence_packet=evidence_packet,
                errors=[f"Missing required fields: {str(e)}"],
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
        
        # Validate output
        errors = validate_grounding(output, evidence_packet)
        if errors:
            logger.warning(f"Validation errors: {errors}")
            return self._retry_with_correction(
                evidence_packet=evidence_packet,
                errors=errors,
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
        
        return output, None
    
    def _retry_with_correction(
        self,
        evidence_packet: EvidencePacket,
        errors: List[str],
        temperature: float,
        max_completion_tokens: int,
    ) -> Tuple[Optional[CommentaryOutput], Optional[str]]:
        """Retry generation with correction prompt."""
        evidence_json = evidence_packet.to_json()
        correction_prompt = format_correction_prompt(
            errors=errors,
            selected_concepts=evidence_packet.selected_concepts,
            evidence_json=evidence_json,
        )
        
        response_text = self._call_api_with_retry(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=correction_prompt,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )
        
        if response_text is None:
            return None, "Correction API call failed"
        
        try:
            data = json.loads(response_text)
            output = CommentaryOutput.from_dict(data)
            
            # Validate again
            new_errors = validate_grounding(output, evidence_packet)
            if new_errors:
                logger.error(f"Validation still failing after correction: {new_errors}")
                return None, f"Validation failed after correction: {new_errors}"
            
            return output, None
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            return None, f"Failed to parse corrected response: {e}"
    
    def _call_api_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_completion_tokens: int,
    ) -> Optional[str]:
        """
        Call OpenAI API with exponential backoff on rate limits.
        
        Returns response text or None on failure.
        """
        backoff = INITIAL_BACKOFF
        
        for attempt in range(MAX_RETRIES):
            try:
                # Use json_object format which is more widely supported than json_schema
                # We'll validate the structure ourselves after parsing
                create_kwargs = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_completion_tokens": max_completion_tokens,
                    "response_format": {"type": "json_object"},
                }
                # Some reasoning models reject temperature; try with it first
                if temperature is not None:
                    create_kwargs["temperature"] = temperature
                try:
                    response = self.client.chat.completions.create(**create_kwargs)
                except Exception as temp_err:
                    if "temperature" in str(temp_err).lower():
                        create_kwargs.pop("temperature", None)
                        response = self.client.chat.completions.create(**create_kwargs)
                    else:
                        raise
                
                # With structured output, the content should be valid JSON
                choice = response.choices[0]
                message = choice.message
                content = message.content
                finish_reason = choice.finish_reason
                
                # Log response details for debugging
                logger.info(f"Finish reason: {finish_reason}")
                logger.info(f"Content type: {type(content)}")
                logger.info(f"Content value: {repr(content[:200]) if content else 'None/Empty'}")
                
                # Check if we hit token limits (especially for reasoning models)
                if finish_reason == 'length':
                    logger.warning(f"Response hit token limit. Used {response.usage.completion_tokens} completion tokens")
                    if hasattr(response.usage, 'completion_tokens_details'):
                        details = response.usage.completion_tokens_details
                        if hasattr(details, 'reasoning_tokens') and details.reasoning_tokens:
                            logger.warning(f"Reasoning tokens used: {details.reasoning_tokens}")
                            logger.error("Model used all tokens for reasoning, no content generated. Increase max_completion_tokens.")
                            return None
                
                if not content:
                    # Check if there's a refusal or other issue
                    if hasattr(message, 'refusal') and message.refusal:
                        logger.error(f"API refused request: {message.refusal}")
                    else:
                        logger.error("Empty response content from API")
                        logger.error(f"Finish reason: {finish_reason}")
                    return None
                
                return content
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check for rate limit errors
                if "rate" in error_str or "429" in error_str:
                    logger.warning(f"Rate limited, backing off for {backoff}s (attempt {attempt + 1})")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, MAX_BACKOFF)
                    continue
                
                # Check for transient errors
                if "timeout" in error_str or "connection" in error_str or "502" in error_str or "503" in error_str:
                    logger.warning(f"Transient error, retrying in {backoff}s: {e}")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, MAX_BACKOFF)
                    continue
                
                # Non-retryable error
                logger.error(f"API error: {e}")
                return None
        
        logger.error(f"Failed after {MAX_RETRIES} retries")
        return None


def generate_move_commentary(
    client: OpenAIClient,
    evidence_packet: EvidencePacket,
) -> Optional[CommentaryOutput]:
    """
    Generate commentary for a single move.
    
    Args:
        client: OpenAI client
        evidence_packet: Evidence packet for the move
        
    Returns:
        CommentaryOutput or None on failure
    """
    output, error = client.generate_commentary(evidence_packet)
    
    if error:
        logger.error(f"Failed to generate commentary for move {evidence_packet.move_number}: {error}")
        return None
    
    return output


def generate_game_commentary(
    game_id: str,
    games_dir: Path,
    html_data_dir: Path,
    cache: Optional[CommentaryCache] = None,
    client: Optional[OpenAIClient] = None,
    max_moves: Optional[int] = None,
    overwrite: bool = False,
    progress_callback: Optional[callable] = None,
) -> List[CommentaryOutput]:
    """
    Generate commentary for all moves in a game.
    
    Args:
        game_id: Game identifier
        games_dir: Directory containing game data (games/<game_id>/snorkel.jsonl)
        html_data_dir: Directory for html data (linear_probes/html_data/<game_id>/concepts.json)
        cache: Optional cache instance (created if not provided)
        client: Optional OpenAI client (created if not provided)
        max_moves: Optional limit on moves to process
        overwrite: Whether to overwrite existing commentary
        progress_callback: Optional callback(move_number, total_moves) for progress
        
    Returns:
        List of generated CommentaryOutput objects
    """
    # Set up paths
    snorkel_path = games_dir / game_id / "snorkel.jsonl"
    concepts_path = html_data_dir / game_id / "concepts.json"
    
    if not snorkel_path.exists():
        raise FileNotFoundError(f"Snorkel file not found: {snorkel_path}")
    if not concepts_path.exists():
        raise FileNotFoundError(f"Concepts file not found: {concepts_path}")
    
    # Initialize cache and client
    if cache is None:
        cache = CommentaryCache(html_data_dir)
    if client is None:
        client = OpenAIClient()
    
    # Clear cache if overwriting
    if overwrite:
        cache.clear_game_cache(game_id)
    
    # Get cached moves
    cached_moves = cache.get_cached_moves(game_id)
    
    # Build evidence packets
    packets = build_all_evidence_packets(
        game_id=game_id,
        snorkel_path=snorkel_path,
        concepts_path=concepts_path,
        max_moves=max_moves,
    )
    
    # A move number alone is not a valid cache key: Snorkel features and concept
    # gates can change. Revalidate cached prose against the current evidence.
    packet_by_move = {packet.move_number: packet for packet in packets}
    valid_cached_moves = {
        move_number for move_number in cached_moves
        if move_number in packet_by_move
        and not validate_grounding(
            cache.get_commentary(game_id, move_number),
            packet_by_move[move_number],
        )
    }
    stale_count = len(cached_moves - valid_cached_moves)
    if stale_count:
        logger.info("Game %s: invalidated %d stale/ungrounded cached comments", game_id, stale_count)

    packets_to_process = [p for p in packets if p.move_number not in valid_cached_moves]
    
    logger.info(f"Game {game_id}: {len(packets)} total moves, {len(cached_moves)} cached, {len(packets_to_process)} to generate")
    
    # Generate commentary for each move
    results: List[CommentaryOutput] = []
    
    for i, packet in enumerate(packets_to_process):
        if progress_callback:
            progress_callback(packet.move_number, len(packets))
        
        output = generate_move_commentary(client, packet)
        
        if output:
            cache.save_commentary(game_id, output)
            results.append(output)
            logger.debug(f"Generated commentary for move {packet.move_number}")
        else:
            logger.warning(f"Failed to generate commentary for move {packet.move_number}")
    
    # Merge with concepts.json
    merge_concepts_with_commentary(
        concepts_path=concepts_path,
        commentary_cache=cache,
        game_id=game_id,
    )
    
    logger.info(f"Game {game_id}: Generated {len(results)} new commentaries")
    
    return results


def discover_games(
    games_dir: Path,
    html_data_dir: Path,
) -> List[str]:
    """
    Discover all games that have both snorkel.jsonl and concepts.json.
    
    Returns list of game_ids.
    """
    games = []
    
    if not html_data_dir.exists():
        return games
    
    for game_dir in html_data_dir.iterdir():
        if not game_dir.is_dir():
            continue
        
        game_id = game_dir.name
        snorkel_path = games_dir / game_id / "snorkel.jsonl"
        concepts_path = game_dir / "concepts.json"
        
        if snorkel_path.exists() and concepts_path.exists():
            games.append(game_id)
    
    return sorted(games)


def generate_all_commentary(
    games_dir: Path,
    html_data_dir: Path,
    game_ids: Optional[List[str]] = None,
    max_moves: Optional[int] = None,
    overwrite: bool = False,
    progress_callback: Optional[callable] = None,
) -> Dict[str, int]:
    """
    Generate commentary for multiple games.
    
    Args:
        games_dir: Directory containing game data
        html_data_dir: Directory for html data
        game_ids: Optional list of specific game IDs (discovers all if None)
        max_moves: Optional limit on moves per game
        overwrite: Whether to overwrite existing commentary
        progress_callback: Optional callback(game_id, game_num, total_games)
        
    Returns:
        Dict mapping game_id to number of commentaries generated
    """
    if game_ids is None:
        game_ids = discover_games(games_dir, html_data_dir)
    
    if not game_ids:
        logger.warning("No games found to process")
        return {}
    
    logger.info(f"Processing {len(game_ids)} games")
    
    # Shared resources
    cache = CommentaryCache(html_data_dir)
    client = OpenAIClient()
    
    results = {}
    
    for i, game_id in enumerate(game_ids):
        if progress_callback:
            progress_callback(game_id, i + 1, len(game_ids))
        
        try:
            commentaries = generate_game_commentary(
                game_id=game_id,
                games_dir=games_dir,
                html_data_dir=html_data_dir,
                cache=cache,
                client=client,
                max_moves=max_moves,
                overwrite=overwrite,
            )
            results[game_id] = len(commentaries)
            
        except Exception as e:
            logger.error(f"Failed to process game {game_id}: {e}")
            results[game_id] = 0
    
    return results
