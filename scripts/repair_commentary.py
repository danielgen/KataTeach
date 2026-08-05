#!/usr/bin/env python3
"""Audit cached commentary against current evidence and repair invalid entries."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from kata_teach.commentary.cache import CommentaryCache, merge_concepts_with_commentary
from kata_teach.commentary.evidence import build_all_evidence_packets
from kata_teach.commentary.grounding import build_grounded_fallback, validate_grounding


def repair_game(game_id: str, games_dir: Path, html_data_dir: Path, apply: bool) -> tuple[int, int]:
    concepts_path = html_data_dir / game_id / "concepts.json"
    packets = build_all_evidence_packets(
        game_id=game_id,
        snorkel_path=games_dir / game_id / "snorkel.jsonl",
        concepts_path=concepts_path,
    )
    cache = CommentaryCache(html_data_dir)
    existing = cache.load_game_cache(game_id)
    repaired = []
    invalid = 0
    for packet in packets:
        output = existing.get(packet.move_number)
        if output is None or validate_grounding(output, packet):
            invalid += 1
            output = build_grounded_fallback(packet)
        repaired.append(output)

    if apply:
        cache.replace_game(game_id, repaired)
        merge_concepts_with_commentary(concepts_path, cache, game_id)
    return invalid, len(packets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--games-dir", type=Path, default=Path("games"))
    parser.add_argument(
        "--html-data-dir",
        type=Path,
        default=Path("daniele_experiment/linear_probes/html_data"),
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    invalid, total = repair_game(
        args.game_id, args.games_dir, args.html_data_dir, args.apply
    )
    action = "Repaired" if args.apply else "Would repair"
    print(f"{action} {invalid}/{total} commentary entries")


if __name__ == "__main__":
    main()
