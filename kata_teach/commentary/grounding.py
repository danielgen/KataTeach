"""Deterministic grounding checks and safe fallback commentary."""
import re
from typing import Dict, List, Tuple

from .schema import CommentaryOutput, EvidencePacket


CLAIM_GATES: List[Tuple[str, str]] = [
    (r"\bcut(?:s|ting)?\b", "cut"),
    (r"\batari\b", "atari"),
    (r"\bforc(?:e|es|ing)\b", "forcing"),
    (r"\btenuki\b", "tenuki"),
    (r"\binvad(?:e|es|ing)|\binvasion\b", "invasion"),
    (r"\bkill(?:s|ing)?\b|threatens? to kill", "killing_attack"),
    (r"\battacks?|\bpressure\b", "attack"),
    (r"\binfluence surge\b", "influence_count_delta"),
    (r"\bconnects?\b|\bmerg(?:e|es|ing)\b", "connection"),
    (r"\bextend(?:s|ing)?\b|\bextension\b", "extension"),
    (r"\boccup(?:y|ies|ying) (?:a |the )?corner\b", "occupy_corner"),
    (r"\bapproach(?:es|ing)? (?:a |the )?corner\b", "approaching_corner"),
    (r"\bdirect sacrifice\b|\bsacrifices? (?:the |a )?stone\b", "direct_sacrifice"),
]


def _gate_is_true(field: str, snorkel: Dict) -> bool:
    value = snorkel.get(field)
    if field == "influence_count_delta":
        return isinstance(value, (int, float)) and value > 0
    return bool(value)


def validate_grounding(output: CommentaryOutput, packet: EvidencePacket) -> List[str]:
    """Reject commentary claims that contradict the current evidence packet."""
    errors = output.validate(
        packet.selected_concepts,
        expected_move_number=packet.move_number,
    )
    text = output.comment.lower()
    for pattern, field in CLAIM_GATES:
        if re.search(pattern, text) and not _gate_is_true(field, packet.snorkel):
            errors.append(f"comment claims {field}, but current evidence does not support it")

    if re.search(r"\b(?:head|loc|index)(?:s|ations?)?\s*(?:at\s*)?\d{1,3}\b", text):
        errors.append("comment exposes a raw internal board index")

    strengthens = re.search(r"\bstrengthen(?:s|ed|ing)?\b|\bgroup strength (?:improves?|increases?)\b", text)
    if strengthens:
        deltas = (
            packet.snorkel.get("current_group_strength_delta", 0),
            packet.snorkel.get("group_strength_delta", 0),
        )
        if not any(isinstance(v, (int, float)) and v > 0 for v in deltas):
            errors.append("comment claims a strength increase without a positive strength delta")

    connectivity = re.search(r"\bconnectivity (?:improves?|increases?)\b|\bimproves? (?:its |the )?connectivity\b", text)
    if connectivity:
        delta = packet.snorkel.get("current_group_connectivity_delta", 0)
        if not isinstance(delta, (int, float)) or delta <= 0:
            errors.append("comment claims a connectivity increase without a positive connectivity delta")

    return list(dict.fromkeys(errors))


def build_grounded_fallback(packet: EvidencePacket) -> CommentaryOutput:
    """Build concise commentary directly from gated concepts, without an LLM."""
    s = packet.snorkel
    phrases = {
        "kill_attack": "threatens to kill an opposing group",
        "atari": "puts an opposing group in atari",
        "must_live": "answers an urgent threat to keep the group alive",
        "cut": "cuts apart opposing stones",
        "multi_connect": "connects several friendly groups",
        "connect": "connects friendly stones",
        "invasion": "invades the opponent's territory",
        "fight_pressure": "puts pressure on an opposing group",
        "fight_wide": "puts pressure on several opposing groups",
        "sacrifice_direct": "sacrifices the played stone",
        "forcing": "creates a forcing threat",
        "occupy_corner": "occupies the corner",
        "approaching_corner": "approaches the corner",
        "territory_building": f"builds territory on {s.get('building_count', 0)} points",
        "territory_securing": f"solidifies {s.get('solidification_count', 0)} points",
        "opponent_reduction": f"reduces the opponent's territory by {s.get('reduction_count', 0)} points",
        "influence_surge": f"expands influence by {s.get('influence_count_delta', 0)} points",
        "extend": "extends from the nearby friendly stones",
        "tenuki": "plays away from the previous local exchange",
        "urgency_peak": "responds in an urgent area",
        "aji_reduction": "reduces weaknesses in the position",
        "group_strength_up": "strengthens the player's groups",
        "group_strength_down": "leaves the player's groups weaker overall",
        "group_strength_shift": "substantially changes group strength",
        "group_connectivity_up": "improves the played group's connectivity",
        "group_connectivity_down": "reduces the played group's connectivity",
        "group_connectivity_shift": "substantially changes the played group's connectivity",
    }
    used = [concept for concept in packet.selected_concepts if concept in phrases][:3]
    if not used:
        comment = "The available analysis does not identify a strong, well-supported concept for this move."
        return CommentaryOutput(packet.move_number, comment, [])

    actor = "Black" if packet.player.lower() == "b" else "White"
    location = f" at {packet.move_coord}" if packet.move_coord else ""
    selected_phrases = [phrases[concept] for concept in used]
    if len(selected_phrases) == 1:
        effects = selected_phrases[0]
    else:
        effects = ", ".join(selected_phrases[:-1]) + f", and {selected_phrases[-1]}"
    return CommentaryOutput(
        move_number=packet.move_number,
        comment=f"{actor}{location} {effects}.",
        concepts_used=used,
    )
