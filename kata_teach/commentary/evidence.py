"""
Evidence packet builder for commentary generation.

Joins snorkel analysis and probe concept data, applies gating rules,
and produces evidence packets for the LLM.
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass

from .schema import SnorkelFields, EvidencePacket


# Gating thresholds (tuneable; aligned with concepts.yaml filters where practical)
GROUP_STRENGTH_THRESHOLD = 0.08
GROUP_CONNECTIVITY_THRESHOLD = 0.08
URGENCY_PEAK_THRESHOLD = 0.80
HIGH_ATTACK_INTENSITY_THRESHOLD = 0.5
FIGHT_PRESSURE_INTENSITY_FLOOR = 0.20
TERRITORY_MIN_MOVE = 8
LOW_TRUST_CONCEPTS: Set[str] = {"sacrifice_commitment", "invasion"}
LOW_TRUST_DELTA_BONUS = 0.5  # require higher delta to rank competitively

# Hard tactical concepts grounded in snorkel booleans — always outrank soft territory/intensity
TACTICAL_PRIORITY: Dict[str, float] = {
    "kill_attack": 100.0,
    "atari": 95.0,
    "must_live": 90.0,
    "cut": 85.0,
    "multi_connect": 82.0,
    "connect": 80.0,
    "invasion": 75.0,
    "fight_pressure": 70.0,
    "fight_wide": 68.0,
    "sacrifice_direct": 65.0,
    "sacrifice_indirect": 60.0,
    "forcing": 55.0,
    "occupy_corner": 50.0,
    "approaching_corner": 48.0,
    # Soft / intensity concepts — supporting only when tactics absent
    "aji_reduction": 35.0,
    "territory_building": 25.0,
    "territory_securing": 24.0,
    "opponent_reduction": 23.0,
    "influence_surge": 22.0,
    "group_strength_up": 20.0,
    "group_strength_down": 20.0,
    "group_connectivity_up": 18.0,
    "group_connectivity_down": 18.0,
    "urgency_peak": 15.0,
    "extend": 8.0,
    "tenuki": 6.0,
    "sacrifice_commitment": 5.0,
}

# Known concepts that may be selected for commentary
KNOWN_CONCEPTS: Set[str] = {
    "cut", "connect", "multi_connect", "extend", "atari", "forcing", "tenuki",
    "invasion", "must_live", "kill_attack", "aji_reduction", "fight_pressure",
    "fight_wide", "territory_building", "territory_securing", "opponent_reduction",
    "influence_surge", "urgency_peak", "group_strength_shift", "group_strength_up",
    "group_strength_down", "group_connectivity_shift", "group_connectivity_up",
    "group_connectivity_down", "sacrifice_direct", "sacrifice_indirect",
    "sacrifice_commitment", "approaching_corner", "occupy_corner",
}

# Snorkel fields relevant to each concept (for slim packets)
CONCEPT_SNORKEL_FIELDS: Dict[str, List[str]] = {
    "cut": ["cut", "cut_groups_created", "cut_regions", "cut_head_coords"],
    "connect": ["connection", "connection_strength_gain", "merged_groups_regions", "merged_groups_head_coords"],
    "multi_connect": ["connection", "connection_strength_gain", "merged_groups_regions", "merged_groups_head_coords"],
    "extend": ["extension"],
    "atari": ["atari"],
    "forcing": ["forcing"],
    "tenuki": ["tenuki"],
    "invasion": ["invasion"],
    "must_live": ["must_live"],
    "kill_attack": [
        "killing_attack", "attack", "max_attack_intensity", "max_attack_intensity_magnitude",
        "attacked_groups_count", "attacked_groups_regions", "attacked_groups_head_coords",
    ],
    "aji_reduction": ["reduce_aji", "aji_reduction_intensity_magnitude"],
    "fight_pressure": [
        "attack", "attacked_groups_count", "max_attack_intensity",
        "max_attack_intensity_magnitude", "attacked_groups_regions", "attacked_groups_head_coords",
    ],
    "fight_wide": ["attack", "attacked_groups_count", "attacked_groups_regions"],
    "territory_building": ["building_count", "building_count_magnitude", "potential_territory"],
    "territory_securing": ["solidification_count", "solidification_count_magnitude", "solid_territory"],
    "opponent_reduction": ["reduction_count", "reduction_count_magnitude"],
    "influence_surge": ["influence_count_delta", "influence_count_delta_magnitude"],
    "urgency_peak": ["urgency_max"],
    "group_strength_shift": ["group_strength_delta", "group_strength_delta_magnitude", "max_group_strength_delta"],
    "group_strength_up": ["group_strength_delta", "group_strength_delta_magnitude"],
    "group_strength_down": ["group_strength_delta", "group_strength_delta_magnitude"],
    "group_connectivity_shift": [
        "current_group_connectivity_delta", "current_group_connectivity_delta_magnitude",
    ],
    "group_connectivity_up": [
        "current_group_connectivity_delta", "current_group_connectivity_delta_magnitude",
    ],
    "group_connectivity_down": [
        "current_group_connectivity_delta", "current_group_connectivity_delta_magnitude",
    ],
    "sacrifice_direct": ["direct_sacrifice", "direct_sacrifice_intensity_magnitude"],
    "sacrifice_indirect": ["indirect_sacrifice", "indirect_sacrifice_intensity_magnitude", "indirect_sacrifice_coords"],
    "sacrifice_commitment": ["direct_sacrifice", "direct_sacrifice_intensity_magnitude"],
    "occupy_corner": ["occupy_corner"],
    "approaching_corner": ["approaching_corner"],
}


def percentile_to_magnitude(percentile: Optional[float]) -> Optional[str]:
    """Convert percentile value to qualitative magnitude term."""
    if percentile is None:
        return None

    if percentile >= 90:
        return "very large"
    elif percentile >= 75:
        return "large"
    elif percentile >= 50:
        return "moderate"
    elif percentile >= 25:
        return "small"
    else:
        return "minimal"


def idx361_to_human(idx: Optional[int], board_size: int = 19) -> Optional[str]:
    """Convert 0..(board_size^2-1) index to human coord (e.g. Q16)."""
    if idx is None:
        return None
    try:
        idx = int(idx)
    except (TypeError, ValueError):
        return None
    if idx < 0 or idx >= board_size * board_size:
        return None
    x = idx % board_size
    y = idx // board_size
    letter = chr(ord("A") + x + (1 if x >= 8 else 0))  # skip I
    number = board_size - y
    return f"{letter}{number}"


def katago_loc_to_human(loc: Optional[int], board_size: int = 19) -> Optional[str]:
    """
    Convert KataGo padded loc to human coord (same as viz.html locToGoCoord).

    dy = size + 1; x = (loc % dy) - 1; y = floor(loc / dy) - 1.
    """
    if loc is None:
        return None
    try:
        loc = int(loc)
    except (TypeError, ValueError):
        return None
    dy = board_size + 1
    x = (loc % dy) - 1
    y = (loc // dy) - 1
    if not (0 <= x < board_size and 0 <= y < board_size):
        return None
    letters = "ABCDEFGHJKLMNOPQRST"
    return f"{letters[x]}{board_size - y}"


def loc_to_human(loc: Optional[int], board_size: int = 19) -> Optional[str]:
    """
    Convert a location that may be KataGo loc OR idx361 to human coords.

    Heuristic: KataGo locs for 19x19 are typically > 361 (padded board) or
    match the (loc % 20) - 1 in-bounds pattern with loc often >= 21.
    Prefer KataGo conversion when it yields a valid coord and loc looks padded.
    """
    if loc is None:
        return None
    try:
        loc = int(loc)
    except (TypeError, ValueError):
        return None
    # Pass
    if loc < 0:
        return None
    kg = katago_loc_to_human(loc, board_size)
    # Typical KataGo board locs are in the padded grid (max ~(size+1)^2 + size)
    if kg and loc >= board_size:
        return kg
    # Fall back to idx361 for probe feature indices 0..360
    return idx361_to_human(loc, board_size) or kg


def idx361_to_region(idx: Optional[int], board_size: int = 19) -> Optional[str]:
    """Map board index / KataGo loc to coarse region name."""
    human = loc_to_human(idx, board_size)
    if not human:
        return None
    # Derive x,y from human
    letters = "ABCDEFGHJKLMNOPQRST"
    letter = human[0]
    try:
        number = int(human[1:])
    except ValueError:
        return None
    if letter not in letters:
        return None
    x = letters.index(letter)
    y = board_size - number
    return _xy_to_region(x, y, board_size)


def _xy_to_region(x: int, y: int, board_size: int = 19) -> str:
    corner = 6
    if x < corner and y < corner:
        return "corner_tl"
    if x >= board_size - corner and y < corner:
        return "corner_tr"
    if x < corner and y >= board_size - corner:
        return "corner_bl"
    if x >= board_size - corner and y >= board_size - corner:
        return "corner_br"
    if x < corner:
        return "side_left"
    if x >= board_size - corner:
        return "side_right"
    if y < corner:
        return "side_top"
    if y >= board_size - corner:
        return "side_bottom"
    return "center"


def locs_to_human(locs: Optional[List[int]], board_size: int = 19) -> List[str]:
    """Convert a list of KataGo locs / indices to human coordinates."""
    if not locs:
        return []
    out = []
    for loc in locs:
        human = loc_to_human(loc, board_size)
        if human:
            out.append(human)
    return out


@dataclass
class ConceptCandidate:
    """A concept candidate with its delta and gate status."""
    concept: str
    delta: float
    gated: bool
    evidence: str
    snorkel_only: bool = False

    @property
    def rank_score(self) -> float:
        """Tactical snorkel priority first; probe delta only breaks ties within a band."""
        base = TACTICAL_PRIORITY.get(self.concept, 12.0)
        delta_term = self.delta if self.delta > 0 else 0.0
        if self.concept in LOW_TRUST_CONCEPTS:
            delta_term = max(0.0, delta_term - LOW_TRUST_DELTA_BONUS)
        # Keep delta as a small tie-breaker so kill/atari always beat territory
        return base + min(delta_term, 5.0) * 0.05 + (0.01 if not self.snorkel_only else 0.0)


def check_concept_gate(
    concept: str,
    snorkel: SnorkelFields,
    move_number: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    Check if a concept passes snorkel gating.

    Unknown concepts are rejected (do not allow).
    Returns (passes_gate, evidence_string).
    """
    move_ok_territory = move_number is None or move_number >= TERRITORY_MIN_MOVE

    gates = {
        "cut": (
            snorkel.cut,
            "cut detected" if snorkel.cut else "",
        ),
        "connect": (
            snorkel.connection,
            (
                f"connection (strength gain: {snorkel.connection_strength_gain:.2f})"
                if snorkel.connection
                else ""
            ),
        ),
        "multi_connect": (
            snorkel.connection and snorkel.connection_strength_gain >= 2.0,
            (
                f"multi-connection (strength gain: {snorkel.connection_strength_gain:.2f})"
                if snorkel.connection and snorkel.connection_strength_gain >= 2.0
                else ""
            ),
        ),
        "extend": (
            snorkel.extension,
            "extension" if snorkel.extension else "",
        ),
        "atari": (
            snorkel.atari,
            "atari" if snorkel.atari else "",
        ),
        "forcing": (
            snorkel.forcing,
            "forcing move" if snorkel.forcing else "",
        ),
        "tenuki": (
            snorkel.tenuki,
            "tenuki (playing elsewhere)" if snorkel.tenuki else "",
        ),
        "invasion": (
            snorkel.invasion,
            "invasion" if snorkel.invasion else "",
        ),
        "must_live": (
            snorkel.must_live,
            "must live" if snorkel.must_live else "",
        ),
        "kill_attack": (
            snorkel.killing_attack,
            (
                "killing attack"
                + (
                    f" (intensity {snorkel.max_attack_intensity:.2f})"
                    if snorkel.max_attack_intensity is not None
                    else ""
                )
                + (
                    f" on group(s) in {', '.join(snorkel.attacked_regions)}"
                    if snorkel.attacked_regions
                    else ""
                )
                if snorkel.killing_attack
                else ""
            ),
        ),
        "aji_reduction": (
            snorkel.reduce_aji,
            "aji reduction" if snorkel.reduce_aji else "",
        ),
        "fight_pressure": (
            (
                snorkel.attacked_groups_count >= 1
                and snorkel.max_attack_intensity is not None
                and snorkel.max_attack_intensity >= FIGHT_PRESSURE_INTENSITY_FLOOR
            ),
            (
                f"fight pressure ({snorkel.attacked_groups_count} group(s), "
                f"intensity: {snorkel.max_attack_intensity:.2f})"
                if (
                    snorkel.attacked_groups_count >= 1
                    and snorkel.max_attack_intensity is not None
                    and snorkel.max_attack_intensity >= FIGHT_PRESSURE_INTENSITY_FLOOR
                )
                else ""
            ),
        ),
        "fight_wide": (
            snorkel.attacked_groups_count >= 2,
            (
                f"wide fight ({snorkel.attacked_groups_count} groups attacked)"
                if snorkel.attacked_groups_count >= 2
                else ""
            ),
        ),
        "territory_building": (
            move_ok_territory and snorkel.building_intensity > 0 and snorkel.building_count > 0,
            (
                f"building territory ({snorkel.building_count} pts)"
                if move_ok_territory and snorkel.building_intensity > 0 and snorkel.building_count > 0
                else ""
            ),
        ),
        "territory_securing": (
            move_ok_territory
            and snorkel.solidification_intensity > 0
            and snorkel.solidification_count > 0,
            (
                f"securing territory ({snorkel.solidification_count} pts solidified)"
                if (
                    move_ok_territory
                    and snorkel.solidification_intensity > 0
                    and snorkel.solidification_count > 0
                )
                else ""
            ),
        ),
        "opponent_reduction": (
            move_ok_territory and snorkel.reduction_intensity > 0 and snorkel.reduction_count > 0,
            (
                f"reducing opponent ({snorkel.reduction_count} pts)"
                if move_ok_territory and snorkel.reduction_intensity > 0 and snorkel.reduction_count > 0
                else ""
            ),
        ),
        "influence_surge": (
            move_ok_territory and snorkel.influence_count_delta > 0,
            (
                f"influence surge (+{snorkel.influence_count_delta} pts)"
                if move_ok_territory and snorkel.influence_count_delta > 0
                else ""
            ),
        ),
        "urgency_peak": (
            snorkel.urgency_max is not None and snorkel.urgency_max >= URGENCY_PEAK_THRESHOLD,
            (
                f"high urgency ({snorkel.urgency_max:.2f})"
                if snorkel.urgency_max and snorkel.urgency_max >= URGENCY_PEAK_THRESHOLD
                else ""
            ),
        ),
        "group_strength_shift": (
            abs(snorkel.group_strength_delta) >= GROUP_STRENGTH_THRESHOLD,
            (
                f"group strength {'improved' if snorkel.group_strength_delta > 0 else 'weakened'}"
                if abs(snorkel.group_strength_delta) >= GROUP_STRENGTH_THRESHOLD
                else ""
            ),
        ),
        "group_strength_up": (
            snorkel.group_strength_delta >= GROUP_STRENGTH_THRESHOLD,
            "group strength improved" if snorkel.group_strength_delta >= GROUP_STRENGTH_THRESHOLD else "",
        ),
        "group_strength_down": (
            snorkel.group_strength_delta <= -GROUP_STRENGTH_THRESHOLD,
            "group strength weakened" if snorkel.group_strength_delta <= -GROUP_STRENGTH_THRESHOLD else "",
        ),
        "group_connectivity_shift": (
            abs(snorkel.current_group_connectivity_delta) >= GROUP_CONNECTIVITY_THRESHOLD,
            (
                f"group connectivity {'improved' if snorkel.current_group_connectivity_delta > 0 else 'weakened'}"
                if abs(snorkel.current_group_connectivity_delta) >= GROUP_CONNECTIVITY_THRESHOLD
                else ""
            ),
        ),
        "group_connectivity_up": (
            snorkel.current_group_connectivity_delta >= GROUP_CONNECTIVITY_THRESHOLD,
            (
                "group connectivity improved"
                if snorkel.current_group_connectivity_delta >= GROUP_CONNECTIVITY_THRESHOLD
                else ""
            ),
        ),
        "group_connectivity_down": (
            snorkel.current_group_connectivity_delta <= -GROUP_CONNECTIVITY_THRESHOLD,
            (
                "group connectivity weakened"
                if snorkel.current_group_connectivity_delta <= -GROUP_CONNECTIVITY_THRESHOLD
                else ""
            ),
        ),
        "sacrifice_direct": (
            snorkel.direct_sacrifice,
            "direct sacrifice" if snorkel.direct_sacrifice else "",
        ),
        "sacrifice_indirect": (
            snorkel.indirect_sacrifice >= 1,
            (
                f"indirect sacrifice ({snorkel.indirect_sacrifice} stones)"
                if snorkel.indirect_sacrifice >= 1
                else ""
            ),
        ),
        "sacrifice_commitment": (
            snorkel.direct_sacrifice,
            "sacrifice commitment" if snorkel.direct_sacrifice else "",
        ),
        "approaching_corner": (
            snorkel.approaching_corner,
            "approaching corner" if snorkel.approaching_corner else "",
        ),
        "occupy_corner": (
            snorkel.occupy_corner,
            "occupy corner" if snorkel.occupy_corner else "",
        ),
    }

    if concept not in gates:
        # Unknown concept — reject
        return False, ""

    passes, evidence = gates[concept]
    return bool(passes), evidence


def _snorkel_primary_concepts(snorkel: SnorkelFields, move_number: Optional[int]) -> List[Tuple[str, str]]:
    """Concepts that pass snorkel gates regardless of probe delta (ordered by priority)."""
    priority = [
        "must_live", "kill_attack", "atari", "cut", "multi_connect", "connect",
        "invasion", "fight_pressure", "fight_wide", "territory_building",
        "territory_securing", "opponent_reduction", "influence_surge",
        "occupy_corner", "approaching_corner", "forcing", "tenuki",
        "sacrifice_direct", "sacrifice_indirect", "sacrifice_commitment",
        "aji_reduction", "group_strength_up", "group_strength_down",
        "group_strength_shift", "group_connectivity_up", "group_connectivity_down",
        "group_connectivity_shift", "extend", "urgency_peak",
    ]
    out = []
    for concept in priority:
        passes, evidence = check_concept_gate(concept, snorkel, move_number)
        if passes:
            out.append((concept, evidence))
    return out


def select_concepts(
    deltas: Dict[str, float],
    snorkel: SnorkelFields,
    max_primary: int = 3,
    max_supporting: int = 2,
    move_number: Optional[int] = None,
) -> Tuple[List[str], Dict[str, float], List[str]]:
    """
    Select concepts for commentary: snorkel-first, probe delta for ranking.

    Admits concepts that pass snorkel gates. Uses probe Δ to rank among gated
    candidates. Strong snorkel-only primaries are kept when Δ ≤ 0.
    """
    candidates: Dict[str, ConceptCandidate] = {}

    # Probe-positive + gated
    for concept, delta in deltas.items():
        if concept not in KNOWN_CONCEPTS:
            continue
        if delta is None:
            continue
        passes_gate, evidence = check_concept_gate(concept, snorkel, move_number)
        if not passes_gate:
            continue
        if delta <= 0 and concept in LOW_TRUST_CONCEPTS:
            continue
        candidates[concept] = ConceptCandidate(
            concept=concept,
            delta=float(delta) if delta > 0 else 0.0,
            gated=True,
            evidence=evidence,
            snorkel_only=delta <= 0,
        )

    # Snorkel-only primaries when probe missed them
    for concept, evidence in _snorkel_primary_concepts(snorkel, move_number):
        if concept in candidates:
            continue
        if concept in LOW_TRUST_CONCEPTS and concept not in ("invasion",):
            # Still allow invasion via snorkel; sacrifice_commitment only if direct
            if concept == "sacrifice_commitment" and not snorkel.direct_sacrifice:
                continue
        candidates[concept] = ConceptCandidate(
            concept=concept,
            delta=0.0,
            gated=True,
            evidence=evidence,
            snorkel_only=True,
        )

    ranked = sorted(candidates.values(), key=lambda c: c.rank_score, reverse=True)

    # When a hard tactic is present, drop weak filler concepts from the shortlist
    hard_tactics = {
        "kill_attack", "atari", "cut", "connect", "multi_connect", "must_live", "invasion",
    }
    has_hard = any(c.concept in hard_tactics for c in ranked)
    filler = {"extend", "tenuki", "urgency_peak"}
    if has_hard:
        ranked = [c for c in ranked if c.concept not in filler] or ranked

    primary = ranked[:max_primary]
    supporting = ranked[max_primary:max_primary + max_supporting]
    selected = primary + supporting

    selected_concepts = [c.concept for c in selected]
    concept_deltas = {c.concept: c.delta for c in selected}

    evidence_highlights = []
    for c in selected:
        if c.evidence:
            evidence_highlights.append(c.evidence)

    # Prefer concrete attack targets in highlights
    if snorkel.killing_attack or snorkel.attack:
        head_coords = locs_to_human(snorkel.attacked_heads)
        if head_coords:
            label = "kills / threatens group at" if snorkel.killing_attack else "attacks group at"
            evidence_highlights.insert(0, f"{label} {', '.join(head_coords)}")
        if snorkel.attacked_regions:
            evidence_highlights.append(
                f"attacked regions: {', '.join(snorkel.attacked_regions)}"
                + (f" ({snorkel.attacked_groups_count} groups)" if snorkel.attacked_groups_count else "")
            )

    if snorkel.new_group:
        evidence_highlights.append("creates new group")
    if snorkel.liberties:
        evidence_highlights.append(f"{snorkel.liberties} liberties")
    if snorkel.cut and snorkel.cut_groups_created > 0:
        cut_info = f"cut ({snorkel.cut_groups_created} groups"
        if snorkel.cut_regions:
            cut_info += f" in {', '.join(snorkel.cut_regions)}"
        cut_info += ")"
        if cut_info not in evidence_highlights:
            evidence_highlights.append(cut_info)
    if snorkel.merged_regions:
        evidence_highlights.append(f"merged regions: {', '.join(snorkel.merged_regions)}")

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for h in evidence_highlights:
        if h not in seen:
            seen.add(h)
            deduped.append(h)

    return selected_concepts, concept_deltas, deduped


def load_snorkel_data(snorkel_path: Path) -> Dict[int, Dict[str, Any]]:
    """Load snorkel JSONL file into a dict keyed by move_number."""
    data = {}
    with open(snorkel_path, "r") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                move_num = entry["move_number"]
                data[move_num] = entry
    return data


def load_concepts_data(concepts_path: Path) -> Dict[str, Any]:
    """Load concepts JSON file."""
    with open(concepts_path, "r") as f:
        return json.load(f)


def load_global_stats(games_dir: Path) -> Optional[Dict[str, Any]]:
    """Load global_stats.json if it exists."""
    stats_path = games_dir / "global_stats.json"
    if stats_path.exists():
        with open(stats_path, "r") as f:
            return json.load(f)
    return None


def compute_percentile_from_stats(value: float, feature_stats: Dict[str, float]) -> Optional[float]:
    """Compute percentile rank for a value given feature statistics."""
    if not feature_stats:
        return None

    p10 = feature_stats.get("p10")
    p25 = feature_stats.get("p25")
    p50 = feature_stats.get("p50")
    p75 = feature_stats.get("p75")
    p90 = feature_stats.get("p90")
    min_val = feature_stats.get("min")
    max_val = feature_stats.get("max")

    if any(x is None for x in [p10, p25, p50, p75, p90, min_val, max_val]):
        return None

    if value <= min_val:
        return 0.0
    if value >= max_val:
        return 100.0

    if value <= p10:
        return 10.0 * (value - min_val) / (p10 - min_val) if p10 > min_val else 0.0
    elif value <= p25:
        return 10.0 + 15.0 * (value - p10) / (p25 - p10) if p25 > p10 else 10.0
    elif value <= p50:
        return 25.0 + 25.0 * (value - p25) / (p50 - p25) if p50 > p25 else 25.0
    elif value <= p75:
        return 50.0 + 25.0 * (value - p50) / (p75 - p50) if p75 > p50 else 50.0
    elif value <= p90:
        return 75.0 + 15.0 * (value - p75) / (p90 - p75) if p90 > p75 else 75.0
    else:
        return 90.0 + 10.0 * (value - p90) / (max_val - p90) if max_val > p90 else 90.0


def _is_meaningful(value: Any) -> bool:
    """True if value should appear in a slim evidence packet."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return abs(value) > 1e-9
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return len(value) > 0
    return True


def _slim_snorkel_summary(
    full: Dict[str, Any],
    selected_concepts: List[str],
) -> Dict[str, Any]:
    """Keep only fields relevant to selected concepts (plus magnitudes when present)."""
    keep_keys: Set[str] = set()
    for concept in selected_concepts:
        for key in CONCEPT_SNORKEL_FIELDS.get(concept, []):
            keep_keys.add(key)

    # Always allow a few contextual fields when true/nonzero
    for key in ("new_group", "liberties", "must_live"):
        if _is_meaningful(full.get(key)):
            keep_keys.add(key)

    slim = {}
    for key in keep_keys:
        if key in full and _is_meaningful(full[key]):
            slim[key] = full[key]
    return slim


def build_evidence_packet(
    game_id: str,
    move_data: Dict[str, Any],
    snorkel_entry: Dict[str, Any],
    max_primary: int = 3,
    max_supporting: int = 2,
    global_stats: Optional[Dict[str, Any]] = None,
    games_dir: Optional[Path] = None,
) -> EvidencePacket:
    """Build an evidence packet for a single move."""
    move_number = move_data["move_number"]
    player = move_data["player"]
    deltas = move_data.get("deltas", {})
    move_loc = move_data.get("move_loc")

    analysis = snorkel_entry.get("analysis", {})
    snorkel_fields = SnorkelFields.from_snorkel_analysis(analysis)

    selected_concepts, concept_deltas, evidence_highlights = select_concepts(
        deltas, snorkel_fields, max_primary, max_supporting, move_number=move_number
    )

    percentiles = analysis.get("percentiles", {})

    if global_stats is None and games_dir is not None:
        global_stats = load_global_stats(games_dir)

    def get_percentile(field_name: str, value: Optional[float] = None) -> Optional[float]:
        if value is not None:
            if abs(value) < 1e-6 and "count" not in field_name.lower():
                return None

        if field_name in percentiles:
            return percentiles[field_name]
        percentile_field = f"{field_name}_percentile"
        if percentile_field in analysis:
            return analysis[percentile_field]

        if global_stats and value is not None:
            features = global_stats.get("features", {})
            if field_name in features:
                return compute_percentile_from_stats(value, features[field_name])
        return None

    move_coord = loc_to_human(move_loc)
    move_region = idx361_to_region(move_loc)

    snorkel_summary_full = {
        "move_coord": move_coord,
        "move_region": move_region,
        "cut": snorkel_fields.cut,
        "cut_groups_created": snorkel_fields.cut_groups_created,
        "cut_regions": snorkel_fields.cut_regions,
        "cut_head_coords": locs_to_human(snorkel_fields.cut_head_locs),
        "connection": snorkel_fields.connection,
        "connection_strength_gain": snorkel_fields.connection_strength_gain,
        "connection_strength_gain_magnitude": percentile_to_magnitude(
            get_percentile("connection_strength_gain", snorkel_fields.connection_strength_gain)
        ),
        "merged_groups_regions": snorkel_fields.merged_regions,
        "merged_groups_head_coords": locs_to_human(snorkel_fields.merged_heads),
        "extension": snorkel_fields.extension,
        "atari": snorkel_fields.atari,
        "forcing": snorkel_fields.forcing,
        "tenuki": snorkel_fields.tenuki,
        "invasion": snorkel_fields.invasion,
        "occupy_corner": snorkel_fields.occupy_corner,
        "approaching_corner": snorkel_fields.approaching_corner,
        "attack": snorkel_fields.attack,
        "killing_attack": snorkel_fields.killing_attack,
        "reduce_aji": snorkel_fields.reduce_aji,
        "aji_reduction_intensity_magnitude": (
            percentile_to_magnitude(
                get_percentile("aji_reduction_intensity", snorkel_fields.aji_reduction_intensity)
            )
            if snorkel_fields.aji_reduction_intensity is not None
            else None
        ),
        "attacked_groups_count": snorkel_fields.attacked_groups_count,
        "max_attack_intensity": (
            round(snorkel_fields.max_attack_intensity, 3)
            if snorkel_fields.max_attack_intensity is not None
            else None
        ),
        "max_attack_intensity_magnitude": (
            percentile_to_magnitude(
                get_percentile("max_attack_intensity", snorkel_fields.max_attack_intensity)
            )
            if snorkel_fields.max_attack_intensity is not None
            else None
        ),
        "attacked_groups_regions": snorkel_fields.attacked_regions,
        "attacked_groups_head_coords": locs_to_human(snorkel_fields.attacked_heads),
        "potential_territory": snorkel_fields.potential_territory,
        "solid_territory": snorkel_fields.solid_territory,
        "building_count": snorkel_fields.building_count,
        "building_count_magnitude": percentile_to_magnitude(
            get_percentile("building_count", float(snorkel_fields.building_count))
        ),
        "solidification_count": snorkel_fields.solidification_count,
        "solidification_count_magnitude": percentile_to_magnitude(
            get_percentile("solidification_count", float(snorkel_fields.solidification_count))
        ),
        "reduction_count": snorkel_fields.reduction_count,
        "reduction_count_magnitude": percentile_to_magnitude(
            get_percentile("reduction_count", float(snorkel_fields.reduction_count))
        ),
        "group_strength_delta": round(snorkel_fields.group_strength_delta, 3),
        "group_strength_delta_magnitude": percentile_to_magnitude(
            get_percentile("group_strength_delta", snorkel_fields.group_strength_delta)
        ),
        "max_group_strength_delta": (
            round(snorkel_fields.max_group_strength_delta, 3)
            if snorkel_fields.max_group_strength_delta is not None
            else None
        ),
        "current_group_connectivity_delta": round(snorkel_fields.current_group_connectivity_delta, 3),
        "current_group_connectivity_delta_magnitude": percentile_to_magnitude(
            get_percentile(
                "current_group_connectivity_delta",
                snorkel_fields.current_group_connectivity_delta,
            )
        ),
        "influence_count_delta": snorkel_fields.influence_count_delta,
        "influence_count_delta_magnitude": percentile_to_magnitude(
            get_percentile("influence_count_delta", float(snorkel_fields.influence_count_delta))
        ),
        "liberties": snorkel_fields.liberties,
        "new_group": snorkel_fields.new_group,
        "urgency_max": round(snorkel_fields.urgency_max, 3) if snorkel_fields.urgency_max else None,
        "direct_sacrifice": snorkel_fields.direct_sacrifice,
        "direct_sacrifice_intensity_magnitude": (
            percentile_to_magnitude(
                get_percentile("direct_sacrifice_intensity", snorkel_fields.direct_sacrifice_intensity)
            )
            if snorkel_fields.direct_sacrifice_intensity is not None
            else None
        ),
        "indirect_sacrifice": snorkel_fields.indirect_sacrifice,
        "indirect_sacrifice_intensity_magnitude": (
            percentile_to_magnitude(
                get_percentile(
                    "indirect_sacrifice_intensity",
                    snorkel_fields.indirect_sacrifice_intensity,
                )
            )
            if snorkel_fields.indirect_sacrifice_intensity is not None
            else None
        ),
        "indirect_sacrifice_coords": locs_to_human(snorkel_fields.indirect_sacrifice_locs),
        "must_live": snorkel_fields.must_live,
    }

    # Always include move location context
    slim = _slim_snorkel_summary(snorkel_summary_full, selected_concepts)
    if move_coord:
        slim["move_coord"] = move_coord
    if move_region:
        slim["move_region"] = move_region

    return EvidencePacket(
        game_id=game_id,
        player=player,
        move_number=move_number,
        selected_concepts=selected_concepts,
        concept_deltas=concept_deltas,
        snorkel=slim,
        evidence_highlights=evidence_highlights,
        move_coord=move_coord,
        move_region=move_region,
        primary_concept=selected_concepts[0] if selected_concepts else None,
    )


def build_all_evidence_packets(
    game_id: str,
    snorkel_path: Path,
    concepts_path: Path,
    max_moves: Optional[int] = None,
) -> List[EvidencePacket]:
    """Build evidence packets for all moves in a game."""
    snorkel_data = load_snorkel_data(snorkel_path)
    concepts_data = load_concepts_data(concepts_path)

    games_dir = snorkel_path.parent.parent
    global_stats = load_global_stats(games_dir)

    packets = []
    moves = concepts_data.get("moves", [])

    if max_moves:
        moves = moves[:max_moves]

    for move_data in moves:
        move_num = move_data["move_number"]
        if move_num not in snorkel_data:
            continue

        packet = build_evidence_packet(
            game_id=game_id,
            move_data=move_data,
            snorkel_entry=snorkel_data[move_num],
            global_stats=global_stats,
            games_dir=games_dir,
        )
        packets.append(packet)

    return packets
