"""
JSON schema definitions for commentary generation output.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import json


@dataclass
class CommentaryOutput:
    """Output schema for generated move commentary."""
    move_number: int
    comment: str  # Short factual comment (1-2 sentences)
    concepts_used: List[str]  # from selected concepts
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CommentaryOutput':
        """Create from dictionary."""
        return cls(
            move_number=data['move_number'],
            comment=data['comment'],
            concepts_used=data.get('concepts_used', []),
        )
    
    def validate(
        self,
        selected_concepts: List[str],
        expected_move_number: Optional[int] = None,
        max_comment_chars: int = 600,
    ) -> List[str]:
        """
        Validate the commentary output.

        Returns a list of validation errors (empty if valid).
        """
        errors = []

        if not isinstance(self.move_number, int):
            errors.append("move_number must be an integer")
        elif expected_move_number is not None and self.move_number != expected_move_number:
            errors.append(
                f"move_number {self.move_number} does not match expected {expected_move_number}"
            )

        if not isinstance(self.comment, str):
            errors.append("comment must be a string")
        else:
            comment = self.comment.strip()
            if not comment:
                errors.append("comment must be non-empty")
            elif len(comment) > max_comment_chars:
                errors.append(f"comment too long ({len(comment)} > {max_comment_chars} chars)")
            # Ban raw KataGo-style numeric indices used as coordinates
            import re
            if re.search(r"\b(?:head|loc|index)\s*\d{1,3}\b", comment, re.IGNORECASE):
                errors.append("comment must not use raw KataGo indices; use human coords or regions")
            if re.search(r"\bat\s+\d{1,3}\b", comment, re.IGNORECASE):
                errors.append("comment must not use numeric stone indices; use human coords (e.g. Q16)")

        if not isinstance(self.concepts_used, list):
            errors.append("concepts_used must be a list")
        else:
            if selected_concepts and len(self.concepts_used) == 0:
                errors.append("concepts_used must be non-empty when selected_concepts is non-empty")
            invalid_concepts = set(self.concepts_used) - set(selected_concepts)
            if invalid_concepts:
                errors.append(f"concepts_used contains invalid concepts: {invalid_concepts}")

        return errors


@dataclass 
class SnorkelFields:
    """Extracted snorkel analysis fields for a move."""
    # Tactics
    cut: bool = False
    cut_groups_created: int = 0
    cut_regions: List[str] = field(default_factory=list)
    cut_head_locs: List[int] = field(default_factory=list)
    connection: bool = False
    extension: bool = False
    atari: bool = False
    forcing: bool = False
    tenuki: bool = False
    invasion: bool = False
    occupy_corner: bool = False
    approaching_corner: bool = False
    
    # Connection details
    connection_strength_gain: float = 0.0
    merged_regions: List[str] = field(default_factory=list)
    merged_heads: List[int] = field(default_factory=list)
    
    # Attack
    attack: bool = False
    killing_attack: bool = False
    reduce_aji: bool = False
    aji_reduction_intensity: Optional[float] = None
    attacked_groups_count: int = 0
    avg_attack_intensity: Optional[float] = None
    max_attack_intensity: Optional[float] = None
    attacked_heads: List[int] = field(default_factory=list)
    attacked_regions: List[str] = field(default_factory=list)
    attacked_groups_strength_deltas: List[float] = field(default_factory=list)
    
    # Territory
    potential_territory: int = 0
    solid_territory: int = 0
    building_count: int = 0
    solidification_count: int = 0
    reduction_count: int = 0
    building_intensity: float = 0.0
    solidification_intensity: float = 0.0
    reduction_intensity: float = 0.0
    
    # Group (all groups average)
    group_strength_delta: float = 0.0
    group_connectivity_delta: float = 0.0
    max_group_strength_delta: Optional[float] = None
    max_group_connectivity_delta: Optional[float] = None
    influence_count_delta: int = 0
    influence_strength_delta: float = 0.0
    
    # Current group (the group containing this move)
    current_group_strength: Optional[float] = None
    current_group_strength_delta: float = 0.0
    current_group_connectivity: Optional[float] = None
    current_group_connectivity_delta: float = 0.0
    current_group_influence_count: int = 0
    current_group_influence_count_delta: int = 0
    current_group_influence_strength: Optional[float] = None
    current_group_influence_strength_delta: float = 0.0
    liberties: int = 0
    new_group: bool = False
    
    # Urgency
    urgency: Dict[str, float] = field(default_factory=dict)
    urgency_max: Optional[float] = None
    
    # Sacrifice
    direct_sacrifice: bool = False
    direct_sacrifice_intensity: Optional[float] = None
    indirect_sacrifice: int = 0
    indirect_sacrifice_intensity: Optional[float] = None
    indirect_sacrifice_locs: List[int] = field(default_factory=list)
    
    # Other
    must_live: bool = False
    
    @classmethod
    def from_snorkel_analysis(cls, analysis: Dict[str, Any]) -> 'SnorkelFields':
        """Extract relevant fields from snorkel analysis dict."""
        urgency = analysis.get('urgency', {})
        urgency_max = max(urgency.values()) if urgency else None
        
        return cls(
            # Tactics
            cut=analysis.get('cut', False),
            cut_groups_created=analysis.get('cut_groups_created', 0),
            cut_regions=analysis.get('cut_regions', []),
            cut_head_locs=analysis.get('cut_head_locs', []),
            connection=analysis.get('connection', False),
            extension=analysis.get('extension', False),
            atari=analysis.get('atari', False),
            forcing=analysis.get('forcing', False),
            tenuki=analysis.get('tenuki', False),
            invasion=analysis.get('invasion', False),
            occupy_corner=analysis.get('occupy_corner', False),
            approaching_corner=analysis.get('approaching_corner', False),
            
            # Connection details
            connection_strength_gain=analysis.get('connection_strength_gain', 0.0),
            merged_regions=analysis.get('merged_groups_regions', []),
            merged_heads=analysis.get('merged_groups_head_locs', []),
            
            # Attack
            attack=analysis.get('attack', False),
            killing_attack=analysis.get('killing_attack', False),
            reduce_aji=analysis.get('reduce_aji', False),
            aji_reduction_intensity=analysis.get('aji_reduction_intensity'),
            attacked_groups_count=analysis.get('attacked_groups_count', 0),
            avg_attack_intensity=analysis.get('avg_attack_intensity'),
            max_attack_intensity=analysis.get('max_attack_intensity'),
            attacked_heads=analysis.get('attacked_groups_head_locs', []),
            attacked_regions=analysis.get('attacked_groups_regions', []),
            attacked_groups_strength_deltas=analysis.get('attacked_groups_strength_deltas', []),
            
            # Territory
            potential_territory=analysis.get('potential_territory', 0),
            solid_territory=analysis.get('solid_territory', 0),
            building_count=analysis.get('building_count', 0),
            solidification_count=analysis.get('solidification_count', 0),
            reduction_count=analysis.get('reduction_count', 0),
            building_intensity=analysis.get('building_intensity', 0.0),
            solidification_intensity=analysis.get('solidification_intensity', 0.0),
            reduction_intensity=analysis.get('reduction_intensity', 0.0),
            
            # Group (all groups average)
            group_strength_delta=analysis.get('group_strength_delta', 0.0),
            group_connectivity_delta=analysis.get('group_connectivity_delta', 0.0),
            max_group_strength_delta=analysis.get('max_group_strength_delta'),
            max_group_connectivity_delta=analysis.get('max_group_connectivity_delta'),
            influence_count_delta=analysis.get('influence_count_delta', 0),
            influence_strength_delta=analysis.get('influence_strength_delta', 0.0),
            
            # Current group (the group containing this move)
            current_group_strength=analysis.get('current_group_strength'),
            current_group_strength_delta=analysis.get('current_group_strength_delta', 0.0),
            current_group_connectivity=analysis.get('current_group_connectivity'),
            current_group_connectivity_delta=analysis.get('current_group_connectivity_delta', 0.0),
            current_group_influence_count=analysis.get('current_group_influence_count', 0),
            current_group_influence_count_delta=analysis.get('current_group_influence_count_delta', 0),
            current_group_influence_strength=analysis.get('current_group_influence_strength'),
            current_group_influence_strength_delta=analysis.get('current_group_influence_strength_delta', 0.0),
            liberties=analysis.get('liberties', 0),
            new_group=analysis.get('creates_new_group', False),
            
            # Urgency
            urgency=urgency,
            urgency_max=urgency_max,
            
            # Sacrifice
            direct_sacrifice=analysis.get('direct_sacrifice', False),
            direct_sacrifice_intensity=analysis.get('direct_sacrifice_intensity'),
            indirect_sacrifice=analysis.get('indirect_sacrifice', 0),
            indirect_sacrifice_intensity=analysis.get('indirect_sacrifice_intensity'),
            indirect_sacrifice_locs=analysis.get('indirect_sacrifice_locs', []),
            
            # Other
            must_live=analysis.get('must_live', False),
        )


@dataclass
class EvidencePacket:
    """Evidence packet sent to the LLM for commentary generation."""
    game_id: str
    player: str  # 'b' or 'w'
    move_number: int
    selected_concepts: List[str]  # Primary + supporting concepts
    concept_deltas: Dict[str, float]  # Only for selected concepts
    snorkel: Dict[str, Any]  # Key fields from snorkel analysis
    evidence_highlights: List[str]  # Short human-readable evidence strings
    move_coord: Optional[str] = None  # e.g. Q16
    move_region: Optional[str] = None  # e.g. corner_tr
    primary_concept: Optional[str] = None  # First selected concept (lead claim)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


# JSON schema for OpenAI structured output
# This schema is passed directly as the "json_schema" value in response_format
COMMENTARY_JSON_SCHEMA = {
    "name": "CommentaryOutput",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "move_number": {
                "type": "integer",
                "description": "The move number in the game"
            },
            "comment": {
                "type": "string",
                "description": "Short factual comment (1-3 sentences)"
            },
            "concepts_used": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of concepts from selected_concepts that apply"
            }
        },
        "required": ["move_number", "comment", "concepts_used"],
        "additionalProperties": False
    }
}
