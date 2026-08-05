from kata_teach.commentary.grounding import build_grounded_fallback, validate_grounding
from kata_teach.commentary.schema import CommentaryOutput, EvidencePacket


def packet(**snorkel):
    return EvidencePacket(
        game_id="game",
        player="w",
        move_number=24,
        selected_concepts=["atari", "forcing", "territory_building"],
        concept_deltas={},
        snorkel={"atari": True, "forcing": True, "building_count": 14, **snorkel},
        evidence_highlights=[],
        move_coord="C13",
        primary_concept="atari",
    )


def test_rejects_tactics_missing_from_current_evidence():
    output = CommentaryOutput(
        24,
        "This move cuts and attacks an opposing group while producing an influence surge.",
        ["atari"],
    )
    errors = validate_grounding(output, packet())
    assert any("cut" in error for error in errors)
    assert any("attack" in error for error in errors)
    assert any("influence_count_delta" in error for error in errors)


def test_rejects_raw_internal_indices():
    output = CommentaryOutput(24, "This attacks the group at head 63.", ["atari"])
    assert any("raw" in error for error in validate_grounding(output, packet(attack=True)))


def test_fallback_uses_only_current_gated_evidence():
    output = build_grounded_fallback(packet())
    assert output.comment == (
        "White at C13 puts an opposing group in atari, creates a forcing threat, "
        "and builds territory on 14 points."
    )
    assert output.concepts_used == ["atari", "forcing", "territory_building"]
    assert validate_grounding(output, packet()) == []
