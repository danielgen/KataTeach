from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from board import Board  # noqa: E402

from daniele_experiment.build_validated_labels import (  # noqa: E402
    rebuild_game_labels,
    required_migration_fields,
)


def _all_legal_output(board, peak_loc=None, peak=0.5):
    legal = [
        board.loc(x, y)
        for y in range(board.size)
        for x in range(board.size)
        if board.would_be_legal(board.pla, board.loc(x, y))
    ]
    weights = {loc: 1.0 for loc in legal}
    weights[Board.PASS_LOC] = 1.0
    if peak_loc is not None:
        weights[peak_loc] = peak * 1000.0
    total = sum(weights.values())
    return [[loc, value / total] for loc, value in weights.items()]


def _raw_game():
    board = Board(19)
    first = board.loc(9, 9)
    board.play(Board.BLACK, first)
    second = board.loc(15, 9)  # distance exactly six
    first_post = _all_legal_output(board, peak_loc=second, peak=0.9)
    board.play(Board.WHITE, second)
    forced_reply = board.loc(15, 10)
    second_post = [[forced_reply, 0.951], [Board.PASS_LOC, 0.049]]
    return [
        {
            "move_number": 1,
            "player": "b",
            "move_loc": first,
            "idx361": 9 * 19 + 9,
            "moves_and_probs0": first_post,
        },
        {
            "move_number": 2,
            "player": "w",
            "move_loc": second,
            "idx361": 9 * 19 + 15,
            "moves_and_probs0": second_post,
        },
    ]


def test_rebuild_uses_previous_record_for_pre_and_current_for_reply():
    moves = _raw_game()
    legacy = [
        {
            "move_number": 1,
            "player": moves[0]["player"],
            "move_loc": moves[0]["move_loc"],
            "analysis": {"cut": True, "tenuki": True},
        },
        {
            "move_number": 2,
            "player": moves[1]["player"],
            "move_loc": moves[1]["move_loc"],
            "analysis": {"cut": False, "forcing": False, "urgency": {"x": 1}},
        },
    ]
    rows, audit = rebuild_game_labels(moves, legacy, ["cut"])

    assert rows[0]["analysis"]["regional_policy_peak"] is None
    assert rows[1]["analysis"]["regional_policy_peak"] is not None
    assert rows[1]["analysis"]["tenuki_distance6"] is True
    assert rows[1]["analysis"]["tenuki_manhattan_distance"] == 6.0
    # The selected move's post-move opponent policy, not the concentrated
    # current-player pre-policy, determines this forcing proxy.
    assert rows[1]["analysis"]["reply_peak_value"] == pytest.approx(0.951)
    assert rows[1]["analysis"]["reply_peak95"] is True
    assert "forcing" not in rows[1]["analysis"]
    assert "urgency" not in rows[1]["analysis"]
    assert rows[1]["analysis"]["cut"] is False
    assert audit["counts"]["missing_pre_policy"] == 1


def test_pass_replay_quarantines_only_legacy_fields_after_pass():
    board = Board(19)
    first = board.loc(3, 3)
    board.play(Board.BLACK, first)
    first_post = _all_legal_output(board)
    board.play(Board.WHITE, Board.PASS_LOC)
    pass_post = _all_legal_output(board)
    third = board.loc(15, 15)
    board.play(Board.BLACK, third)
    third_post = _all_legal_output(board)
    moves = [
        {"move_number": 1, "player": "b", "move_loc": first, "idx361": 60,
         "moves_and_probs0": first_post},
        {"move_number": 2, "player": "w", "move_loc": 0, "idx361": 361,
         "moves_and_probs0": pass_post},
        {"move_number": 3, "player": "b", "move_loc": third, "idx361": 300,
         "moves_and_probs0": third_post},
    ]
    legacy = [
        {
            "move_number": index,
            "player": moves[index - 1]["player"],
            "move_loc": moves[index - 1]["move_loc"],
            "analysis": {"cut": True},
        }
        for index in range(1, 4)
    ]
    rows, audit = rebuild_game_labels(moves, legacy, ["cut"])
    assert rows[0]["analysis"]["cut"] is True
    assert "cut" not in rows[1]["analysis"]
    assert "cut" not in rows[2]["analysis"]
    assert rows[2]["analysis"]["regional_policy_peak"] is not None
    assert rows[2]["analysis"]["tenuki_distance6"] is True
    assert rows[2]["analysis"]["tenuki_manhattan_distance"] == 24.0
    assert audit["counts"]["legacy_rows_excluded_for_pass_alignment"] == 2


def test_migration_whitelist_excludes_central_contract_sources(tmp_path):
    path = tmp_path / "concepts.yaml"
    path.write_text(
        """
concepts:
  tenuki: {type: binary, source: tenuki_distance6}
  forcing: {type: binary, source: reply_peak95}
  urgency_peak: {type: quantile, source: regional_policy_peak}
  cut:
    type: binary
    source: cut
    filters:
      - {column: attacked_groups_count, operator: '>=', value: 1}
  sacrifice_direct: {type: binary, source: direct_sacrifice}
  sacrifice_commitment:
    type: quantile
    source: sacrifice_intensity
    filters:
      - {column: label_sacrifice_direct, operator: '==', value: 1}
"""
    )
    assert required_migration_fields(path) == (
        "attacked_groups_count",
        "cut",
        "direct_sacrifice",
        "sacrifice_intensity",
    )


def test_coordinate_mismatch_fails_closed():
    moves = _raw_game()
    moves[1]["idx361"] = moves[1]["move_loc"]
    with pytest.raises(ValueError, match="Coordinate mismatch"):
        rebuild_game_labels(moves, [], [])


def test_migration_fails_closed_when_archived_action_is_shifted():
    moves = _raw_game()
    legacy = [
        {
            "move_number": move["move_number"],
            "player": move["player"],
            "move_loc": move["move_loc"],
            "analysis": {"cut": True},
        }
        for move in moves
    ]
    legacy[1]["move_loc"] = moves[0]["move_loc"]
    with pytest.raises(ValueError, match="Archived label alignment mismatch"):
        rebuild_game_labels(moves, legacy, ["cut"])


def test_fresh_holdout_recomputes_canonical_fields_without_legacy_rows():
    rows, audit = rebuild_game_labels(_raw_game(), [], [])
    assert rows[1]["analysis"]["tenuki_distance6"] is True
    assert rows[1]["analysis"]["reply_peak95"] is True
    assert rows[1]["analysis"]["regional_policy_peak"] is not None
    assert "cut" not in rows[1]["analysis"]
    assert audit["counts"]["migrated_fields_missing"] == 0
