"""Invariant tests for the versioned operational-definition contracts."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from board import Board  # noqa: E402

from daniele_experiment.operational_definitions import (  # noqa: E402
    CONTRACTS,
    IneligiblePositionError,
    MissingCandidateScoresError,
    OperationalDefinitionError,
    PolicySupport,
    PolicyTime,
    TransitionContext,
    classify_region_xy,
    get_contract,
    legal_board_mask,
    legal_policy_view,
    manhattan_distance,
    region_masses,
    regional_peak_contrast_mask,
    regional_peak_target_mask,
    regional_policy_peak_observed_label,
    regional_policy_peak_observed_value,
    regional_policy_readouts,
    reply_peak95_candidate_readouts,
    reply_peak95_observed_label,
    reply_peak95_observed_value,
    reply_peak95_target_mask,
    rms_contrast_mask,
    tenuki_action_predicate,
    tenuki_contrast_mask,
    tenuki_observed_label,
    tenuki_policy_readouts,
    tenuki_target_mask,
)


def _policy(board, pairs, *, time, support, player=None):
    return legal_policy_view(
        pairs,
        board,
        time=time,
        support=support,
        player_to_move=player,
        coordinate_system="board_loc",
    )


def _board_with_previous_move():
    board = Board(19)
    previous = board.loc(9, 9)
    board.play(Board.BLACK, previous)
    assert board.pla == Board.WHITE
    return board, previous


def test_legal_policy_filters_illegal_actions_and_normalizes_explicit_support():
    board, occupied = _board_with_previous_move()
    first = board.loc(3, 3)
    second = board.loc(15, 15)
    pairs = [(occupied, 0.8), (first, 0.1), (second, 0.1), (Board.PASS_LOC, 0.2)]

    full = _policy(
        board,
        pairs,
        time=PolicyTime.PRE_MOVE,
        support=PolicySupport.LEGAL_PLUS_PASS,
    )
    assert occupied not in full.probabilities
    assert full.probability(first) == pytest.approx(0.25)
    assert full.probability(second) == pytest.approx(0.25)
    assert full.pass_probability == pytest.approx(0.5)
    assert sum(full.probabilities.values()) == pytest.approx(1.0)

    board_only = _policy(
        board,
        pairs,
        time=PolicyTime.PRE_MOVE,
        support=PolicySupport.LEGAL_BOARD_CONDITIONAL,
    )
    assert Board.PASS_LOC not in board_only.probabilities
    assert board_only.probability(first) == pytest.approx(0.5)
    assert board_only.probability(second) == pytest.approx(0.5)


def test_policy_ingestion_rejects_dense_or_ambiguous_coordinates():
    board = Board(19)
    kwargs = {
        "time": PolicyTime.PRE_MOVE,
        "support": PolicySupport.LEGAL_BOARD_CONDITIONAL,
        "coordinate_system": "board_loc",
    }
    with pytest.raises(OperationalDefinitionError, match="Dense policy0"):
        legal_policy_view({"policy0": np.ones(362)}, board, **kwargs)
    with pytest.raises(OperationalDefinitionError, match="coordinate-ambiguous"):
        legal_policy_view(np.ones(362), board, **kwargs)
    with pytest.raises(OperationalDefinitionError, match="coordinate_system='board_loc'"):
        legal_policy_view(
            [(board.loc(3, 3), 1.0)],
            board,
            time=PolicyTime.PRE_MOVE,
            support=PolicySupport.LEGAL_BOARD_CONDITIONAL,
            coordinate_system="idx361",
        )


def test_region_partition_has_exact_19x19_boundaries_and_complete_coverage():
    assert classify_region_xy(5, 5) == "corner_tl"
    assert classify_region_xy(6, 5) == "side_top"
    assert classify_region_xy(12, 5) == "side_top"
    assert classify_region_xy(13, 5) == "corner_tr"
    assert classify_region_xy(5, 6) == "side_left"
    assert classify_region_xy(6, 6) == "center"
    assert classify_region_xy(12, 12) == "center"
    assert classify_region_xy(13, 13) == "corner_br"

    assignments = [classify_region_xy(x, y) for y in range(19) for x in range(19)]
    assert len(assignments) == 361
    assert set(assignments) == {
        "corner_tl",
        "corner_tr",
        "corner_bl",
        "corner_br",
        "side_left",
        "side_right",
        "side_top",
        "side_bottom",
        "center",
    }


def test_tenuki_label_action_set_and_policy_readout_share_one_predicate():
    board, previous = _board_with_previous_move()
    near = board.loc(14, 9)  # Manhattan distance five.
    far = board.loc(15, 9)   # Manhattan distance six.
    pre_policy = _policy(
        board,
        [(near, 0.3), (far, 0.7)],
        time=PolicyTime.PRE_MOVE,
        support=PolicySupport.LEGAL_BOARD_CONDITIONAL,
    )

    far_context = TransitionContext(board, board.pla, previous, far, pre_policy)
    near_context = TransitionContext(board, board.pla, previous, near, pre_policy)
    target = tenuki_target_mask(board, previous)

    assert manhattan_distance(board, previous, near) == 5
    assert manhattan_distance(board, previous, far) == 6
    assert tenuki_action_predicate(board, previous, near) is False
    assert tenuki_action_predicate(board, previous, far) is True
    assert tenuki_observed_label(near_context) == int(target[9, 14]) == 0
    assert tenuki_observed_label(far_context) == int(target[9, 15]) == 1

    readouts = tenuki_policy_readouts(pre_policy, board, previous)
    assert readouts["tenuki_distance6_policy_mass"] == pytest.approx(0.7)
    assert readouts["tenuki_distance6_complement_mass"] == pytest.approx(0.3)
    assert readouts["tenuki_expected_manhattan_distance"] == pytest.approx(5.7)


def test_tenuki_missing_anchor_is_ineligible_not_negative():
    board = Board(19)
    selected = board.loc(3, 3)
    policy = _policy(
        board,
        [(selected, 1.0)],
        time=PolicyTime.PRE_MOVE,
        support=PolicySupport.LEGAL_BOARD_CONDITIONAL,
    )
    context = TransitionContext(board, board.pla, None, selected, policy)
    assert tenuki_observed_label(context) is None
    with pytest.raises(IneligiblePositionError):
        tenuki_target_mask(board, None)

    pass_context = TransitionContext(board, board.pla, Board.PASS_LOC, selected, policy)
    assert tenuki_observed_label(pass_context) is None


def test_tenuki_geometry_is_invariant_under_board_symmetries():
    board = Board(19)
    previous_xy = (4, 7)
    candidate_xy = (15, 12)
    transforms = (
        lambda x, y: (x, y),
        lambda x, y: (18 - x, y),
        lambda x, y: (x, 18 - y),
        lambda x, y: (18 - y, x),
    )
    outcomes = []
    distances = []
    for transform in transforms:
        px, py = transform(*previous_xy)
        cx, cy = transform(*candidate_xy)
        previous = board.loc(px, py)
        candidate = board.loc(cx, cy)
        outcomes.append(tenuki_action_predicate(board, previous, candidate))
        distances.append(manhattan_distance(board, previous, candidate))
    assert len(set(outcomes)) == 1
    assert len(set(distances)) == 1


def test_semantic_contrast_masks_exclude_illegal_points_and_are_normalized():
    board, previous = _board_with_previous_move()
    near = board.loc(14, 9)
    far = board.loc(15, 9)
    mask = tenuki_contrast_mask(board, previous)

    assert mask[board.loc_y(previous), board.loc_x(previous)] == 0.0
    assert mask[board.loc_y(far), board.loc_x(far)] > 0.0
    assert mask[board.loc_y(near), board.loc_x(near)] < 0.0
    active = legal_board_mask(board)
    assert np.isclose(mask[active].mean(), 0.0, atol=1e-6)
    assert np.isclose(np.sqrt(np.mean(mask[active] * mask[active])), 1.0, atol=1e-6)


def test_rms_contrast_ignores_inactive_points_in_normalization():
    target = np.zeros((5, 5), dtype=bool)
    comparison = np.zeros((5, 5), dtype=bool)
    target[0, 0] = True
    comparison[0, 1] = True
    mask = rms_contrast_mask(target, comparison)
    active = target | comparison

    np.testing.assert_array_equal(mask[~active], 0.0)
    assert mask[active].mean() == pytest.approx(0.0)
    assert np.sqrt(np.mean(np.square(mask[active]))) == pytest.approx(1.0)
    # Board-wide RMS is smaller because inactive points are deliberately not
    # part of the intervention's semantic/legal action support.
    assert np.sqrt(np.mean(np.square(mask))) == pytest.approx(np.sqrt(2.0 / 25.0))


def _forcing_context(reply_top_probability: float):
    board = Board(19)
    selected = board.loc(3, 3)
    alternative = board.loc(15, 15)
    # A highly concentrated current-player policy must not determine forcingness.
    pre_policy = _policy(
        board,
        [(selected, 0.999), (alternative, 0.001)],
        time=PolicyTime.PRE_MOVE,
        support=PolicySupport.LEGAL_BOARD_CONDITIONAL,
    )
    board_after = board.copy()
    board_after.play(Board.BLACK, selected)
    reply = board_after.loc(4, 4)
    reply_policy = _policy(
        board_after,
        [(reply, reply_top_probability), (Board.PASS_LOC, 1.0 - reply_top_probability)],
        time=PolicyTime.POST_MOVE_REPLY,
        support=PolicySupport.LEGAL_PLUS_PASS,
    )
    return TransitionContext(
        board_before=board,
        player=Board.BLACK,
        previous_move=None,
        selected_move=selected,
        pre_policy=pre_policy,
        board_after=board_after,
        reply_policy=reply_policy,
    )


def test_reply_peak95_uses_post_move_opponent_policy_and_strict_threshold():
    exact = _forcing_context(0.95)
    above = _forcing_context(0.951)
    diffuse = _forcing_context(0.60)

    assert reply_peak95_observed_value(exact) == pytest.approx(0.95)
    assert reply_peak95_observed_label(exact) == 0
    assert reply_peak95_observed_label(above) == 1
    assert reply_peak95_observed_label(diffuse) == 0


def test_missing_initial_pre_policy_is_missing_only_for_pre_policy_contract():
    context = _forcing_context(0.96)
    missing_pre = TransitionContext(
        board_before=context.board_before,
        player=context.player,
        previous_move=context.previous_move,
        selected_move=context.selected_move,
        pre_policy=None,
        board_after=context.board_after,
        reply_policy=context.reply_policy,
    )
    assert reply_peak95_observed_label(missing_pre) == 1
    assert regional_policy_peak_observed_value(missing_pre) is None


def test_reply_peak95_candidate_readout_is_exact_and_rejects_missing_scores():
    board = Board(19)
    forcing_candidate = board.loc(3, 3)
    ordinary_candidate = board.loc(15, 15)
    current = _policy(
        board,
        [(forcing_candidate, 0.25), (ordinary_candidate, 0.75)],
        time=PolicyTime.PRE_MOVE,
        support=PolicySupport.LEGAL_BOARD_CONDITIONAL,
    )
    readouts = reply_peak95_candidate_readouts(
        current,
        {forcing_candidate: 0.96, ordinary_candidate: 0.30},
    )
    assert readouts["reply_peak95_action_mass"] == pytest.approx(0.25)
    assert readouts["expected_reply_peak"] == pytest.approx(0.465)

    with pytest.raises(MissingCandidateScoresError):
        reply_peak95_candidate_readouts(current, {forcing_candidate: 0.96})


def test_reply_peak95_spatial_target_uses_candidate_counterfactuals():
    board = Board(19)
    legal = legal_board_mask(board)
    forcing_candidate = board.loc(3, 3)
    scores = {
        board.loc(x, y): (0.96 if board.loc(x, y) == forcing_candidate else 0.25)
        for y in range(board.size)
        for x in range(board.size)
        if legal[y, x]
    }
    target = reply_peak95_target_mask(board, scores)
    assert target.sum() == 1
    assert target[3, 3]


def test_regional_policy_peak_uses_pre_move_board_conditional_policy_only():
    board = Board(19)
    corner = board.loc(3, 3)
    center = board.loc(9, 9)
    side = board.loc(9, 3)
    pre_policy = _policy(
        board,
        [(corner, 0.7), (center, 0.2), (side, 0.1), (Board.PASS_LOC, 0.9)],
        time=PolicyTime.PRE_MOVE,
        support=PolicySupport.LEGAL_BOARD_CONDITIONAL,
    )
    # The pass mass was explicitly conditioned away, preserving 0.7/0.2/0.1.
    masses = region_masses(pre_policy, board)
    assert sum(masses.values()) == pytest.approx(1.0)
    assert masses["corner_tl"] == pytest.approx(0.7)
    assert masses["center"] == pytest.approx(0.2)
    assert masses["side_top"] == pytest.approx(0.1)

    selected = corner
    context = TransitionContext(board, Board.BLACK, None, selected, pre_policy)
    assert regional_policy_peak_observed_value(context) == pytest.approx(0.7)
    assert regional_policy_peak_observed_label(context, high_threshold=0.7) == 1
    assert regional_policy_peak_observed_label(context, high_threshold=0.71) == 0

    readouts = regional_policy_readouts(pre_policy, board)
    assert readouts["regional_policy_peak"] == pytest.approx(0.7)
    assert readouts["regional_policy_peak_region"] == "corner_tl"
    assert readouts["regional_policy_margin"] == pytest.approx(0.5)


def test_regional_peak_mask_and_fixed_anchor_use_canonical_regions():
    board = Board(19)
    corner = board.loc(3, 3)
    center = board.loc(9, 9)
    policy = _policy(
        board,
        [(corner, 0.8), (center, 0.2)],
        time=PolicyTime.PRE_MOVE,
        support=PolicySupport.LEGAL_BOARD_CONDITIONAL,
    )
    target = regional_peak_target_mask(board, policy)
    assert target[3, 3]
    assert not target[9, 9]

    mask = regional_peak_contrast_mask(board, policy, anchor_region="corner_tl")
    assert mask[3, 3] > 0.0
    assert mask[9, 9] < 0.0
    active = legal_board_mask(board)
    assert np.isclose(mask[active].mean(), 0.0, atol=1e-6)
    assert np.isclose(np.sqrt(np.mean(mask[active] * mask[active])), 1.0, atol=1e-6)


def test_contract_registry_is_versioned_and_hashes_semantics():
    assert set(CONTRACTS) == {
        "tenuki_distance6@2",
        "reply_peak95@2",
        "regional_policy_peak@2",
    }
    assert get_contract("tenuki").definition_id == "tenuki_distance6@2"
    assert get_contract("forcing").definition_id == "reply_peak95@2"
    assert get_contract("urgency_peak").definition_id == "regional_policy_peak@2"

    hashes = {contract.contract_hash for contract in CONTRACTS.values()}
    assert len(hashes) == 3
    assert all(len(value) == 64 for value in hashes)
    assert get_contract("tenuki").metadata()["parameters"] == {
        "min_manhattan_distance": 6,
        "anchor": "most_recent_nonpass_move",
    }
