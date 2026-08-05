from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import joblib
import torch
import yaml

from daniele_experiment import validated_causal_eval as causal


class FakeBoard:
    PASS_LOC = 0

    def __init__(self, size: int = 3):
        self.size = size
        self.pla = 1
        self._occupied = set()

    def loc(self, x, y):
        return (int(y) + 1) * (self.size + 1) + int(x) + 1

    def loc_x(self, loc):
        return (int(loc) % (self.size + 1)) - 1

    def loc_y(self, loc):
        return (int(loc) // (self.size + 1)) - 1

    def is_on_board(self, loc):
        if int(loc) == self.PASS_LOC:
            return False
        return 0 <= self.loc_x(loc) < self.size and 0 <= self.loc_y(loc) < self.size

    def would_be_legal(self, _player, loc):
        return self.is_on_board(loc) and int(loc) not in self._occupied


def output(board, probabilities, pass_probability=0.0):
    pairs = [(board.loc(x, y), float(probabilities[y][x]))
             for y in range(board.size) for x in range(board.size)]
    pairs.append((board.PASS_LOC, float(pass_probability)))
    return {"moves_and_probs0": pairs}


class FakePolicyHead(torch.nn.Module):
    """Small deterministic head with the same call contract as KataGo."""

    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, x, mask, mask_sum_hw, mask_sum, extra_outputs):
        del mask_sum_hw, mask_sum, extra_outputs
        self.batch_sizes.append(int(x.shape[0]))
        spatial = (x[:, 0] + 2.0 * x[:, 1]) - (1.0 - mask[:, 0]) * 5000.0
        pass_logit = x.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        logits = torch.cat((spatial.flatten(1), pass_logit), dim=1)
        return logits.unsqueeze(1)


class FakeHeadModel(torch.nn.Module):
    def __init__(self, board_size=2):
        super().__init__()
        self.policy_head = FakePolicyHead()
        self.pos_len = int(board_size)
        self.device = torch.device("cpu")


class FakeFullState:
    def __init__(self, board, trunkfinal, *, policy_error=0.0, activation_error=0.0):
        self.board = board
        self._trunkfinal = np.asarray(trunkfinal, dtype=np.float32)
        self.policy_error = float(policy_error)
        self.activation_error = float(activation_error)

    def get_model_outputs(self, model, extra_output_names=None):
        del extra_output_names
        activation = torch.tensor(self._trunkfinal[None], dtype=torch.float32)
        mask = torch.ones((1, 1, self.board.size, self.board.size), dtype=torch.float32)
        with torch.no_grad():
            logits = model.policy_head(
                activation,
                mask=mask,
                mask_sum_hw=mask.sum(dim=(2, 3), keepdim=True),
                mask_sum=mask.sum(),
                extra_outputs=None,
            )
            policy = torch.softmax(logits[0, 0], dim=0).cpu().numpy()
        if self.policy_error:
            policy = policy.copy()
            policy[0] += self.policy_error
            policy /= policy.sum()
        return {
            "policy0": policy,
            "trunkfinal": self._trunkfinal + self.activation_error,
        }


def fake_replay_position(position_id, trunkfinal, *, policy_error=0.0, activation_error=0.0):
    board = FakeBoard(2)
    frozen = np.asarray(trunkfinal, dtype=np.float32).copy()
    frozen.setflags(write=False)
    state = FakeFullState(
        board,
        frozen,
        policy_error=policy_error,
        activation_error=activation_error,
    )
    return causal.ReplayPosition(
        position_id=position_id,
        game_id=position_id.split(":")[0],
        move_number=int(position_id.split(":")[1]),
        idx361=0,
        move_loc=board.loc(0, 0),
        previous_move=None,
        split_role="causal_test",
        label=1,
        game_state=state,
        baseline={},
        spatial_mask=None,
        anchor_region=None,
        candidate_reply_peaks=None,
        trunkfinal_path=Path(f"/fake/{position_id}.npy"),
        trunkfinal_sha256="a" * 64,
        trunkfinal=frozen,
        model_board_mask=np.ones((2, 2), dtype=np.float32),
    )


def test_idx361_conversion_is_not_padded_move_loc():
    # Internal location uses stride size+1 and a one-cell padding border.
    assert causal.flat_index_from_internal_loc(5, 3) == 0
    assert causal.flat_index_from_internal_loc(15, 3) == 8
    assert causal.flat_index_from_internal_loc(0, 3) is None
    with pytest.raises(causal.CausalValidationError):
        causal.flat_index_from_internal_loc(4, 3)


def test_policy_head_backend_batches_saved_activations_and_exact_intervention():
    model = FakeHeadModel(2)
    backend = causal.PolicyHeadOnlyBackend(
        model, channels=2, board_size=2, batch_size=2
    )
    positions = [
        fake_replay_position(
            f"g:{index}",
            np.arange(8, dtype=np.float32).reshape(2, 2, 2) * (index + 1) / 10.0,
        )
        for index in range(3)
    ]
    positions[0].board._occupied.add(positions[0].board.loc(1, 1))
    baseline = backend.baseline_outputs(positions)
    assert model.policy_head.batch_sizes == [2, 1]
    assert all(item["evaluation_backend"] == "saved_trunkfinal_policy_head" for item in baseline)
    assert all(np.asarray(item["policy0"]).sum() == pytest.approx(1.0) for item in baseline)
    illegal = positions[0].board.loc(1, 1)
    assert illegal not in dict(baseline[0]["moves_and_probs0"])
    assert 0 in dict(baseline[0]["moves_and_probs0"])  # pass

    direction = causal.InterventionDirection(
        "global",
        2,
        np.asarray([0.25, -0.5], dtype=np.float32),
        None,
        raw_norm=1.0,
    )
    steered = backend.evaluate(positions, direction=direction, dose=2.0)
    manual_activation = (
        np.asarray(positions[0].trunkfinal)
        + 2.0 * direction.global_delta[:, None, None]
    )
    mask = torch.ones((1, 1, 2, 2), dtype=torch.float32)
    with torch.no_grad():
        logits = model.policy_head(
            torch.as_tensor(manual_activation[None]),
            mask=mask,
            mask_sum_hw=mask.sum(dim=(2, 3), keepdim=True),
            mask_sum=mask.sum(),
            extra_outputs=None,
        )
        expected = torch.softmax(logits[0, 0], dim=0).numpy()
    np.testing.assert_allclose(steered[0]["policy0"], expected, rtol=0, atol=1e-7)


def test_policy_head_equivalence_audit_passes_and_fails_closed():
    trunk = np.arange(8, dtype=np.float32).reshape(2, 2, 2) / 10.0
    model = FakeHeadModel(2)
    backend = causal.PolicyHeadOnlyBackend(
        model,
        channels=2,
        board_size=2,
        batch_size=4,
        policy_atol=1e-7,
        activation_atol=1e-7,
    )
    positions = [fake_replay_position("g:1", trunk), fake_replay_position("h:2", trunk + 1)]
    report = backend.validate_equivalence(positions, seed=9, sample_size=2)
    assert report["status"] == "validated"
    assert report["full_network_forwards"] == 2
    assert backend.validated is True

    bad_policy_backend = causal.PolicyHeadOnlyBackend(
        FakeHeadModel(2), channels=2, board_size=2, policy_atol=1e-7
    )
    with pytest.raises(causal.CausalValidationError, match="policy-head equivalence failed"):
        bad_policy_backend.validate_equivalence(
            [fake_replay_position("bad:1", trunk, policy_error=1e-3)],
            seed=9,
            sample_size=1,
        )
    assert bad_policy_backend.validated is False

    bad_activation_backend = causal.PolicyHeadOnlyBackend(
        FakeHeadModel(2), channels=2, board_size=2, activation_atol=1e-7
    )
    with pytest.raises(causal.CausalValidationError, match="does not reproduce"):
        bad_activation_backend.validate_equivalence(
            [fake_replay_position("bad:2", trunk, activation_error=1e-3)],
            seed=9,
            sample_size=1,
        )


def test_saved_trunk_loader_hashes_exact_bytes_and_freezes_array(tmp_path):
    game_dir = tmp_path / "games" / "g" / "trunkfinal"
    game_dir.mkdir(parents=True)
    path = game_dir / "move_003.npy"
    array = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    np.save(path, array)
    run = SimpleNamespace(
        games_dir=tmp_path / "games",
        channels=2,
        board_size=2,
        build_manifest={
            "input_provenance": {
                "games": {
                    "g": {
                        "files": {
                            "move_003.npy": {
                                "identity": "g/trunkfinal/move_003.npy",
                                "bytes": path.stat().st_size,
                                "sha256": causal.sha256_file(path),
                            }
                        }
                    }
                }
            }
        },
    )
    observed_path, digest, loaded = causal._load_bound_trunkfinal(run, "g", 3)
    assert observed_path == path.resolve()
    assert digest == causal.sha256_file(path)
    np.testing.assert_array_equal(loaded, array)
    assert loaded.flags.writeable is False
    np.save(path, array + 1.0)
    with pytest.raises(causal.CausalValidationError, match="build-time leaf"):
        causal._load_bound_trunkfinal(run, "g", 3)


def test_incomplete_probe_run_name_is_rejected_before_use(tmp_path):
    with pytest.raises(causal.CausalValidationError, match="incomplete probe runs"):
        causal.load_validated_run(
            tmp_path / "validity_v5_incomplete", "forcing", "global"
        )


def test_split_overlap_rejected():
    causal.assert_disjoint_protocol(["train"], ["cal"], ["test"])
    with pytest.raises(causal.CausalValidationError, match="Split leakage"):
        causal.assert_disjoint_protocol(["same"], ["cal"], ["same"])


def test_probe_direction_respects_explicit_ablation_blocks():
    probe = SimpleNamespace(coef_=np.asarray([[2.0, 4.0, 6.0, 8.0]]))
    scaler = SimpleNamespace(scale_=np.asarray([2.0, 2.0, 2.0, 2.0]))
    direction = causal.InterventionDirection.from_probe_objects(
        probe, scaler, representation="combined", channels=2
    )
    raw = np.asarray([1.0, 2.0, 3.0, 4.0])
    expected = raw / (raw @ raw)
    np.testing.assert_allclose(direction.global_delta, expected[:2])
    np.testing.assert_allclose(direction.local_delta, expected[2:])
    assert direction.representation == "combined"
    metadata = direction.metadata()
    assert metadata["raw_space_gradient"] == "g = beta / sigma (elementwise)"
    assert "concatenated [global, local]" in metadata["combined_denominator"]
    assert "RMS over the active legal support" in metadata["mask_scaling"]

    with pytest.raises(causal.CausalValidationError, match="expected 2"):
        causal.InterventionDirection.from_probe_objects(
            probe, scaler, representation="global", channels=2
        )


def test_policy_disruption_renormalizes_legal_plus_pass():
    board = FakeBoard(2)
    baseline = output(board, [[2.0, 2.0], [0.0, 0.0]], pass_probability=0.0)
    same_distribution_different_scale = output(
        board, [[20.0, 20.0], [0.0, 0.0]], pass_probability=0.0
    )
    effect = causal.policy_disruption(baseline, same_distribution_different_scale, board)
    assert effect["policy_js"] == pytest.approx(0.0)
    assert effect["policy_l1"] == pytest.approx(0.0)


def test_forcing_readout_uses_candidate_post_reply_map():
    board = FakeBoard(2)
    locations = [board.loc(0, 0), board.loc(1, 0), board.loc(0, 1), board.loc(1, 1)]
    current = output(board, [[0.70, 0.20], [0.08, 0.02]])
    candidate_reply_peaks = {
        locations[0]: 0.10,
        locations[1]: 0.96,
        locations[2]: 0.99,
        locations[3]: 0.50,
    }
    readout = causal.concept_policy_readouts(
        "forcing",
        current,
        board,
        previous_move=None,
        candidate_reply_peaks=candidate_reply_peaks,
    )
    # The current policy itself is concentrated at .70, but the exact proxy is
    # mass on candidates whose *opponent reply* peak exceeds .95.
    assert readout["reply_peak95_action_mass"] == pytest.approx(0.28)
    assert readout["expected_reply_peak"] == pytest.approx(
        0.70 * 0.10 + 0.20 * 0.96 + 0.08 * 0.99 + 0.02 * 0.50
    )


def test_operational_alignment_audits_every_selected_urgency_value(tmp_path):
    position = fake_replay_position(
        "g:1", np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    )
    position.baseline = output(position.board, [[0.7, 0.1], [0.1, 0.1]])
    position.label = 1
    observed = causal.concept_policy_readouts(
        "urgency_peak",
        position.baseline,
        position.board,
        previous_move=None,
    )["regional_policy_peak"]
    labels_dir = tmp_path / "labels"
    (labels_dir / "g").mkdir(parents=True)
    (labels_dir / "g" / "snorkel.jsonl").write_text(
        '{"move_number":1,"analysis":{"regional_policy_peak":0.7}}\n'
    )
    dataset = pd.DataFrame([{
        "row_id": "g:1",
        "game_id": "g",
        "move_number": 1,
        "move_loc": position.move_loc,
        "game_phase": "early",
        "rawval_urgency_peak": observed,
        "label_urgency_peak": 1.0,
    }])
    run = SimpleNamespace(
        dataset=dataset,
        labels_dir=labels_dir,
        concept="urgency_peak",
        probe_metadata={
            "concept": {
                "name": "urgency_peak",
                "type": "quantile",
                "direction": "high",
                "no_drop": True,
                "filters": [],
                "use_abs": False,
            },
            "final_fit": {"quantile_thresholds": [0.2, 0.6]},
        },
    )
    report = causal.audit_operational_alignment(run, [position], value_atol=1e-8)
    assert report["status"] == "validated"
    assert report["positions_checked"] == 1

    run.dataset.loc[0, "rawval_urgency_peak"] = observed - 0.1
    failed = causal.audit_operational_alignment(run, [position], value_atol=1e-8)
    assert failed["status"] == "failed"
    assert failed["failed_positions"] == 1


def test_operational_alignment_reapplies_forcing_strict_threshold(tmp_path):
    position = fake_replay_position(
        "g:1", np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    )
    position.label = 1
    position.candidate_reply_peaks = {position.move_loc: 0.96}
    labels_dir = tmp_path / "labels"
    (labels_dir / "g").mkdir(parents=True)
    (labels_dir / "g" / "snorkel.jsonl").write_text(
        '{"move_number":1,"analysis":{"reply_peak95":true,"reply_peak_value":0.96}}\n'
    )
    run = SimpleNamespace(
        dataset=pd.DataFrame([{
            "row_id": "g:1",
            "game_id": "g",
            "move_number": 1,
            "rawval_forcing": True,
            "label_forcing": 1.0,
        }]),
        labels_dir=labels_dir,
        concept="forcing",
    )
    report = causal.audit_operational_alignment(run, [position], value_atol=1e-8)
    assert report["status"] == "validated"
    position.candidate_reply_peaks = {position.move_loc: 0.95}
    failed = causal.audit_operational_alignment(run, [position], value_atol=0.02)
    assert failed["status"] == "failed"
    assert failed["positions"][0]["threshold_label_agrees"] is False


def test_position_selection_anchors_to_immediately_previous_raw_row():
    frame = pd.DataFrame([
        {"row_id": "g:1", "game_id": "g", "move_number": 1, "move_loc": 5,
         "idx361": 0, "has_local": True, "split_role": "causal_test", "label_forcing": np.nan},
        {"row_id": "g:2", "game_id": "g", "move_number": 2, "move_loc": 6,
         "idx361": 1, "has_local": True, "split_role": "causal_test", "label_forcing": np.nan},
        {"row_id": "g:3", "game_id": "g", "move_number": 3, "move_loc": 9,
         "idx361": 3, "has_local": True, "split_role": "causal_test", "label_forcing": 1.0},
    ])
    run = SimpleNamespace(dataset=frame, concept="forcing", representation="local")
    selected = causal.select_positions(run, "causal_test", 1, seed=3)
    row3 = selected.loc[selected["move_number"].eq(3)].iloc[0]
    assert int(row3["previous_move_loc"]) == 6


def test_position_selection_anchors_to_most_recent_nonpass():
    frame = pd.DataFrame([
        {"row_id": "g:1", "game_id": "g", "move_number": 1, "move_loc": 5,
         "idx361": 0, "has_local": True, "split_role": "causal_test", "label_forcing": np.nan},
        {"row_id": "g:2", "game_id": "g", "move_number": 2, "move_loc": 0,
         "idx361": 9, "has_local": False, "split_role": "causal_test", "label_forcing": np.nan},
        {"row_id": "g:3", "game_id": "g", "move_number": 3, "move_loc": 9,
         "idx361": 3, "has_local": True, "split_role": "causal_test", "label_forcing": 1.0},
    ])
    run = SimpleNamespace(dataset=frame, concept="forcing", representation="local")
    selected = causal.select_positions(run, "causal_test", 1, seed=3)
    row3 = selected.loc[selected["move_number"].eq(3)].iloc[0]
    assert int(row3["previous_move_loc"]) == 5


def test_quantile_causal_labels_use_frozen_development_threshold():
    frame = pd.DataFrame([
        {"row_id": "g:1", "game_id": "g", "move_number": 1, "move_loc": 5,
         "idx361": 0, "has_local": True, "game_phase": "early",
         "split_role": "causal_test", "rawval_urgency_peak": 0.79},
        {"row_id": "h:1", "game_id": "h", "move_number": 1, "move_loc": 6,
         "idx361": 1, "has_local": True, "game_phase": "early",
         "split_role": "causal_test", "rawval_urgency_peak": 0.80},
    ])
    run = SimpleNamespace(
        dataset=frame,
        concept="urgency_peak",
        representation="local",
        probe_metadata={
            "concept": {
                "name": "urgency_peak",
                "type": "quantile",
                "direction": "high",
                "no_drop": True,
                "filters": [],
                "use_abs": False,
            },
            "final_fit": {"quantile_thresholds": [0.20, 0.80]},
        },
    )
    selected = causal.select_positions(run, "causal_test", 2, seed=3)
    assert sorted(selected["label_urgency_peak"].tolist()) == [0.0, 1.0]
    assert selected["game_id"].nunique() == 2
    assert set(selected["selection_unit"]) == {"one_position_per_game"}
    assert sorted(selected["selection_stratum"].tolist()) == [0, 1]


def test_position_selection_fails_instead_of_reusing_one_game():
    frame = pd.DataFrame([
        {"row_id": "g:1", "game_id": "g", "move_number": 1, "move_loc": 5,
         "idx361": 0, "has_local": True, "split_role": "causal_test",
         "label_forcing": 0.0},
        {"row_id": "g:2", "game_id": "g", "move_number": 2, "move_loc": 6,
         "idx361": 1, "has_local": True, "split_role": "causal_test",
         "label_forcing": 1.0},
    ])
    run = SimpleNamespace(dataset=frame, concept="forcing", representation="global")
    with pytest.raises(causal.CausalValidationError, match="independent"):
        causal.select_positions(run, "causal_test", 2, seed=3)


def test_causal_checkpoint_must_equal_fresh_generation_checkpoint():
    run = SimpleNamespace(fresh_holdout={"checkpoint_sha256": "a" * 64})
    causal._require_fresh_checkpoint(run, "a" * 64)
    with pytest.raises(causal.CausalValidationError, match="does not match"):
        causal._require_fresh_checkpoint(run, "b" * 64)


def test_candidate_cache_duplicate_actions_fail_closed():
    frame = pd.DataFrame([
        {"position_id": "g:1", "candidate_loc": 5, "reply_peak": 0.2},
        {"position_id": "g:1", "candidate_loc": 5, "reply_peak": 0.3},
    ])
    with pytest.raises(causal.CausalValidationError, match="duplicate"):
        causal._candidate_maps_from_frame(frame)


def test_control_families_enforce_counts_and_are_deterministic():
    learned = causal.InterventionDirection(
        "local", 2, None, np.asarray([0.3, -0.4], dtype=np.float32), 2.0
    )
    with pytest.raises(causal.CausalValidationError, match="100 random"):
        causal.control_specs(learned, seed=1, spatial_shuffles=50, random_directions=99)
    controls_a = causal.control_specs(
        learned, seed=1, spatial_shuffles=50, random_directions=100
    )
    controls_b = causal.control_specs(
        learned, seed=1, spatial_shuffles=50, random_directions=100
    )
    assert len(controls_a) == 150
    random_a = next(item for item in controls_a if item.kind == "random_direction")
    random_b = next(item for item in controls_b if item.control_id == random_a.control_id)
    np.testing.assert_array_equal(random_a.direction.local_delta, random_b.direction.local_delta)

    combined = causal.InterventionDirection(
        "combined",
        2,
        np.asarray([3.0, 4.0], dtype=np.float32),
        np.asarray([0.0, 12.0], dtype=np.float32),
        1.0,
    )
    combined_random = combined.random_control("random_000", seed=5)
    assert np.linalg.norm(combined_random.global_delta) == pytest.approx(5.0)
    assert np.linalg.norm(combined_random.local_delta) == pytest.approx(12.0)
    assert np.linalg.norm(combined_random.flattened()) == pytest.approx(13.0)


def test_calibration_matches_downstream_js_not_behavior(monkeypatch):
    learned = causal.InterventionDirection(
        "global", 1, np.asarray([1.0], dtype=np.float32), None, 1.0, "trained"
    )
    control_direction = causal.InterventionDirection(
        "global", 1, np.asarray([0.5], dtype=np.float32), None, 1.0, "random_000"
    )
    control = causal.ControlSpec("random_000", "random_direction", control_direction)

    def fake_evaluate(_model, _positions, _concept, direction, dose, **_kwargs):
        strength = float(direction.global_delta[0])
        # Target at dose 2 is JS=4.  Half-strength control therefore needs a
        # multiplier of 2.  No concept-behavior metric enters this callback.
        return [{"policy_js": (strength * float(dose)) ** 2, "policy_l1": abs(strength * dose)}]

    monkeypatch.setattr(causal, "_evaluate_intervention", fake_evaluate)
    matches, targets = causal.calibrate_controls(
        object(), "tenuki", [object()], learned, [control], [2.0], seed=4
    )
    assert targets[2.0]["mean_policy_js"] == pytest.approx(4.0)
    assert matches[("random_000", 2.0)]["matched"] is True
    assert matches[("random_000", 2.0)]["dose_multiplier"] == pytest.approx(2.0, rel=0.03)


def test_causal_summary_distinguishes_calibration_success_from_test_residual():
    common = {
        "position_id": "g:1",
        "game_id": "g",
        "nominal_dose": 2.0,
        "dose_multiplier": 1.0,
        "effective_dose": 2.0,
        "policy_l1": 0.2,
        "top_move_flip": 0.0,
        "calibration_match_succeeded": True,
        "calibration_target_mean_policy_js": 0.10,
        "calibration_achieved_mean_policy_js": 0.10,
    }
    rows = [
        {**common, "control_id": "trained", "control_kind": "trained", "policy_js": 0.20},
        {
            **common,
            "control_id": "random_000",
            "control_kind": "random_direction",
            "policy_js": 0.14,
            "calibration_achieved_mean_policy_js": 0.099,
        },
    ]
    summary = causal.summarize_causal_rows(rows)
    random = next(item for item in summary if item["control_id"] == "random_000")
    assert random["calibration_match_succeeded"] is True
    assert random["causal_test_observed_mean_policy_js"] == pytest.approx(0.14)
    assert random["causal_test_js_residual_from_calibration_target"] == pytest.approx(0.04)
    assert random["causal_test_js_residual_vs_trained"] == pytest.approx(-0.06)
    assert "policy_disruption_matched" not in random


def test_forward_estimate_exposes_cost_bounds():
    estimate = causal.estimate_protocol_forwards(
        calibration_positions=10,
        causal_test_positions=20,
        representation="local",
        doses=[-2, 0, 2],
        spatial_shuffles=50,
        random_directions=100,
        head_batch_size=8,
        equivalence_sample_size=6,
    )
    assert estimate["total_controls"] == 150
    assert estimate["nonzero_doses"] == 2
    assert estimate["fixed_components"]["baseline_forwards"] == 30
    assert estimate["evaluation_backend"] == "saved_trunkfinal_policy_head"
    assert estimate["full_network_forward_evaluations"]["equivalence_audit"] == 6
    assert estimate["full_network_forward_evaluations"][
        "causal_interventions_and_controls"
    ] == 0
    assert (
        estimate["estimated_total_forwards"]["lower_bound"]
        < estimate["estimated_total_forwards"]["planning_assumption"]
        < estimate["estimated_total_forwards"]["upper_bound"]
    )
    assert (
        estimate["estimated_policy_head_batch_forwards"]["planning_assumption"]
        < estimate["estimated_policy_head_position_evaluations"]["planning_assumption"]
    )


def test_partial_output_is_explicitly_marked_failed(tmp_path):
    output_dir = tmp_path / "partial"
    _manifest, estimate_path = causal._start_running_output(
        output_dir,
        kind="causal_evaluation",
        provenance={"test": True},
        estimate={"estimated_total_forwards": 123},
    )
    assert estimate_path.is_file()
    causal._mark_running_output_failed(output_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"


def test_validated_run_loader_checks_run_scoped_label_hash(tmp_path):
    run_dir = tmp_path / "run"
    games_dir = tmp_path / "games"
    (run_dir / "frozen_config").mkdir(parents=True)
    (run_dir / "labels" / "games").mkdir(parents=True)
    games_dir.mkdir()
    concepts = {
        "concepts": {
            "forcing": {
                "type": "binary",
                "source": "reply_peak95",
                "contract_id": "reply_peak95@2",
                "feature_mode": "pre",
            }
        }
    }
    concepts_path = run_dir / "frozen_config" / "concepts.yaml"
    concepts_path.write_text(yaml.safe_dump(concepts))
    split = pd.DataFrame([
        {"game_id": "dev", "split_role": "development", "outer_fold": 0},
        {"game_id": "cal", "split_role": "control_calibration", "outer_fold": None},
        {"game_id": "test", "split_role": "causal_test", "outer_fold": None},
    ])
    splits_path = run_dir / "splits.parquet"
    split.to_parquet(splits_path, index=False)
    checkpoint_hash = "f" * 64
    cohort = "fixture-fresh-holdout"
    game_seeds = {"cal": 101, "test": 202}
    seed_digest = hashlib.sha256(
        ",".join(map(str, sorted(game_seeds.values()))).encode("ascii")
    ).hexdigest()
    generator_path = tmp_path / "fixture_generate_games_dataset.py"
    common_utils_path = tmp_path / "fixture_common_utils.py"
    generator_path.write_text("# frozen fixture generator\n")
    common_utils_path.write_text("# frozen fixture common utils\n")
    generator_hash = causal.sha256_file(generator_path)
    common_utils_hash = causal.sha256_file(common_utils_path)
    frozen_causal_sources = causal._current_causal_source_hashes()
    validator_hash = causal.sha256_file(
        Path(causal.__file__).parent / "checkpoint_activation_fidelity.py"
    )
    protocol_path = tmp_path / "fresh_protocol.json"
    protocol = {
        "status": "frozen_before_fresh_data_generation",
        "fresh_holdout": {
            "cohort": cohort,
            "games": 2,
            "game_seed_set_sha256": seed_digest,
            "split_seed": 1,
        },
        "checkpoint": {"sha256": checkpoint_hash},
        "source_sha256": {
            "daniele_experiment/generate_games_dataset.py": generator_hash,
            "daniele_experiment/common_utils.py": common_utils_hash,
            "daniele_experiment/checkpoint_activation_fidelity.py": validator_hash,
            **frozen_causal_sources,
        },
        "development_activation_fidelity_gate": {
            "required_before_training": True,
            "expected_games": 1,
            "absolute_max_error_tolerance": 1e-5,
        },
    }
    protocol_path.write_text(json.dumps(protocol))
    protocol_hash = causal.sha256_file(protocol_path)
    source_hashes = causal._current_probe_source_hashes()
    manifest = {
        "schema_version": 1,
        "pipeline": "validated_probe_pipeline",
        "seed": 1,
        "source_games_dir": str(games_dir),
        "source_code_sha256": source_hashes,
        "contract_implementation_sha256": source_hashes[
            "daniele_experiment/operational_definitions.py"
        ],
        "fresh_holdout": {
            "cohort": cohort,
            "game_ids": ["cal", "test"],
            "games": 2,
            "protocol_manifest_sha256": protocol_hash,
            "checkpoint_sha256": checkpoint_hash,
            "generator_source_sha256": generator_hash,
            "common_utils_source_sha256": common_utils_hash,
            "protocol_source_sha256": {
                "daniele_experiment/generate_games_dataset.py": generator_hash,
                "daniele_experiment/common_utils.py": common_utils_hash,
                "daniele_experiment/checkpoint_activation_fidelity.py": validator_hash,
                **frozen_causal_sources,
            },
            "rng_seed_set_sha256": seed_digest,
            "protocol_path": str(protocol_path.resolve()),
            "protocol_split_seed": 1,
            "created_at_utc_min": "2026-01-01T00:00:00+00:00",
            "created_at_utc_max": "2026-01-01T00:00:01+00:00",
        },
        "artifacts": {
            "concepts_yaml_sha256": causal.sha256_file(concepts_path),
            "splits_sha256": causal.sha256_file(splits_path),
            "labels_games_dir": "labels/games",
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    input_records = {}
    for game in split["game_id"]:
        source_game = games_dir / game
        source_game.mkdir()
        moves_path = source_game / "moves.jsonl"
        moves_path.write_text(
            '{"move_number":1,"player":"b","move_loc":5,"idx361":0}\n'
        )
        trunk_dir = source_game / "trunkfinal"
        trunk_dir.mkdir()
        trunk_path = trunk_dir / "move_001.npy"
        np.save(trunk_path, np.zeros((2, 3, 3), dtype=np.float32))
        identity = f"{game}/trunkfinal/move_001.npy"
        payload = trunk_path.read_bytes()
        bytes_digest = hashlib.sha256()
        causal._update_length_prefixed(bytes_digest, identity.encode())
        bytes_digest.update(len(payload).to_bytes(8, "big"))
        bytes_digest.update(payload)
        stat = trunk_path.stat()
        stat_digest = hashlib.sha256()
        causal._update_length_prefixed(stat_digest, identity.encode())
        stat_digest.update(stat.st_size.to_bytes(8, "big"))
        stat_digest.update(stat.st_mtime_ns.to_bytes(8, "big"))
        meta_path = source_game / "meta.json"
        if game in game_seeds:
            meta_path.write_text(json.dumps({
                "cohort": cohort,
                "created_at_utc": (
                    "2026-01-01T00:00:00+00:00"
                    if game == "cal"
                    else "2026-01-01T00:00:01+00:00"
                ),
                "immutable_outputs": True,
                "protocol_manifest": {
                    "path": str(protocol_path.resolve()),
                    "sha256": protocol_hash,
                },
                "checkpoint": {
                    "sha256": checkpoint_hash,
                    "use_swa": False,
                    "selected_weights": "raw_model",
                },
                "generator": {
                    "source": str(generator_path.resolve()),
                    "source_sha256": generator_hash,
                    "common_utils_source": str(common_utils_path.resolve()),
                    "common_utils_source_sha256": common_utils_hash,
                },
                "rng": {"game_seed": game_seeds[game]},
            }))
        input_records[game] = {
            "source_game_dir": str(source_game.resolve()),
            "moves_path": str(moves_path.resolve()),
            "moves_sha256": causal.sha256_file(moves_path),
            "moves_bytes": moves_path.stat().st_size,
            "trunkfinal_dir": str(trunk_dir.resolve()),
            "file_count": 1,
            "total_bytes": len(payload),
            "identity_bytes_sha256": bytes_digest.hexdigest(),
            "identity_stat_sha256": stat_digest.hexdigest(),
            "files": {
                "move_001.npy": {
                    "identity": identity,
                    "bytes": len(payload),
                    "sha256": causal.sha256_file(trunk_path),
                }
            },
            "meta_path": str(meta_path.resolve()) if meta_path.is_file() else None,
            "meta_sha256": (
                causal.sha256_file(meta_path) if meta_path.is_file() else None
            ),
        }
        label_dir = run_dir / "labels" / "games" / game
        label_dir.mkdir()
        (label_dir / "snorkel.jsonl").write_text('{"move_number":1,"analysis":{}}\n')
    labels_manifest = {
        "pipeline": "validated_label_builder",
        "status": "complete",
        "run_manifest_sha256": causal.sha256_file(run_dir / "manifest.json"),
        "split_manifest_sha256": causal.sha256_file(splits_path),
        "concepts_yaml_sha256": causal.sha256_file(concepts_path),
    }
    labels_manifest_path = run_dir / "labels_manifest.json"
    labels_manifest_path.write_text(json.dumps(labels_manifest))
    dataset = pd.DataFrame([
        {"row_id": f"{game}:1", "game_id": game, "move_number": 1,
         "move_loc": 5, "idx361": 0, "has_local": True,
         "split_role": role, "label_forcing": 1.0}
        for game, role in zip(split["game_id"], split["split_role"])
    ])
    dataset_path = run_dir / "dataset.parquet"
    dataset.to_parquet(dataset_path, index=False)
    input_provenance = {
        "schema_version": 1,
        "source_games_dir": str(games_dir.resolve()),
        "trunk_hash_scheme": "test fixture",
        "trunk_identity_bytes_sha256": causal._overall_trunk_commitment(
            str(games_dir.resolve()), input_records
        ),
        "trunk_file_count": len(input_records),
        "trunk_total_bytes": sum(item["total_bytes"] for item in input_records.values()),
        "games": input_records,
    }
    build = {
        "schema_version": 1,
        "pipeline": "validated_probe_pipeline",
        "dataset": "dataset.parquet",
        "dataset_sha256": causal.sha256_file(dataset_path),
        "split_manifest_sha256": causal.sha256_file(splits_path),
        "concepts_yaml_sha256": causal.sha256_file(concepts_path),
        "labels_games_dir": "labels/games",
        "labels_sha256": causal._hash_run_labels(run_dir / "labels" / "games", split["game_id"]),
        "labels_manifest": "labels_manifest.json",
        "labels_manifest_sha256": causal.sha256_file(labels_manifest_path),
        "source_code_sha256": source_hashes,
        "contract_implementation_sha256": source_hashes[
            "daniele_experiment/operational_definitions.py"
        ],
        "board_size": 3,
        "trunk_channels": 2,
        "games": 3,
        "input_provenance": input_provenance,
        "input_provenance_sha256": hashlib.sha256(
            causal._canonical_bytes(input_provenance)
        ).hexdigest(),
    }
    (run_dir / "build_manifest.json").write_text(json.dumps(build))
    fidelity_report_path = run_dir / "checkpoint_activation_fidelity.json"
    fidelity_report = {
        "status": "passed",
        "validator": "checkpoint_activation_fidelity",
        "validator_source_sha256": validator_hash,
        "run": {
            "manifest_sha256": causal.sha256_file(run_dir / "manifest.json"),
            "build_manifest_sha256": causal.sha256_file(
                run_dir / "build_manifest.json"
            ),
        },
        "checkpoint": {"sha256": checkpoint_hash},
        "sampling": {
            "algorithm": "one_deterministic_position_per_game_v1",
            "split_role_filter": "development",
            "requested_sample_count": 1,
        },
        "aggregate_errors": {"sample_count": 1, "max_abs_error": 0.0},
        "tolerance": {"absolute_tolerance": 1e-5},
        "samples": [{"game_id": "dev", "move_number": 1}],
        "claim_scope": "fixture development activation fidelity",
    }
    fidelity_report_path.write_text(json.dumps(fidelity_report))
    fidelity_record = {
        "path": "checkpoint_activation_fidelity.json",
        "sha256": causal.sha256_file(fidelity_report_path),
        "checkpoint_sha256": checkpoint_hash,
        "sample_count": 1,
        "sampling_algorithm": "one_deterministic_position_per_game_v1",
        "absolute_tolerance": 1e-5,
        "observed_max_abs_error": 0.0,
        "claim_scope": "fixture development activation fidelity",
    }
    probe_dir = run_dir / "probes" / "global"
    probe_dir.mkdir(parents=True)
    probe_path = probe_dir / "probe_forcing.joblib"
    scaler_path = probe_dir / "scaler_forcing.joblib"
    metadata_path = probe_dir / "probe_forcing.meta.json"
    joblib.dump(SimpleNamespace(coef_=np.asarray([[1.0, 1.0]])), probe_path)
    joblib.dump(SimpleNamespace(scale_=np.asarray([1.0, 1.0])), scaler_path)
    metadata = {
        "representation": "global",
        "concept": {"name": "forcing", "source": "reply_peak95"},
        "contract_id": causal.get_contract("forcing").definition_id,
        "contract_hash": causal.get_contract("forcing").contract_hash,
        "feature_mode": "pre",
        "training_role": "development",
        "excluded_roles": ["control_calibration", "causal_test"],
        "dataset_sha256": build["dataset_sha256"],
        "split_manifest_sha256": build["split_manifest_sha256"],
        "concepts_yaml_sha256": build["concepts_yaml_sha256"],
        "labels_manifest_sha256": build["labels_manifest_sha256"],
        "input_provenance_sha256": build["input_provenance_sha256"],
        "trunk_identity_bytes_sha256": build["input_provenance"][
            "trunk_identity_bytes_sha256"
        ],
        "source_code_sha256": source_hashes,
        "checkpoint_activation_fidelity": fidelity_record,
        "contract_implementation_sha256": build[
            "contract_implementation_sha256"
        ],
        "final_fit": {"training_game_ids": ["dev"]},
        "n_features": 2,
    }
    metadata_path.write_text(json.dumps(metadata))
    artifacts = {
        str(path.relative_to(run_dir)): causal.sha256_file(path)
        for path in (probe_path, scaler_path, metadata_path)
    }
    training = {
        "schema_version": 1,
        "pipeline": "validated_probe_pipeline",
        "training_role": "development",
        "dataset_sha256": build["dataset_sha256"],
        "build_manifest_sha256": causal.sha256_file(run_dir / "build_manifest.json"),
        "labels_manifest_sha256": build["labels_manifest_sha256"],
        "split_manifest_sha256": build["split_manifest_sha256"],
        "input_provenance_sha256": build["input_provenance_sha256"],
        "trunk_identity_bytes_sha256": build["input_provenance"][
            "trunk_identity_bytes_sha256"
        ],
        "source_code_sha256": source_hashes,
        "checkpoint_activation_fidelity": fidelity_record,
        "contract_implementation_sha256": build[
            "contract_implementation_sha256"
        ],
        "concepts": ["forcing"],
        "representations": ["global"],
        "artifacts": artifacts,
    }
    (run_dir / "training_manifest.json").write_text(json.dumps(training))
    loaded = causal.load_validated_run(run_dir, "forcing", "global")
    assert loaded.concept == "forcing"
    (run_dir / "labels" / "games" / "test" / "snorkel.jsonl").write_text("tampered\n")
    with pytest.raises(causal.CausalValidationError, match="label hash"):
        causal.load_validated_run(run_dir, "forcing", "global")
