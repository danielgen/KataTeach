import json
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import pytest

from daniele_experiment.activation_manipulation import (
    ConceptIntervention,
    dose_response,
    list_intervenable_concepts,
    policy_effect,
)
from daniele_experiment.position_causal_eval import (
    _eligible_corner_regions,
    _region_for_loc,
    concept_local_mask,
    concept_metrics,
    spatial_intervention_mask,
)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.act_trunkfinal = torch.nn.Identity()


def _artifacts(tmp_path: Path, feature_mode: str = "pre", concept: str = "shape"):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(100, 3)) * np.array([2.0, 1.0, 0.5])
    y = (x[:, 0] - x[:, 1] > 0).astype(int)
    scaler = StandardScaler().fit(x)
    probe = LogisticRegression().fit(scaler.transform(x), y)
    joblib.dump(probe, tmp_path / f"probe_{concept}.joblib")
    joblib.dump(scaler, tmp_path / f"scaler_{concept}.joblib")
    config = {
        "concepts": {
            concept: {
                "type": "binary",
                "source": concept,
                "feature_mode": feature_mode,
            }
        },
        "feature_extraction": {
            "aggregation": "global_pool",
            "pool_type": "mean",
            "include_move_location": False,
            "trunk_channels": 3,
        },
    }
    yaml_path = tmp_path / "concepts.yaml"
    yaml_path.write_text(yaml.safe_dump(config))
    return probe, scaler, yaml_path


def test_hook_changes_probe_score_by_requested_dose(tmp_path):
    _probe, _scaler, yaml_path = _artifacts(tmp_path)
    intervention = ConceptIntervention.load("shape", tmp_path, yaml_path)
    model = TinyModel()
    activation = torch.randn(1, 3, 5, 5)
    before = activation.mean((2, 3)).numpy()[0] @ intervention.probe_raw_direction
    with intervention.apply(model, 1.75):
        after_tensor = model.act_trunkfinal(activation)
    after = after_tensor.mean((2, 3)).numpy()[0] @ intervention.probe_raw_direction
    assert np.isclose(after - before, 1.75, atol=1e-5)
    assert len(model.act_trunkfinal._forward_hooks) == 0


def test_rejects_post_feature_mode(tmp_path):
    _artifacts(tmp_path, feature_mode="post", concept="cut")
    with pytest.raises(ValueError, match="feature_mode='post'"):
        ConceptIntervention.load("cut", tmp_path, tmp_path / "concepts.yaml")


def test_list_intervenable_concepts_filters_to_pre(tmp_path):
    _artifacts(tmp_path, feature_mode="pre", concept="forcing")
    joblib.dump(joblib.load(tmp_path / "probe_forcing.joblib"), tmp_path / "probe_cut.joblib")
    joblib.dump(joblib.load(tmp_path / "scaler_forcing.joblib"), tmp_path / "scaler_cut.joblib")
    config = yaml.safe_load((tmp_path / "concepts.yaml").read_text())
    config["concepts"]["cut"] = {
        "type": "binary",
        "source": "cut",
        "feature_mode": "post",
    }
    (tmp_path / "concepts.yaml").write_text(yaml.safe_dump(config))
    assert list_intervenable_concepts(tmp_path, tmp_path / "concepts.yaml") == ["forcing"]


class FakeBoard:
    size = 2
    PASS_LOC = -1

    @staticmethod
    def loc_x(loc): return loc % 2

    @staticmethod
    def loc_y(loc): return loc // 2


def test_policy_effect_detects_policy_and_value_change():
    baseline = {"moves_and_probs0": [(0, .8), (1, .2)], "value": np.array([.4]), "scoremean": 1.0}
    changed = {"moves_and_probs0": [(0, .1), (1, .9)], "value": np.array([.6]), "scoremean": 2.5}
    result = policy_effect(baseline, changed, FakeBoard())
    assert result["policy_js"] > 0
    assert result["top_move_changed"] == 1
    assert np.isclose(result["winrate_delta"], .2)


def test_dose_response_reports_slope():
    rows = [
        {"dose": -1, "area_score_black_minus_white": -2, "winrate_delta": -.1, "scoremean_delta": -.5},
        {"dose": 0, "area_score_black_minus_white": 0, "winrate_delta": 0, "scoremean_delta": 0},
        {"dose": 1, "area_score_black_minus_white": 2, "winrate_delta": .1, "scoremean_delta": .5},
    ]
    summary = dose_response(rows)
    assert np.isclose(summary["area_score_black_minus_white_slope"], 2.0)


def test_local_hook_only_changes_masked_locations():
    intervention = ConceptIntervention(
        "shape", np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float32), np.ones(3, dtype=np.float32),
    )
    model = TinyModel()
    activation = torch.zeros(1, 3, 2, 2)
    mask = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    with intervention.apply(model, 2.0, component="local", spatial_mask=mask):
        changed = model.act_trunkfinal(activation)
    assert changed[0, 0, 0, 1] == 2
    assert changed[0, 0, 1, 0] == -2
    assert torch.count_nonzero(changed[:, 1:]) == 0


def test_direction_override_replaces_trained_local_direction():
    intervention = ConceptIntervention(
        "shape", np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float32), np.ones(3, dtype=np.float32),
    )
    model = TinyModel()
    override = np.array([0.0, 2.0, 0.0], dtype=np.float32)
    with intervention.apply(
        model, 1.0, component="local", spatial_mask=np.ones((2, 2)),
        direction_override=override,
    ):
        changed = model.act_trunkfinal(torch.zeros(1, 3, 2, 2))
    assert torch.count_nonzero(changed[:, 0]) == 0
    assert torch.all(changed[:, 1] == 2)


class SpatialBoard:
    size = 19
    PASS_LOC = 0

    @staticmethod
    def loc_x(loc): return (loc % 20) - 1

    @staticmethod
    def loc_y(loc): return (loc // 20) - 1


def test_contrast_mask_is_zero_mean_and_rms_normalized():
    # Internal location for (9,9) on a board with stride 20 and one-cell border.
    loc = (9 + 1) * 20 + (9 + 1)
    mask = spatial_intervention_mask(SpatialBoard(), loc, "local-contrast", 4, 6)
    assert np.isclose(mask.mean(), 0.0, atol=1e-6)
    assert np.isclose(np.sqrt(np.mean(mask ** 2)), 1.0, atol=1e-6)
    assert mask[9, 9] < 0
    assert mask[0, 0] > 0
    # (dx,dy)=(4,4) is distance 8 under Snorkel's Manhattan convention,
    # despite having Chebyshev distance 4.
    assert mask[13, 13] > 0


def test_corner_regions_and_eligibility_match_snorkel_boundaries():
    from board import Board

    board = Board(19)
    assert _region_for_loc(board, board.loc(5, 5)) == "corner_tl"
    assert _region_for_loc(board, board.loc(6, 5)) == "side_top"
    assert set(_eligible_corner_regions(board, "occupy_corner")) == {
        "corner_tl", "corner_tr", "corner_bl", "corner_br"
    }
    board.play(Board.BLACK, board.loc(3, 3))
    # White can approach the lone black stone; that corner is no longer empty.
    assert "corner_tl" in _eligible_corner_regions(board, "approaching_corner")
    assert "corner_tl" not in _eligible_corner_regions(board, "occupy_corner")


def test_forcing_metrics_and_mask_use_baseline_top_candidate():
    from board import Board

    board = Board(19)
    outputs = {
        "moves_and_probs0": [(board.loc(3, 3), .96), (board.loc(10, 10), .04)],
    }
    metrics = concept_metrics("forcing", outputs, board, {"move_loc": board.loc(3, 3)})
    assert metrics["forcing_threshold_crossed"] == 1
    assert np.isclose(metrics["forcing_top_margin"], .92)
    mask = concept_local_mask("forcing", outputs, board, {})
    assert np.isclose(mask.mean(), 0, atol=1e-6)
    assert mask[3, 3] > 0
    assert mask[10, 10] < 0
