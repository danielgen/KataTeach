import json

import numpy as np
import pytest

from daniele_experiment.causal_controls import (
    MIN_CONTROL_REPEATS,
    ManifestValidationError,
    artifact_record,
    control_ids,
    match_mean_policy_js,
    new_run_manifest,
    position_shuffle_seed,
    random_direction_control_ids,
    sha256_file,
    shuffled_position_mask,
    stable_sha256_seed,
    summarize_policy_disruption,
    update_run_status,
    validate_run_manifest,
    write_run_manifest,
)


def test_stable_seeds_are_typed_and_position_specific():
    seed = stable_sha256_seed("spatial-mask-shuffle", 7, "shuffle_003", "g:12")
    assert seed == stable_sha256_seed("spatial-mask-shuffle", 7, "shuffle_003", "g:12")
    assert seed != stable_sha256_seed("spatial-mask-shuffle", "7", "shuffle_003", "g:12")
    assert position_shuffle_seed(7, "shuffle_003", "g:12") != position_shuffle_seed(
        7, "shuffle_003", "g:13"
    )


def test_repeated_shuffle_is_reproducible_and_preserves_values():
    mask = np.arange(25, dtype=np.float32).reshape(5, 5)
    first = shuffled_position_mask(
        mask, base_seed=123, repeat_id="shuffle_000", position_id="game-a:4"
    )
    repeated = shuffled_position_mask(
        mask, base_seed=123, repeat_id="shuffle_000", position_id="game-a:4"
    )
    other = shuffled_position_mask(
        mask, base_seed=123, repeat_id="shuffle_001", position_id="game-a:4"
    )
    np.testing.assert_array_equal(first, repeated)
    np.testing.assert_array_equal(np.sort(first.ravel()), np.sort(mask.ravel()))
    assert not np.array_equal(first, other)


def test_shuffle_preserves_inactive_support_exactly():
    mask = np.asarray([
        [0.0, 2.0, 0.0, -1.0],
        [3.0, 0.0, -4.0, 0.0],
        [0.0, 5.0, -2.0, 0.0],
    ], dtype=np.float32)
    shuffled = shuffled_position_mask(
        mask, base_seed=7, repeat_id="shuffle_011", position_id="game:9"
    )
    np.testing.assert_array_equal(shuffled != 0, mask != 0)
    np.testing.assert_array_equal(
        np.sort(shuffled[mask != 0]), np.sort(mask[mask != 0])
    )
    np.testing.assert_array_equal(shuffled[mask == 0], np.zeros(np.sum(mask == 0)))


def test_control_ids_enforce_an_empirical_control_set_of_at_least_fifty():
    ids = random_direction_control_ids()
    assert len(ids) == 100
    assert len(set(ids)) == 100
    assert control_ids("shuffle", MIN_CONTROL_REPEATS)[-1] == "shuffle_049"
    with pytest.raises(ValueError, match="At least 50"):
        control_ids("random", 49)


def test_aggregate_policy_js_matching_reports_effective_signed_dose():
    calls = []

    def calibrate(multiplier):
        calls.append(multiplier)
        return {
            "mean_policy_js": (multiplier / 2.0) ** 2,
            "mean_policy_l1": multiplier / 3.0,
        }

    result = match_mean_policy_js(
        calibrate,
        0.09,
        nominal_dose=-5.0,
        relative_tolerance=1e-5,
        absolute_tolerance=1e-8,
    )
    assert result.matched
    assert result.status == "matched"
    assert result.dose_multiplier == pytest.approx(0.6, rel=1e-4)
    assert result.effective_dose == pytest.approx(-3.0, rel=1e-4)
    assert result.achieved_mean_policy_js == pytest.approx(0.09, rel=1e-5)
    assert result.achieved_mean_policy_l1 == pytest.approx(0.2, rel=1e-4)
    # A scalar is evaluated once for the aggregate callback, rather than once
    # independently for every position.
    assert len(calls) == len(set(calls))


def test_policy_js_matching_marks_an_unbracketed_control_invalid():
    result = match_mean_policy_js(
        lambda multiplier: {"mean_policy_js": min(multiplier, 1.0) * 0.01},
        0.1,
        nominal_dose=5,
        maximum_multiplier=2,
    )
    assert not result.matched
    assert result.status == "target_not_bracketed"
    assert result.effective_dose == 10

    with pytest.raises(ValueError, match="zero nominal dose"):
        match_mean_policy_js(lambda _multiplier: 0.1, 0.1, nominal_dose=0)


def test_policy_disruption_summary_includes_js_l1_and_effective_dose():
    summary = summarize_policy_disruption(
        [
            {"policy_js": 0.01, "policy_l1": 0.2, "top_move_flip": 0},
            {"policy_js": 0.03, "policy_l1": 0.4, "top_move_flip": 1},
        ],
        control_id="random_000",
        nominal_dose=-5,
        dose_multiplier=1.5,
        target_mean_policy_js=0.021,
        matched=True,
    )
    assert summary["mean_policy_js"] == pytest.approx(0.02)
    assert summary["mean_policy_l1"] == pytest.approx(0.3)
    assert summary["effective_dose"] == -7.5
    assert summary["top_move_flip_rate"] == 0.5
    assert summary["policy_disruption_matched"] is True


def test_manifest_hash_validation_and_status_transition(tmp_path):
    run_dir = tmp_path / "run-a.incomplete"
    artifact = run_dir / "causal" / "tenuki.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"ok": true}\n', encoding="utf-8")
    record = artifact_record(artifact, run_dir=run_dir)
    manifest = new_run_manifest("run-a", artifacts=[record])
    manifest_path = write_run_manifest(run_dir / "manifest.json", manifest)

    loaded = validate_run_manifest(
        manifest_path, allowed_statuses=("running",), verify_artifacts=True
    )
    assert loaded["run_id"] == "run-a"
    updated = update_run_status(manifest_path, "validated", expected_current_status="running")
    assert updated["status"] == "validated"
    assert validate_run_manifest(run_dir)["status"] == "validated"

    artifact.write_text('{"ok": false}\n', encoding="utf-8")
    with pytest.raises(ManifestValidationError, match="mismatch"):
        validate_run_manifest(run_dir)


def test_manifest_rejects_invalid_status_and_path_escape(tmp_path):
    run_dir = tmp_path / "run-b"
    run_dir.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(ManifestValidationError, match="outside"):
        artifact_record(outside, run_dir=run_dir)

    manifest = new_run_manifest("run-b", status="invalid_do_not_use")
    manifest["artifacts"] = [{
        "path": "../outside.json",
        "size_bytes": outside.stat().st_size,
        "sha256": sha256_file(outside),
    }]
    write_run_manifest(run_dir / "manifest.json", manifest)
    with pytest.raises(ManifestValidationError, match="status"):
        validate_run_manifest(run_dir)
    with pytest.raises(ManifestValidationError, match="escapes"):
        validate_run_manifest(
            run_dir,
            allowed_statuses=("invalid_do_not_use",),
            verify_artifacts=True,
        )


def test_manifest_json_is_plain_and_inspectable(tmp_path):
    path = tmp_path / "manifest.json"
    write_run_manifest(path, new_run_manifest("inspectable"))
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed["schema_version"] == 1
    assert parsed["status"] == "running"
