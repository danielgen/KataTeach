import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from daniele_experiment.causal_controls import artifact_record
from daniele_experiment.operational_definitions import get_contract
from daniele_experiment.validated_causal_results_report import (
    CausalResultsValidationError,
    _bootstrap_views,
    _empirical_control_comparison,
    current_causal_source_hashes,
    generate_causal_results_report,
    validate_causal_inputs,
    write_causal_results_report,
)
from daniele_experiment.validated_results_report import current_pipeline_source_hashes


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _causal_rows(test_positions, controls, doses):
    rows = []
    for dose in doses:
        for control_id, kind, control_index in controls:
            trained = kind == "trained"
            for position_index, position in enumerate(test_positions):
                baseline_mass = 0.30 + 0.01 * position_index
                if trained:
                    mass_delta = float(dose) * 0.05
                    policy_js = 0.02 if dose else 0.0
                    policy_l1 = 0.20 if dose else 0.0
                    match_status = "trained_reference"
                else:
                    mass_delta = float(dose) * (0.001 + control_index * 0.00001)
                    # Deliberately differs on the untouched test set despite an
                    # exact calibration target/achievement of 0.02 below.
                    policy_js = 0.03 if dose else 0.0
                    policy_l1 = 0.30 if dose else 0.0
                    match_status = "matched" if dose else "zero_dose_exact"
                effects = {
                    "tenuki_distance6_policy_mass": (baseline_mass, mass_delta),
                    "tenuki_distance6_complement_mass": (
                        1.0 - baseline_mass,
                        -mass_delta,
                    ),
                    "tenuki_expected_manhattan_distance": (
                        4.0 + 0.1 * position_index,
                        mass_delta * 4.0,
                    ),
                }
                row = {
                    **position,
                    "split_role": "causal_test",
                    "control_id": control_id,
                    "control_kind": kind,
                    "nominal_dose": float(dose),
                    "dose_multiplier": 1.0,
                    "effective_dose": float(dose),
                    "policy_js": policy_js,
                    "policy_l1": policy_l1,
                    "top_move_flip": float(policy_js > 0.025),
                    "calibration_match_succeeded": True,
                    "calibration_match_status": match_status,
                    "calibration_target_mean_policy_js": 0.02 if dose else 0.0,
                    "calibration_achieved_mean_policy_js": 0.02 if dose else 0.0,
                }
                for name, (baseline, delta) in effects.items():
                    row[f"baseline_{name}"] = baseline
                    row[f"steered_{name}"] = baseline + delta
                    row[f"delta_{name}"] = delta
                rows.append(row)
    return pd.DataFrame(rows)


def _make_causal_output(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    protocol_path = tmp_path / "validity_protocol.json"
    fresh_game_ids = [
        *(f"cal-{index}" for index in range(4)),
        *(f"test-{index}" for index in range(10)),
    ]
    seed_set_hash = "e" * 64
    protocol = {
        "schema_version": 1,
        "status": "frozen_before_fresh_data_generation",
        "fresh_holdout": {
            "cohort": "fresh-synthetic",
            "games": len(fresh_game_ids),
            "game_seed_set_sha256": seed_set_hash,
        },
        "checkpoint": {"sha256": _sha(checkpoint)},
        "development_activation_fidelity_gate": {
            "required_before_training": True,
            "expected_games": 2,
            "absolute_max_error_tolerance": 1e-4,
        },
        "causal": {
            "primary_hypothesis": {
                "concept": "tenuki",
                "representation": "local",
                "readout": "tenuki_distance6_policy_mass",
                "statistic": {
                    "name": "label_balanced_ols_slope_across_all_frozen_doses",
                    "predictor": "nominal_dose",
                    "outcome": "paired readout delta",
                    "intercept": True,
                    "dose_set": "all values in causal.doses",
                    "label_weights": {"0": 0.5, "1": 0.5},
                },
                "decision_rule": {
                    "alpha": 0.05,
                    "expected_trained_slope": "positive",
                    "random_direction_test": "one-sided finite-control empirical p <= alpha",
                    "spatial_shuffle_test": "one-sided finite-control empirical p <= alpha",
                    "headline_support_requires": (
                        "trained slope > 0 AND random-direction p <= alpha AND "
                        "spatial-shuffle p <= alpha"
                    ),
                },
            },
            "secondary_exploratory_concepts": ["forcing", "urgency_peak"],
            "secondary_representation": "local",
            "secondary_evaluation_rules": {
                "forcing": {
                    "scope": "best_effort_exploratory",
                    "confirmatory_gate": False,
                    "infeasibility_rule": (
                        "If mask-ineligible, omit analysis; do not replace/refill games."
                    ),
                },
                "urgency_peak": {
                    "scope": "exploratory",
                    "confirmatory_gate": False,
                },
            },
            "doses": [-1.0, 0.0, 1.0],
            "one_position_per_game": True,
            "maximum_calibration_positions": 4,
            "maximum_test_positions": 10,
            "spatial_shuffle_controls": 50,
            "random_direction_controls": 100,
            "causal_seed": 123,
            "policy_head_batch_size": 64,
            "full_vs_head_equivalence_sample_size": 6,
            "policy_equivalence_absolute_tolerance": 1e-6,
            "activation_equivalence_absolute_tolerance": 1e-5,
            "control_matching": (
                "control_calibration games: mean legal-plus-pass policy "
                "Jensen-Shannon divergence"
            ),
        },
        "inference": {
            "unit": "game",
            "bootstrap_replicates": 2_000,
            "confidence_level": 0.95,
        },
    }
    source_dir = Path(__file__).resolve().parent
    protocol["source_sha256"] = {
        **current_pipeline_source_hashes(),
        **current_causal_source_hashes(),
        "daniele_experiment/validated_results_report.py": _sha(
            source_dir / "validated_results_report.py"
        ),
        "daniele_experiment/validated_causal_results_report.py": _sha(
            source_dir / "validated_causal_results_report.py"
        ),
        "daniele_experiment/checkpoint_activation_fidelity.py": _sha(
            source_dir / "checkpoint_activation_fidelity.py"
        ),
        "daniele_experiment/build_validated_labels.py": _sha(
            source_dir / "build_validated_labels.py"
        ),
    }
    _write_json(protocol_path, protocol)
    probe_run = tmp_path / "validated-probe-run"
    probe_run.mkdir()
    (probe_run / "frozen_config").mkdir()
    (probe_run / "frozen_config" / "concepts.yaml").write_text(
        "concepts:\n"
        "  tenuki:\n"
        "    enabled: true\n"
        "    type: binary\n"
        "    source: tenuki_distance6\n"
        "    contract_id: tenuki_distance6@2\n"
        "    feature_mode: pre\n"
    )
    split_rows = [
        {"game_id": "dev-0", "split_role": "development", "outer_fold": 0},
        {"game_id": "dev-1", "split_role": "development", "outer_fold": 1},
    ]
    split_rows.extend(
        {"game_id": f"cal-{index}", "split_role": "control_calibration", "outer_fold": None}
        for index in range(4)
    )
    split_rows.extend(
        {"game_id": f"test-{index}", "split_role": "causal_test", "outer_fold": None}
        for index in range(10)
    )
    pd.DataFrame(split_rows).to_parquet(probe_run / "splits.parquet", index=False)
    probe_sources = current_pipeline_source_hashes()
    run_manifest = {
        "schema_version": 1,
        "pipeline": "validated_probe_pipeline",
        "status": "complete",
        "seed": 123,
        "source_code_sha256": probe_sources,
        "source_games_dir": str(tmp_path / "raw-games"),
        "fresh_holdout": {
            "cohort": "fresh-synthetic",
            "game_ids": fresh_game_ids,
            "games": len(fresh_game_ids),
            "checkpoint_sha256": _sha(checkpoint),
            "protocol_manifest_sha256": _sha(protocol_path),
            "generator_source_sha256": "f" * 64,
            "common_utils_source_sha256": "1" * 64,
            "protocol_source_sha256": protocol["source_sha256"],
            "rng_seed_set_sha256": seed_set_hash,
            "protocol_path": str(protocol_path),
        },
        "artifacts": {
            "concepts_yaml_sha256": _sha(
                probe_run / "frozen_config" / "concepts.yaml"
            ),
            "splits_sha256": _sha(probe_run / "splits.parquet"),
        },
    }
    _write_json(probe_run / "manifest.json", run_manifest)
    labels_manifest = {
        "schema_version": 1,
        "pipeline": "validated_label_builder",
        "status": "complete",
        "run_manifest_sha256": _sha(probe_run / "manifest.json"),
        "split_manifest_sha256": _sha(probe_run / "splits.parquet"),
        "concepts_yaml_sha256": _sha(
            probe_run / "frozen_config" / "concepts.yaml"
        ),
        "recomputed_fields": ["tenuki_distance6"],
        "migrated_legacy_fields": [],
        "builder_source_sha256": _sha(
            Path(__file__).resolve().parent / "build_validated_labels.py"
        ),
        "operational_definitions_source_sha256": _sha(
            Path(__file__).resolve().parent / "operational_definitions.py"
        ),
    }
    _write_json(probe_run / "labels_manifest.json", labels_manifest)
    (probe_run / "dataset.parquet").write_bytes(b"synthetic corrected dataset")
    input_hash = "a" * 64
    trunk_hash = "b" * 64
    build_manifest = {
        "schema_version": 1,
        "pipeline": "validated_probe_pipeline",
        "status": "complete",
        "dataset": "dataset.parquet",
        "dataset_sha256": _sha(probe_run / "dataset.parquet"),
        "labels_manifest": "labels_manifest.json",
        "labels_manifest_sha256": _sha(probe_run / "labels_manifest.json"),
        "labels_sha256": "c" * 64,
        "split_manifest_sha256": _sha(probe_run / "splits.parquet"),
        "concepts_yaml_sha256": _sha(
            probe_run / "frozen_config" / "concepts.yaml"
        ),
        "source_code_sha256": probe_sources,
        "input_provenance_sha256": input_hash,
        "input_provenance": {"trunk_identity_bytes_sha256": trunk_hash},
        "concepts": ["tenuki"],
    }
    _write_json(probe_run / "build_manifest.json", build_manifest)

    fidelity_report = {
        "validator": "checkpoint_activation_fidelity",
        "validator_source_sha256": protocol["source_sha256"][
            "daniele_experiment/checkpoint_activation_fidelity.py"
        ],
        "status": "passed",
        "run": {
            "manifest_sha256": _sha(probe_run / "manifest.json"),
            "build_manifest_sha256": _sha(probe_run / "build_manifest.json"),
        },
        "checkpoint": {"sha256": _sha(checkpoint)},
        "sampling": {
            "algorithm": "one_deterministic_position_per_game_v1",
            "split_role_filter": "development",
            "requested_sample_count": 2,
        },
        "tolerance": {"absolute_tolerance": 1e-4},
        "aggregate_errors": {"sample_count": 2, "max_abs_error": 1e-6},
        "samples": [{"game_id": "dev-0"}, {"game_id": "dev-1"}],
    }
    _write_json(
        probe_run / "checkpoint_activation_fidelity.json", fidelity_report
    )
    fidelity_gate = {
        "path": "checkpoint_activation_fidelity.json",
        "sha256": _sha(probe_run / "checkpoint_activation_fidelity.json"),
        "checkpoint_sha256": _sha(checkpoint),
        "sample_count": 2,
        "sampling_algorithm": "one_deterministic_position_per_game_v1",
        "absolute_tolerance": 1e-4,
        "observed_max_abs_error": 1e-6,
        "claim_scope": "synthetic compatibility gate",
    }

    probe_dir = probe_run / "probes" / "local"
    probe_dir.mkdir(parents=True)
    (probe_dir / "probe_tenuki.joblib").write_bytes(b"probe")
    (probe_dir / "scaler_tenuki.joblib").write_bytes(b"scaler")
    contract = get_contract("tenuki")
    probe_metadata = {
        "concept": {"name": "tenuki", "source": "tenuki_distance6"},
        "representation": "local",
        "training_role": "development",
        "excluded_roles": ["control_calibration", "causal_test"],
        "contract_id": contract.definition_id,
        "contract_hash": contract.contract_hash,
        "checkpoint_activation_fidelity": fidelity_gate,
    }
    _write_json(probe_dir / "probe_tenuki.meta.json", probe_metadata)
    training_artifacts = {
        str(path.relative_to(probe_run)): _sha(path)
        for path in (
            probe_dir / "probe_tenuki.joblib",
            probe_dir / "scaler_tenuki.joblib",
            probe_dir / "probe_tenuki.meta.json",
        )
    }
    training_manifest = {
        "schema_version": 1,
        "pipeline": "validated_probe_pipeline",
        "status": "complete",
        "training_role": "development",
        "concepts": ["tenuki"],
        "representations": ["global", "local", "combined"],
        "dataset_sha256": build_manifest["dataset_sha256"],
        "build_manifest_sha256": _sha(probe_run / "build_manifest.json"),
        "labels_manifest_sha256": _sha(probe_run / "labels_manifest.json"),
        "split_manifest_sha256": _sha(probe_run / "splits.parquet"),
        "source_code_sha256": probe_sources,
        "checkpoint_activation_fidelity": fidelity_gate,
        "artifacts": training_artifacts,
    }
    _write_json(probe_run / "training_manifest.json", training_manifest)

    causal_dir = probe_run / "causal" / "tenuki-local"
    causal_dir.mkdir(parents=True)
    calibration_positions = [
        {
            "position_id": f"cal-{game}:{move}",
            "row_id": f"cal-{game}:{move}",
            "game_id": f"cal-{game}",
            "move_number": move,
            "split_role": "control_calibration",
            "causal_protocol_role": "control_calibration",
            "label_tenuki": label,
            "selection_stratum": label,
            "selection_quota": 2,
            "selection_unit": "one_position_per_game",
        }
        for game in range(4)
        for move, label in ((1, game % 2),)
    ]
    test_positions = [
        {
            "position_id": f"test-{game}:{move}",
            "row_id": f"test-{game}:{move}",
            "game_id": f"test-{game}",
            "move_number": move,
            "split_role": "causal_test",
            "causal_protocol_role": "causal_test",
            "label_tenuki": label,
            "label": label,
            "selection_stratum": label,
            "selection_quota": 5,
            "selection_unit": "one_position_per_game",
        }
        for game in range(10)
        for move, label in ((1, game % 2),)
    ]
    pd.DataFrame(calibration_positions + test_positions).to_parquet(
        causal_dir / "selected_positions.parquet", index=False
    )
    pd.DataFrame([
        {
            "position_id": row["position_id"],
            "game_id": row["game_id"],
            "move_number": row["move_number"],
            "split_role": row["causal_protocol_role"],
            "trunkfinal_sha256": "d" * 64,
        }
        for row in calibration_positions + test_positions
    ]).to_parquet(causal_dir / "activation_bindings.parquet", index=False)

    controls = [("trained", "trained", 0)] + [
        (f"random_{index:03d}", "random_direction", index)
        for index in range(100)
    ] + [
        (f"shuffle_{index:03d}", "spatial_shuffle", index)
        for index in range(50)
    ]
    doses = (-1.0, 0.0, 1.0)
    row_frame = _causal_rows(test_positions, controls, doses)
    row_frame.to_parquet(causal_dir / "causal_test_rows.parquet", index=False)
    matches = []
    for control_id, kind, _index in controls[1:]:
        for dose in doses:
            matches.append({
                "control_id": control_id,
                "control_kind": kind,
                "target_mean_policy_js": 0.02 if dose else 0.0,
                "dose_multiplier": 1.0,
                "nominal_dose": dose,
                "effective_dose": dose,
                "achieved_mean_policy_js": 0.02 if dose else 0.0,
                "achieved_mean_policy_l1": 0.2 if dose else 0.0,
                "matched": True,
                "status": "matched" if dose else "zero_dose_exact",
                "iterations": 1 if dose else 0,
                "bracket_low": 0.0,
                "bracket_high": 1.0,
                "absolute_js_error": 0.0,
            })
    calibration = {
        "split_role": "control_calibration",
        "games": 4,
        "positions": 4,
        "trained_targets_by_nominal_dose": {
            str(dose): {
                "mean_policy_js": 0.02 if dose else 0.0,
                "mean_policy_l1": 0.2 if dose else 0.0,
            }
            for dose in doses
        },
        "matches": matches,
    }
    _write_json(causal_dir / "control_calibration.json", calibration)
    causal_sources = current_causal_source_hashes()
    summary = {
        "schema_version": 1,
        "pipeline": "validated_causal_eval",
        "concept": "tenuki",
        "representation": "local",
        "contract_hash": contract.contract_hash,
        "final_evaluation_role": "causal_test",
        "selection_unit": "exactly_one_position_per_fresh_holdout_game",
        "calibration_positions": 4,
        "causal_test_positions": 10,
        "doses": list(doses),
        "spatial_controls_applicable": True,
        "spatial_shuffle_repeats": 50,
        "random_direction_repeats": 100,
        "calibration_selection_strata": {"0": 2, "1": 2},
        "causal_test_selection_strata": {"0": 5, "1": 5},
        "source_code_sha256": causal_sources,
        "producer_source_sha256": causal_sources,
        "controls_policy_matched_on": protocol["causal"]["control_matching"],
        "evaluation_backend": {
            "batch_size": 64,
            "equivalence_sample_size": 6,
        },
        "fresh_holdout": {
            key: run_manifest["fresh_holdout"][key]
            for key in (
                "cohort", "game_ids", "checkpoint_sha256",
                "protocol_manifest_sha256", "generator_source_sha256",
                "common_utils_source_sha256", "protocol_source_sha256",
                "rng_seed_set_sha256",
            )
        },
        "checkpoint_activation_fidelity": fidelity_gate,
    }
    _write_json(causal_dir / "summary.json", summary)
    _write_json(causal_dir / "policy_head_equivalence.json", {"status": "validated"})
    _write_json(causal_dir / "operational_alignment.json", {
        "status": "validated",
        "concept": "tenuki",
        "contract_id": contract.definition_id,
        "contract_hash": contract.contract_hash,
        "positions_checked": len(calibration_positions) + len(test_positions),
        "games_checked": len(calibration_positions) + len(test_positions),
        "failed_positions": 0,
        "errors": [],
        "positions": [
            {"position_id": row["position_id"], "status": "validated"}
            for row in calibration_positions + test_positions
        ],
    })
    _write_json(causal_dir / "protocol_estimate.json", {"status": "estimate"})

    provenance = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha(checkpoint),
        "probe_run": str(probe_run),
        "probe_run_manifest_sha256": _sha(probe_run / "manifest.json"),
        "build_manifest_sha256": _sha(probe_run / "build_manifest.json"),
        "training_manifest_sha256": _sha(probe_run / "training_manifest.json"),
        "dataset_sha256": build_manifest["dataset_sha256"],
        "input_provenance_sha256": input_hash,
        "trunk_identity_bytes_sha256": trunk_hash,
        "labels_sha256": build_manifest["labels_sha256"],
        "splits_sha256": _sha(probe_run / "splits.parquet"),
        "probe_sha256": training_artifacts["probes/local/probe_tenuki.joblib"],
        "scaler_sha256": training_artifacts["probes/local/scaler_tenuki.joblib"],
        "probe_metadata_sha256": training_artifacts[
            "probes/local/probe_tenuki.meta.json"
        ],
        "concept": "tenuki",
        "representation": "local",
        "contract_id": contract.definition_id,
        "contract_hash": contract.contract_hash,
        "seed": 123,
        "policy_head_batch_size": 64,
        "equivalence_sample_size": 6,
        "policy_equivalence_atol": 1e-6,
        "activation_equivalence_atol": 1e-5,
        "fresh_holdout": {
            key: run_manifest["fresh_holdout"][key]
            for key in (
                "cohort", "game_ids", "checkpoint_sha256",
                "protocol_manifest_sha256", "generator_source_sha256",
                "common_utils_source_sha256", "protocol_source_sha256",
                "rng_seed_set_sha256",
            )
        },
        "checkpoint_activation_fidelity": fidelity_gate,
        "source_code_sha256": causal_sources,
        "producer_source_sha256": causal_sources,
    }
    produced = [
        causal_dir / name for name in (
            "protocol_estimate.json",
            "selected_positions.parquet",
            "activation_bindings.parquet",
            "policy_head_equivalence.json",
            "operational_alignment.json",
            "control_calibration.json",
            "causal_test_rows.parquet",
            "summary.json",
        )
    ]
    manifest = {
        "schema_version": 1,
        "run_id": causal_dir.name,
        "status": "validated",
        "pipeline": "validated_causal_eval",
        "kind": "causal_evaluation",
        "provenance": provenance,
        "operational_alignment": {
            "status": "validated",
            "report_sha256": _sha(causal_dir / "operational_alignment.json"),
            "positions_checked": len(calibration_positions) + len(test_positions),
            "failed_positions": 0,
        },
        "artifacts": [
            artifact_record(path, run_dir=causal_dir) for path in produced
        ],
    }
    _write_json(causal_dir / "manifest.json", manifest)
    return causal_dir


def test_game_cluster_bootstrap_is_deterministic_and_moves_same_game_together():
    frame = pd.DataFrame({
        "position_id": ["a:1", "b:1", "c:1", "d:1"],
        "game_id": ["a", "b", "c", "d"],
        "label": [0, 1, 0, 1],
        "baseline_x": [1.0, 1.0, 1.0, 1.0],
        "steered_x": [1.0, 1.0, 3.0, 3.0],
        "delta_x": [0.0, 0.0, 2.0, 2.0],
    })
    repeated = pd.concat([
        frame,
        frame.assign(position_id=["a:2", "b:2", "c:2", "d:2"]),
    ], ignore_index=True)
    duplicated = _bootstrap_views(
        repeated,
        readout="x",
        replicates=200,
        seed=7,
        confidence_level=0.95,
        expected_sign=1,
    )
    collapsed = _bootstrap_views(
        frame,
        readout="x",
        replicates=200,
        seed=7,
        confidence_level=0.95,
        expected_sign=1,
    )
    again = _bootstrap_views(
        repeated,
        readout="x",
        replicates=200,
        seed=7,
        confidence_level=0.95,
        expected_sign=1,
    )
    assert duplicated == again
    assert (
        duplicated["sample_position_weighted"]["game_cluster_bootstrap"]
        ["paired_absolute_delta"]
        == collapsed["sample_position_weighted"]["game_cluster_bootstrap"]
        ["paired_absolute_delta"]
    )


def test_empirical_p_has_finite_floor_and_negative_dose_reverses_sign():
    result = _empirical_control_comparison(
        -0.5,
        [-0.1, 0.0, 0.1],
        expected_sign=-1,
        family="random_direction",
    )
    assert result["expected_raw_sign"] == "decrease"
    assert result["n_controls_at_least_as_extreme"] == 0
    assert result["one_sided_empirical_p"] == pytest.approx(1 / 4)
    assert result["minimum_attainable_p"] == pytest.approx(1 / 4)


def test_full_report_is_deterministic_directional_and_exposes_test_js_residual(tmp_path):
    causal_dir = _make_causal_output(tmp_path)
    first = generate_causal_results_report(
        causal_dir, bootstrap_replicates=50, bootstrap_seed=55
    )
    second = generate_causal_results_report(
        causal_dir, bootstrap_replicates=50, bootstrap_seed=55
    )
    assert first["trained_direction_by_dose"] == second["trained_direction_by_dose"]
    negative = first["trained_direction_by_dose"]["-1.0"]["readouts"]
    mass = negative["tenuki_distance6_policy_mass"]
    assert mass["directionality"]["predicted_raw_change"] == "decrease"
    assert mass["label_balanced"]["paired_absolute_delta"] == pytest.approx(-0.05)
    comparison = first["empirical_control_comparisons"]["-1.0"][
        "tenuki_distance6_policy_mass"
    ]["random_direction"]
    assert comparison["one_sided_empirical_p"] == pytest.approx(1 / 101)
    assert comparison["conservative_empirical_percentile"] == 100.0
    diagnostics = first["control_calibration_and_test_disruption"]
    assert diagnostics["causal_test_matching_claim"] == "none"
    record = next(
        item for item in diagnostics["by_control_and_dose"]
        if item["nominal_dose"] == 1.0
    )
    assert record["calibration"]["success"] is True
    assert record["causal_test_observed_disruption"][
        "js_residual_control_minus_trained"
    ] == pytest.approx(0.01)
    assert record["causal_test_observed_disruption"]["matching_claim"] == "none_on_causal_test"
    primary = first["sole_primary_confirmatory_test"]
    assert primary["status"] == "diagnostic_protocol_deviation"
    assert primary["statistic"]["slope"] == pytest.approx(0.05)
    assert primary["headline_decision"] is None
    assert primary["empirical_nulls"]["random_direction"][
        "one_sided_empirical_p"
    ] == pytest.approx(1 / 101)
    assert primary["empirical_nulls"]["spatial_shuffle"][
        "one_sided_empirical_p"
    ] == pytest.approx(1 / 51)
    dose_scope = mass["analysis_scope"]["scope"]
    assert dose_scope == "diagnostic_protocol_deviation"


def test_artifact_hash_tamper_fails_closed(tmp_path):
    causal_dir = _make_causal_output(tmp_path)
    rows_path = causal_dir / "causal_test_rows.parquet"
    rows = pd.read_parquet(rows_path)
    rows.loc[0, "policy_js"] = 0.5
    rows.to_parquet(rows_path, index=False)
    with pytest.raises(CausalResultsValidationError, match="manifest validation failed"):
        validate_causal_inputs(causal_dir)


def test_failed_operational_alignment_is_rejected_even_if_rehashed(tmp_path):
    causal_dir = _make_causal_output(tmp_path)
    alignment_path = causal_dir / "operational_alignment.json"
    alignment = json.loads(alignment_path.read_text())
    alignment["status"] = "failed"
    alignment["failed_positions"] = 1
    alignment["positions"][0]["status"] = "failed"
    _write_json(alignment_path, alignment)
    manifest_path = causal_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for record in manifest["artifacts"]:
        if record["path"] == "operational_alignment.json":
            record.update({
                "sha256": _sha(alignment_path),
                "size_bytes": alignment_path.stat().st_size,
            })
    manifest["operational_alignment"].update({
        "status": "failed",
        "report_sha256": _sha(alignment_path),
        "failed_positions": 1,
    })
    _write_json(manifest_path, manifest)
    with pytest.raises(CausalResultsValidationError, match="Operational-alignment"):
        validate_causal_inputs(causal_dir)


def test_primary_decision_requires_both_control_families(tmp_path):
    causal_dir = _make_causal_output(tmp_path)
    rows_path = causal_dir / "causal_test_rows.parquet"
    rows = pd.read_parquet(rows_path)
    spatial = rows["control_kind"].astype(str).eq("spatial_shuffle")
    dose = rows.loc[spatial, "nominal_dose"].to_numpy(float)
    mass_delta = 0.08 * dose
    rows.loc[spatial, "delta_tenuki_distance6_policy_mass"] = mass_delta
    rows.loc[spatial, "steered_tenuki_distance6_policy_mass"] = (
        rows.loc[spatial, "baseline_tenuki_distance6_policy_mass"].to_numpy(float)
        + mass_delta
    )
    rows.loc[spatial, "delta_tenuki_distance6_complement_mass"] = -mass_delta
    rows.loc[spatial, "steered_tenuki_distance6_complement_mass"] = (
        rows.loc[spatial, "baseline_tenuki_distance6_complement_mass"].to_numpy(float)
        - mass_delta
    )
    rows.loc[spatial, "delta_tenuki_expected_manhattan_distance"] = 4.0 * mass_delta
    rows.loc[spatial, "steered_tenuki_expected_manhattan_distance"] = (
        rows.loc[spatial, "baseline_tenuki_expected_manhattan_distance"].to_numpy(float)
        + 4.0 * mass_delta
    )
    rows.to_parquet(rows_path, index=False)
    manifest_path = causal_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for record in manifest["artifacts"]:
        if record["path"] == "causal_test_rows.parquet":
            record.update({
                "sha256": _sha(rows_path),
                "size_bytes": rows_path.stat().st_size,
            })
    _write_json(manifest_path, manifest)
    report = generate_causal_results_report(
        causal_dir, bootstrap_replicates=2_000, bootstrap_seed=31
    )
    primary = report["sole_primary_confirmatory_test"]
    assert primary["status"] == "confirmatory_complete"
    assert primary["decision_rule"]["criteria"][
        "random_direction_one_sided_empirical_p_at_most_alpha"
    ] is True
    assert primary["decision_rule"]["criteria"][
        "spatial_shuffle_one_sided_empirical_p_at_most_alpha"
    ] is False
    assert primary["headline_decision"] == (
        "does_not_pass_predeclared_headline_support_criterion"
    )


def test_fidelity_record_tamper_fails_closed(tmp_path):
    causal_dir = _make_causal_output(tmp_path)
    fidelity = causal_dir.parents[1] / "checkpoint_activation_fidelity.json"
    report = json.loads(fidelity.read_text())
    report["aggregate_errors"]["max_abs_error"] = 0.5
    _write_json(fidelity, report)
    with pytest.raises(CausalResultsValidationError, match="fidelity|Hash mismatch"):
        validate_causal_inputs(causal_dir)


def test_write_is_append_only_and_hashes_report(tmp_path):
    causal_dir = _make_causal_output(tmp_path)
    report, manifest = write_causal_results_report(
        causal_dir, bootstrap_replicates=2_000, bootstrap_seed=9
    )
    declared = json.loads(manifest.read_text())
    assert declared["report_sha256"] == _sha(report)
    assert declared["upstream_causal_manifest_sha256"] == _sha(
        causal_dir / "manifest.json"
    )
    saved = json.loads(report.read_text())
    primary = saved["sole_primary_confirmatory_test"]
    assert primary["status"] == "confirmatory_complete"
    assert primary["headline_decision"] == (
        "passes_predeclared_headline_support_criterion"
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        write_causal_results_report(
            causal_dir, bootstrap_replicates=2_000, bootstrap_seed=9
        )
