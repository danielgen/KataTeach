import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from daniele_experiment.validated_results_report import (
    ResultsValidationError,
    _fresh_frozen_inference,
    _metric_values,
    _weighted_metric_matrix,
    current_pipeline_source_hashes,
    generate_results_report,
    validate_results_inputs,
    write_results_report,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def _make_validated_run(run: Path) -> Path:
    run.mkdir(parents=True)
    (run / "frozen_config").mkdir()
    (run / "frozen_config" / "concepts.yaml").write_text(
        "concepts:\n  signal:\n    enabled: true\n    type: binary\n    source: signal\n"
    )
    games = run.parent / "raw-games"
    games.mkdir(exist_ok=True)
    split_rows = []
    for fold in range(2):
        for game_index in range(4):
            split_rows.append({
                "game_id": f"dev-{fold}-{game_index}",
                "split_role": "development",
                "outer_fold": fold,
            })
    split_rows.extend([
        {"game_id": "calibration-0", "split_role": "control_calibration", "outer_fold": None},
        {"game_id": "causal-0", "split_role": "causal_test", "outer_fold": None},
    ])
    pd.DataFrame(split_rows).to_parquet(run / "splits.parquet", index=False)

    sources = current_pipeline_source_hashes()
    run_manifest = {
        "schema_version": 1,
        "pipeline": "validated_probe_pipeline",
        "seed": 1234,
        "source_games_dir": str(games),
        "source_code_sha256": sources,
        "artifacts": {
            "concepts_yaml_sha256": _sha(run / "frozen_config" / "concepts.yaml"),
            "splits_sha256": _sha(run / "splits.parquet"),
        },
    }
    _write_json(run / "manifest.json", run_manifest)
    labels_manifest = {
        "schema_version": 1,
        "pipeline": "validated_label_builder",
        "status": "complete",
        "run_manifest_sha256": _sha(run / "manifest.json"),
        "split_manifest_sha256": _sha(run / "splits.parquet"),
        "concepts_yaml_sha256": _sha(run / "frozen_config" / "concepts.yaml"),
        "recomputed_fields": [],
        "migrated_legacy_fields": ["signal"],
        "builder_source_sha256": _sha(
            Path(__file__).resolve().parent.parent / "build_validated_labels.py"
        ),
        "operational_definitions_source_sha256": _sha(
            Path(__file__).resolve().parent.parent / "operational_definitions.py"
        ),
    }
    _write_json(run / "labels_manifest.json", labels_manifest)
    (run / "dataset.parquet").write_bytes(b"synthetic rebuilt dataset")
    build_manifest = {
        "schema_version": 1,
        "pipeline": "validated_probe_pipeline",
        "dataset": "dataset.parquet",
        "dataset_sha256": _sha(run / "dataset.parquet"),
        "split_manifest_sha256": _sha(run / "splits.parquet"),
        "concepts_yaml_sha256": _sha(run / "frozen_config" / "concepts.yaml"),
        "labels_manifest_sha256": _sha(run / "labels_manifest.json"),
        "source_code_sha256": sources,
        "concepts": ["signal"],
    }
    _write_json(run / "build_manifest.json", build_manifest)

    predictions_dir = run / "outer_predictions"
    predictions_dir.mkdir()
    probe_results = {}
    nested_rows = []
    training_artifact_paths = []
    for representation in ("global", "local", "combined"):
        rows = []
        for fold in range(2):
            for game_index in range(4):
                game_id = f"dev-{fold}-{game_index}"
                for move_number, label in enumerate((0, 1, 0, 1), start=1):
                    rows.append({
                        "concept": "signal",
                        "representation": representation,
                        "row_id": f"{game_id}:{move_number}",
                        "game_id": game_id,
                        "move_number": move_number,
                        "outer_fold": fold,
                        "label": label,
                        "probability": 0.9 if label else 0.1,
                        "prediction": label,
                        "f1_threshold": 0.5,
                    })
            nested_rows.append({
                "concept": "signal",
                "representation": representation,
                "outer_fold": fold,
                "n_train": 16,
                "n_test": 16,
                "positive_train": 8,
                "positive_test": 8,
                "best_C": 0.1,
                "f1_threshold": 0.5,
                "roc_auc": 1.0,
                "average_precision": 1.0,
                "f1": 1.0,
                "balanced_accuracy": 1.0,
                "converged": True,
            })
        prediction_path = predictions_dir / f"signal__{representation}.parquet"
        pd.DataFrame(rows).to_parquet(prediction_path, index=False)
        training_artifact_paths.append(prediction_path)

        probe_dir = run / "probes" / representation
        probe_dir.mkdir(parents=True)
        for filename, payload in (
            ("probe_signal.joblib", b"probe"),
            ("scaler_signal.joblib", b"scaler"),
            ("probe_signal.meta.json", b"{}"),
        ):
            path = probe_dir / filename
            path.write_bytes(payload)
            training_artifact_paths.append(path)
        probe_results[representation] = {"outer_metrics": {"mean_roc_auc": 1.0}}

    nested_path = run / "nested_cv_results.parquet"
    pd.DataFrame(nested_rows).to_parquet(nested_path, index=False)
    results_path = run / "probe_results.json"
    _write_json(results_path, {"signal": probe_results})
    training_artifact_paths.extend((nested_path, results_path))
    training_manifest = {
        "schema_version": 1,
        "pipeline": "validated_probe_pipeline",
        "concepts": ["signal"],
        "representations": ["global", "local", "combined"],
        "outer_folds": 2,
        "inner_folds": 2,
        "training_role": "development",
        "dataset_sha256": _sha(run / "dataset.parquet"),
        "build_manifest_sha256": _sha(run / "build_manifest.json"),
        "labels_manifest_sha256": _sha(run / "labels_manifest.json"),
        "split_manifest_sha256": _sha(run / "splits.parquet"),
        "source_code_sha256": sources,
        "artifacts": {
            str(path.relative_to(run)): _sha(path) for path in training_artifact_paths
        },
    }
    _write_json(run / "training_manifest.json", training_manifest)
    return run


def test_vectorised_weighted_metrics_match_reference_with_probability_ties():
    labels = np.asarray([0, 1, 0, 1, 1, 0])
    probabilities = np.asarray([0.2, 0.8, 0.2, 0.6, 0.8, 0.4])
    predictions = (probabilities >= 0.5).astype(int)
    weights = np.asarray([
        [1, 1, 1, 1, 1, 1],
        [2, 0, 1, 3, 1, 2],
        [0, 4, 2, 1, 3, 1],
    ], dtype=float)
    vectorised = _weighted_metric_matrix(
        labels, probabilities, predictions, weights
    )
    for row, sample_weight in enumerate(weights):
        reference = _metric_values(
            labels,
            probabilities,
            predictions,
            sample_weight=sample_weight,
        )
        assert reference is not None
        for metric, expected in reference.items():
            assert vectorised[metric][row] == pytest.approx(expected)


def test_reports_fold_sd_clustered_intervals_counts_and_paired_ablations(tmp_path):
    run = _make_validated_run(tmp_path / "validated-run")
    first = generate_results_report(
        run, bootstrap_replicates=50, bootstrap_seed=77
    )
    second = generate_results_report(
        run, bootstrap_replicates=50, bootstrap_seed=77
    )
    assert first["status"] == "complete"
    assert first["bootstrap"]["unit"] == "game"
    assert first["bootstrap"]["refits_models"] is False
    assert "fixed nested-CV out-of-fold predictions" in first["bootstrap"]["conditioning"]
    assert first["concepts"] == second["concepts"]

    signal = first["concepts"]["signal"]
    assert signal["label_validation"]["tier"] == "whitelisted_migrated_exploratory"
    assert signal["label_validation"]["presentation_scope"] == "exploratory_only"
    for representation in ("global", "local", "combined"):
        result = signal["representations"][representation]
        assert result["counts"] == {
            "n_positions": 32,
            "n_games": 8,
            "n_positives": 16,
            "n_negatives": 16,
            "n_games_with_positive": 8,
        }
        assert result["fold_metrics"]["roc_auc"]["mean"] == 1.0
        assert result["fold_metrics"]["roc_auc"]["sd"] == 0.0
        interval = result["game_cluster_bootstrap"]["roc_auc"]
        assert interval["status"] == "ok"
        assert interval["lower"] == interval["upper"] == 1.0

    ablation = signal["ablations"]["combined_minus_local"]
    assert ablation["paired_outer_fold_deltas"]["average_precision"]["mean_delta"] == 0.0
    interval = ablation["paired_game_cluster_bootstrap"]["average_precision"]
    assert interval["status"] == "ok"
    assert interval["lower"] == interval["upper"] == 0.0


def test_fails_closed_when_a_hashed_prediction_is_modified(tmp_path):
    run = _make_validated_run(tmp_path / "validated-run")
    path = run / "outer_predictions" / "signal__local.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "probability"] = 0.2
    frame.to_parquet(path, index=False)
    with pytest.raises(ResultsValidationError, match="Hash mismatch"):
        validate_results_inputs(run)


def test_fails_closed_when_ablation_rows_differ_even_with_updated_hash(tmp_path):
    run = _make_validated_run(tmp_path / "validated-run")
    relative = "outer_predictions/signal__local.parquet"
    path = run / relative
    frame = pd.read_parquet(path)
    frame.loc[0, "row_id"] = "different-row"
    frame.to_parquet(path, index=False)
    training_path = run / "training_manifest.json"
    training = json.loads(training_path.read_text())
    training["artifacts"][relative] = _sha(path)
    _write_json(training_path, training)
    with pytest.raises(ResultsValidationError, match="same rows/labels/folds"):
        validate_results_inputs(run)


def test_fails_closed_when_an_enabled_concept_was_not_built(tmp_path):
    run = _make_validated_run(tmp_path / "validated-run")
    build_path = run / "build_manifest.json"
    build = json.loads(build_path.read_text())
    build["concepts"] = []
    _write_json(build_path, build)
    training_path = run / "training_manifest.json"
    training = json.loads(training_path.read_text())
    training["build_manifest_sha256"] = _sha(build_path)
    _write_json(training_path, training)
    with pytest.raises(ResultsValidationError, match="Build manifest has missing"):
        validate_results_inputs(run)


def test_rejects_archive_and_incomplete_run_paths(tmp_path):
    archived = _make_validated_run(tmp_path / "archive" / "old-run")
    with pytest.raises(ResultsValidationError, match="archive"):
        validate_results_inputs(archived)
    incomplete = _make_validated_run(tmp_path / "new-run.incomplete")
    with pytest.raises(ResultsValidationError, match="incomplete"):
        validate_results_inputs(incomplete)

    underscored_incomplete = _make_validated_run(tmp_path / "new-run_incomplete")
    with pytest.raises(ResultsValidationError, match="incomplete"):
        validate_results_inputs(underscored_incomplete)


def test_write_is_append_only_and_hashes_the_report(tmp_path):
    run = _make_validated_run(tmp_path / "validated-run")
    report_path, manifest_path = write_results_report(
        run, bootstrap_replicates=10, bootstrap_seed=9
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "complete"
    assert manifest["report_sha256"] == _sha(report_path)
    with pytest.raises(FileExistsError, match="overwrite"):
        write_results_report(run, bootstrap_replicates=10, bootstrap_seed=9)


def test_fresh_frozen_inference_requires_exact_confirmatory_settings(tmp_path):
    protocol = tmp_path / "protocol.json"
    _write_json(protocol, {
        "inference": {"bootstrap_replicates": 2000, "confidence_level": 0.95}
    })
    manifest = {"fresh_holdout": {"protocol_path": str(protocol)}}
    assert _fresh_frozen_inference(manifest) == {
        "bootstrap_replicates": 2000,
        "confidence_level": 0.95,
    }

    _write_json(protocol, {
        "inference": {"bootstrap_replicates": 1999, "confidence_level": 0.95}
    })
    with pytest.raises(ResultsValidationError, match="exactly 2000"):
        _fresh_frozen_inference(manifest)
