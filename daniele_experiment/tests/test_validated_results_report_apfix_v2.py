import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from daniele_experiment.tests.test_validated_results_report import (
    _make_validated_run,
    _write_json,
)
from daniele_experiment.validated_results_report import (
    _metric_values,
    write_results_report,
)
from daniele_experiment.validated_results_report_apfix_v2 import (
    CORRECTION_RELATIVE_DIR,
    MANIFEST_FILENAME,
    PIPELINE_NAME,
    REPORT_FILENAME,
    generate_corrected_results_report,
    weighted_metric_matrix_apfix_v2,
    write_corrected_results_report,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _introduce_omittable_leading_score_group(run: Path) -> None:
    prediction_paths = sorted((run / "outer_predictions").glob("*.parquet"))
    for path in prediction_paths:
        frame = pd.read_parquet(path)
        for fold in sorted(frame["outer_fold"].unique()):
            fold_mask = frame["outer_fold"].eq(fold)
            leading_game = sorted(frame.loc[fold_mask, "game_id"].unique())[0]
            positive = fold_mask & frame["label"].eq(1)
            frame.loc[positive, "probability"] = 0.8
            frame.loc[positive & frame["game_id"].eq(leading_game), "probability"] = 0.95
        frame.to_parquet(path, index=False)

    manifest_path = run / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for path in prediction_paths:
        manifest["artifacts"][str(path.relative_to(run))] = _sha(path)
    _write_json(manifest_path, manifest)


def test_apfix_handles_zero_weight_leading_probability_group():
    labels = np.asarray([1, 1, 0])
    probabilities = np.asarray([0.9, 0.8, 0.1])
    predictions = (probabilities >= 0.5).astype(int)
    weights = np.asarray([[0.0, 1.0, 1.0]])

    corrected = weighted_metric_matrix_apfix_v2(
        labels, probabilities, predictions, weights
    )
    reference = _metric_values(
        labels,
        probabilities,
        predictions,
        sample_weight=weights[0],
    )

    assert reference is not None
    assert reference["average_precision"] == pytest.approx(1.0)
    assert corrected["average_precision"][0] == pytest.approx(
        reference["average_precision"]
    )
    for metric in ("roc_auc", "f1", "balanced_accuracy"):
        assert corrected[metric][0] == pytest.approx(reference[metric])


def test_writes_append_only_versioned_correction_without_touching_original(tmp_path):
    run = _make_validated_run(tmp_path / "validated-run")
    _introduce_omittable_leading_score_group(run)
    original_report_path, original_manifest_path = write_results_report(
        run,
        bootstrap_replicates=50,
        bootstrap_seed=77,
    )
    original_report_bytes = original_report_path.read_bytes()
    original_manifest_bytes = original_manifest_path.read_bytes()
    original = json.loads(original_report_bytes)
    assert all(
        result["game_cluster_bootstrap"]["average_precision"]["status"]
        == "not_estimable"
        for result in original["concepts"]["signal"]["representations"].values()
    )

    generated = generate_corrected_results_report(run)
    assert generated["pipeline"] == PIPELINE_NAME
    assert generated["status"] == "complete_post_freeze_reporting_correction"
    assert generated["correction"]["post_freeze_correction"] is True
    assert generated["correction"]["part_of_original_prospective_protocol"] is False
    assert generated["correction"]["non_ap_fields_reproduced_exactly"] is True
    assert generated["correction"]["reused_bootstrap_settings"] == {
        "replicates": 50,
        "confidence_level": 0.95,
        "base_seed": 77,
    }
    assert all(
        result["game_cluster_bootstrap"]["average_precision"]["status"] == "ok"
        and result["game_cluster_bootstrap"]["average_precision"][
            "replicates_valid"
        ]
        == 50
        for result in generated["concepts"]["signal"]["representations"].values()
    )

    corrected_report_path, corrected_manifest_path = write_corrected_results_report(run)
    assert corrected_report_path == run / CORRECTION_RELATIVE_DIR / REPORT_FILENAME
    assert corrected_manifest_path == run / CORRECTION_RELATIVE_DIR / MANIFEST_FILENAME
    corrected = json.loads(corrected_report_path.read_bytes())
    correction_manifest = json.loads(corrected_manifest_path.read_bytes())
    assert correction_manifest["report_sha256"] == _sha(corrected_report_path)
    assert correction_manifest["original_frozen_report_sha256"] == _sha(
        original_report_path
    )
    assert corrected["correction"]["original_report_sha256"] == _sha(
        original_report_path
    )
    assert corrected["correction"]["original_report_manifest_sha256"] == _sha(
        original_manifest_path
    )
    assert original_report_path.read_bytes() == original_report_bytes
    assert original_manifest_path.read_bytes() == original_manifest_bytes

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_corrected_results_report(run)
