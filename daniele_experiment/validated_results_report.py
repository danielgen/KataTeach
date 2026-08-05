#!/usr/bin/env python3
"""Produce uncertainty-aware reports from validated probe runs only.

The reporter is deliberately downstream-only: it never discovers legacy probe
directories and never accepts an archive as a run.  Before reading predictions
it verifies the complete run -> labels -> build -> training hash chain and every
artifact declared by the training manifest.

Confidence intervals use a game-cluster bootstrap conditional on the fixed
nested-CV out-of-fold predictions.  Games are resampled with replacement
*within each outer fold*, each fold metric is recomputed, and the fold metrics
are averaged.  Models are not refitted in bootstrap replicates.  The same draws
are used for global, local, and combined probes, making the reported ablation
intervals paired.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


SCHEMA_VERSION = 1
PIPELINE_NAME = "validated_results_report"
SOURCE_PIPELINE = "validated_probe_pipeline"
REPRESENTATIONS = ("global", "local", "combined")
METRICS = ("roc_auc", "average_precision", "f1", "balanced_accuracy")
DEFAULT_BOOTSTRAP_REPLICATES = 2_000
DEFAULT_CONFIDENCE_LEVEL = 0.95
PIPELINE_SOURCE_FILES = {
    "daniele_experiment/validated_probe_pipeline.py": "validated_probe_pipeline.py",
    "daniele_experiment/operational_definitions.py": "operational_definitions.py",
}


class ResultsValidationError(ValueError):
    """Raised when a purported validated run fails a provenance/content check."""


@dataclass(frozen=True)
class ValidatedResultsInputs:
    run_dir: Path
    run_manifest: Mapping[str, Any]
    build_manifest: Mapping[str, Any]
    labels_manifest: Mapping[str, Any]
    training_manifest: Mapping[str, Any]
    splits: pd.DataFrame
    nested_results: pd.DataFrame
    predictions: Mapping[Tuple[str, str], pd.DataFrame]
    frozen_concepts: Mapping[str, Mapping[str, Any]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(_json_safe(value), sort_keys=True, indent=2) + "\n").encode()


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _read_json(path: Path, description: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultsValidationError(f"Invalid {description}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ResultsValidationError(f"{description} must be a JSON object: {path}")
    return value


def _is_archive_component(component: str) -> bool:
    value = component.casefold()
    return value in {"archive", "archived"} or value.startswith("archive_")


def _reject_archive_path(path: Path, description: str) -> None:
    if any(_is_archive_component(part) for part in path.parts):
        raise ResultsValidationError(f"{description} points into an archive: {path}")


def _inside_run(run_dir: Path, relative: str, description: str) -> Path:
    candidate_relative = Path(str(relative))
    if candidate_relative.is_absolute() or ".." in candidate_relative.parts:
        raise ResultsValidationError(
            f"{description} must be a relative path inside the run: {relative!r}"
        )
    _reject_archive_path(candidate_relative, description)
    candidate = (run_dir / candidate_relative).resolve()
    try:
        candidate.relative_to(run_dir)
    except ValueError as exc:
        raise ResultsValidationError(f"{description} escapes the run: {relative!r}") from exc
    _reject_archive_path(candidate, description)
    return candidate


def _require_hash(path: Path, expected: Any, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ResultsValidationError(f"Missing/invalid declared hash for {description}")
    observed = _sha256(path)
    if observed != expected:
        raise ResultsValidationError(
            f"Hash mismatch for {description}: expected {expected}, observed {observed}"
        )


def current_pipeline_source_hashes() -> Dict[str, str]:
    """Return the exact source hash set frozen by validated probe runs."""
    source_dir = Path(__file__).resolve().parent
    result = {}
    for identity, filename in PIPELINE_SOURCE_FILES.items():
        path = source_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing validated pipeline source: {path}")
        result[identity] = _sha256(path)
    return result


def _label_validation_tier(
    concept: str,
    spec: Mapping[str, Any],
    labels_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    """Classify label provenance without conflating rebuilt and migrated fields."""

    source = str(spec.get("source", ""))
    if not source:
        raise ResultsValidationError(f"Concept {concept!r} has no frozen label source")
    recomputed = set(map(str, labels_manifest.get("recomputed_fields") or ()))
    migrated = set(map(str, labels_manifest.get("migrated_legacy_fields") or ()))
    overlap = recomputed & migrated
    if overlap:
        raise ResultsValidationError(
            "Labels manifest places fields in both validation tiers: "
            + ", ".join(sorted(overlap))
        )
    if source in recomputed:
        return {
            "tier": "canonical_recomputed",
            "source": source,
            "presentation_scope": "primary",
            "description": (
                "Recomputed from raw move records under the frozen operational "
                "definition; no archived central label value was reused."
            ),
        }
    if source in migrated:
        return {
            "tier": "whitelisted_migrated_exploratory",
            "source": source,
            "presentation_scope": "exploratory_only",
            "description": (
                "Whitelisted non-central field copied from the quarantined legacy "
                "label corpus after source-hash verification; it is not a freshly "
                "recomputed canonical label."
            ),
        }
    raise ResultsValidationError(
        f"Concept {concept!r} source {source!r} is absent from both declared "
        "label-validation tiers"
    )


def _require_complete_if_declared(document: Mapping[str, Any], description: str) -> None:
    status = document.get("status")
    if status is not None and str(status).casefold() not in {"complete", "validated"}:
        raise ResultsValidationError(
            f"{description} declares non-complete status {status!r}"
        )


def _validate_checkpoint_fidelity_gate(
    run_dir: Path,
    run_manifest: Mapping[str, Any],
    training_manifest: Mapping[str, Any],
) -> Optional[Mapping[str, Any]]:
    """Re-hash the development checkpoint-compatibility gate for fresh runs."""

    fresh = run_manifest.get("fresh_holdout")
    gate = training_manifest.get("checkpoint_activation_fidelity")
    if not isinstance(fresh, Mapping):
        if gate is not None:
            raise ResultsValidationError(
                "Non-fresh run unexpectedly claims a fresh checkpoint-fidelity gate"
            )
        return None
    if not isinstance(gate, Mapping):
        raise ResultsValidationError(
            "Fresh run training manifest lacks checkpoint_activation_fidelity"
        )
    protocol_value = fresh.get("protocol_path")
    if not isinstance(protocol_value, str) or not protocol_value:
        raise ResultsValidationError("Fresh run lacks frozen protocol path")
    protocol_path = Path(protocol_value).resolve()
    _reject_archive_path(protocol_path, "Frozen protocol")
    _require_hash(
        protocol_path,
        fresh.get("protocol_manifest_sha256"),
        "frozen protocol",
    )
    protocol = _read_json(protocol_path, "frozen protocol")
    protocol_gate = protocol.get("development_activation_fidelity_gate")
    protocol_sources = protocol.get("source_sha256")
    if (
        protocol.get("status") != "frozen_before_fresh_data_generation"
        or not isinstance(protocol_gate, Mapping)
        or protocol_gate.get("required_before_training") is not True
        or not isinstance(protocol_sources, Mapping)
    ):
        raise ResultsValidationError(
            "Frozen protocol lacks a mandatory fidelity gate/source commitment"
        )
    reporter_identity = "daniele_experiment/validated_results_report.py"
    if protocol_sources.get(reporter_identity) != _sha256(Path(__file__).resolve()):
        raise ResultsValidationError(
            "Current probe results reporter differs from prospectively frozen bytes"
        )
    repository = Path(__file__).resolve().parent.parent
    for identity, expected_hash in protocol_sources.items():
        relative = Path(str(identity))
        if relative.is_absolute() or ".." in relative.parts:
            raise ResultsValidationError(
                f"Frozen protocol has unsafe source identity {identity!r}"
            )
        source_path = (repository / relative).resolve()
        try:
            source_path.relative_to(repository)
        except ValueError as exc:
            raise ResultsValidationError(
                f"Frozen protocol source escapes repository: {identity!r}"
            ) from exc
        _require_hash(
            source_path, expected_hash, f"frozen protocol source {identity}"
        )
    path = _inside_run(run_dir, str(gate.get("path", "")), "Fidelity report")
    _require_hash(path, gate.get("sha256"), "checkpoint activation-fidelity report")
    report = _read_json(path, "checkpoint activation-fidelity report")
    if (
        report.get("status") != "passed"
        or report.get("validator") != "checkpoint_activation_fidelity"
        or (report.get("checkpoint") or {}).get("sha256")
        != fresh.get("checkpoint_sha256")
        or gate.get("checkpoint_sha256") != fresh.get("checkpoint_sha256")
    ):
        raise ResultsValidationError(
            "Checkpoint activation-fidelity gate is failed or checkpoint-mismatched"
        )
    if report.get("validator_source_sha256") != protocol_sources.get(
        "daniele_experiment/checkpoint_activation_fidelity.py"
    ):
        raise ResultsValidationError(
            "Checkpoint-fidelity validator differs from prospectively frozen bytes"
        )
    report_run = report.get("run") or {}
    if (
        report_run.get("manifest_sha256") != _sha256(run_dir / "manifest.json")
        or report_run.get("build_manifest_sha256")
        != _sha256(run_dir / "build_manifest.json")
    ):
        raise ResultsValidationError(
            "Checkpoint activation-fidelity report is bound to another run/build"
        )
    sampling = report.get("sampling") or {}
    aggregate = report.get("aggregate_errors") or {}
    if (
        sampling.get("algorithm") != gate.get("sampling_algorithm")
        or int(aggregate.get("sample_count", -1)) != int(gate.get("sample_count", -2))
        or float((report.get("tolerance") or {}).get("absolute_tolerance", math.inf))
        != float(gate.get("absolute_tolerance", math.nan))
    ):
        raise ResultsValidationError(
            "Checkpoint activation-fidelity report differs from its training binding"
        )
    frozen_tolerance = float(
        protocol_gate.get("absolute_max_error_tolerance", -1)
    )
    if float(gate.get("absolute_tolerance", math.inf)) > frozen_tolerance:
        raise ResultsValidationError(
            "Checkpoint activation-fidelity tolerance is looser than frozen"
        )
    splits = pd.read_parquet(run_dir / "splits.parquet")
    development_games = set(
        splits.loc[splits["split_role"].astype(str).eq("development"), "game_id"]
        .astype(str)
    )
    sampled_games = [str(item.get("game_id")) for item in report.get("samples") or ()]
    if (
        int(protocol_gate.get("expected_games", -1)) != len(development_games)
        or len(sampled_games) != len(development_games)
        or len(set(sampled_games)) != len(development_games)
        or set(sampled_games) != development_games
    ):
        raise ResultsValidationError(
            "Checkpoint activation-fidelity samples do not cover every development game"
        )
    observed = float(aggregate.get("max_abs_error", math.inf))
    if not math.isfinite(observed) or observed != float(
        gate.get("observed_max_abs_error", math.nan)
    ):
        raise ResultsValidationError(
            "Checkpoint activation-fidelity maximum error binding is inconsistent"
        )
    return gate


def _fresh_frozen_inference(
    run_manifest: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """Return and validate the prospectively frozen reporting settings."""

    fresh = run_manifest.get("fresh_holdout")
    if not isinstance(fresh, Mapping):
        return None
    protocol_value = fresh.get("protocol_path")
    if not isinstance(protocol_value, str) or not protocol_value:
        raise ResultsValidationError("Fresh run lacks frozen protocol path")
    protocol = _read_json(Path(protocol_value).resolve(), "frozen protocol")
    inference = protocol.get("inference")
    if not isinstance(inference, Mapping):
        raise ResultsValidationError("Frozen protocol lacks inference settings")
    try:
        replicates = int(inference["bootstrap_replicates"])
        confidence_level = float(inference["confidence_level"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ResultsValidationError(
            "Frozen protocol has invalid bootstrap inference settings"
        ) from exc
    if (
        replicates != DEFAULT_BOOTSTRAP_REPLICATES
        or not math.isclose(
            confidence_level,
            DEFAULT_CONFIDENCE_LEVEL,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise ResultsValidationError(
            "Fresh protocol must freeze exactly 2000 bootstrap replicates and "
            "a 0.95 confidence level"
        )
    return {
        "bootstrap_replicates": replicates,
        "confidence_level": confidence_level,
    }


def _validate_binary_predictions(frame: pd.DataFrame, identity: str) -> pd.DataFrame:
    required = {
        "concept",
        "representation",
        "row_id",
        "game_id",
        "move_number",
        "outer_fold",
        "label",
        "probability",
        "prediction",
        "f1_threshold",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ResultsValidationError(f"{identity} lacks columns: {missing}")
    if frame.empty:
        raise ResultsValidationError(f"{identity} is empty")
    if frame["row_id"].astype(str).duplicated().any():
        raise ResultsValidationError(f"{identity} contains duplicate row_id values")
    for column in ("label", "prediction"):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or not set(np.unique(values)).issubset({0.0, 1.0}):
            raise ResultsValidationError(f"{identity} has non-binary {column} values")
    probabilities = pd.to_numeric(frame["probability"], errors="coerce").to_numpy(float)
    thresholds = pd.to_numeric(frame["f1_threshold"], errors="coerce").to_numpy(float)
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ResultsValidationError(f"{identity} has invalid probabilities")
    if not np.isfinite(thresholds).all() or np.any((thresholds < 0) | (thresholds > 1)):
        raise ResultsValidationError(f"{identity} has invalid F1 thresholds")
    observed = frame["prediction"].to_numpy(dtype=int)
    expected = (probabilities >= thresholds).astype(int)
    if not np.array_equal(observed, expected):
        raise ResultsValidationError(
            f"{identity} predictions do not reproduce probability >= saved fold threshold"
        )
    folds = pd.to_numeric(frame["outer_fold"], errors="coerce").to_numpy(float)
    if not np.isfinite(folds).all() or np.any(folds != np.floor(folds)):
        raise ResultsValidationError(f"{identity} has invalid outer-fold identifiers")
    return frame.copy()


def _metric_values(
    labels: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    *,
    sample_weight: Optional[np.ndarray] = None,
) -> Optional[Dict[str, float]]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = np.asarray(predictions, dtype=int)
    if sample_weight is None:
        weights = np.ones(len(labels), dtype=float)
    else:
        weights = np.asarray(sample_weight, dtype=float)
    if len(labels) == 0 or weights.shape != labels.shape:
        return None
    positive_weight = float(weights[labels == 1].sum())
    negative_weight = float(weights[labels == 0].sum())
    if positive_weight <= 0 or negative_weight <= 0:
        return None
    return {
        "roc_auc": float(
            roc_auc_score(labels, probabilities, sample_weight=weights)
        ),
        "average_precision": float(
            average_precision_score(labels, probabilities, sample_weight=weights)
        ),
        "f1": float(
            f1_score(labels, predictions, sample_weight=weights, zero_division=0)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions, sample_weight=weights)
        ),
    }


def validate_results_inputs(run_dir: Path) -> ValidatedResultsInputs:
    """Validate the complete immutable probe artifact chain and load predictions."""
    run_dir = Path(run_dir).resolve()
    _reject_archive_path(run_dir, "Run directory")
    if run_dir.name.casefold().endswith((".incomplete", "_incomplete")):
        raise ResultsValidationError(f"Refusing to report an incomplete run: {run_dir}")
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    run_path = run_dir / "manifest.json"
    run_manifest = _read_json(run_path, "run manifest")
    if (
        run_manifest.get("schema_version") != SCHEMA_VERSION
        or run_manifest.get("pipeline") != SOURCE_PIPELINE
    ):
        raise ResultsValidationError(
            f"Run was not created by schema-{SCHEMA_VERSION} {SOURCE_PIPELINE}"
        )
    _require_complete_if_declared(run_manifest, "Run manifest")
    declared_sources = run_manifest.get("source_code_sha256")
    current_sources = current_pipeline_source_hashes()
    if declared_sources != current_sources:
        raise ResultsValidationError(
            "Validated pipeline source hashes differ from the run manifest"
        )
    source_games_dir = run_manifest.get("source_games_dir")
    if isinstance(source_games_dir, str) and source_games_dir:
        _reject_archive_path(Path(source_games_dir).resolve(), "Raw games directory")

    run_artifacts = run_manifest.get("artifacts")
    if not isinstance(run_artifacts, Mapping):
        raise ResultsValidationError("Run manifest lacks an artifact mapping")
    concepts_path = _inside_run(run_dir, "frozen_config/concepts.yaml", "Frozen concepts")
    splits_path = _inside_run(run_dir, "splits.parquet", "Frozen splits")
    _require_hash(
        concepts_path, run_artifacts.get("concepts_yaml_sha256"), "frozen concepts"
    )
    _require_hash(splits_path, run_artifacts.get("splits_sha256"), "frozen splits")
    try:
        with concepts_path.open(encoding="utf-8") as handle:
            frozen_config = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ResultsValidationError(
            f"Invalid frozen concept configuration {concepts_path}: {exc}"
        ) from exc
    frozen_concepts = frozen_config.get("concepts")
    if not isinstance(frozen_concepts, Mapping):
        raise ResultsValidationError("Frozen concept configuration lacks a concept mapping")

    splits = pd.read_parquet(splits_path)
    required_splits = {"game_id", "split_role", "outer_fold"}
    if not required_splits.issubset(splits.columns):
        raise ResultsValidationError(
            f"Frozen splits lack columns: {sorted(required_splits - set(splits.columns))}"
        )
    if splits["game_id"].astype(str).duplicated().any():
        raise ResultsValidationError("Frozen splits contain duplicate game IDs")
    valid_roles = {"development", "control_calibration", "causal_test"}
    unknown_roles = sorted(set(splits["split_role"].astype(str)) - valid_roles)
    if unknown_roles:
        raise ResultsValidationError(f"Frozen splits contain unknown roles: {unknown_roles}")

    labels_path = _inside_run(run_dir, "labels_manifest.json", "Labels manifest")
    labels_manifest = _read_json(labels_path, "labels manifest")
    if labels_manifest.get("pipeline") != "validated_label_builder":
        raise ResultsValidationError("Labels were not produced by validated_label_builder")
    if labels_manifest.get("status") != "complete":
        raise ResultsValidationError("Labels manifest is not complete")
    source_dir = Path(__file__).resolve().parent
    _require_hash(
        source_dir / "build_validated_labels.py",
        labels_manifest.get("builder_source_sha256"),
        "validated label-builder source",
    )
    _require_hash(
        source_dir / "operational_definitions.py",
        labels_manifest.get("operational_definitions_source_sha256"),
        "label operational-definition source",
    )
    for key, observed in (
        ("run_manifest_sha256", _sha256(run_path)),
        ("split_manifest_sha256", run_artifacts.get("splits_sha256")),
        ("concepts_yaml_sha256", run_artifacts.get("concepts_yaml_sha256")),
    ):
        if labels_manifest.get(key) != observed:
            raise ResultsValidationError(f"Labels provenance mismatch for {key}")

    build_path = _inside_run(run_dir, "build_manifest.json", "Build manifest")
    build_manifest = _read_json(build_path, "build manifest")
    if (
        build_manifest.get("schema_version") != SCHEMA_VERSION
        or build_manifest.get("pipeline") != SOURCE_PIPELINE
    ):
        raise ResultsValidationError("Invalid build-manifest producer or schema")
    _require_complete_if_declared(build_manifest, "Build manifest")
    for key, expected in (
        ("split_manifest_sha256", run_artifacts.get("splits_sha256")),
        ("concepts_yaml_sha256", run_artifacts.get("concepts_yaml_sha256")),
        ("labels_manifest_sha256", _sha256(labels_path)),
    ):
        if build_manifest.get(key) != expected:
            raise ResultsValidationError(f"Build provenance mismatch for {key}")
    if build_manifest.get("source_code_sha256") != declared_sources:
        raise ResultsValidationError("Build source hashes differ from the prepared run")
    dataset_path = _inside_run(
        run_dir, str(build_manifest.get("dataset", "dataset.parquet")), "Dataset"
    )
    _require_hash(dataset_path, build_manifest.get("dataset_sha256"), "rebuilt dataset")

    training_path = _inside_run(run_dir, "training_manifest.json", "Training manifest")
    training_manifest = _read_json(training_path, "training manifest")
    if (
        training_manifest.get("schema_version") != SCHEMA_VERSION
        or training_manifest.get("pipeline") != SOURCE_PIPELINE
    ):
        raise ResultsValidationError("Invalid training-manifest producer or schema")
    _require_complete_if_declared(training_manifest, "Training manifest")
    if training_manifest.get("training_role") != "development":
        raise ResultsValidationError("Probes were not fitted exclusively on development games")
    for key, expected in (
        ("dataset_sha256", build_manifest.get("dataset_sha256")),
        ("build_manifest_sha256", _sha256(build_path)),
        ("labels_manifest_sha256", _sha256(labels_path)),
        ("split_manifest_sha256", run_artifacts.get("splits_sha256")),
        ("source_code_sha256", declared_sources),
    ):
        if training_manifest.get(key) != expected:
            raise ResultsValidationError(f"Training provenance mismatch for {key}")
    fidelity_gate = _validate_checkpoint_fidelity_gate(
        run_dir, run_manifest, training_manifest
    )

    concepts = tuple(map(str, training_manifest.get("concepts") or ()))
    representations = tuple(map(str, training_manifest.get("representations") or ()))
    if not concepts or len(set(concepts)) != len(concepts):
        raise ResultsValidationError("Training manifest has missing/duplicate concepts")
    build_concepts = tuple(map(str, build_manifest.get("concepts") or ()))
    if not build_concepts or len(set(build_concepts)) != len(build_concepts):
        raise ResultsValidationError("Build manifest has missing/duplicate concepts")
    enabled_concepts = tuple(
        str(name)
        for name, spec in frozen_concepts.items()
        if isinstance(spec, Mapping) and bool(spec.get("enabled", True))
    )
    if build_concepts != enabled_concepts or concepts != enabled_concepts:
        raise ResultsValidationError(
            "Complete reporting requires build and training concepts to equal every "
            "enabled frozen concept, in frozen order"
        )
    for concept in concepts:
        spec = frozen_concepts.get(concept)
        if not isinstance(spec, Mapping):
            raise ResultsValidationError(
                f"Trained concept {concept!r} is absent from frozen concepts"
            )
        _label_validation_tier(concept, spec, labels_manifest)
    if set(representations) != set(REPRESENTATIONS) or len(representations) != 3:
        raise ResultsValidationError(
            "Results reporting requires exactly global, local, and combined probe ablations"
        )
    outer_folds = int(training_manifest.get("outer_folds", 0))
    if outer_folds < 2:
        raise ResultsValidationError("Training manifest has fewer than two outer folds")

    training_artifacts = training_manifest.get("artifacts")
    if not isinstance(training_artifacts, Mapping) or not training_artifacts:
        raise ResultsValidationError("Training manifest lacks a complete artifact hash map")
    for relative, expected_hash in training_artifacts.items():
        artifact_path = _inside_run(run_dir, str(relative), "Training artifact")
        _require_hash(artifact_path, expected_hash, f"training artifact {relative}")

    required_artifacts = {"nested_cv_results.parquet", "probe_results.json"}
    for concept in concepts:
        for representation in REPRESENTATIONS:
            required_artifacts.update({
                f"outer_predictions/{concept}__{representation}.parquet",
                f"probes/{representation}/probe_{concept}.joblib",
                f"probes/{representation}/scaler_{concept}.joblib",
                f"probes/{representation}/probe_{concept}.meta.json",
            })
    missing_artifacts = sorted(required_artifacts - set(map(str, training_artifacts)))
    if missing_artifacts:
        raise ResultsValidationError(
            f"Training manifest omits required artifacts: {missing_artifacts}"
        )
    for concept in concepts:
        for representation in REPRESENTATIONS:
            metadata_path = _inside_run(
                run_dir,
                f"probes/{representation}/probe_{concept}.meta.json",
                "Probe metadata",
            )
            metadata = _read_json(metadata_path, "probe metadata")
            if fidelity_gate is not None and metadata.get(
                "checkpoint_activation_fidelity"
            ) != fidelity_gate:
                raise ResultsValidationError(
                    f"Probe metadata {concept}/{representation} is not bound to the "
                    "validated checkpoint activation-fidelity gate"
                )

    nested_path = _inside_run(run_dir, "nested_cv_results.parquet", "Nested CV results")
    nested_results = pd.read_parquet(nested_path)
    required_nested = {
        "concept",
        "representation",
        "outer_fold",
        "n_test",
        "positive_test",
        *METRICS,
    }
    if not required_nested.issubset(nested_results.columns):
        raise ResultsValidationError(
            "Nested CV table lacks columns: "
            f"{sorted(required_nested - set(nested_results.columns))}"
        )
    nested_keys = nested_results[["concept", "representation", "outer_fold"]]
    if nested_keys.duplicated().any():
        raise ResultsValidationError("Nested CV table has duplicate concept/representation/fold")

    predictions: Dict[Tuple[str, str], pd.DataFrame] = {}
    role_by_game = splits.set_index(splits["game_id"].astype(str))["split_role"].to_dict()
    fold_by_game = splits.set_index(splits["game_id"].astype(str))["outer_fold"].to_dict()
    for concept in concepts:
        reference: Optional[pd.DataFrame] = None
        for representation in REPRESENTATIONS:
            relative = f"outer_predictions/{concept}__{representation}.parquet"
            path = _inside_run(run_dir, relative, "Outer predictions")
            frame = _validate_binary_predictions(
                pd.read_parquet(path), f"outer predictions {concept}/{representation}"
            )
            if set(frame["concept"].astype(str)) != {concept}:
                raise ResultsValidationError(f"Prediction concept mismatch in {relative}")
            if set(frame["representation"].astype(str)) != {representation}:
                raise ResultsValidationError(f"Prediction representation mismatch in {relative}")
            game_ids = frame["game_id"].astype(str)
            roles = game_ids.map(role_by_game)
            if roles.isna().any() or set(roles.astype(str)) != {"development"}:
                raise ResultsValidationError(
                    f"{relative} contains non-development or unknown games"
                )
            expected_folds = pd.to_numeric(game_ids.map(fold_by_game), errors="coerce")
            observed_folds = pd.to_numeric(frame["outer_fold"], errors="coerce")
            if not np.array_equal(
                expected_folds.to_numpy(dtype=float), observed_folds.to_numpy(dtype=float)
            ):
                raise ResultsValidationError(f"{relative} disagrees with frozen outer folds")
            observed_fold_ids = set(observed_folds.astype(int))
            if observed_fold_ids != set(range(outer_folds)):
                raise ResultsValidationError(
                    f"{relative} does not contain every expected outer fold"
                )

            ordered = frame.sort_values("row_id").reset_index(drop=True)
            if reference is None:
                reference = ordered
            else:
                for column in ("row_id", "game_id", "move_number", "outer_fold", "label"):
                    if not np.array_equal(
                        reference[column].astype(str).to_numpy(),
                        ordered[column].astype(str).to_numpy(),
                    ):
                        raise ResultsValidationError(
                            f"Ablations for {concept} do not use the same rows/labels/folds"
                        )
            predictions[(concept, representation)] = ordered

            nested_subset = nested_results.loc[
                nested_results["concept"].astype(str).eq(concept)
                & nested_results["representation"].astype(str).eq(representation)
            ]
            if set(nested_subset["outer_fold"].astype(int)) != set(range(outer_folds)):
                raise ResultsValidationError(
                    f"Nested results are incomplete for {concept}/{representation}"
                )
            for fold in range(outer_folds):
                prediction_fold = frame.loc[frame["outer_fold"].astype(int).eq(fold)]
                saved_fold = nested_subset.loc[
                    nested_subset["outer_fold"].astype(int).eq(fold)
                ].iloc[0]
                values = _metric_values(
                    prediction_fold["label"].to_numpy(int),
                    prediction_fold["probability"].to_numpy(float),
                    prediction_fold["prediction"].to_numpy(int),
                )
                if values is None:
                    raise ResultsValidationError(
                        f"Fold {fold} lacks both classes for {concept}/{representation}"
                    )
                if int(saved_fold["n_test"]) != len(prediction_fold):
                    raise ResultsValidationError(
                        f"n_test mismatch for {concept}/{representation}/fold {fold}"
                    )
                if int(saved_fold["positive_test"]) != int(
                    prediction_fold["label"].astype(int).sum()
                ):
                    raise ResultsValidationError(
                        f"positive_test mismatch for {concept}/{representation}/fold {fold}"
                    )
                for metric, recomputed in values.items():
                    saved = float(saved_fold[metric])
                    if not math.isclose(saved, recomputed, rel_tol=1e-10, abs_tol=1e-12):
                        raise ResultsValidationError(
                            f"Saved {metric} does not reproduce outer predictions for "
                            f"{concept}/{representation}/fold {fold}: {saved} vs {recomputed}"
                        )

    expected_nested = {
        (concept, representation, fold)
        for concept in concepts
        for representation in REPRESENTATIONS
        for fold in range(outer_folds)
    }
    observed_nested = {
        (str(row.concept), str(row.representation), int(row.outer_fold))
        for row in nested_results.itertuples(index=False)
    }
    if observed_nested != expected_nested:
        raise ResultsValidationError("Nested CV table has missing or undeclared result rows")

    return ValidatedResultsInputs(
        run_dir=run_dir,
        run_manifest=run_manifest,
        build_manifest=build_manifest,
        labels_manifest=labels_manifest,
        training_manifest=training_manifest,
        splits=splits,
        nested_results=nested_results,
        predictions=predictions,
        frozen_concepts=frozen_concepts,
    )


def _fold_summary(frame: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    by_metric: Dict[str, list[float]] = {metric: [] for metric in METRICS}
    fold_ids = sorted(frame["outer_fold"].astype(int).unique())
    for fold in fold_ids:
        part = frame.loc[frame["outer_fold"].astype(int).eq(fold)]
        values = _metric_values(
            part["label"].to_numpy(int),
            part["probability"].to_numpy(float),
            part["prediction"].to_numpy(int),
        )
        if values is None:  # guarded during validation
            raise ResultsValidationError(f"Outer fold {fold} lacks both classes")
        for metric in METRICS:
            by_metric[metric].append(values[metric])
    arrays = {metric: np.asarray(values, dtype=float) for metric, values in by_metric.items()}
    report = {
        metric: {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
            "sd_ddof": 1,
            "values_by_outer_fold": {
                str(fold): float(value) for fold, value in zip(fold_ids, values)
            },
        }
        for metric, values in arrays.items()
    }
    return report, arrays


def _derived_seed(base_seed: int, identity: str) -> int:
    digest = hashlib.sha256(f"{int(base_seed)}:{identity}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


def _weighted_metric_matrix(
    labels: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    weights: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Vectorised binary metrics for many bootstrap weight vectors.

    Probability order and ties do not change across bootstrap samples.  Using
    this fixed ordering avoids calling/sorting through sklearn once per sample,
    which would make a 2,000-replicate report unnecessarily expensive.
    """
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = np.asarray(predictions, dtype=int)
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 2 or weights.shape[1] != len(labels):
        raise ValueError("Bootstrap weights must have shape replicates x observations")

    positive = labels == 1
    negative = ~positive
    positive_total = weights @ positive.astype(float)
    negative_total = weights @ negative.astype(float)
    valid = (positive_total > 0) & (negative_total > 0)

    true_positive = weights @ (positive & (predictions == 1)).astype(float)
    false_positive = weights @ (negative & (predictions == 1)).astype(float)
    false_negative = weights @ (positive & (predictions == 0)).astype(float)
    true_negative = weights @ (negative & (predictions == 0)).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = 2.0 * true_positive / (
            2.0 * true_positive + false_positive + false_negative
        )
        balanced = 0.5 * (
            true_positive / positive_total + true_negative / negative_total
        )

    ascending = np.argsort(probabilities, kind="stable")
    sorted_probability = probabilities[ascending]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_probability) != 0) + 1]
    ascending_weights = weights[:, ascending]
    positive_group = np.add.reduceat(
        ascending_weights * positive[ascending], starts, axis=1
    )
    negative_group = np.add.reduceat(
        ascending_weights * negative[ascending], starts, axis=1
    )
    negative_before = np.cumsum(negative_group, axis=1) - negative_group
    with np.errstate(divide="ignore", invalid="ignore"):
        roc_auc = np.sum(
            positive_group * (negative_before + 0.5 * negative_group), axis=1
        ) / (positive_total * negative_total)

    descending = ascending[::-1]
    descending_probability = probabilities[descending]
    starts = np.r_[0, np.flatnonzero(np.diff(descending_probability) != 0) + 1]
    descending_weights = weights[:, descending]
    positive_group = np.add.reduceat(
        descending_weights * positive[descending], starts, axis=1
    )
    total_group = np.add.reduceat(descending_weights, starts, axis=1)
    cumulative_positive = np.cumsum(positive_group, axis=1)
    cumulative_total = np.cumsum(total_group, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = cumulative_positive / cumulative_total
        average_precision = np.sum(
            (positive_group / positive_total[:, None]) * precision, axis=1
        )

    result = {
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "f1": f1,
        "balanced_accuracy": balanced,
    }
    for values in result.values():
        values[~valid] = np.nan
    return result


def _bootstrap_metrics(
    frames: Mapping[str, pd.DataFrame],
    *,
    replicates: int,
    seed: int,
    confidence_level: float,
) -> Dict[str, Dict[str, np.ndarray]]:
    if replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")
    reference = frames[REPRESENTATIONS[0]]
    fold_ids = sorted(reference["outer_fold"].astype(int).unique())
    rng = np.random.default_rng(seed)
    fold_state: Dict[int, Dict[str, Any]] = {}
    for fold in fold_ids:
        mask = reference["outer_fold"].astype(int).eq(fold).to_numpy()
        games = reference.loc[mask, "game_id"].astype(str)
        unique_games = sorted(games.unique())
        if len(unique_games) < 2:
            raise ResultsValidationError(
                f"Outer fold {fold} has fewer than two game clusters"
            )
        game_codes = pd.Categorical(games, categories=unique_games).codes
        counts = rng.multinomial(
            len(unique_games),
            np.full(len(unique_games), 1.0 / len(unique_games)),
            size=replicates,
        )
        fold_state[fold] = {
            "mask": mask,
            "game_codes": game_codes,
            "counts": counts,
        }

    result = {
        representation: {
            metric: np.zeros(replicates, dtype=float) for metric in METRICS
        }
        for representation in REPRESENTATIONS
    }
    valid = {
        representation: {
            metric: np.ones(replicates, dtype=bool) for metric in METRICS
        }
        for representation in REPRESENTATIONS
    }
    arrays = {
        representation: {
            "label": frame["label"].to_numpy(int),
            "probability": frame["probability"].to_numpy(float),
            "prediction": frame["prediction"].to_numpy(int),
        }
        for representation, frame in frames.items()
    }
    # Bound the largest temporary matrix to roughly chunk_size x positions in
    # one fold.  This keeps full 50k-position reports comfortably below a few
    # hundred MB while retaining vectorised metric calculations.
    chunk_size = 128
    for fold in fold_ids:
        state = fold_state[fold]
        mask = state["mask"]
        for start in range(0, replicates, chunk_size):
            stop = min(start + chunk_size, replicates)
            weights = state["counts"][start:stop, state["game_codes"]].astype(float)
            for representation in REPRESENTATIONS:
                values = _weighted_metric_matrix(
                    arrays[representation]["label"][mask],
                    arrays[representation]["probability"][mask],
                    arrays[representation]["prediction"][mask],
                    weights,
                )
                for metric in METRICS:
                    finite = np.isfinite(values[metric])
                    result[representation][metric][start:stop] += np.where(
                        finite, values[metric], 0.0
                    )
                    valid[representation][metric][start:stop] &= finite
    for representation in REPRESENTATIONS:
        for metric in METRICS:
            result[representation][metric] /= len(fold_ids)
            result[representation][metric][~valid[representation][metric]] = np.nan
    return result


def _interval_record(
    values: np.ndarray,
    *,
    point_estimate: float,
    confidence_level: float,
    requested: int,
) -> Dict[str, Any]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    minimum_valid = max(1, int(math.ceil(0.8 * requested)))
    if len(finite) < minimum_valid:
        return {
            "status": "not_estimable",
            "reason": "too_many_bootstrap_samples_lacked_both_classes_in_every_fold",
            "point_estimate": float(point_estimate),
            "replicates_requested": int(requested),
            "replicates_valid": int(len(finite)),
        }
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(finite, [alpha, 1.0 - alpha])
    return {
        "status": "ok",
        "point_estimate": float(point_estimate),
        "bootstrap_mean": float(finite.mean()),
        "lower": float(lower),
        "upper": float(upper),
        "confidence_level": float(confidence_level),
        "replicates_requested": int(requested),
        "replicates_valid": int(len(finite)),
    }


def generate_results_report(
    run_dir: Path,
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: Optional[int] = None,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> Dict[str, Any]:
    """Validate a run and return fold summaries, clustered CIs, and ablations."""
    validated = validate_results_inputs(run_dir)
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")
    frozen_inference = _fresh_frozen_inference(validated.run_manifest)
    inference_protocol_conforms = frozen_inference is None or (
        int(bootstrap_replicates) == int(frozen_inference["bootstrap_replicates"])
        and math.isclose(
            float(confidence_level),
            float(frozen_inference["confidence_level"]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
    )
    base_seed = (
        int(bootstrap_seed)
        if bootstrap_seed is not None
        else _derived_seed(int(validated.run_manifest["seed"]), PIPELINE_NAME)
    )
    concepts = tuple(map(str, validated.training_manifest["concepts"]))
    concept_reports: Dict[str, Any] = {}
    pairs = (
        ("combined_minus_global", "combined", "global"),
        ("combined_minus_local", "combined", "local"),
        ("local_minus_global", "local", "global"),
    )

    for concept in concepts:
        frames = {
            representation: validated.predictions[(concept, representation)]
            for representation in REPRESENTATIONS
        }
        concept_seed = _derived_seed(base_seed, concept)
        bootstrap = _bootstrap_metrics(
            frames,
            replicates=int(bootstrap_replicates),
            seed=concept_seed,
            confidence_level=confidence_level,
        )
        representation_reports: Dict[str, Any] = {}
        fold_arrays: Dict[str, Dict[str, np.ndarray]] = {}
        for representation in REPRESENTATIONS:
            frame = frames[representation]
            fold_metrics, arrays = _fold_summary(frame)
            fold_arrays[representation] = arrays
            counts = {
                "n_positions": int(len(frame)),
                "n_games": int(frame["game_id"].astype(str).nunique()),
                "n_positives": int(frame["label"].astype(int).sum()),
                "n_negatives": int(len(frame) - frame["label"].astype(int).sum()),
                "n_games_with_positive": int(
                    frame.groupby(frame["game_id"].astype(str))["label"].max().sum()
                ),
            }
            intervals = {
                metric: _interval_record(
                    bootstrap[representation][metric],
                    point_estimate=fold_metrics[metric]["mean"],
                    confidence_level=confidence_level,
                    requested=int(bootstrap_replicates),
                )
                for metric in METRICS
            }
            representation_reports[representation] = {
                "counts": counts,
                "fold_metrics": fold_metrics,
                "game_cluster_bootstrap": intervals,
            }

        ablations: Dict[str, Any] = {}
        for name, left, right in pairs:
            paired_folds: Dict[str, Any] = {}
            paired_bootstrap: Dict[str, Any] = {}
            for metric in METRICS:
                fold_delta = fold_arrays[left][metric] - fold_arrays[right][metric]
                point = float(fold_delta.mean())
                paired_folds[metric] = {
                    "mean_delta": point,
                    "sd_delta": (
                        float(fold_delta.std(ddof=1)) if len(fold_delta) > 1 else None
                    ),
                    "sd_ddof": 1,
                    "delta_by_outer_fold": {
                        str(fold): float(value) for fold, value in enumerate(fold_delta)
                    },
                }
                delta_draws = bootstrap[left][metric] - bootstrap[right][metric]
                paired_bootstrap[metric] = _interval_record(
                    delta_draws,
                    point_estimate=point,
                    confidence_level=confidence_level,
                    requested=int(bootstrap_replicates),
                )
            ablations[name] = {
                "left": left,
                "right": right,
                "pairing": "same outer-fold rows and same resampled game multiplicities",
                "paired_outer_fold_deltas": paired_folds,
                "paired_game_cluster_bootstrap": paired_bootstrap,
            }

        concept_reports[concept] = {
            "label_validation": _label_validation_tier(
                concept,
                validated.frozen_concepts[concept],
                validated.labels_manifest,
            ),
            "bootstrap_seed": int(concept_seed),
            "representations": representation_reports,
            "ablations": ablations,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline": PIPELINE_NAME,
        "source_pipeline": SOURCE_PIPELINE,
        "status": "complete",
        "created_at_utc": _utc_now(),
        "run_dir": str(validated.run_dir),
        "provenance": {
            "run_manifest_sha256": _sha256(validated.run_dir / "manifest.json"),
            "build_manifest_sha256": _sha256(validated.run_dir / "build_manifest.json"),
            "labels_manifest_sha256": _sha256(validated.run_dir / "labels_manifest.json"),
            "training_manifest_sha256": _sha256(
                validated.run_dir / "training_manifest.json"
            ),
            "reporter_source_sha256": _sha256(Path(__file__).resolve()),
            "checkpoint_activation_fidelity": validated.training_manifest.get(
                "checkpoint_activation_fidelity"
            ),
        },
        "bootstrap": {
            "unit": "game",
            "stratification": "outer_fold",
            "procedure": (
                "sample games with replacement within each outer fold; recompute each "
                "fold metric; average fold metrics; use identical draws for ablations"
            ),
            "conditioning": (
                "Intervals are conditional on the fixed nested-CV out-of-fold "
                "predictions and fitted fold-specific thresholds; bootstrap "
                "replicates do not refit probes, scalers, hyperparameters, or thresholds."
            ),
            "refits_models": False,
            "interval": "percentile",
            "confidence_level": float(confidence_level),
            "replicates": int(bootstrap_replicates),
            "base_seed": int(base_seed),
            "frozen_inference": frozen_inference,
            "inference_protocol_conforms": bool(inference_protocol_conforms),
        },
        "metric_notes": {
            "average_precision": "non-interpolated average precision",
            "f1": "uses each outer fold's inner-CV-selected threshold",
            "balanced_accuracy": "uses each outer fold's inner-CV-selected threshold",
            "fold_sd": "sample standard deviation across outer folds (ddof=1)",
        },
        "concepts": concept_reports,
    }


def write_results_report(
    run_dir: Path,
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: Optional[int] = None,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> Tuple[Path, Path]:
    """Write an append-only report and a small manifest binding its hash."""
    run_dir = Path(run_dir).resolve()
    report_path = run_dir / "validated_results_report.json"
    manifest_path = run_dir / "validated_results_report_manifest.json"
    if report_path.exists() or manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing validated results report in {run_dir}"
        )
    report = generate_results_report(
        run_dir,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    if not bool(report["bootstrap"]["inference_protocol_conforms"]):
        raise ResultsValidationError(
            "Append-only fresh validated reports must use the frozen 2000 bootstrap "
            "replicates and 0.95 confidence level"
        )
    with report_path.open("xb") as handle:
        handle.write(_canonical_json_bytes(report))
    report_path.chmod(0o444)
    report_manifest = {
        "schema_version": SCHEMA_VERSION,
        "pipeline": PIPELINE_NAME,
        "status": "complete",
        "created_at_utc": _utc_now(),
        "report": report_path.name,
        "report_sha256": _sha256(report_path),
        "reporter_source_sha256": _sha256(Path(__file__).resolve()),
        "upstream_training_manifest_sha256": report["provenance"][
            "training_manifest_sha256"
        ],
    }
    with manifest_path.open("xb") as handle:
        handle.write(_canonical_json_bytes(report_manifest))
    manifest_path.chmod(0o444)
    return report_path, manifest_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    parser.add_argument("--bootstrap-seed", type=int)
    parser.add_argument("--confidence-level", type=float, default=DEFAULT_CONFIDENCE_LEVEL)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write append-only report files in the run instead of printing JSON",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    if args.write:
        report, manifest = write_results_report(
            args.run_dir,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
            confidence_level=args.confidence_level,
        )
        print(json.dumps({"report": str(report), "manifest": str(manifest)}, indent=2))
    else:
        report = generate_results_report(
            args.run_dir,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
            confidence_level=args.confidence_level,
        )
        print(json.dumps(_json_safe(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
