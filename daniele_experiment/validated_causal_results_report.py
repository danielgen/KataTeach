#!/usr/bin/env python3
"""Report held-out causal effects from validated causal-evaluation artifacts.

The reporter is downstream-only and fail-closed.  It verifies the causal
manifest, every declared causal artifact, the causal producer source hashes,
and the complete probe-run manifest chain before reading results.  Only rows
explicitly assigned to ``causal_test`` are accepted.

Confidence intervals resample whole games.  Because causal positions were
selected with label-stratified sampling, the report presents label-specific
effects and an equal-label-weighted estimate in addition to the raw selected-
position mean.  Empirical control tests use the label-balanced effect and the
finite-sample correction ``(1 + # at least as extreme) / (1 + K)``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from .causal_controls import validate_run_manifest
    from .operational_definitions import get_contract
    from .validated_results_report import (
        _validate_checkpoint_fidelity_gate,
        current_pipeline_source_hashes,
    )
except ImportError:  # pragma: no cover - direct CLI execution.
    from causal_controls import validate_run_manifest
    from operational_definitions import get_contract
    from validated_results_report import (
        _validate_checkpoint_fidelity_gate,
        current_pipeline_source_hashes,
    )


SCHEMA_VERSION = 1
PIPELINE_NAME = "validated_causal_results_report"
SOURCE_PIPELINE = "validated_causal_eval"
PROBE_PIPELINE = "validated_probe_pipeline"
DEFAULT_BOOTSTRAP_REPLICATES = 2_000
DEFAULT_CONFIDENCE_LEVEL = 0.95
MIN_SPATIAL_SHUFFLES = 50
MIN_RANDOM_DIRECTIONS = 100
REQUIRED_CAUSAL_ARTIFACTS = frozenset({
    "protocol_estimate.json",
    "selected_positions.parquet",
    "activation_bindings.parquet",
    "policy_head_equivalence.json",
    "operational_alignment.json",
    "control_calibration.json",
    "causal_test_rows.parquet",
    "summary.json",
})
CAUSAL_SOURCE_FILES = {
    "daniele_experiment/validated_causal_eval.py": "validated_causal_eval.py",
    "daniele_experiment/causal_controls.py": "causal_controls.py",
    "daniele_experiment/operational_definitions.py": "operational_definitions.py",
}


class CausalResultsValidationError(ValueError):
    """Raised when a causal result cannot support a validated report."""


@dataclass(frozen=True)
class ValidatedCausalInputs:
    causal_dir: Path
    probe_run_dir: Path
    causal_manifest: Mapping[str, Any]
    run_manifest: Mapping[str, Any]
    build_manifest: Mapping[str, Any]
    labels_manifest: Mapping[str, Any]
    training_manifest: Mapping[str, Any]
    probe_metadata: Mapping[str, Any]
    summary: Mapping[str, Any]
    calibration: Mapping[str, Any]
    selected_positions: pd.DataFrame
    rows: pd.DataFrame
    concept: str
    representation: str
    contract_name: str
    readout_directions: Mapping[str, Mapping[str, Any]]
    frozen_protocol: Mapping[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(_json_safe(value), sort_keys=True, indent=2) + "\n").encode()


def _read_json(path: Path, description: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CausalResultsValidationError(
            f"Invalid {description} {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise CausalResultsValidationError(
            f"{description} must contain a JSON object: {path}"
        )
    return value


def _is_archive_component(part: str) -> bool:
    value = part.casefold()
    return value in {"archive", "archives", "archived"} or value.startswith("archive_")


def _reject_unsafe_path(path: Path, description: str) -> None:
    resolved = path.resolve()
    if any(_is_archive_component(part) for part in resolved.parts):
        raise CausalResultsValidationError(f"{description} points into an archive: {path}")
    if any(
        part.casefold().endswith(("_incomplete", ".incomplete", "-incomplete"))
        for part in resolved.parts
    ):
        raise CausalResultsValidationError(
            f"{description} points into an incomplete run: {path}"
        )


def _inside(root: Path, relative: Any, description: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise CausalResultsValidationError(f"Missing relative path for {description}")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise CausalResultsValidationError(
            f"{description} must be a relative path inside {root}: {relative!r}"
        )
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise CausalResultsValidationError(f"{description} escapes {root}") from exc
    _reject_unsafe_path(resolved, description)
    return resolved


def _require_hash(path: Path, expected: Any, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    if not isinstance(expected, str) or len(expected) != 64:
        raise CausalResultsValidationError(
            f"Missing or invalid declared hash for {description}"
        )
    observed = _sha256(path)
    if observed != expected.lower():
        raise CausalResultsValidationError(
            f"Hash mismatch for {description}: expected {expected}, observed {observed}"
        )


def current_causal_source_hashes() -> Dict[str, str]:
    source_dir = Path(__file__).resolve().parent
    return {
        identity: _sha256(source_dir / filename)
        for identity, filename in CAUSAL_SOURCE_FILES.items()
    }


def _require_terminal_success(document: Mapping[str, Any], description: str) -> None:
    status = document.get("status")
    if status is not None and status not in {"complete", "validated"}:
        raise CausalResultsValidationError(
            f"{description} has non-success status {status!r}"
        )


def _artifact_map(manifest: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    records = manifest.get("artifacts")
    if not isinstance(records, list):
        raise CausalResultsValidationError("Causal artifact manifest is not a list")
    result: Dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise CausalResultsValidationError("Invalid causal artifact record")
        path = str(record["path"])
        if path in result:
            raise CausalResultsValidationError(f"Duplicate causal artifact record: {path}")
        result[path] = record
    return result


def _readout_direction_specs(contract_name: str) -> Mapping[str, Mapping[str, Any]]:
    """Return the declared sign under a positive operational-label direction."""

    specs: Dict[str, Dict[str, Mapping[str, Any]]] = {
        "tenuki_distance6": {
            "tenuki_distance6_policy_mass": {
                "positive_label_sign": 1,
                "basis": "exact policy mass on actions satisfying distance >= 6",
            },
            "tenuki_distance6_complement_mass": {
                "positive_label_sign": -1,
                "basis": "complement of the exact distance >= 6 action mass",
            },
            "tenuki_expected_manhattan_distance": {
                "positive_label_sign": 1,
                "basis": "distance-associated secondary readout",
            },
        },
        "reply_peak95": {
            "reply_peak95_action_mass": {
                "positive_label_sign": 1,
                "basis": "exact mass on actions with frozen opponent reply peak > 0.95",
            },
            "expected_reply_peak": {
                "positive_label_sign": 1,
                "basis": "reply-peak-associated secondary readout",
            },
        },
        "regional_policy_peak": {
            "regional_policy_peak": {
                "positive_label_sign": 1,
                "basis": "exact operational variable used by the high-quantile label",
            },
            "regional_policy_margin": {
                "positive_label_sign": 1,
                "basis": "concentration-associated secondary readout",
            },
            "regional_policy_entropy": {
                "positive_label_sign": -1,
                "basis": "entropy decreases as regional concentration increases",
            },
            "regional_policy_anchor_mass": {
                "positive_label_sign": 1,
                "basis": "mass in the baseline-peak region used by the aligned mask",
            },
        },
    }
    if contract_name not in specs:
        raise CausalResultsValidationError(
            f"No causal readout direction registry for contract {contract_name!r}"
        )
    return specs[contract_name]


def _validate_frozen_causal_design(
    protocol: Mapping[str, Any],
    provenance: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> None:
    causal = protocol.get("causal")
    inference = protocol.get("inference")
    if not isinstance(causal, Mapping) or not isinstance(inference, Mapping):
        raise CausalResultsValidationError(
            "Frozen protocol lacks causal/inference declarations"
        )
    primary = causal.get("primary_hypothesis")
    statistic = primary.get("statistic") if isinstance(primary, Mapping) else None
    decision = primary.get("decision_rule") if isinstance(primary, Mapping) else None
    if (
        not isinstance(statistic, Mapping)
        or statistic.get("name")
        != "label_balanced_ols_slope_across_all_frozen_doses"
        or statistic.get("predictor") != "nominal_dose"
        or statistic.get("outcome") != "paired readout delta"
        or statistic.get("intercept") is not True
        or statistic.get("dose_set") != "all values in causal.doses"
        or statistic.get("label_weights") != {"0": 0.5, "1": 0.5}
        or not isinstance(decision, Mapping)
        or float(decision.get("alpha", math.nan)) != 0.05
        or decision.get("expected_trained_slope") != "positive"
        or decision.get("random_direction_test")
        != "one-sided finite-control empirical p <= alpha"
        or decision.get("spatial_shuffle_test")
        != "one-sided finite-control empirical p <= alpha"
        or decision.get("headline_support_requires")
        != (
            "trained slope > 0 AND random-direction p <= alpha AND "
            "spatial-shuffle p <= alpha"
        )
    ):
        raise CausalResultsValidationError(
            "Frozen protocol lacks the exact sole primary slope statistic/decision rule"
        )
    if list(map(float, summary.get("doses") or ())) != list(
        map(float, causal.get("doses") or ())
    ):
        raise CausalResultsValidationError(
            "Evaluated doses differ from the prospectively frozen causal doses"
        )
    for summary_key, protocol_key in (
        ("spatial_shuffle_repeats", "spatial_shuffle_controls"),
        ("random_direction_repeats", "random_direction_controls"),
        ("calibration_positions", "maximum_calibration_positions"),
        ("causal_test_positions", "maximum_test_positions"),
    ):
        if int(summary.get(summary_key, -1)) != int(causal.get(protocol_key, -2)):
            raise CausalResultsValidationError(
                f"Causal result differs from frozen protocol for {summary_key}"
            )
    if int(provenance.get("seed", -1)) != int(causal.get("causal_seed", -2)):
        raise CausalResultsValidationError(
            "Causal seed differs from the prospectively frozen protocol"
        )
    for provenance_key, protocol_key in (
        ("policy_head_batch_size", "policy_head_batch_size"),
        ("equivalence_sample_size", "full_vs_head_equivalence_sample_size"),
    ):
        if int(provenance.get(provenance_key, -1)) != int(
            causal.get(protocol_key, -2)
        ):
            raise CausalResultsValidationError(
                f"Causal provenance differs from frozen protocol for {provenance_key}"
            )
    for provenance_key, protocol_key in (
        ("policy_equivalence_atol", "policy_equivalence_absolute_tolerance"),
        ("activation_equivalence_atol", "activation_equivalence_absolute_tolerance"),
    ):
        if not math.isclose(
            float(provenance.get(provenance_key, math.nan)),
            float(causal.get(protocol_key, math.nan)),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise CausalResultsValidationError(
                f"Causal provenance differs from frozen protocol for {provenance_key}"
            )
    backend = summary.get("evaluation_backend") or {}
    if (
        int(backend.get("batch_size", -1))
        != int(causal.get("policy_head_batch_size", -2))
        or int(backend.get("equivalence_sample_size", -1))
        != int(causal.get("full_vs_head_equivalence_sample_size", -2))
    ):
        raise CausalResultsValidationError(
            "Causal summary backend differs from frozen batch/equivalence protocol"
        )
    if summary.get("controls_policy_matched_on") != causal.get("control_matching"):
        raise CausalResultsValidationError(
            "Causal summary control-matching target differs from frozen protocol"
        )
    if causal.get("one_position_per_game") is not True:
        raise CausalResultsValidationError(
            "Frozen protocol does not authorize the confirmatory selection unit"
        )
    if inference.get("unit") != "game":
        raise CausalResultsValidationError("Frozen inference unit is not game")
    secondary_rules = causal.get("secondary_evaluation_rules") or {}
    forcing_rule = secondary_rules.get("forcing") or {}
    urgency_rule = secondary_rules.get("urgency_peak") or {}
    if (
        forcing_rule.get("scope") != "best_effort_exploratory"
        or forcing_rule.get("confirmatory_gate") is not False
        or "do not replace/refill games" not in str(
            forcing_rule.get("infeasibility_rule", "")
        )
        or urgency_rule.get("scope") != "exploratory"
        or urgency_rule.get("confirmatory_gate") is not False
    ):
        raise CausalResultsValidationError(
            "Frozen protocol lacks exact secondary exploratory feasibility rules"
        )


def _analysis_scope(
    protocol: Mapping[str, Any], concept: str, representation: str, readout: str
) -> Dict[str, Any]:
    causal = protocol.get("causal") or {}
    primary = causal.get("primary_hypothesis") or {}
    secondary = set(map(str, causal.get("secondary_exploratory_concepts") or ()))
    if (
        concept == str(primary.get("concept"))
        and representation == str(primary.get("representation"))
        and readout == str(primary.get("readout"))
    ):
        scope = "descriptive_component_of_confirmatory_aggregate"
        reason = (
            "This is the frozen primary readout, but no individual dose is a "
            "confirmatory test; only the all-dose slope conjunction is primary."
        )
    elif concept in secondary and representation == str(
        causal.get("secondary_representation")
    ):
        scope = "secondary_exploratory_concept"
        reason = "Concept was prospectively designated secondary exploratory."
    elif (
        concept == str(primary.get("concept"))
        and representation == str(primary.get("representation"))
    ):
        scope = "secondary_exploratory_readout"
        reason = "Same primary intervention, but this readout was not the frozen primary readout."
    else:
        scope = "outside_frozen_confirmatory_scope"
        reason = "Concept/representation/readout tuple is not a frozen confirmatory hypothesis."
    result = {
        "scope": scope,
        "reason": reason,
        "multiplicity": (
            "Dose/readout/control-family p-values are reported individually without a "
            "multiple-testing correction and are descriptive/exploratory. Never infer "
            "a headline from the minimum p-value or substitute one for the frozen slope."
        ),
    }
    secondary_rule = (causal.get("secondary_evaluation_rules") or {}).get(concept)
    if isinstance(secondary_rule, Mapping):
        result["frozen_secondary_evaluation_rule"] = dict(secondary_rule)
    return result


def _finite_numeric(frame: pd.DataFrame, columns: Iterable[str], description: str) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise CausalResultsValidationError(
                f"{description} column {column!r} contains non-finite values"
            )


def validate_causal_inputs(causal_dir: Path) -> ValidatedCausalInputs:
    """Verify a causal evaluation and all upstream inputs before reporting."""

    causal_dir = Path(causal_dir).resolve()
    _reject_unsafe_path(causal_dir, "Causal result directory")
    if not causal_dir.is_dir():
        raise FileNotFoundError(f"Causal result directory does not exist: {causal_dir}")
    try:
        causal_manifest = validate_run_manifest(
            causal_dir,
            allowed_statuses=("validated",),
            verify_artifacts=True,
            require_artifacts=True,
        )
    except (ValueError, OSError) as exc:
        raise CausalResultsValidationError(
            f"Causal manifest validation failed: {exc}"
        ) from exc
    if (
        causal_manifest.get("pipeline") != SOURCE_PIPELINE
        or causal_manifest.get("kind") != "causal_evaluation"
    ):
        raise CausalResultsValidationError(
            "Input is not a validated causal_evaluation output"
        )
    artifacts = _artifact_map(causal_manifest)
    missing = sorted(REQUIRED_CAUSAL_ARTIFACTS - set(artifacts))
    if missing:
        raise CausalResultsValidationError(
            f"Causal manifest omits required artifacts: {missing}"
        )

    provenance = causal_manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise CausalResultsValidationError("Causal manifest lacks provenance")
    current_causal_sources = current_causal_source_hashes()
    if provenance.get("source_code_sha256") != current_causal_sources:
        raise CausalResultsValidationError(
            "Causal producer sources differ from the currently audited sources"
        )
    if provenance.get("producer_source_sha256") != current_causal_sources:
        raise CausalResultsValidationError(
            "Causal producer source alias is missing or inconsistent"
        )

    probe_run_value = provenance.get("probe_run")
    if not isinstance(probe_run_value, str) or not probe_run_value:
        raise CausalResultsValidationError("Causal provenance lacks probe_run")
    probe_run_dir = Path(probe_run_value).resolve()
    _reject_unsafe_path(probe_run_dir, "Upstream probe run")
    if not probe_run_dir.is_dir():
        raise FileNotFoundError(f"Upstream probe run does not exist: {probe_run_dir}")
    try:
        causal_dir.relative_to(probe_run_dir)
    except ValueError as exc:
        raise CausalResultsValidationError(
            "Causal output is not contained in its declared probe run"
        ) from exc

    run_path = probe_run_dir / "manifest.json"
    build_path = probe_run_dir / "build_manifest.json"
    training_path = probe_run_dir / "training_manifest.json"
    run_manifest = _read_json(run_path, "probe run manifest")
    build_manifest = _read_json(build_path, "build manifest")
    training_manifest = _read_json(training_path, "training manifest")
    for document, description in (
        (run_manifest, "Probe run manifest"),
        (build_manifest, "Build manifest"),
        (training_manifest, "Training manifest"),
    ):
        _require_terminal_success(document, description)
        if document.get("pipeline") != PROBE_PIPELINE or document.get("schema_version") != 1:
            raise CausalResultsValidationError(
                f"{description} has an invalid producer or schema"
            )

    for key, path in (
        ("probe_run_manifest_sha256", run_path),
        ("build_manifest_sha256", build_path),
        ("training_manifest_sha256", training_path),
    ):
        _require_hash(path, provenance.get(key), f"causal provenance {key}")

    current_probe_sources = current_pipeline_source_hashes()
    for document, description in (
        (run_manifest, "Probe run"),
        (build_manifest, "Build"),
        (training_manifest, "Training"),
    ):
        if document.get("source_code_sha256") != current_probe_sources:
            raise CausalResultsValidationError(
                f"{description} source hashes differ from current audited sources"
            )

    run_artifacts = run_manifest.get("artifacts")
    if not isinstance(run_artifacts, Mapping):
        raise CausalResultsValidationError("Probe run lacks artifact hashes")
    concepts_path = _inside(
        probe_run_dir, "frozen_config/concepts.yaml", "frozen concepts"
    )
    splits_path = _inside(probe_run_dir, "splits.parquet", "frozen splits")
    _require_hash(
        concepts_path, run_artifacts.get("concepts_yaml_sha256"), "frozen concepts"
    )
    _require_hash(splits_path, run_artifacts.get("splits_sha256"), "frozen splits")

    labels_relative = build_manifest.get("labels_manifest", "labels_manifest.json")
    labels_path = _inside(probe_run_dir, labels_relative, "labels manifest")
    labels_manifest = _read_json(labels_path, "labels manifest")
    if (
        labels_manifest.get("pipeline") != "validated_label_builder"
        or labels_manifest.get("status") != "complete"
    ):
        raise CausalResultsValidationError("Labels manifest is not a completed rebuild")
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
    _require_hash(
        labels_path,
        build_manifest.get("labels_manifest_sha256"),
        "labels manifest",
    )
    for key, expected in (
        ("run_manifest_sha256", _sha256(run_path)),
        ("split_manifest_sha256", run_artifacts.get("splits_sha256")),
        ("concepts_yaml_sha256", run_artifacts.get("concepts_yaml_sha256")),
    ):
        if labels_manifest.get(key) != expected:
            raise CausalResultsValidationError(
                f"Labels manifest provenance mismatch for {key}"
            )

    if build_manifest.get("split_manifest_sha256") != run_artifacts.get("splits_sha256"):
        raise CausalResultsValidationError("Build manifest is bound to different splits")
    if build_manifest.get("concepts_yaml_sha256") != run_artifacts.get(
        "concepts_yaml_sha256"
    ):
        raise CausalResultsValidationError("Build manifest is bound to different concepts")
    dataset_path = _inside(
        probe_run_dir,
        str(build_manifest.get("dataset", "dataset.parquet")),
        "rebuilt dataset",
    )
    _require_hash(dataset_path, build_manifest.get("dataset_sha256"), "rebuilt dataset")

    for key, expected in (
        ("dataset_sha256", build_manifest.get("dataset_sha256")),
        ("build_manifest_sha256", _sha256(build_path)),
        ("labels_manifest_sha256", _sha256(labels_path)),
        ("split_manifest_sha256", run_artifacts.get("splits_sha256")),
    ):
        if training_manifest.get(key) != expected:
            raise CausalResultsValidationError(
                f"Training manifest provenance mismatch for {key}"
            )
    try:
        fidelity_gate = _validate_checkpoint_fidelity_gate(
            probe_run_dir, run_manifest, training_manifest
        )
    except (ValueError, OSError) as exc:
        raise CausalResultsValidationError(
            f"Checkpoint activation-fidelity validation failed: {exc}"
        ) from exc
    if training_manifest.get("training_role") != "development":
        raise CausalResultsValidationError("Probe was not trained on development games only")

    for key, expected in (
        ("dataset_sha256", build_manifest.get("dataset_sha256")),
        ("labels_sha256", build_manifest.get("labels_sha256")),
        ("splits_sha256", run_artifacts.get("splits_sha256")),
        ("input_provenance_sha256", build_manifest.get("input_provenance_sha256")),
        (
            "trunk_identity_bytes_sha256",
            (build_manifest.get("input_provenance") or {}).get(
                "trunk_identity_bytes_sha256"
            ),
        ),
    ):
        if provenance.get(key) != expected:
            raise CausalResultsValidationError(
                f"Causal provenance differs from upstream for {key}"
            )

    concept = str(provenance.get("concept", ""))
    representation = str(provenance.get("representation", ""))
    if not concept or representation not in {"global", "local", "combined"}:
        raise CausalResultsValidationError("Invalid causal concept or representation")
    if concept not in set(map(str, training_manifest.get("concepts") or ())):
        raise CausalResultsValidationError("Causal concept was not in probe training")
    if representation not in set(map(str, training_manifest.get("representations") or ())):
        raise CausalResultsValidationError("Causal representation was not in probe training")

    training_artifacts = training_manifest.get("artifacts")
    if not isinstance(training_artifacts, Mapping):
        raise CausalResultsValidationError("Training manifest lacks artifact hashes")
    for relative, expected_hash in training_artifacts.items():
        artifact_path = _inside(
            probe_run_dir, str(relative), "upstream training artifact"
        )
        _require_hash(
            artifact_path, expected_hash, f"upstream training artifact {relative}"
        )
    probe_relative = f"probes/{representation}/probe_{concept}.joblib"
    scaler_relative = f"probes/{representation}/scaler_{concept}.joblib"
    metadata_relative = f"probes/{representation}/probe_{concept}.meta.json"
    for relative, provenance_key in (
        (probe_relative, "probe_sha256"),
        (scaler_relative, "scaler_sha256"),
        (metadata_relative, "probe_metadata_sha256"),
    ):
        if relative not in training_artifacts:
            raise CausalResultsValidationError(
                f"Training manifest does not bind causal input {relative}"
            )
        path = _inside(probe_run_dir, relative, "causal probe input")
        _require_hash(path, training_artifacts[relative], f"training artifact {relative}")
        if provenance.get(provenance_key) != training_artifacts[relative]:
            raise CausalResultsValidationError(
                f"Causal provenance mismatch for {provenance_key}"
            )
    probe_metadata = _read_json(
        probe_run_dir / metadata_relative, "causal probe metadata"
    )
    contract = get_contract(concept)
    for source, description in (
        (provenance, "Causal provenance"),
        (probe_metadata, "Probe metadata"),
    ):
        if source.get("contract_id") != contract.definition_id:
            raise CausalResultsValidationError(
                f"{description} contract ID differs from canonical definition"
            )
        if source.get("contract_hash") != contract.contract_hash:
            raise CausalResultsValidationError(
                f"{description} contract hash differs from canonical definition"
            )
    if probe_metadata.get("representation") != representation:
        raise CausalResultsValidationError("Probe metadata representation mismatch")
    if probe_metadata.get("training_role") != "development":
        raise CausalResultsValidationError("Probe metadata training role is not development")
    if fidelity_gate is not None and probe_metadata.get(
        "checkpoint_activation_fidelity"
    ) != fidelity_gate:
        raise CausalResultsValidationError(
            "Causal probe metadata is not bound to the validated checkpoint "
            "activation-fidelity gate"
        )
    if set(probe_metadata.get("excluded_roles") or ()) != {
        "control_calibration", "causal_test"
    }:
        raise CausalResultsValidationError(
            "Probe metadata does not exclude both held-out causal roles"
        )

    checkpoint_value = provenance.get("checkpoint")
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        raise CausalResultsValidationError("Causal provenance lacks checkpoint path")
    checkpoint_path = Path(checkpoint_value).resolve()
    _reject_unsafe_path(checkpoint_path, "Causal checkpoint")
    _require_hash(checkpoint_path, provenance.get("checkpoint_sha256"), "causal checkpoint")

    fresh = run_manifest.get("fresh_holdout")
    causal_fresh = provenance.get("fresh_holdout")
    if not isinstance(fresh, Mapping) or not isinstance(causal_fresh, Mapping):
        raise CausalResultsValidationError(
            "Confirmatory reporting requires checkpoint-bound fresh-holdout provenance"
        )
    fresh_keys = (
        "cohort", "game_ids", "checkpoint_sha256", "protocol_manifest_sha256",
        "generator_source_sha256", "common_utils_source_sha256",
        "protocol_source_sha256", "rng_seed_set_sha256",
    )
    for key in fresh_keys:
        if causal_fresh.get(key) != fresh.get(key):
            raise CausalResultsValidationError(
                f"Causal fresh-holdout provenance mismatch for {key}"
            )
    if provenance.get("checkpoint_sha256") != fresh.get("checkpoint_sha256"):
        raise CausalResultsValidationError(
            "Causal checkpoint differs from the prospectively frozen holdout checkpoint"
        )
    protocol_value = fresh.get("protocol_path")
    if not isinstance(protocol_value, str) or not protocol_value:
        raise CausalResultsValidationError("Fresh holdout lacks frozen protocol path")
    protocol_path = Path(protocol_value).resolve()
    _reject_unsafe_path(protocol_path, "Frozen causal protocol")
    _require_hash(
        protocol_path,
        fresh.get("protocol_manifest_sha256"),
        "frozen causal protocol",
    )
    frozen_protocol = _read_json(protocol_path, "frozen causal protocol")
    if frozen_protocol.get("status") != "frozen_before_fresh_data_generation":
        raise CausalResultsValidationError(
            "Causal protocol was not frozen before holdout generation"
        )
    protocol_fresh = frozen_protocol.get("fresh_holdout")
    protocol_sources = frozen_protocol.get("source_sha256")
    if not isinstance(protocol_fresh, Mapping):
        raise CausalResultsValidationError("Frozen protocol lacks fresh_holdout")
    if not isinstance(protocol_sources, Mapping):
        raise CausalResultsValidationError("Frozen protocol lacks source hash mapping")
    if (
        protocol_fresh.get("cohort") != fresh.get("cohort")
        or int(protocol_fresh.get("games", -1)) != int(fresh.get("games", -2))
        or protocol_fresh.get("game_seed_set_sha256")
        != fresh.get("rng_seed_set_sha256")
        or (frozen_protocol.get("checkpoint") or {}).get("sha256")
        != fresh.get("checkpoint_sha256")
    ):
        raise CausalResultsValidationError(
            "Fresh-holdout declaration does not reproduce the frozen protocol"
        )
    if dict(fresh.get("protocol_source_sha256") or {}) != dict(protocol_sources):
        raise CausalResultsValidationError(
            "Run fresh-holdout source commitment differs from the frozen protocol"
        )
    if any(
        protocol_sources.get(identity) != observed
        for identity, observed in current_causal_sources.items()
    ):
        raise CausalResultsValidationError(
            "Current causal producer sources differ from prospectively frozen bytes"
        )
    reporter_identity = "daniele_experiment/validated_causal_results_report.py"
    if protocol_sources.get(reporter_identity) != _sha256(Path(__file__).resolve()):
        raise CausalResultsValidationError(
            "Current causal reporter differs from the prospectively frozen source"
        )
    if provenance.get("checkpoint_activation_fidelity") != fidelity_gate:
        raise CausalResultsValidationError(
            "Causal provenance is not bound to the checkpoint activation-fidelity gate"
        )

    summary_path = causal_dir / "summary.json"
    calibration_path = causal_dir / "control_calibration.json"
    positions_path = causal_dir / "selected_positions.parquet"
    rows_path = causal_dir / "causal_test_rows.parquet"
    summary = _read_json(summary_path, "causal summary")
    calibration = _read_json(calibration_path, "control calibration")
    if summary.get("pipeline") != SOURCE_PIPELINE or summary.get("schema_version") != 1:
        raise CausalResultsValidationError("Invalid causal summary producer or schema")
    if summary.get("concept") != concept or summary.get("representation") != representation:
        raise CausalResultsValidationError("Causal summary identity mismatch")
    if summary.get("contract_hash") != contract.contract_hash:
        raise CausalResultsValidationError("Causal summary contract hash mismatch")
    if summary.get("fresh_holdout") != provenance.get("fresh_holdout"):
        raise CausalResultsValidationError(
            "Causal summary and manifest bind different fresh holdouts"
        )
    if summary.get("checkpoint_activation_fidelity") != fidelity_gate:
        raise CausalResultsValidationError(
            "Causal summary is not bound to the checkpoint activation-fidelity gate"
        )
    if summary.get("source_code_sha256") != current_causal_sources:
        raise CausalResultsValidationError("Causal summary source hashes mismatch")
    if summary.get("producer_source_sha256") != current_causal_sources:
        raise CausalResultsValidationError("Causal summary source alias mismatch")
    if summary.get("final_evaluation_role") != "causal_test":
        raise CausalResultsValidationError("Causal summary is not a causal-test evaluation")
    _validate_frozen_causal_design(frozen_protocol, provenance, summary)
    if calibration.get("split_role") != "control_calibration":
        raise CausalResultsValidationError("Controls were not calibrated on calibration games")
    equivalence = _read_json(
        causal_dir / "policy_head_equivalence.json", "policy-head equivalence audit"
    )
    if equivalence.get("status") != "validated":
        raise CausalResultsValidationError("Policy-head equivalence audit did not validate")
    alignment = _read_json(
        causal_dir / "operational_alignment.json", "operational-alignment audit"
    )
    if (
        alignment.get("status") != "validated"
        or int(alignment.get("failed_positions", -1)) != 0
        or alignment.get("contract_id") != contract.definition_id
        or alignment.get("contract_hash") != contract.contract_hash
    ):
        raise CausalResultsValidationError(
            "Operational-alignment audit is failed or bound to another contract"
        )
    alignment_binding = causal_manifest.get("operational_alignment")
    if (
        not isinstance(alignment_binding, Mapping)
        or alignment_binding.get("status") != "validated"
        or alignment_binding.get("report_sha256")
        != _sha256(causal_dir / "operational_alignment.json")
        or int(alignment_binding.get("positions_checked", -1))
        != int(alignment.get("positions_checked", -2))
        or int(alignment_binding.get("failed_positions", -1)) != 0
    ):
        raise CausalResultsValidationError(
            "Causal manifest does not bind the validated operational-alignment audit"
        )

    selected_positions = pd.read_parquet(positions_path)
    rows = pd.read_parquet(rows_path)
    _validate_causal_tables(
        selected_positions,
        rows,
        probe_run_dir=probe_run_dir,
        splits_path=splits_path,
        calibration=calibration,
        summary=summary,
        representation=representation,
        fresh_holdout=fresh,
    )
    if int(alignment.get("positions_checked", -1)) != len(selected_positions):
        raise CausalResultsValidationError(
            "Operational-alignment coverage does not equal all selected positions"
        )
    alignment_positions = alignment.get("positions")
    if not isinstance(alignment_positions, list) or {
        str(record.get("position_id"))
        for record in alignment_positions
        if isinstance(record, Mapping)
    } != set(selected_positions["position_id"].astype(str)):
        raise CausalResultsValidationError(
            "Operational-alignment audit does not identify every selected position"
        )
    readout_columns = sorted(column[6:] for column in rows if column.startswith("delta_"))
    direction_specs = _readout_direction_specs(contract.name)
    if set(readout_columns) != set(direction_specs):
        raise CausalResultsValidationError(
            "Causal readout schema differs from the contract direction registry: "
            f"observed={readout_columns}, expected={sorted(direction_specs)}"
        )
    return ValidatedCausalInputs(
        causal_dir=causal_dir,
        probe_run_dir=probe_run_dir,
        causal_manifest=causal_manifest,
        run_manifest=run_manifest,
        build_manifest=build_manifest,
        labels_manifest=labels_manifest,
        training_manifest=training_manifest,
        probe_metadata=probe_metadata,
        summary=summary,
        calibration=calibration,
        selected_positions=selected_positions,
        rows=rows,
        concept=concept,
        representation=representation,
        contract_name=contract.name,
        readout_directions=direction_specs,
        frozen_protocol=frozen_protocol,
    )


def _validate_causal_tables(
    selected: pd.DataFrame,
    rows: pd.DataFrame,
    *,
    probe_run_dir: Path,
    splits_path: Path,
    calibration: Mapping[str, Any],
    summary: Mapping[str, Any],
    representation: str,
    fresh_holdout: Mapping[str, Any],
) -> None:
    required_selected = {
        "position_id", "game_id", "move_number", "split_role",
        "causal_protocol_role", "selection_stratum", "selection_quota",
        "selection_unit",
    }
    missing = sorted(required_selected - set(selected.columns))
    if missing or selected.empty:
        raise CausalResultsValidationError(
            f"Selected-position table is empty or lacks columns: {missing}"
        )
    if selected["position_id"].astype(str).duplicated().any():
        raise CausalResultsValidationError("Selected positions contain duplicate IDs")
    if not np.array_equal(
        selected["split_role"].astype(str).to_numpy(),
        selected["causal_protocol_role"].astype(str).to_numpy(),
    ):
        raise CausalResultsValidationError(
            "Selected-position split_role and causal_protocol_role disagree"
        )
    allowed_roles = {"control_calibration", "causal_test"}
    if set(selected["causal_protocol_role"].astype(str)) != allowed_roles:
        raise CausalResultsValidationError(
            "Selected positions must contain both and only held-out causal roles"
        )
    if selected["game_id"].astype(str).duplicated().any():
        raise CausalResultsValidationError(
            "Confirmatory selection must contain exactly one position per holdout game"
        )
    if set(selected["selection_unit"].astype(str)) != {"one_position_per_game"}:
        raise CausalResultsValidationError(
            "Selected positions do not preserve the one-position-per-game unit"
        )
    strata = pd.to_numeric(selected["selection_stratum"], errors="coerce")
    quotas = pd.to_numeric(selected["selection_quota"], errors="coerce")
    if (
        strata.isna().any()
        or quotas.isna().any()
        or set(strata.astype(int)) != {0, 1}
        or (quotas <= 0).any()
    ):
        raise CausalResultsValidationError("Invalid selected label strata/quotas")
    for role, part in selected.groupby("causal_protocol_role", sort=False):
        observed_counts = part["selection_stratum"].astype(int).value_counts()
        for label, label_rows in part.groupby("selection_stratum", sort=False):
            if set(label_rows["selection_quota"].astype(int)) != {
                int(observed_counts.loc[int(label)])
            }:
                raise CausalResultsValidationError(
                    f"Selection quota metadata is inconsistent in {role}"
                )
    if summary.get("selection_unit") != "exactly_one_position_per_fresh_holdout_game":
        raise CausalResultsValidationError(
            "Causal summary does not declare the frozen one-position-per-game unit"
        )

    splits = pd.read_parquet(splits_path)
    if not {"game_id", "split_role"}.issubset(splits.columns):
        raise CausalResultsValidationError("Frozen splits lack game_id/split_role")
    if splits["game_id"].astype(str).duplicated().any():
        raise CausalResultsValidationError("Frozen splits contain duplicate games")
    role_by_game = splits.set_index(splits["game_id"].astype(str))["split_role"].to_dict()
    observed_roles = selected["game_id"].astype(str).map(role_by_game)
    if observed_roles.isna().any() or not np.array_equal(
        observed_roles.astype(str).to_numpy(),
        selected["causal_protocol_role"].astype(str).to_numpy(),
    ):
        raise CausalResultsValidationError(
            "Selected positions disagree with frozen game roles"
        )
    frozen_holdout_games = set(map(str, fresh_holdout.get("game_ids") or ()))
    selected_games = set(selected["game_id"].astype(str))
    split_holdout_games = set(
        splits.loc[splits["split_role"].isin(allowed_roles), "game_id"].astype(str)
    )
    if not frozen_holdout_games:
        raise CausalResultsValidationError("Fresh holdout cohort has no game IDs")
    if selected_games != frozen_holdout_games:
        raise CausalResultsValidationError(
            "Selected games do not exactly cover the fresh holdout cohort"
        )
    if split_holdout_games != frozen_holdout_games:
        raise CausalResultsValidationError(
            "Frozen splits do not exactly assign the declared fresh holdout cohort"
        )

    required_rows = {
        "position_id", "game_id", "move_number", "split_role", "label",
        "control_id", "control_kind", "nominal_dose", "dose_multiplier",
        "effective_dose", "policy_js", "policy_l1", "top_move_flip",
        "calibration_match_succeeded", "calibration_match_status",
        "calibration_target_mean_policy_js",
        "calibration_achieved_mean_policy_js",
    }
    missing = sorted(required_rows - set(rows.columns))
    if missing or rows.empty:
        raise CausalResultsValidationError(
            f"Causal-test row table is empty or lacks columns: {missing}"
        )
    if set(rows["split_role"].astype(str)) != {"causal_test"}:
        raise CausalResultsValidationError(
            "Reporter accepts only rows explicitly marked causal_test"
        )
    test_selected = selected.loc[
        selected["causal_protocol_role"].astype(str).eq("causal_test")
    ]
    test_ids = set(test_selected["position_id"].astype(str))
    if set(rows["position_id"].astype(str)) != test_ids:
        raise CausalResultsValidationError(
            "Causal-test rows do not exactly cover selected causal-test positions"
        )
    game_by_position = test_selected.set_index(
        test_selected["position_id"].astype(str)
    )["game_id"].astype(str).to_dict()
    if not np.array_equal(
        rows["position_id"].astype(str).map(game_by_position).to_numpy(),
        rows["game_id"].astype(str).to_numpy(),
    ):
        raise CausalResultsValidationError("Causal row game IDs disagree with selection")
    stratum_by_position = test_selected.set_index(
        test_selected["position_id"].astype(str)
    )["selection_stratum"].astype(int).to_dict()
    if not np.array_equal(
        rows["position_id"].astype(str).map(stratum_by_position).to_numpy(dtype=int),
        pd.to_numeric(rows["label"]).to_numpy(dtype=int),
    ):
        raise CausalResultsValidationError(
            "Causal row labels disagree with frozen selection strata"
        )
    for role, summary_key in (
        ("control_calibration", "calibration_selection_strata"),
        ("causal_test", "causal_test_selection_strata"),
    ):
        expected = {
            str(int(label)): int(count)
            for label, count in selected.loc[
                selected["causal_protocol_role"].astype(str).eq(role),
                "selection_stratum",
            ].astype(int).value_counts().sort_index().items()
        }
        if summary.get(summary_key) != expected:
            raise CausalResultsValidationError(
                f"Causal summary label strata differ for {role}"
            )

    numeric = [
        "label", "nominal_dose", "dose_multiplier", "effective_dose",
        "policy_js", "policy_l1", "top_move_flip",
        "calibration_target_mean_policy_js",
        "calibration_achieved_mean_policy_js",
    ]
    readouts = sorted(column[6:] for column in rows if column.startswith("delta_"))
    if not readouts:
        raise CausalResultsValidationError("Causal rows contain no behavioral readouts")
    for readout in readouts:
        for prefix in ("baseline_", "steered_", "delta_"):
            column = prefix + readout
            if column not in rows:
                raise CausalResultsValidationError(
                    f"Readout {readout!r} lacks {column!r}"
                )
            numeric.append(column)
    _finite_numeric(rows, numeric, "Causal-test rows")
    labels = set(pd.to_numeric(rows["label"]).astype(int))
    if labels != {0, 1}:
        raise CausalResultsValidationError(
            "Label-balanced causal reporting requires both binary label strata"
        )
    if (pd.to_numeric(rows["policy_js"]) < 0).any() or (
        pd.to_numeric(rows["policy_l1"]) < 0
    ).any():
        raise CausalResultsValidationError("Policy disruption cannot be negative")
    if not set(pd.to_numeric(rows["top_move_flip"]).astype(int)).issubset({0, 1}):
        raise CausalResultsValidationError("top_move_flip must be binary")
    multipliers = pd.to_numeric(rows["dose_multiplier"]).to_numpy(float)
    if (multipliers < 0).any():
        raise CausalResultsValidationError("Dose multipliers cannot be negative")
    if not np.allclose(
        pd.to_numeric(rows["effective_dose"]).to_numpy(float),
        pd.to_numeric(rows["nominal_dose"]).to_numpy(float) * multipliers,
        rtol=0.0,
        atol=1e-12,
    ):
        raise CausalResultsValidationError("Effective doses do not equal nominal*multiplier")
    for readout in readouts:
        baseline = pd.to_numeric(rows[f"baseline_{readout}"]).to_numpy(float)
        steered = pd.to_numeric(rows[f"steered_{readout}"]).to_numpy(float)
        delta = pd.to_numeric(rows[f"delta_{readout}"]).to_numpy(float)
        if not np.allclose(delta, steered - baseline, rtol=1e-10, atol=1e-12):
            raise CausalResultsValidationError(
                f"Saved deltas do not reproduce paired readout {readout!r}"
            )

    rows = rows.copy()
    rows["control_id"] = rows["control_id"].astype(str)
    rows["control_kind"] = rows["control_kind"].astype(str)
    allowed_kinds = {"trained", "spatial_shuffle", "random_direction"}
    if not set(rows["control_kind"]).issubset(allowed_kinds):
        raise CausalResultsValidationError("Unknown causal control kind")
    trained = rows.loc[rows["control_kind"].eq("trained")]
    if set(trained["control_id"]) != {"trained"}:
        raise CausalResultsValidationError("Trained rows have an invalid control ID")
    if (rows["control_id"].eq("trained") != rows["control_kind"].eq("trained")).any():
        raise CausalResultsValidationError("Trained control identity/kind mismatch")
    control_kind_counts = (
        rows.loc[~rows["control_kind"].eq("trained"), ["control_id", "control_kind"]]
        .drop_duplicates()
        .groupby("control_kind")["control_id"].nunique()
        .to_dict()
    )
    random_count = int(control_kind_counts.get("random_direction", 0))
    shuffle_count = int(control_kind_counts.get("spatial_shuffle", 0))
    if random_count < MIN_RANDOM_DIRECTIONS:
        raise CausalResultsValidationError(
            f"At least {MIN_RANDOM_DIRECTIONS} random directions are required"
        )
    spatial_applicable = bool(summary.get("spatial_controls_applicable"))
    if spatial_applicable and shuffle_count < MIN_SPATIAL_SHUFFLES:
        raise CausalResultsValidationError(
            f"At least {MIN_SPATIAL_SHUFFLES} spatial shuffles are required"
        )
    if not spatial_applicable and shuffle_count:
        raise CausalResultsValidationError(
            "Spatial shuffles are present although the summary marks them inapplicable"
        )
    if representation == "global" and spatial_applicable:
        raise CausalResultsValidationError("Global intervention cannot have spatial controls")

    doses = sorted(pd.to_numeric(rows["nominal_dose"]).astype(float).unique())
    if not any(dose != 0.0 for dose in doses):
        raise CausalResultsValidationError("Causal evaluation has no nonzero dose")
    summary_doses = sorted(map(float, summary.get("doses") or ()))
    if doses != summary_doses:
        raise CausalResultsValidationError("Causal row doses differ from summary doses")
    key_columns = ["control_id", "nominal_dose", "position_id"]
    if rows.duplicated(key_columns).any():
        raise CausalResultsValidationError("Duplicate control/dose/position causal rows")
    expected_positions = len(test_ids)
    for (control_id, dose), part in rows.groupby(
        ["control_id", "nominal_dose"], sort=False
    ):
        if len(part) != expected_positions or set(part["position_id"].astype(str)) != test_ids:
            raise CausalResultsValidationError(
                f"Control {control_id} dose {dose} does not cover the same test positions"
            )
        if part["game_id"].astype(str).nunique() < 2:
            raise CausalResultsValidationError(
                "Game-cluster intervals require at least two causal-test games"
            )
    expected_groups = (
        rows[["control_id", "control_kind"]].drop_duplicates().shape[0] * len(doses)
    )
    if rows.groupby(["control_id", "nominal_dose"]).ngroups != expected_groups:
        raise CausalResultsValidationError("Causal control-by-dose grid is incomplete")

    # A paired comparison requires a single immutable baseline per position/readout.
    for readout in readouts:
        spread = rows.groupby("position_id")[f"baseline_{readout}"].agg(
            lambda values: float(np.max(values) - np.min(values))
        )
        if (spread > 1e-12).any():
            raise CausalResultsValidationError(
                f"Baseline {readout!r} changes across interventions"
            )

    matches = calibration.get("matches")
    if not isinstance(matches, list):
        raise CausalResultsValidationError("Control calibration lacks match records")
    match_by_key: Dict[Tuple[str, float], Mapping[str, Any]] = {}
    raw_targets = calibration.get("trained_targets_by_nominal_dose")
    if not isinstance(raw_targets, Mapping):
        raise CausalResultsValidationError(
            "Control calibration lacks trained disruption targets"
        )
    try:
        targets = {float(key): value for key, value in raw_targets.items()}
    except (TypeError, ValueError) as exc:
        raise CausalResultsValidationError(
            "Invalid trained calibration target dose"
        ) from exc
    if set(targets) != set(doses):
        raise CausalResultsValidationError(
            "Trained calibration targets do not exactly cover causal doses"
        )
    if (
        int(calibration.get("games", -1))
        != int(
            selected.loc[
                selected["causal_protocol_role"].astype(str).eq(
                    "control_calibration"
                ),
                "game_id",
            ].astype(str).nunique()
        )
        or int(calibration.get("positions", -1))
        != int(
            selected["causal_protocol_role"].astype(str).eq(
                "control_calibration"
            ).sum()
        )
    ):
        raise CausalResultsValidationError(
            "Calibration counts differ from selected calibration positions"
        )
    for match in matches:
        if not isinstance(match, Mapping):
            raise CausalResultsValidationError("Invalid calibration match record")
        key = (str(match.get("control_id")), float(match.get("nominal_dose")))
        if key in match_by_key:
            raise CausalResultsValidationError(f"Duplicate calibration match {key}")
        if not bool(match.get("matched")):
            raise CausalResultsValidationError(
                f"Calibration did not succeed for control {key[0]} at dose {key[1]}"
            )
        target = targets.get(key[1])
        if (
            not isinstance(target, Mapping)
            or not math.isclose(
                float(match.get("target_mean_policy_js", math.nan)),
                float(target.get("mean_policy_js", math.nan)),
                rel_tol=1e-10,
                abs_tol=1e-12,
            )
        ):
            raise CausalResultsValidationError(
                f"Control match target differs from trained target for dose {key[1]}"
            )
        match_by_key[key] = match
    controls = rows.loc[~rows["control_kind"].eq("trained"), "control_id"].unique()
    expected_matches = {(str(control), float(dose)) for control in controls for dose in doses}
    if set(match_by_key) != expected_matches:
        raise CausalResultsValidationError(
            "Calibration match records do not exactly cover the control/dose grid"
        )
    for dose, part in rows.loc[
        rows["control_kind"].eq("trained")
    ].groupby("nominal_dose", sort=False):
        target_js = float(targets[float(dose)]["mean_policy_js"])
        if (
            not part["calibration_match_succeeded"].astype(bool).all()
            or not np.allclose(
                pd.to_numeric(part["calibration_target_mean_policy_js"]).to_numpy(float),
                target_js,
                rtol=1e-10,
                atol=1e-12,
            )
            or not np.allclose(
                pd.to_numeric(part["calibration_achieved_mean_policy_js"]).to_numpy(float),
                target_js,
                rtol=1e-10,
                atol=1e-12,
            )
        ):
            raise CausalResultsValidationError(
                f"Trained rows do not reproduce calibration target at dose {dose}"
            )
    for (control_id, dose), part in rows.loc[
        ~rows["control_kind"].eq("trained")
    ].groupby(["control_id", "nominal_dose"], sort=False):
        match = match_by_key[(str(control_id), float(dose))]
        if not part["calibration_match_succeeded"].astype(bool).all():
            raise CausalResultsValidationError(
                f"Rows do not retain calibration success for {control_id}/{dose}"
            )
        if set(part["calibration_match_status"].astype(str)) != {str(match["status"])}:
            raise CausalResultsValidationError(
                f"Row calibration status differs from calibration artifact for {control_id}/{dose}"
            )
        for column, key in (
            ("dose_multiplier", "dose_multiplier"),
            ("calibration_target_mean_policy_js", "target_mean_policy_js"),
            ("calibration_achieved_mean_policy_js", "achieved_mean_policy_js"),
        ):
            if column not in part:
                raise CausalResultsValidationError(
                    f"Causal rows lack calibration field {column!r}"
                )
            if not np.allclose(
                pd.to_numeric(part[column]).to_numpy(float),
                float(match[key]),
                rtol=1e-10,
                atol=1e-12,
            ):
                raise CausalResultsValidationError(
                    f"Row {column} differs from calibration artifact for {control_id}/{dose}"
                )


def _derived_seed(base_seed: int, *parts: Any) -> int:
    digest = hashlib.sha256()
    digest.update(str(int(base_seed)).encode("ascii"))
    for part in parts:
        payload = str(part).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return int.from_bytes(digest.digest()[:8], "big") & 0x7FFFFFFFFFFFFFFF


def _interval(
    values: np.ndarray,
    *,
    point: Optional[float],
    replicates: int,
    confidence_level: float,
) -> Optional[Dict[str, Any]]:
    if point is None:
        return None
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    required = max(1, int(math.ceil(0.8 * replicates)))
    if len(finite) < required:
        raise CausalResultsValidationError(
            "Too few valid game-cluster bootstrap draws for a required interval: "
            f"{len(finite)}/{replicates}"
        )
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(finite, [alpha, 1.0 - alpha])
    return {
        "method": "percentile_game_cluster_bootstrap",
        "point_estimate": float(point),
        "bootstrap_mean": float(finite.mean()),
        "lower": float(lower),
        "upper": float(upper),
        "confidence_level": float(confidence_level),
        "replicates_requested": int(replicates),
        "replicates_valid": int(len(finite)),
    }


def _point_effect(
    frame: pd.DataFrame,
    *,
    readout: str,
    expected_sign: Optional[int],
) -> Dict[str, Any]:
    baseline = frame[f"baseline_{readout}"].to_numpy(float)
    steered = frame[f"steered_{readout}"].to_numpy(float)
    delta = frame[f"delta_{readout}"].to_numpy(float)
    mean_baseline = float(baseline.mean())
    mean_steered = float(steered.mean())
    paired_delta = float(delta.mean())
    relative = None
    if not math.isclose(mean_baseline, 0.0, rel_tol=0.0, abs_tol=1e-15):
        relative = float(paired_delta / mean_baseline)
    oriented = None if expected_sign is None else expected_sign * delta
    return {
        "n_positions": int(frame["position_id"].astype(str).nunique()),
        "n_games": int(frame["game_id"].astype(str).nunique()),
        "mean_baseline": mean_baseline,
        "mean_steered": mean_steered,
        # "Absolute" means original readout units (rather than a relative
        # percentage); the sign is retained to distinguish increases/decreases.
        "paired_absolute_delta": paired_delta,
        "paired_absolute_delta_magnitude": abs(paired_delta),
        "relative_delta_from_mean_baseline": relative,
        "predicted_direction_proportion": (
            None if oriented is None else float(np.mean(oriented > 0.0))
        ),
        "zero_delta_proportion": float(np.mean(delta == 0.0)),
        "direction_test_is_strict": True,
    }


def _bootstrap_views(
    frame: pd.DataFrame,
    *,
    readout: str,
    replicates: int,
    seed: int,
    confidence_level: float,
    expected_sign: Optional[int],
) -> Dict[str, Any]:
    """Resample whole games and return pooled, by-label, and balanced effects."""

    if replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    games = frame["game_id"].astype(str)
    labels = frame["label"].to_numpy(int)
    game_label_counts = frame.assign(_game_id=games).groupby("_game_id")[
        "label"
    ].nunique()
    if (game_label_counts != 1).any():
        raise CausalResultsValidationError(
            "A game cluster spans label strata; the frozen one-position-per-game "
            "stratified bootstrap cannot be reproduced"
        )
    rng = np.random.default_rng(int(seed))
    weights = np.zeros((int(replicates), len(frame)), dtype=float)
    for label in (0, 1):
        mask = labels == label
        stratum_games = games.loc[mask]
        unique_games = sorted(stratum_games.unique())
        if len(unique_games) < 2:
            raise CausalResultsValidationError(
                f"Game-cluster bootstrap requires at least two games in label stratum {label}"
            )
        game_codes = pd.Categorical(stratum_games, categories=unique_games).codes
        cluster_counts = rng.multinomial(
            len(unique_games),
            np.full(len(unique_games), 1.0 / len(unique_games)),
            size=int(replicates),
        )
        weights[:, np.flatnonzero(mask)] = cluster_counts[:, game_codes]
    baseline = frame[f"baseline_{readout}"].to_numpy(float)
    steered = frame[f"steered_{readout}"].to_numpy(float)
    delta = frame[f"delta_{readout}"].to_numpy(float)

    def draws(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        subset_weights = weights[:, mask]
        denominator = subset_weights.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            baseline_draw = (subset_weights @ baseline[mask]) / denominator
            steered_draw = (subset_weights @ steered[mask]) / denominator
            delta_draw = (subset_weights @ delta[mask]) / denominator
            relative_draw = delta_draw / baseline_draw
        for values in (baseline_draw, steered_draw, delta_draw, relative_draw):
            values[denominator <= 0.0] = np.nan
        relative_draw[np.isclose(baseline_draw, 0.0, rtol=0.0, atol=1e-15)] = np.nan
        return baseline_draw, steered_draw, delta_draw, relative_draw

    def attach_intervals(
        point: Dict[str, Any], arrays: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> Dict[str, Any]:
        baseline_draw, steered_draw, delta_draw, relative_draw = arrays
        point = dict(point)
        point["game_cluster_bootstrap"] = {
            "mean_baseline": _interval(
                baseline_draw,
                point=point["mean_baseline"],
                replicates=replicates,
                confidence_level=confidence_level,
            ),
            "mean_steered": _interval(
                steered_draw,
                point=point["mean_steered"],
                replicates=replicates,
                confidence_level=confidence_level,
            ),
            "paired_absolute_delta": _interval(
                delta_draw,
                point=point["paired_absolute_delta"],
                replicates=replicates,
                confidence_level=confidence_level,
            ),
            "relative_delta_from_mean_baseline": _interval(
                relative_draw,
                point=point["relative_delta_from_mean_baseline"],
                replicates=replicates,
                confidence_level=confidence_level,
            ),
        }
        return point

    pooled_mask = np.ones(len(frame), dtype=bool)
    pooled = attach_intervals(
        _point_effect(frame, readout=readout, expected_sign=expected_sign),
        draws(pooled_mask),
    )
    by_label: Dict[str, Any] = {}
    label_arrays: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    label_points: Dict[int, Dict[str, Any]] = {}
    for label in (0, 1):
        mask = labels == label
        if not mask.any():  # guarded in validation, retained for direct unit tests.
            raise CausalResultsValidationError(f"Causal sample lacks label stratum {label}")
        part = frame.loc[mask]
        label_points[label] = _point_effect(
            part, readout=readout, expected_sign=expected_sign
        )
        label_arrays[label] = draws(mask)
        by_label[str(label)] = attach_intervals(label_points[label], label_arrays[label])

    balanced_point = {
        key: 0.5 * (float(label_points[0][key]) + float(label_points[1][key]))
        for key in ("mean_baseline", "mean_steered", "paired_absolute_delta")
    }
    balanced_point["paired_absolute_delta_magnitude"] = abs(
        balanced_point["paired_absolute_delta"]
    )
    balanced_point["relative_delta_from_mean_baseline"] = (
        balanced_point["paired_absolute_delta"] / balanced_point["mean_baseline"]
        if not math.isclose(
            balanced_point["mean_baseline"], 0.0, rel_tol=0.0, abs_tol=1e-15
        )
        else None
    )
    balanced_point.update({
        "n_positions": int(frame["position_id"].astype(str).nunique()),
        "n_games": int(frame["game_id"].astype(str).nunique()),
        "label_weighting": {"0": 0.5, "1": 0.5},
        "predicted_direction_proportion": (
            None
            if expected_sign is None
            else 0.5 * (
                label_points[0]["predicted_direction_proportion"]
                + label_points[1]["predicted_direction_proportion"]
            )
        ),
        "zero_delta_proportion": 0.5 * (
            label_points[0]["zero_delta_proportion"]
            + label_points[1]["zero_delta_proportion"]
        ),
        "direction_test_is_strict": True,
    })
    balanced_arrays = tuple(
        0.5 * (label_arrays[0][index] + label_arrays[1][index])
        for index in range(3)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        balanced_relative = balanced_arrays[2] / balanced_arrays[0]
    balanced_relative[np.isclose(
        balanced_arrays[0], 0.0, rtol=0.0, atol=1e-15
    )] = np.nan
    label_balanced = attach_intervals(
        balanced_point,
        (
            balanced_arrays[0], balanced_arrays[1], balanced_arrays[2],
            balanced_relative,
        ),
    )
    return {
        "sample_position_weighted": pooled,
        "by_label": by_label,
        "label_balanced": label_balanced,
        "bootstrap_seed": int(seed),
        "cluster_sampling": (
            "within each frozen label stratum, sample the observed number of "
            "causal-test games with replacement; every position from a sampled "
            "game receives the same multiplicity"
        ),
    }


def _label_balanced_mean(frame: pd.DataFrame, column: str) -> float:
    means = frame.groupby(frame["label"].astype(int))[column].mean()
    if set(map(int, means.index)) != {0, 1}:
        raise CausalResultsValidationError(
            "Label-balanced statistic requires both binary strata"
        )
    return float(0.5 * (float(means.loc[0]) + float(means.loc[1])))


def _ols_slope(predictor: Sequence[float], outcome: Sequence[float]) -> float:
    """Least-squares slope with an intercept."""

    x = np.asarray(predictor, dtype=float)
    y = np.asarray(outcome, dtype=float)
    if (
        x.ndim != 1
        or y.ndim != 1
        or len(x) != len(y)
        or len(x) < 2
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise CausalResultsValidationError("Primary slope inputs are invalid")
    centered = x - x.mean()
    denominator = float(centered @ centered)
    if denominator <= 0.0:
        raise CausalResultsValidationError(
            "Primary slope needs at least two distinct frozen doses"
        )
    return float(centered @ y / denominator)


def _label_balanced_slope(
    frame: pd.DataFrame,
    *,
    readout: str,
    doses: Sequence[float],
) -> Tuple[float, Mapping[str, float]]:
    dose_means: Dict[str, float] = {}
    values = []
    for dose in doses:
        part = frame.loc[
            pd.to_numeric(frame["nominal_dose"]).astype(float).eq(float(dose))
        ]
        if part.empty:
            raise CausalResultsValidationError(
                f"Primary slope lacks frozen dose {dose}"
            )
        value = _label_balanced_mean(part, f"delta_{readout}")
        dose_means[str(float(dose))] = value
        values.append(value)
    return _ols_slope(doses, values), dose_means


def _bootstrap_label_balanced_slope(
    frame: pd.DataFrame,
    *,
    readout: str,
    doses: Sequence[float],
    replicates: int,
    seed: int,
    confidence_level: float,
) -> Dict[str, Any]:
    """Bootstrap the all-dose slope with shared game draws across doses."""

    doses = tuple(map(float, doses))
    delta_column = f"delta_{readout}"
    reference_ids: Optional[Tuple[str, ...]] = None
    for dose in doses:
        part = frame.loc[
            pd.to_numeric(frame["nominal_dose"]).astype(float).eq(dose)
        ]
        ids = tuple(sorted(part["position_id"].astype(str)))
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise CausalResultsValidationError(
                "Primary all-dose slope does not use identical positions at every dose"
            )
    point, dose_means = _label_balanced_slope(
        frame, readout=readout, doses=doses
    )
    rng = np.random.default_rng(int(seed))
    label_draw_means = []
    for label in (0, 1):
        part = frame.loc[frame["label"].astype(int).eq(label)].copy()
        pivot = part.pivot(
            index="game_id", columns="nominal_dose", values=delta_column
        )
        pivot.index = pivot.index.astype(str)
        pivot = pivot.sort_index()
        missing_doses = sorted(set(doses) - set(map(float, pivot.columns)))
        if missing_doses or pivot.isna().any().any():
            raise CausalResultsValidationError(
                f"Primary slope label stratum {label} lacks a complete game/dose grid"
            )
        pivot = pivot.loc[:, list(doses)]
        n_games = len(pivot)
        if n_games < 2:
            raise CausalResultsValidationError(
                f"Primary slope bootstrap needs two games in label stratum {label}"
            )
        counts = rng.multinomial(
            n_games,
            np.full(n_games, 1.0 / n_games),
            size=int(replicates),
        )
        label_draw_means.append(counts @ pivot.to_numpy(float) / n_games)
    balanced_draw_means = 0.5 * (label_draw_means[0] + label_draw_means[1])
    x = np.asarray(doses, dtype=float)
    centered = x - x.mean()
    denominator = float(centered @ centered)
    slope_draws = balanced_draw_means @ centered / denominator
    return {
        "statistic": "label_balanced_ols_slope_across_all_frozen_doses",
        "predictor": "nominal_dose",
        "intercept": True,
        "doses": list(doses),
        "label_weights": {"0": 0.5, "1": 0.5},
        "label_balanced_mean_delta_by_dose": dose_means,
        "slope": float(point),
        "game_cluster_bootstrap": _interval(
            slope_draws,
            point=point,
            replicates=int(replicates),
            confidence_level=float(confidence_level),
        ),
        "bootstrap_seed": int(seed),
        "resampling": (
            "sample games within each frozen label stratum and reuse each game's "
            "bootstrap multiplicity across every dose"
        ),
    }


def _empirical_control_comparison(
    trained_effect: float,
    control_effects: Sequence[float],
    *,
    expected_sign: int,
    family: str,
    statistic: str = "label_balanced_mean_paired_delta",
) -> Dict[str, Any]:
    """One-sided finite-control comparison in the predeclared direction."""

    values = np.asarray(control_effects, dtype=float)
    if values.ndim != 1 or len(values) < 1 or not np.isfinite(values).all():
        raise CausalResultsValidationError(
            f"Invalid empirical {family} control distribution"
        )
    if expected_sign not in {-1, 1}:
        raise ValueError("expected_sign must be -1 or +1 for a nonzero dose")
    oriented_trained = float(expected_sign * float(trained_effect))
    oriented_controls = expected_sign * values
    at_least = int(np.count_nonzero(oriented_controls >= oriented_trained))
    strictly_less = int(np.count_nonzero(oriented_controls < oriented_trained))
    count = int(len(values))
    empirical_p = float((1 + at_least) / (1 + count))
    conservative_percentile = float(100.0 * (1 + strictly_less) / (1 + count))
    quantile_levels = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
    raw_quantiles = np.quantile(values, quantile_levels)
    oriented_quantiles = np.quantile(oriented_controls, quantile_levels)
    labels = ("0", "0.05", "0.25", "0.5", "0.75", "0.95", "1")
    return {
        "control_family": family,
        "statistic": statistic,
        "alternative": "trained effect is farther in the dose/readout-predicted direction",
        "expected_raw_sign": "increase" if expected_sign > 0 else "decrease",
        "trained_effect": float(trained_effect),
        "oriented_trained_effect": oriented_trained,
        "n_controls": count,
        "n_controls_at_least_as_extreme": at_least,
        "conservative_rank_from_most_extreme": int(1 + at_least),
        "conservative_empirical_percentile": conservative_percentile,
        "one_sided_empirical_p": empirical_p,
        "p_formula": "(1 + #controls at least as extreme) / (1 + n_controls)",
        "minimum_attainable_p": float(1.0 / (1 + count)),
        "raw_control_effect": {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if count > 1 else None,
            "quantiles": {
                label: float(value) for label, value in zip(labels, raw_quantiles)
            },
        },
        "direction_oriented_control_effect": {
            "mean": float(oriented_controls.mean()),
            "sd": float(oriented_controls.std(ddof=1)) if count > 1 else None,
            "quantiles": {
                label: float(value)
                for label, value in zip(labels, oriented_quantiles)
            },
        },
    }


def _primary_confirmatory_test(
    validated: ValidatedCausalInputs,
    rows: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    confidence_level: float,
    inference_conforms: bool,
) -> Dict[str, Any]:
    """Evaluate the one prospectively frozen headline statistic and conjunction."""

    causal = validated.frozen_protocol.get("causal") or {}
    primary = causal.get("primary_hypothesis") or {}
    identity = {
        "concept": validated.concept,
        "representation": validated.representation,
    }
    expected_identity = {
        "concept": str(primary.get("concept")),
        "representation": str(primary.get("representation")),
    }
    if identity != expected_identity:
        return {
            "status": "not_applicable_secondary_or_nonprimary_output",
            "sole_confirmatory_test": True,
            "frozen_primary_identity": expected_identity,
            "current_identity": identity,
            "headline_decision": None,
        }

    readout = str(primary.get("readout"))
    if readout not in validated.readout_directions:
        raise CausalResultsValidationError(
            "Frozen primary readout is absent from causal output"
        )
    if int(validated.readout_directions[readout]["positive_label_sign"]) != 1:
        raise CausalResultsValidationError(
            "Frozen primary slope expects a positive-oriented readout"
        )
    doses = tuple(map(float, causal.get("doses") or ()))
    trained = rows.loc[rows["control_kind"].astype(str).eq("trained")]
    trained_statistic = _bootstrap_label_balanced_slope(
        trained,
        readout=readout,
        doses=doses,
        replicates=int(bootstrap_replicates),
        seed=_derived_seed(bootstrap_seed, "sole-primary-slope", readout),
        confidence_level=float(confidence_level),
    )
    trained_slope = float(trained_statistic["slope"])
    family_comparisons: Dict[str, Any] = {}
    controls = rows.loc[~rows["control_kind"].astype(str).eq("trained")]
    for family, family_rows in controls.groupby("control_kind", sort=True):
        slopes = []
        for _control_id, control_rows in family_rows.groupby("control_id", sort=True):
            slope, _dose_means = _label_balanced_slope(
                control_rows, readout=readout, doses=doses
            )
            slopes.append(slope)
        family_comparisons[str(family)] = _empirical_control_comparison(
            trained_slope,
            slopes,
            expected_sign=1,
            family=str(family),
            statistic="label_balanced_ols_slope_across_all_frozen_doses",
        )
    required_families = {"random_direction", "spatial_shuffle"}
    if set(family_comparisons) != required_families:
        raise CausalResultsValidationError(
            "Frozen primary conjunction requires both random-direction and "
            "spatial-shuffle empirical null families"
        )
    alpha = float((primary.get("decision_rule") or {})["alpha"])
    criteria = {
        "trained_slope_strictly_positive": trained_slope > 0.0,
        "random_direction_one_sided_empirical_p_at_most_alpha": (
            float(family_comparisons["random_direction"]["one_sided_empirical_p"])
            <= alpha
        ),
        "spatial_shuffle_one_sided_empirical_p_at_most_alpha": (
            float(family_comparisons["spatial_shuffle"]["one_sided_empirical_p"])
            <= alpha
        ),
    }
    passed = all(criteria.values())
    if not inference_conforms:
        status = "diagnostic_protocol_deviation"
        decision = None
    else:
        status = "confirmatory_complete"
        decision = (
            "passes_predeclared_headline_support_criterion"
            if passed
            else "does_not_pass_predeclared_headline_support_criterion"
        )
    return {
        "status": status,
        "sole_confirmatory_test": True,
        "frozen_primary_identity": {
            **expected_identity,
            "readout": readout,
        },
        "statistic": trained_statistic,
        "empirical_nulls": family_comparisons,
        "decision_rule": {
            "alpha": alpha,
            "logic": (
                "trained slope > 0 AND random-direction one-sided empirical p <= "
                "alpha AND spatial-shuffle one-sided empirical p <= alpha"
            ),
            "criteria": criteria,
            "all_criteria_satisfied": bool(passed),
        },
        "headline_decision": decision,
        "no_substitution": (
            "Individual doses, secondary readouts, and minimum p-values are "
            "descriptive/exploratory and cannot replace this conjunction."
        ),
    }


def _calibration_index(calibration: Mapping[str, Any]) -> Dict[Tuple[str, float], Mapping[str, Any]]:
    result = {}
    for record in calibration["matches"]:
        key = (str(record["control_id"]), float(record["nominal_dose"]))
        result[key] = record
    return result


def _control_disruption_diagnostics(
    rows: pd.DataFrame,
    calibration: Mapping[str, Any],
) -> Sequence[Mapping[str, Any]]:
    """Keep calibration success separate from untouched-test residual mismatch."""

    matches = _calibration_index(calibration)
    trained_means = (
        rows.loc[rows["control_kind"].astype(str).eq("trained")]
        .groupby("nominal_dose")[["policy_js", "policy_l1"]]
        .mean()
    )
    records = []
    controls = rows.loc[~rows["control_kind"].astype(str).eq("trained")]
    for (control_id, kind, dose), group in controls.groupby(
        ["control_id", "control_kind", "nominal_dose"], sort=True
    ):
        dose = float(dose)
        match = matches[(str(control_id), dose)]
        test_js = float(group["policy_js"].mean())
        test_l1 = float(group["policy_l1"].mean())
        trained_js = float(trained_means.loc[dose, "policy_js"])
        trained_l1 = float(trained_means.loc[dose, "policy_l1"])
        records.append({
            "control_id": str(control_id),
            "control_kind": str(kind),
            "nominal_dose": dose,
            "effective_dose": float(group["effective_dose"].iloc[0]),
            "calibration": {
                "split_role": "control_calibration",
                "success": bool(match["matched"]),
                "status": str(match["status"]),
                "target_mean_policy_js": float(match["target_mean_policy_js"]),
                "achieved_mean_policy_js": float(match["achieved_mean_policy_js"]),
                "absolute_js_error": float(match["absolute_js_error"]),
                "achieved_mean_policy_l1": (
                    None
                    if match.get("achieved_mean_policy_l1") is None
                    else float(match["achieved_mean_policy_l1"])
                ),
                "dose_multiplier_frozen_after_calibration": float(
                    match["dose_multiplier"]
                ),
            },
            "causal_test_observed_disruption": {
                "control_mean_policy_js": test_js,
                "trained_mean_policy_js": trained_js,
                "js_residual_control_minus_trained": test_js - trained_js,
                "absolute_js_residual": abs(test_js - trained_js),
                "control_mean_policy_l1": test_l1,
                "trained_mean_policy_l1": trained_l1,
                "l1_residual_control_minus_trained": test_l1 - trained_l1,
                "absolute_l1_residual": abs(test_l1 - trained_l1),
                "matching_claim": "none_on_causal_test",
                "interpretation": (
                    "Diagnostic after applying the frozen calibration multiplier; "
                    "the causal-test disruption was not re-matched."
                ),
            },
        })
    return records


def generate_causal_results_report(
    causal_dir: Path,
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: Optional[int] = None,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> Dict[str, Any]:
    validated = validate_causal_inputs(causal_dir)
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    causal_seed = int((validated.causal_manifest.get("provenance") or {}).get("seed"))
    base_seed = (
        int(bootstrap_seed)
        if bootstrap_seed is not None
        else _derived_seed(causal_seed, PIPELINE_NAME)
    )
    rows = validated.rows.copy()
    rows["nominal_dose"] = pd.to_numeric(rows["nominal_dose"]).astype(float)
    readouts = sorted(validated.readout_directions)
    trained = rows.loc[rows["control_kind"].astype(str).eq("trained")].copy()
    doses = sorted(trained["nominal_dose"].unique())
    trained_reports: Dict[str, Any] = {}
    empirical: Dict[str, Any] = {}
    frozen_inference = validated.frozen_protocol.get("inference") or {}
    inference_conforms = (
        int(bootstrap_replicates)
        == int(frozen_inference.get("bootstrap_replicates", -1))
        and math.isclose(
            float(confidence_level),
            float(frozen_inference.get("confidence_level", math.nan)),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )

    for dose in doses:
        dose = float(dose)
        dose_key = str(dose)
        trained_dose = trained.loc[trained["nominal_dose"].eq(dose)].sort_values(
            "position_id"
        )
        dose_sign = 0 if dose == 0.0 else (1 if dose > 0.0 else -1)
        readout_reports: Dict[str, Any] = {}
        dose_empirical: Dict[str, Any] = {}
        for readout in readouts:
            base_sign = int(
                validated.readout_directions[readout]["positive_label_sign"]
            )
            expected_sign = None if dose_sign == 0 else base_sign * dose_sign
            seed = _derived_seed(
                base_seed,
                validated.concept,
                validated.representation,
                dose,
                readout,
            )
            effect_views = _bootstrap_views(
                trained_dose,
                readout=readout,
                replicates=int(bootstrap_replicates),
                seed=seed,
                confidence_level=float(confidence_level),
                expected_sign=expected_sign,
            )
            effect_views["directionality"] = {
                **validated.readout_directions[readout],
                "dose_sign": dose_sign,
                "predicted_raw_change": (
                    "none_at_zero_dose"
                    if expected_sign is None
                    else ("increase" if expected_sign > 0 else "decrease")
                ),
                "rule": (
                    "positive dose follows the positive-class probe coefficient; "
                    "negative dose reverses the declared readout sign"
                ),
            }
            effect_views["analysis_scope"] = _analysis_scope(
                validated.frozen_protocol,
                validated.concept,
                validated.representation,
                readout,
            )
            if not inference_conforms:
                effect_views["analysis_scope"] = {
                    "scope": "diagnostic_protocol_deviation",
                    "reason": (
                        "Bootstrap replicate count or confidence level differs from the "
                        "prospectively frozen inference protocol."
                    ),
                    "otherwise": effect_views["analysis_scope"],
                }
            readout_reports[readout] = effect_views

            if expected_sign is not None:
                trained_statistic = float(
                    effect_views["label_balanced"]["paired_absolute_delta"]
                )
                family_reports = {}
                controls_at_dose = rows.loc[
                    rows["nominal_dose"].eq(dose)
                    & ~rows["control_kind"].astype(str).eq("trained")
                ]
                for family, family_rows in controls_at_dose.groupby(
                    "control_kind", sort=True
                ):
                    effects = []
                    for _control_id, control_rows in family_rows.groupby(
                        "control_id", sort=True
                    ):
                        effects.append(
                            _label_balanced_mean(
                                control_rows, f"delta_{readout}"
                            )
                        )
                    comparison = _empirical_control_comparison(
                        trained_statistic,
                        effects,
                        expected_sign=expected_sign,
                        family=str(family),
                    )
                    comparison["analysis_scope"] = (
                        "descriptive_or_exploratory_individual_dose; not a "
                        "confirmatory decision and not a substitute for the all-dose slope"
                    )
                    family_reports[str(family)] = comparison
                dose_empirical[readout] = family_reports
        trained_reports[dose_key] = {
            "nominal_dose": dose,
            "n_positions": int(trained_dose["position_id"].nunique()),
            "n_games": int(trained_dose["game_id"].astype(str).nunique()),
            "label_stratum_counts": {
                str(int(label)): {
                    "n_positions": int(part["position_id"].nunique()),
                    "n_games": int(part["game_id"].astype(str).nunique()),
                }
                for label, part in trained_dose.groupby("label", sort=True)
            },
            "policy_disruption": {
                "mean_policy_js": float(trained_dose["policy_js"].mean()),
                "mean_policy_l1": float(trained_dose["policy_l1"].mean()),
                "top_move_flip_rate": float(trained_dose["top_move_flip"].mean()),
            },
            "readouts": readout_reports,
        }
        if dose_empirical:
            empirical[dose_key] = dose_empirical

    control_diagnostics = _control_disruption_diagnostics(
        rows, validated.calibration
    )
    primary_test = _primary_confirmatory_test(
        validated,
        rows,
        bootstrap_replicates=int(bootstrap_replicates),
        bootstrap_seed=int(base_seed),
        confidence_level=float(confidence_level),
        inference_conforms=bool(inference_conforms),
    )
    calibration_successes = sum(
        bool(record["calibration"]["success"]) for record in control_diagnostics
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline": PIPELINE_NAME,
        "source_pipeline": SOURCE_PIPELINE,
        "status": "complete",
        "created_at_utc": _utc_now(),
        "causal_dir": str(validated.causal_dir),
        "concept": validated.concept,
        "representation": validated.representation,
        "contract": {
            "name": validated.contract_name,
            "definition_id": get_contract(validated.concept).definition_id,
            "contract_hash": get_contract(validated.concept).contract_hash,
        },
        "analysis_scope": {
            "frozen_protocol_sha256": _sha256(
                Path(str(validated.run_manifest["fresh_holdout"]["protocol_path"]))
            ),
            "primary_hypothesis": (
                validated.frozen_protocol.get("causal") or {}
            ).get("primary_hypothesis"),
            "current_tuple": {
                "concept": validated.concept,
                "representation": validated.representation,
            },
            "inference_protocol_conforms": bool(inference_conforms),
            "multiplicity_caution": (
                "Only the exact frozen all-dose slope and its two-family conjunction is "
                "confirmatory. Every individual-dose effect/p-value, other readout, and "
                "secondary concept is descriptive or exploratory; do not select a "
                "minimum p-value as a headline result."
            ),
        },
        "provenance": {
            "causal_manifest_sha256": _sha256(validated.causal_dir / "manifest.json"),
            "probe_run_manifest_sha256": _sha256(
                validated.probe_run_dir / "manifest.json"
            ),
            "build_manifest_sha256": _sha256(
                validated.probe_run_dir / "build_manifest.json"
            ),
            "labels_manifest_sha256": _sha256(
                validated.probe_run_dir / str(
                    validated.build_manifest.get("labels_manifest", "labels_manifest.json")
                )
            ),
            "training_manifest_sha256": _sha256(
                validated.probe_run_dir / "training_manifest.json"
            ),
            "causal_rows_sha256": _sha256(
                validated.causal_dir / "causal_test_rows.parquet"
            ),
            "causal_producer_source_sha256": current_causal_source_hashes(),
            "reporter_source_sha256": _sha256(Path(__file__).resolve()),
            "checkpoint_activation_fidelity": (
                validated.training_manifest.get("checkpoint_activation_fidelity")
            ),
        },
        "sampling_and_estimand": {
            "selection": "deterministic label-stratified sampling on frozen causal_test games",
            "primary_reported_estimand": "equal-label-weighted mean paired policy-readout change",
            "label_weights": {"0": 0.5, "1": 0.5},
            "caution": (
                "The raw selected-position mean reflects the stratified evaluation sample, "
                "not the eligible-role prevalence. Label-balanced and by-label results are "
                "therefore reported explicitly."
            ),
        },
        "bootstrap": {
            "unit": "game",
            "procedure": (
                "within each frozen label stratum, sample causal-test games with "
                "replacement and move all positions from a sampled game together; "
                "percentile intervals on paired effects"
            ),
            "replicates": int(bootstrap_replicates),
            "confidence_level": float(confidence_level),
            "base_seed": int(base_seed),
        },
        "trained_direction_by_dose": trained_reports,
        "sole_primary_confirmatory_test": primary_test,
        "empirical_control_comparisons": empirical,
        "control_calibration_and_test_disruption": {
            "calibration_split": "control_calibration",
            "calibration_successes": int(calibration_successes),
            "calibration_records": int(len(control_diagnostics)),
            "causal_test_matching_claim": "none",
            "note": (
                "Calibration success describes only the calibration split. Test JS/L1 "
                "residuals are reported diagnostically and are never relabelled as a match."
            ),
            "by_control_and_dose": control_diagnostics,
        },
    }


def write_causal_results_report(
    causal_dir: Path,
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: Optional[int] = None,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> Tuple[Path, Path]:
    """Write an append-only JSON report plus a manifest binding its hash."""

    causal_dir = Path(causal_dir).resolve()
    report_path = causal_dir / "validated_causal_results_report.json"
    manifest_path = causal_dir / "validated_causal_results_report_manifest.json"
    if report_path.exists() or manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing causal results report in {causal_dir}"
        )
    report = generate_causal_results_report(
        causal_dir,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    if not bool(report["analysis_scope"]["inference_protocol_conforms"]):
        raise CausalResultsValidationError(
            "Append-only validated reports must use the frozen bootstrap replicate "
            "count and confidence level"
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
        "upstream_causal_manifest_sha256": report["provenance"][
            "causal_manifest_sha256"
        ],
        "upstream_causal_rows_sha256": report["provenance"][
            "causal_rows_sha256"
        ],
    }
    with manifest_path.open("xb") as handle:
        handle.write(_canonical_json_bytes(report_manifest))
    manifest_path.chmod(0o444)
    return report_path, manifest_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal-dir", type=Path, required=True)
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    parser.add_argument("--bootstrap-seed", type=int)
    parser.add_argument(
        "--confidence-level", type=float, default=DEFAULT_CONFIDENCE_LEVEL
    )
    parser.add_argument(
        "--write", action="store_true",
        help="Write append-only report files rather than printing JSON",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    if args.write:
        report, manifest = write_causal_results_report(
            args.causal_dir,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
            confidence_level=args.confidence_level,
        )
        print(json.dumps({"report": str(report), "manifest": str(manifest)}, indent=2))
    else:
        report = generate_causal_results_report(
            args.causal_dir,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
            confidence_level=args.confidence_level,
        )
        print(json.dumps(_json_safe(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
