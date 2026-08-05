"""Append-only AP-bootstrap correction for frozen validated probe reports.

The validity-v5 prospective protocol commits ``validated_results_report.py``
byte-for-byte.  That frozen reporter must therefore remain unchanged even
though its vectorised average-precision bootstrap mishandles a resample that
assigns zero weight to the highest score group.  This versioned correction
invokes the frozen validator and report generator, replacing only the weighted
metric helper while the report is recomputed.  It writes to a separate
correction namespace and records both the original frozen provenance and this
post-freeze correction's source hash.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
from unittest.mock import patch

import numpy as np

try:  # pragma: no cover - supports direct script execution
    from . import validated_results_report as frozen_reporter
except ImportError:  # pragma: no cover
    import validated_results_report as frozen_reporter


SCHEMA_VERSION = 1
PIPELINE_NAME = "validated_results_report_apfix_v2"
CORRECTION_ID = "average_precision_zero_weight_leading_score_group_v2"
CORRECTION_RELATIVE_DIR = Path("corrections") / PIPELINE_NAME
REPORT_FILENAME = "corrected_results_report.json"
MANIFEST_FILENAME = "corrected_results_report_manifest.json"

_FROZEN_WEIGHTED_METRIC_MATRIX = frozen_reporter._weighted_metric_matrix


class ResultsCorrectionError(ValueError):
    """Raised when the frozen report cannot be corrected without ambiguity."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def _read_json(path: Path, description: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultsCorrectionError(f"Could not read {description}: {path}") from exc
    if not isinstance(value, dict):
        raise ResultsCorrectionError(f"{description} is not a JSON object: {path}")
    return value


def weighted_metric_matrix_apfix_v2(
    labels: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    weights: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Recompute AP safely while preserving every other frozen metric.

    A bootstrap can omit all observations in one or more leading probability
    groups while retaining both outcome classes overall.  Those empty groups
    have cumulative total weight zero.  Their precision is mathematically
    irrelevant because their positive increment is also zero, so it is set to
    zero rather than allowing ``0 / 0`` to poison the complete AP sum.
    """

    result = _FROZEN_WEIGHTED_METRIC_MATRIX(
        labels, probabilities, predictions, weights
    )
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    weights = np.asarray(weights, dtype=float)

    positive = labels == 1
    negative = ~positive
    positive_total = weights @ positive.astype(float)
    negative_total = weights @ negative.astype(float)
    valid = (positive_total > 0) & (negative_total > 0)

    ascending = np.argsort(probabilities, kind="stable")
    descending = ascending[::-1]
    descending_probability = probabilities[descending]
    starts = np.r_[
        0,
        np.flatnonzero(np.diff(descending_probability) != 0) + 1,
    ]
    descending_weights = weights[:, descending]
    positive_group = np.add.reduceat(
        descending_weights * positive[descending], starts, axis=1
    )
    total_group = np.add.reduceat(descending_weights, starts, axis=1)
    cumulative_positive = np.cumsum(positive_group, axis=1)
    cumulative_total = np.cumsum(total_group, axis=1)
    precision = np.divide(
        cumulative_positive,
        cumulative_total,
        out=np.zeros_like(cumulative_positive, dtype=float),
        where=cumulative_total > 0,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        average_precision = np.sum(
            (positive_group / positive_total[:, None]) * precision,
            axis=1,
        )
    average_precision[~valid] = np.nan

    corrected = dict(result)
    corrected["average_precision"] = average_precision
    return corrected


def _validate_original_report_binding(
    run_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, str]]:
    report_path = run_dir / "validated_results_report.json"
    manifest_path = run_dir / "validated_results_report_manifest.json"
    training_manifest_path = run_dir / "training_manifest.json"
    frozen_source_path = Path(frozen_reporter.__file__).resolve()

    report = _read_json(report_path, "original frozen results report")
    manifest = _read_json(manifest_path, "original frozen report manifest")
    report_sha256 = _sha256(report_path)
    manifest_sha256 = _sha256(manifest_path)
    frozen_source_sha256 = _sha256(frozen_source_path)
    training_manifest_sha256 = _sha256(training_manifest_path)

    if (
        manifest.get("pipeline") != frozen_reporter.PIPELINE_NAME
        or manifest.get("status") != "complete"
        or manifest.get("report") != report_path.name
        or manifest.get("report_sha256") != report_sha256
    ):
        raise ResultsCorrectionError(
            "Original frozen report manifest does not bind the existing report"
        )
    if (
        manifest.get("reporter_source_sha256") != frozen_source_sha256
        or manifest.get("upstream_training_manifest_sha256")
        != training_manifest_sha256
    ):
        raise ResultsCorrectionError(
            "Original frozen report manifest does not bind current frozen source/upstream"
        )
    provenance = report.get("provenance")
    if (
        report.get("pipeline") != frozen_reporter.PIPELINE_NAME
        or report.get("status") != "complete"
        or not isinstance(provenance, Mapping)
        or provenance.get("reporter_source_sha256") != frozen_source_sha256
        or provenance.get("training_manifest_sha256") != training_manifest_sha256
    ):
        raise ResultsCorrectionError(
            "Original report provenance is inconsistent with its frozen inputs"
        )

    hashes = {
        "original_report_sha256": report_sha256,
        "original_report_manifest_sha256": manifest_sha256,
        "original_frozen_reporter_source_sha256": frozen_source_sha256,
        "upstream_training_manifest_sha256": training_manifest_sha256,
    }
    return report, manifest, hashes


def _replace_ap_bootstrap_records(
    target: Dict[str, Any], source: Mapping[str, Any]
) -> None:
    if set(target.get("concepts", {})) != set(source.get("concepts", {})):
        raise ResultsCorrectionError("Corrected and original concept sets differ")
    for concept in source["concepts"]:
        target_concept = target["concepts"][concept]
        source_concept = source["concepts"][concept]
        if set(target_concept["representations"]) != set(
            source_concept["representations"]
        ) or set(target_concept["ablations"]) != set(source_concept["ablations"]):
            raise ResultsCorrectionError(
                f"Corrected and original result structure differs for {concept}"
            )
        for representation in source_concept["representations"]:
            target_concept["representations"][representation][
                "game_cluster_bootstrap"
            ]["average_precision"] = copy.deepcopy(
                source_concept["representations"][representation][
                    "game_cluster_bootstrap"
                ]["average_precision"]
            )
        for ablation in source_concept["ablations"]:
            target_concept["ablations"][ablation][
                "paired_game_cluster_bootstrap"
            ]["average_precision"] = copy.deepcopy(
                source_concept["ablations"][ablation][
                    "paired_game_cluster_bootstrap"
                ]["average_precision"]
            )


def _assert_only_ap_bootstrap_records_changed(
    corrected: Mapping[str, Any], original: Mapping[str, Any]
) -> None:
    comparison = copy.deepcopy(corrected)
    comparison["created_at_utc"] = original.get("created_at_utc")
    _replace_ap_bootstrap_records(comparison, original)
    if comparison != original:
        raise ResultsCorrectionError(
            "Recomputed report differs outside AP bootstrap interval records"
        )

    for concept, original_concept in original["concepts"].items():
        corrected_concept = corrected["concepts"][concept]
        for representation, original_representation in original_concept[
            "representations"
        ].items():
            before = original_representation["game_cluster_bootstrap"][
                "average_precision"
            ]
            after = corrected_concept["representations"][representation][
                "game_cluster_bootstrap"
            ]["average_precision"]
            if before.get("point_estimate") != after.get("point_estimate"):
                raise ResultsCorrectionError(
                    f"AP point estimate changed for {concept}/{representation}"
                )
        for ablation, original_ablation in original_concept["ablations"].items():
            before = original_ablation["paired_game_cluster_bootstrap"][
                "average_precision"
            ]
            after = corrected_concept["ablations"][ablation][
                "paired_game_cluster_bootstrap"
            ]["average_precision"]
            if before.get("point_estimate") != after.get("point_estimate"):
                raise ResultsCorrectionError(
                    f"AP ablation point estimate changed for {concept}/{ablation}"
                )


def _ap_status_counts(report: Mapping[str, Any]) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {
        "representations": {},
        "ablations": {},
    }
    for concept in report["concepts"].values():
        for representation in concept["representations"].values():
            status = str(
                representation["game_cluster_bootstrap"]["average_precision"].get(
                    "status"
                )
            )
            counts["representations"][status] = (
                counts["representations"].get(status, 0) + 1
            )
        for ablation in concept["ablations"].values():
            status = str(
                ablation["paired_game_cluster_bootstrap"]["average_precision"].get(
                    "status"
                )
            )
            counts["ablations"][status] = counts["ablations"].get(status, 0) + 1
    return counts


def generate_corrected_results_report(run_dir: Path) -> Dict[str, Any]:
    """Recompute only AP bootstrap records under frozen report settings."""

    run_dir = Path(run_dir).resolve()
    # This is the original fail-closed validator, including the prospective
    # byte-identity check for fresh validity-v5 runs.
    frozen_reporter.validate_results_inputs(run_dir)
    original, _original_manifest, hashes = _validate_original_report_binding(run_dir)

    bootstrap = original.get("bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise ResultsCorrectionError("Original report lacks bootstrap settings")
    try:
        replicates = int(bootstrap["replicates"])
        confidence_level = float(bootstrap["confidence_level"])
        base_seed = int(bootstrap["base_seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ResultsCorrectionError(
            "Original report has invalid bootstrap settings"
        ) from exc

    with patch.object(
        frozen_reporter,
        "_weighted_metric_matrix",
        weighted_metric_matrix_apfix_v2,
    ):
        corrected = frozen_reporter.generate_results_report(
            run_dir,
            bootstrap_replicates=replicates,
            bootstrap_seed=base_seed,
            confidence_level=confidence_level,
        )
    _assert_only_ap_bootstrap_records_changed(corrected, original)

    correction_source_path = Path(__file__).resolve()
    correction_source_sha256 = _sha256(correction_source_path)
    original_status_counts = _ap_status_counts(original)
    corrected_status_counts = _ap_status_counts(corrected)
    frozen_reporter_sha256 = corrected["provenance"].pop(
        "reporter_source_sha256"
    )
    corrected["provenance"].update(
        {
            "frozen_reporter_source_sha256": frozen_reporter_sha256,
            "correction_reporter_source_sha256": correction_source_sha256,
            "original_frozen_report_sha256": hashes["original_report_sha256"],
            "original_frozen_report_manifest_sha256": hashes[
                "original_report_manifest_sha256"
            ],
        }
    )
    corrected.update(
        {
            "pipeline": PIPELINE_NAME,
            "source_pipeline": frozen_reporter.PIPELINE_NAME,
            "status": "complete_post_freeze_reporting_correction",
            "correction": {
                "correction_id": CORRECTION_ID,
                "reason": (
                    "The frozen vectorised AP bootstrap evaluated precision as 0/0 "
                    "when a game-cluster resample assigned zero weight to leading "
                    "probability groups. Those empty groups have zero positive "
                    "increment and must contribute zero, not NaN, to AP."
                ),
                "scope": (
                    "Only representation game_cluster_bootstrap.average_precision "
                    "and ablation paired_game_cluster_bootstrap.average_precision "
                    "records are recomputed. Fold point estimates and every non-AP "
                    "field reproduce the original report exactly."
                ),
                "changed_field_patterns": [
                    "concepts.*.representations.*.game_cluster_bootstrap.average_precision",
                    "concepts.*.ablations.*.paired_game_cluster_bootstrap.average_precision",
                ],
                "post_freeze_correction": True,
                "part_of_original_prospective_protocol": False,
                "frozen_validator_invoked": True,
                "non_ap_fields_reproduced_exactly": True,
                "original_ap_status_counts": original_status_counts,
                "corrected_ap_status_counts": corrected_status_counts,
                "original_report": "validated_results_report.json",
                "original_report_sha256": hashes["original_report_sha256"],
                "original_report_manifest": "validated_results_report_manifest.json",
                "original_report_manifest_sha256": hashes[
                    "original_report_manifest_sha256"
                ],
                "original_frozen_reporter_source_sha256": hashes[
                    "original_frozen_reporter_source_sha256"
                ],
                "correction_reporter_source_sha256": correction_source_sha256,
                "upstream_training_manifest_sha256": hashes[
                    "upstream_training_manifest_sha256"
                ],
                "reused_bootstrap_settings": {
                    "replicates": replicates,
                    "confidence_level": confidence_level,
                    "base_seed": base_seed,
                },
            },
        }
    )
    return corrected


def write_corrected_results_report(run_dir: Path) -> Tuple[Path, Path]:
    """Write the versioned correction without touching frozen report files."""

    run_dir = Path(run_dir).resolve()
    report = generate_corrected_results_report(run_dir)
    correction_dir = run_dir / CORRECTION_RELATIVE_DIR
    report_path = correction_dir / REPORT_FILENAME
    manifest_path = correction_dir / MANIFEST_FILENAME
    if report_path.exists() or manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing AP correction in {correction_dir}"
        )

    report_bytes = _canonical_json_bytes(report)
    report_sha256 = _sha256_bytes(report_bytes)
    correction = report["correction"]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "pipeline": PIPELINE_NAME,
        "status": "complete_post_freeze_reporting_correction",
        "created_at_utc": _utc_now(),
        "correction_id": CORRECTION_ID,
        "report": str(report_path.relative_to(run_dir)),
        "report_sha256": report_sha256,
        "correction_reporter_source_sha256": correction[
            "correction_reporter_source_sha256"
        ],
        "original_frozen_report_sha256": correction["original_report_sha256"],
        "original_frozen_report_manifest_sha256": correction[
            "original_report_manifest_sha256"
        ],
        "original_frozen_reporter_source_sha256": correction[
            "original_frozen_reporter_source_sha256"
        ],
        "upstream_training_manifest_sha256": correction[
            "upstream_training_manifest_sha256"
        ],
    }
    manifest_bytes = _canonical_json_bytes(manifest)

    correction_dir.mkdir(parents=True, exist_ok=True)
    with report_path.open("xb") as handle:
        handle.write(report_bytes)
    report_path.chmod(0o444)
    with manifest_path.open("xb") as handle:
        handle.write(manifest_bytes)
    manifest_path.chmod(0o444)
    return report_path, manifest_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write append-only correction files instead of printing JSON",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    if args.write:
        report, manifest = write_corrected_results_report(args.run_dir)
        print(json.dumps({"report": str(report), "manifest": str(manifest)}, indent=2))
    else:
        report = generate_corrected_results_report(args.run_dir)
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
