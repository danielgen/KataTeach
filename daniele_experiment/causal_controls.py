"""Reusable controls and provenance helpers for causal evaluations.

The helpers in this module are deliberately independent of KataGo and PyTorch.
Model-facing code supplies an aggregate calibration callback and per-position
policy-effect rows; this module handles deterministic control identities,
downstream-disruption matching, and fail-closed run-manifest validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np


MIN_CONTROL_REPEATS = 50
MANIFEST_SCHEMA_VERSION = 1
RUN_STATUSES = frozenset({"running", "failed", "validated", "invalid_do_not_use"})
VALID_STATUS_TRANSITIONS = {
    "running": frozenset({"failed", "validated", "invalid_do_not_use"}),
    "failed": frozenset(),
    "validated": frozenset(),
    "invalid_do_not_use": frozenset(),
}


class ManifestValidationError(ValueError):
    """Raised when a run manifest is unsafe or inconsistent with its files."""


def _stable_part_bytes(value: Any) -> bytes:
    """Encode one seed component canonically, including an unambiguous type."""
    if isinstance(value, Path):
        value = str(value)
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Seed component is not canonically JSON-serializable: {value!r}") from exc
    return f"{type(value).__name__}:".encode("ascii") + payload


def stable_sha256_seed(*parts: Any, bits: int = 64) -> int:
    """Derive a process-independent integer seed from structured components.

    Python's built-in ``hash`` is randomized between processes.  A sum of
    character codes also collides readily.  Length-prefixed, typed components
    hashed with SHA-256 give stable seeds while avoiding concatenation
    ambiguities such as ``("ab", "c")`` versus ``("a", "bc")``.
    """
    if bits <= 0 or bits > 256 or bits % 8:
        raise ValueError("bits must be a positive multiple of 8 no greater than 256")
    digest = hashlib.sha256()
    digest.update(b"katateach-causal-seed-v1\0")
    for part in parts:
        encoded = _stable_part_bytes(part)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[: bits // 8], "big", signed=False)


def position_shuffle_seed(base_seed: int, repeat_id: str | int, position_id: str) -> int:
    """Return the stable seed for one shuffle repeat at one saved position."""
    return stable_sha256_seed("spatial-mask-shuffle", int(base_seed), repeat_id, position_id)


def control_ids(kind: str, count: int = MIN_CONTROL_REPEATS) -> tuple[str, ...]:
    """Create zero-padded IDs for an adequately sized empirical control set."""
    normalized = kind.strip().lower().replace("-", "_")
    if not normalized or not all(ch.isalnum() or ch == "_" for ch in normalized):
        raise ValueError(f"Invalid control kind: {kind!r}")
    if count < MIN_CONTROL_REPEATS:
        raise ValueError(
            f"At least {MIN_CONTROL_REPEATS} controls are required; received {count}"
        )
    width = max(3, len(str(count - 1)))
    return tuple(f"{normalized}_{index:0{width}d}" for index in range(count))


def shuffle_control_ids(count: int = MIN_CONTROL_REPEATS) -> tuple[str, ...]:
    """IDs for repeated spatial-mask permutations."""
    return control_ids("shuffle", count)


def random_direction_control_ids(count: int = 100) -> tuple[str, ...]:
    """IDs for random channel-direction controls."""
    return control_ids("random", count)


def shuffled_position_mask(
    mask: np.ndarray,
    *,
    base_seed: int,
    repeat_id: str | int,
    position_id: str,
) -> np.ndarray:
    """Permute a mask reproducibly for one repeat and position.

    Values are permuted only within the mask's active support (``mask != 0``).
    This preserves every value, mean, RMS, activation-space norm, and the exact
    legal-action support while disrupting only semantic spatial arrangement.
    Different positions receive different stable permutations within the same
    repeat.
    """
    values = np.asarray(mask)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2-D spatial mask, got shape {values.shape}")
    seed = position_shuffle_seed(base_seed, repeat_id, position_id)
    rng = np.random.default_rng(seed)
    active = values != 0
    shuffled = np.zeros_like(values)
    shuffled[active] = rng.permutation(values[active])
    return shuffled


@dataclass(frozen=True)
class PolicyMatchResult:
    """Result of calibrating one aggregate control to a target mean policy JS."""

    target_mean_policy_js: float
    dose_multiplier: float
    nominal_dose: float
    effective_dose: float
    achieved_mean_policy_js: float
    achieved_mean_policy_l1: Optional[float]
    matched: bool
    status: str
    iterations: int
    bracket_low: float
    bracket_high: float
    absolute_js_error: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _policy_means(value: Mapping[str, Any] | float) -> tuple[float, Optional[float]]:
    if isinstance(value, Mapping):
        if "mean_policy_js" in value:
            mean_js = value["mean_policy_js"]
        elif "policy_js" in value:
            mean_js = value["policy_js"]
        else:
            raise KeyError("Calibration result needs mean_policy_js or policy_js")
        mean_l1 = value.get("mean_policy_l1", value.get("policy_l1"))
    else:
        mean_js = value
        mean_l1 = None
    mean_js = float(mean_js)
    mean_l1 = None if mean_l1 is None else float(mean_l1)
    if not math.isfinite(mean_js) or mean_js < 0:
        raise ValueError(f"Calibration returned invalid mean policy JS: {mean_js!r}")
    if mean_l1 is not None and (not math.isfinite(mean_l1) or mean_l1 < 0):
        raise ValueError(f"Calibration returned invalid mean policy L1: {mean_l1!r}")
    return mean_js, mean_l1


def match_mean_policy_js(
    calibration_callback: Callable[[float], Mapping[str, Any] | float],
    target_mean_policy_js: float,
    *,
    nominal_dose: float,
    initial_multiplier: float = 1.0,
    maximum_multiplier: float = 128.0,
    relative_tolerance: float = 0.02,
    absolute_tolerance: float = 1e-6,
    max_bracket_steps: int = 16,
    max_bisection_steps: int = 30,
) -> PolicyMatchResult:
    """Match a control's *aggregate* mean policy JS by scalar bisection.

    ``calibration_callback(multiplier)`` must evaluate the same multiplier on
    every position in a calibration set and return their aggregate mean policy
    JS (and optionally aggregate mean policy L1).  The helper never performs
    per-position scaling.  The callback must not consult the concept-specific
    behavioral outcome used for the final causal test.

    The caller should use calibration games disjoint from final evaluation
    games.  The signed effective dose is ``nominal_dose * multiplier``.
    Matching assumes mean JS is broadly non-decreasing with non-negative
    multiplier.  When the target cannot be bracketed, the result is explicitly
    marked unmatched instead of silently treating the nearest endpoint as a
    valid match.
    """
    target = float(target_mean_policy_js)
    nominal = float(nominal_dose)
    if not math.isfinite(target) or target < 0:
        raise ValueError("target_mean_policy_js must be finite and non-negative")
    if not math.isfinite(nominal):
        raise ValueError("nominal_dose must be finite")
    if nominal == 0 and target > 0:
        raise ValueError("A zero nominal dose cannot match non-zero policy disruption")
    if initial_multiplier <= 0 or not math.isfinite(initial_multiplier):
        raise ValueError("initial_multiplier must be finite and positive")
    if maximum_multiplier < initial_multiplier or not math.isfinite(maximum_multiplier):
        raise ValueError("maximum_multiplier must be finite and >= initial_multiplier")
    if relative_tolerance < 0 or absolute_tolerance < 0:
        raise ValueError("matching tolerances must be non-negative")
    if max_bracket_steps < 0 or max_bisection_steps < 0:
        raise ValueError("iteration limits must be non-negative")

    tolerance = max(float(absolute_tolerance), target * float(relative_tolerance))
    evaluations: dict[float, tuple[float, Optional[float]]] = {}

    def evaluate(multiplier: float) -> tuple[float, Optional[float]]:
        key = float(multiplier)
        if key not in evaluations:
            evaluations[key] = _policy_means(calibration_callback(key))
        return evaluations[key]

    low = 0.0
    low_js, low_l1 = evaluate(low)
    best_multiplier = low
    best_js, best_l1 = low_js, low_l1

    def update_best(multiplier: float, mean_js: float, mean_l1: Optional[float]) -> None:
        nonlocal best_multiplier, best_js, best_l1
        if abs(mean_js - target) < abs(best_js - target):
            best_multiplier, best_js, best_l1 = multiplier, mean_js, mean_l1

    if abs(low_js - target) <= tolerance:
        return PolicyMatchResult(
            target, low, nominal, nominal * low, low_js, low_l1, True,
            "matched", 0, low, low, abs(low_js - target),
        )
    if low_js > target:
        return PolicyMatchResult(
            target, low, nominal, nominal * low, low_js, low_l1, False,
            "target_below_zero_dose_disruption", 0, low, low,
            abs(low_js - target),
        )

    high = min(float(initial_multiplier), float(maximum_multiplier))
    high_js, high_l1 = evaluate(high)
    update_best(high, high_js, high_l1)
    bracket_iterations = 0
    while high_js < target and high < maximum_multiplier and bracket_iterations < max_bracket_steps:
        low, low_js, low_l1 = high, high_js, high_l1
        high = min(high * 2.0, float(maximum_multiplier))
        high_js, high_l1 = evaluate(high)
        update_best(high, high_js, high_l1)
        bracket_iterations += 1

    if abs(high_js - target) <= tolerance:
        return PolicyMatchResult(
            target, high, nominal, nominal * high, high_js, high_l1, True,
            "matched", bracket_iterations, low, high, abs(high_js - target),
        )
    if high_js < target:
        return PolicyMatchResult(
            target, high, nominal, nominal * high,
            high_js, high_l1, False, "target_not_bracketed", bracket_iterations,
            low, high, abs(high_js - target),
        )

    bisection_iterations = 0
    for bisection_iterations in range(1, max_bisection_steps + 1):
        middle = 0.5 * (low + high)
        middle_js, middle_l1 = evaluate(middle)
        update_best(middle, middle_js, middle_l1)
        if abs(middle_js - target) <= tolerance:
            return PolicyMatchResult(
                target, middle, nominal, nominal * middle, middle_js, middle_l1,
                True, "matched", bracket_iterations + bisection_iterations,
                low, high, abs(middle_js - target),
            )
        if middle_js < target:
            low, low_js, low_l1 = middle, middle_js, middle_l1
        else:
            high, high_js, high_l1 = middle, middle_js, middle_l1

    matched = abs(best_js - target) <= tolerance
    return PolicyMatchResult(
        target, best_multiplier, nominal, nominal * best_multiplier,
        best_js, best_l1, matched,
        "matched" if matched else "bisection_tolerance_not_reached",
        bracket_iterations + bisection_iterations, low, high,
        abs(best_js - target),
    )


def summarize_policy_disruption(
    rows: Iterable[Mapping[str, Any]],
    *,
    control_id: str,
    nominal_dose: float,
    dose_multiplier: float = 1.0,
    target_mean_policy_js: Optional[float] = None,
    matched: Optional[bool] = None,
) -> dict[str, Any]:
    """Summarize downstream disruption for one control/dose evaluation."""
    materialized = list(rows)
    if not materialized:
        raise ValueError("Cannot summarize an empty policy-effect collection")
    js_values = [float(row["policy_js"]) for row in materialized]
    l1_values = [float(row["policy_l1"]) for row in materialized]
    if not all(math.isfinite(value) and value >= 0 for value in js_values + l1_values):
        raise ValueError("Policy JS and L1 values must be finite and non-negative")
    nominal = float(nominal_dose)
    multiplier = float(dose_multiplier)
    if not math.isfinite(nominal) or not math.isfinite(multiplier) or multiplier < 0:
        raise ValueError("Dose and non-negative dose multiplier must be finite")
    mean_js = math.fsum(js_values) / len(js_values)
    mean_l1 = math.fsum(l1_values) / len(l1_values)
    summary: dict[str, Any] = {
        "control_id": control_id,
        "n": len(materialized),
        "nominal_dose": nominal,
        "dose_multiplier": multiplier,
        "effective_dose": nominal * multiplier,
        "mean_policy_js": mean_js,
        "mean_policy_l1": mean_l1,
    }
    if target_mean_policy_js is not None:
        target = float(target_mean_policy_js)
        summary["target_mean_policy_js"] = target
        summary["absolute_js_error"] = abs(mean_js - target)
        summary["relative_js_error"] = (
            abs(mean_js - target) / target if target > 0 else 0.0 if mean_js == 0 else math.inf
        )
    if matched is not None:
        summary["policy_disruption_matched"] = bool(matched)
    if all("top_move_flip" in row for row in materialized):
        summary["top_move_flip_rate"] = math.fsum(
            float(row["top_move_flip"]) for row in materialized
        ) / len(materialized)
    return summary


def sha256_file(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""
    resolved = Path(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value using canonical serialization."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def artifact_record(path: Path | str, *, run_dir: Path | str) -> dict[str, Any]:
    """Build a hash/size record for a file contained inside ``run_dir``."""
    root = Path(run_dir).resolve()
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ManifestValidationError(f"Artifact is outside run directory: {resolved}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": relative.as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def new_run_manifest(
    run_id: str,
    *,
    status: str = "running",
    provenance: Optional[Mapping[str, Any]] = None,
    artifacts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Create the minimal manifest contract shared by result producers."""
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a non-empty single path component")
    if status not in RUN_STATUSES:
        raise ValueError(f"Unknown run status: {status!r}")
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": dict(provenance or {}),
        "artifacts": [dict(record) for record in artifacts],
    }


def write_run_manifest(path: Path | str, manifest: Mapping[str, Any]) -> Path:
    """Write a manifest atomically in its destination directory."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(manifest), sort_keys=True, indent=2, allow_nan=False) + "\n"
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def load_run_manifest(path_or_dir: Path | str) -> tuple[dict[str, Any], Path, Path]:
    """Load a manifest and return ``(manifest, manifest_path, run_dir)``."""
    supplied = Path(path_or_dir)
    manifest_path = supplied / "manifest.json" if supplied.is_dir() else supplied
    if not manifest_path.is_file():
        raise ManifestValidationError(f"Missing run manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestValidationError(f"Cannot read run manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ManifestValidationError("Run manifest root must be a JSON object")
    return manifest, manifest_path, manifest_path.parent.resolve()


def _validate_artifact_path(run_dir: Path, stored_path: Any) -> Path:
    if not isinstance(stored_path, str) or not stored_path:
        raise ManifestValidationError("Every artifact needs a non-empty relative path")
    relative = Path(stored_path)
    if relative.is_absolute():
        raise ManifestValidationError(f"Absolute artifact path is forbidden: {stored_path}")
    resolved = (run_dir / relative).resolve()
    try:
        resolved.relative_to(run_dir)
    except ValueError as exc:
        raise ManifestValidationError(f"Artifact escapes run directory: {stored_path}") from exc
    return resolved


def validate_run_manifest(
    path_or_dir: Path | str,
    *,
    allowed_statuses: Sequence[str] = ("validated",),
    verify_artifacts: bool = True,
    require_artifacts: bool = True,
) -> dict[str, Any]:
    """Validate status and artifact hashes, failing closed on stale results."""
    manifest, manifest_path, run_dir = load_run_manifest(path_or_dir)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ManifestValidationError(
            f"Unsupported manifest schema in {manifest_path}: "
            f"{manifest.get('schema_version')!r}"
        )
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ManifestValidationError("Manifest needs a non-empty run_id")
    status = manifest.get("status")
    if status not in RUN_STATUSES:
        raise ManifestValidationError(f"Unknown run status: {status!r}")
    allowed = frozenset(allowed_statuses)
    unknown_allowed = allowed - RUN_STATUSES
    if unknown_allowed:
        raise ValueError(f"Unknown allowed statuses: {sorted(unknown_allowed)}")
    if status not in allowed:
        raise ManifestValidationError(
            f"Run {run_id!r} has status {status!r}; allowed: {sorted(allowed)}"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ManifestValidationError("Manifest artifacts must be a list")
    if require_artifacts and not artifacts:
        raise ManifestValidationError("Validated consumption requires at least one artifact")
    seen_paths: set[str] = set()
    for record in artifacts:
        if not isinstance(record, Mapping):
            raise ManifestValidationError("Every artifact record must be an object")
        stored_path = record.get("path")
        if stored_path in seen_paths:
            raise ManifestValidationError(f"Duplicate artifact path: {stored_path}")
        seen_paths.add(stored_path)
        resolved = _validate_artifact_path(run_dir, stored_path)
        if not resolved.is_file():
            raise ManifestValidationError(f"Missing artifact: {stored_path}")
        expected_size = record.get("size_bytes")
        expected_hash = record.get("sha256")
        if not isinstance(expected_size, int) or expected_size < 0:
            raise ManifestValidationError(f"Invalid artifact size for {stored_path}")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ManifestValidationError(f"Invalid SHA-256 for {stored_path}")
        if verify_artifacts:
            actual_size = resolved.stat().st_size
            if actual_size != expected_size:
                raise ManifestValidationError(
                    f"Artifact size mismatch for {stored_path}: "
                    f"expected {expected_size}, got {actual_size}"
                )
            actual_hash = sha256_file(resolved)
            if actual_hash != expected_hash.lower():
                raise ManifestValidationError(f"Artifact hash mismatch for {stored_path}")
    return manifest


def update_run_status(
    path_or_dir: Path | str,
    new_status: str,
    *,
    expected_current_status: Optional[str] = None,
    verify_before_validation: bool = True,
) -> dict[str, Any]:
    """Apply an allowed terminal status transition and atomically persist it."""
    if new_status not in RUN_STATUSES:
        raise ValueError(f"Unknown run status: {new_status!r}")
    manifest, manifest_path, _run_dir = load_run_manifest(path_or_dir)
    current = manifest.get("status")
    if current not in RUN_STATUSES:
        raise ManifestValidationError(f"Unknown current run status: {current!r}")
    if expected_current_status is not None and current != expected_current_status:
        raise ManifestValidationError(
            f"Expected status {expected_current_status!r}, found {current!r}"
        )
    if new_status != current and new_status not in VALID_STATUS_TRANSITIONS[current]:
        raise ManifestValidationError(
            f"Forbidden run status transition: {current!r} -> {new_status!r}"
        )
    if new_status == "validated" and verify_before_validation:
        # Validate the same artifact contract while it is still in its expected
        # running state; status is advanced only after every hash passes.
        validate_run_manifest(
            manifest_path,
            allowed_statuses=(current,),
            verify_artifacts=True,
            require_artifacts=True,
        )
    manifest["status"] = new_status
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_run_manifest(manifest_path, manifest)
    return manifest
