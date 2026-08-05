#!/usr/bin/env python3
"""Held-out, run-scoped causal evaluation for validated KataGo probes.

This module is intentionally separate from the legacy causal scripts.  It only
accepts artifacts produced inside a ``validated_probe_pipeline`` run, verifies
their hashes, and evaluates positions from the frozen ``control_calibration``
and ``causal_test`` game splits.  Development games used to fit a probe are
rejected at the causal boundary.

The three supported operational contracts are imported from
``operational_definitions``.  In particular, ``reply_peak95`` (the forcing
proxy) is evaluated with a frozen candidate-action map: every legal board move
is played once, the opponent's post-move legal reply policy is evaluated, and
the current policy mass assigned to actions with reply peak > .95 is measured.
Current-policy concentration is never substituted for this quantity.

Controls are calibrated on held-out ``control_calibration`` games to match the
trained intervention's *mean downstream policy Jensen--Shannon divergence*.
The matched controls are then evaluated, without further tuning, on untouched
``causal_test`` games.  Local and combined interventions require at least 50
deterministic spatial shuffles; every representation requires at least 100
deterministic random channel directions.

Intervention and control policies are computed in batches by applying KataGo's
policy head directly to the exact saved pre-move ``act_trunkfinal`` tensors.
Every selected tensor must match an individually addressable build-time
SHA-256 leaf.  Before the backend is trusted, a deterministic sample is replayed
through the complete checkpoint and both saved activations and channel-0 policy
probabilities must agree within tight tolerances.  Full-network replay remains
necessary for the forcing proxy's post-candidate opponent-reply cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
PYTHON_DIR = REPO_DIR / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.append(str(PYTHON_DIR))

try:  # Package import.
    from .causal_controls import (
        MIN_CONTROL_REPEATS,
        artifact_record,
        match_mean_policy_js,
        new_run_manifest,
        random_direction_control_ids,
        sha256_file,
        shuffle_control_ids,
        shuffled_position_mask,
        stable_sha256_seed,
        update_run_status,
        write_run_manifest,
    )
    from .operational_definitions import (
        PolicySupport,
        PolicyTime,
        PolicyView,
        get_contract,
        legal_policy_view,
        manhattan_distance,
        regional_peak_contrast_mask,
        regional_policy_readouts,
        reply_peak,
        reply_peak95_candidate_readouts,
        reply_peak95_contrast_mask,
        tenuki_contrast_mask,
        tenuki_policy_readouts,
    )
except ImportError:  # pragma: no cover - direct CLI execution.
    from causal_controls import (
        MIN_CONTROL_REPEATS,
        artifact_record,
        match_mean_policy_js,
        new_run_manifest,
        random_direction_control_ids,
        sha256_file,
        shuffle_control_ids,
        shuffled_position_mask,
        stable_sha256_seed,
        update_run_status,
        write_run_manifest,
    )
    from operational_definitions import (
        PolicySupport,
        PolicyTime,
        PolicyView,
        get_contract,
        legal_policy_view,
        manhattan_distance,
        regional_peak_contrast_mask,
        regional_policy_readouts,
        reply_peak,
        reply_peak95_candidate_readouts,
        reply_peak95_contrast_mask,
        tenuki_contrast_mask,
        tenuki_policy_readouts,
    )


SCHEMA_VERSION = 1
PIPELINE_NAME = "validated_causal_eval"
PROBE_PIPELINE_NAME = "validated_probe_pipeline"
REPRESENTATIONS = ("global", "local", "combined")
CAUSAL_ROLES = ("control_calibration", "causal_test")
DEFAULT_SEED = 20260730
DEFAULT_SHUFFLES = 50
DEFAULT_RANDOM_DIRECTIONS = 100
DEFAULT_DOSES = (-2.0, -1.0, 0.0, 1.0, 2.0)
DEFAULT_HEAD_BATCH_SIZE = 64
DEFAULT_EQUIVALENCE_SAMPLE_SIZE = 6
DEFAULT_POLICY_EQUIVALENCE_ATOL = 1e-6
DEFAULT_ACTIVATION_EQUIVALENCE_ATOL = 1e-5
DEFAULT_OPERATIONAL_ALIGNMENT_ATOL = 1e-6
DEFAULT_CALIBRATION_POSITIONS = 50
DEFAULT_CAUSAL_TEST_POSITIONS = 100
FRESH_PROVENANCE_KEYS = (
    "cohort",
    "game_ids",
    "checkpoint_sha256",
    "protocol_manifest_sha256",
    "generator_source_sha256",
    "common_utils_source_sha256",
    "protocol_source_sha256",
    "rng_seed_set_sha256",
)


class CausalValidationError(ValueError):
    """Raised when provenance, split isolation, or semantics fail closed."""


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


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(_json_safe(value), sort_keys=True, indent=2) + "\n").encode("utf-8")


def _write_new_json(path: Path, value: Any, *, readonly: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_canonical_bytes(value))
    if readonly:
        path.chmod(0o444)


def _read_json(path: Path, description: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CausalValidationError(f"Invalid JSON in {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CausalValidationError(f"{description} must contain a JSON object: {path}")
    return value


def _inside(path: Path, root: Path, description: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise CausalValidationError(f"{description} escapes run directory: {path}") from exc
    return resolved


def _reject_archive_path(path: Path, description: str) -> None:
    suspicious = {"archive", "archives", "archived"}
    if any(part.lower() in suspicious for part in path.resolve().parts):
        raise CausalValidationError(
            f"{description} points into an archive, which is forbidden: {path}"
        )


def _require_hash(path: Path, expected: Any, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    if not isinstance(expected, str) or len(expected) != 64:
        raise CausalValidationError(f"Invalid expected SHA-256 for {description}")
    observed = sha256_file(path)
    if observed != expected.lower():
        raise CausalValidationError(
            f"SHA-256 mismatch for {description}: expected {expected}, got {observed}"
        )


def _update_length_prefixed(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big", signed=False))
    digest.update(value)


def _overall_trunk_commitment(
    source_games_dir: str, records: Mapping[str, Mapping[str, Any]]
) -> str:
    """Reconstruct the build manifest's per-game aggregate commitment."""

    digest = hashlib.sha256()
    _update_length_prefixed(digest, str(source_games_dir).encode("utf-8"))
    for game_id in sorted(map(str, records)):
        record = records[game_id]
        try:
            game_digest = bytes.fromhex(str(record["identity_bytes_sha256"]))
            file_count = int(record["file_count"])
            total_bytes = int(record["total_bytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CausalValidationError(
                f"Invalid activation input-provenance record for {game_id}"
            ) from exc
        if len(game_digest) != 32 or file_count < 0 or total_bytes < 0:
            raise CausalValidationError(
                f"Invalid activation input-provenance record for {game_id}"
            )
        _update_length_prefixed(digest, game_id.encode("utf-8"))
        digest.update(game_digest)
        digest.update(file_count.to_bytes(8, "big", signed=False))
        digest.update(total_bytes.to_bytes(8, "big", signed=False))
    return digest.hexdigest()


def _verify_input_provenance_commitment(
    run_manifest: Mapping[str, Any],
    build_manifest: Mapping[str, Any],
    game_ids: Iterable[str],
) -> Mapping[str, Any]:
    """Verify immutable build-time provenance without claiming a Merkle proof."""

    provenance = build_manifest.get("input_provenance")
    if not isinstance(provenance, Mapping):
        raise CausalValidationError("Build manifest lacks activation input provenance")
    observed_hash = hashlib.sha256(_canonical_bytes(provenance)).hexdigest()
    if observed_hash != build_manifest.get("input_provenance_sha256"):
        raise CausalValidationError("Build input-provenance canonical hash mismatch")
    games_dir = Path(str(run_manifest.get("source_games_dir", ""))).resolve()
    if provenance.get("source_games_dir") != str(games_dir):
        raise CausalValidationError("Activation provenance source root differs from run")
    records = provenance.get("games")
    if not isinstance(records, Mapping):
        raise CausalValidationError("Activation provenance lacks per-game records")
    expected_games = sorted(map(str, game_ids))
    if sorted(map(str, records)) != expected_games:
        raise CausalValidationError(
            "Activation provenance games differ from the frozen split"
        )
    overall = _overall_trunk_commitment(str(games_dir), records)
    if overall != provenance.get("trunk_identity_bytes_sha256"):
        raise CausalValidationError(
            "Activation provenance aggregate content commitment is inconsistent"
        )
    total_files = sum(int(records[game]["file_count"]) for game in expected_games)
    total_bytes = sum(int(records[game]["total_bytes"]) for game in expected_games)
    for game_id in expected_games:
        record = records[game_id]
        leaves = record.get("files")
        if not isinstance(leaves, Mapping) or len(leaves) != int(record["file_count"]):
            raise CausalValidationError(
                f"Activation provenance leaves are missing or incomplete for {game_id}"
            )
        for filename, leaf in leaves.items():
            if not isinstance(leaf, Mapping):
                raise CausalValidationError(
                    f"Invalid activation provenance leaf for {game_id}/{filename}"
                )
            expected_identity = f"{game_id}/trunkfinal/{filename}"
            declared_hash = str(leaf.get("sha256", ""))
            try:
                decoded = bytes.fromhex(declared_hash)
            except ValueError as exc:
                raise CausalValidationError(
                    f"Invalid activation leaf SHA-256 for {expected_identity}"
                ) from exc
            if (
                leaf.get("identity") != expected_identity
                or int(leaf.get("bytes", -1)) < 0
                or len(decoded) != 32
            ):
                raise CausalValidationError(
                    f"Invalid activation provenance leaf for {expected_identity}"
                )
    if total_files != int(provenance.get("trunk_file_count", -1)):
        raise CausalValidationError("Activation provenance file count is inconsistent")
    if total_bytes != int(provenance.get("trunk_total_bytes", -1)):
        raise CausalValidationError("Activation provenance byte count is inconsistent")
    return provenance


def _verify_current_game_sources(
    run: "ValidatedRun", game_id: str, moves: Sequence[Mapping[str, Any]]
) -> None:
    """Recheck one selected game's paths, moves, and activation stat digest.

    The build's per-game content digest is an aggregate, not a Merkle tree, so
    this function deliberately does not claim that an individual selected-file
    digest can be derived from it.  Instead, the causal run records exact hashes
    for selected files and separately requires the upstream aggregate plus this
    current identity/size/mtime check.
    """

    provenance = run.build_manifest.get("input_provenance") or {}
    records = provenance.get("games") or {}
    record = records.get(str(game_id))
    if not isinstance(record, Mapping):
        raise CausalValidationError(
            f"Activation provenance lacks selected game {game_id}"
        )
    game_dir = (run.games_dir / str(game_id)).resolve()
    moves_path = game_dir / "moves.jsonl"
    trunk_dir = (game_dir / "trunkfinal").resolve()
    for key, observed in (
        ("source_game_dir", str(game_dir)),
        ("moves_path", str(moves_path)),
        ("trunkfinal_dir", str(trunk_dir)),
    ):
        if record.get(key) != observed:
            raise CausalValidationError(
                f"Selected game {game_id} source path mismatch for {key}"
            )
    moves_payload = moves_path.read_bytes()
    if hashlib.sha256(moves_payload).hexdigest() != record.get("moves_sha256"):
        raise CausalValidationError(
            f"Raw moves changed after feature building for selected game {game_id}"
        )
    if len(moves_payload) != int(record.get("moves_bytes", -1)):
        raise CausalValidationError(
            f"Raw move byte count changed for selected game {game_id}"
        )
    stat_digest = hashlib.sha256()
    identities: set[str] = set()
    total_bytes = 0
    leaf_records = record.get("files")
    if not isinstance(leaf_records, Mapping):
        raise CausalValidationError(
            f"Activation provenance lacks file leaves for selected game {game_id}"
        )
    expected_filenames = {
        f"move_{int(move['move_number']):03d}.npy" for move in moves
    }
    if set(map(str, leaf_records)) != expected_filenames:
        raise CausalValidationError(
            f"Activation provenance leaf identities changed for selected game {game_id}"
        )
    for move in moves:
        move_number = int(move["move_number"])
        filename = f"move_{move_number:03d}.npy"
        identity = f"{game_id}/trunkfinal/{filename}"
        if identity in identities:
            raise CausalValidationError(
                f"Duplicate saved activation identity {identity}"
            )
        identities.add(identity)
        path = run.games_dir / identity
        if not path.is_file():
            raise FileNotFoundError(f"Missing saved activation: {path}")
        stat = path.stat()
        leaf = leaf_records[filename]
        if (
            not isinstance(leaf, Mapping)
            or leaf.get("identity") != identity
            or int(leaf.get("bytes", -1)) != int(stat.st_size)
        ):
            raise CausalValidationError(
                f"Activation build-time leaf mismatch for selected input {identity}"
            )
        _update_length_prefixed(stat_digest, identity.encode("utf-8"))
        stat_digest.update(int(stat.st_size).to_bytes(8, "big", signed=False))
        stat_digest.update(int(stat.st_mtime_ns).to_bytes(8, "big", signed=False))
        total_bytes += int(stat.st_size)
    if stat_digest.hexdigest() != record.get("identity_stat_sha256"):
        raise CausalValidationError(
            f"Activation identities, sizes, or mtimes changed for selected game {game_id}"
        )
    if len(identities) != int(record.get("file_count", -1)):
        raise CausalValidationError(
            f"Activation count changed for selected game {game_id}"
        )
    if total_bytes != int(record.get("total_bytes", -1)):
        raise CausalValidationError(
            f"Activation byte count changed for selected game {game_id}"
        )
    meta_path = game_dir / "meta.json"
    if meta_path.is_file():
        if record.get("meta_path") != str(meta_path) or sha256_file(meta_path) != record.get(
            "meta_sha256"
        ):
            raise CausalValidationError(
                f"Generator metadata changed for selected game {game_id}"
            )
    elif record.get("meta_path") is not None or record.get("meta_sha256") is not None:
        raise CausalValidationError(
            f"Generator metadata disappeared for selected game {game_id}"
        )


def _hash_run_labels(labels_dir: Path, game_ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for game_id in sorted(str(game) for game in game_ids):
        path = _inside(labels_dir / game_id / "snorkel.jsonl", labels_dir, "label file")
        if not path.is_file():
            raise FileNotFoundError(f"Missing run-scoped label file: {path}")
        _reject_archive_path(path, "Run-scoped label file")
        digest.update(game_id.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _current_probe_source_hashes() -> Dict[str, str]:
    return {
        "daniele_experiment/validated_probe_pipeline.py": sha256_file(
            SCRIPT_DIR / "validated_probe_pipeline.py"
        ),
        "daniele_experiment/operational_definitions.py": sha256_file(
            SCRIPT_DIR / "operational_definitions.py"
        ),
    }


def _current_causal_source_hashes() -> Dict[str, str]:
    """Freeze producer, matching, and operational semantics for causal artifacts."""

    return {
        "daniele_experiment/validated_causal_eval.py": sha256_file(Path(__file__).resolve()),
        "daniele_experiment/causal_controls.py": sha256_file(
            SCRIPT_DIR / "causal_controls.py"
        ),
        "daniele_experiment/operational_definitions.py": sha256_file(
            SCRIPT_DIR / "operational_definitions.py"
        ),
    }


def _valid_sha256(value: Any, description: str) -> str:
    candidate = str(value or "")
    try:
        decoded = bytes.fromhex(candidate)
    except ValueError as exc:
        raise CausalValidationError(f"Invalid SHA-256 for {description}") from exc
    if len(decoded) != 32:
        raise CausalValidationError(f"Invalid SHA-256 for {description}")
    return candidate.lower()


def _verify_fresh_holdout_manifest(
    run_manifest: Mapping[str, Any],
    build_manifest: Mapping[str, Any],
    splits: pd.DataFrame,
    games_dir: Path,
) -> Dict[str, Any]:
    """Require a checkpoint-bound fresh cohort for confirmatory causal work."""

    fresh = run_manifest.get("fresh_holdout")
    if not isinstance(fresh, Mapping):
        raise CausalValidationError(
            "Confirmatory causal evaluation requires run_manifest.fresh_holdout"
        )
    cohort = fresh.get("cohort")
    if not isinstance(cohort, str) or not cohort.strip():
        raise CausalValidationError("Fresh holdout cohort must be a non-empty string")
    holdout_games = sorted(
        splits.loc[
            splits["split_role"].isin(CAUSAL_ROLES), "game_id"
        ].astype(str)
    )
    declared_games = sorted(map(str, fresh.get("game_ids") or []))
    if declared_games != holdout_games or int(fresh.get("games", -1)) != len(holdout_games):
        raise CausalValidationError(
            "Fresh holdout game IDs must exactly equal control_calibration + causal_test"
        )
    checkpoint_hash = _valid_sha256(
        fresh.get("checkpoint_sha256"), "fresh holdout checkpoint"
    )
    protocol_hash = _valid_sha256(
        fresh.get("protocol_manifest_sha256"), "fresh holdout protocol manifest"
    )
    generator_hash = _valid_sha256(
        fresh.get("generator_source_sha256"), "fresh holdout generator source"
    )
    declared_common_hash = _valid_sha256(
        fresh.get("common_utils_source_sha256"), "fresh holdout common_utils source"
    )
    input_records = ((build_manifest.get("input_provenance") or {}).get("games") or {})
    common_hashes: set[str] = set()
    seeds: set[int] = set()
    protocol_paths: set[str] = set()
    generator_paths: set[str] = set()
    common_paths: set[str] = set()
    created_at: List[str] = []
    for game_id in holdout_games:
        meta_path = (games_dir / game_id / "meta.json").resolve()
        if not meta_path.is_file():
            raise FileNotFoundError(f"Fresh holdout metadata is missing: {meta_path}")
        meta_hash = sha256_file(meta_path)
        input_record = input_records.get(game_id)
        if (
            not isinstance(input_record, Mapping)
            or input_record.get("meta_path") != str(meta_path)
            or input_record.get("meta_sha256") != meta_hash
        ):
            raise CausalValidationError(
                f"Fresh holdout metadata is not bound by build provenance for {game_id}"
            )
        metadata = _read_json(meta_path, f"fresh holdout metadata {game_id}")
        if metadata.get("cohort") != cohort:
            raise CausalValidationError(f"Fresh cohort mismatch for {game_id}")
        if metadata.get("immutable_outputs") is not True:
            raise CausalValidationError(f"Fresh holdout outputs are not immutable for {game_id}")
        protocol = metadata.get("protocol_manifest")
        checkpoint = metadata.get("checkpoint")
        generator = metadata.get("generator")
        rng = metadata.get("rng")
        if not all(isinstance(item, Mapping) for item in (protocol, checkpoint, generator, rng)):
            raise CausalValidationError(
                f"Fresh holdout metadata is incomplete for {game_id}"
            )
        if _valid_sha256(protocol.get("sha256"), f"protocol {game_id}") != protocol_hash:
            raise CausalValidationError(f"Protocol hash mismatch for fresh game {game_id}")
        if _valid_sha256(checkpoint.get("sha256"), f"checkpoint {game_id}") != checkpoint_hash:
            raise CausalValidationError(f"Checkpoint hash mismatch for fresh game {game_id}")
        if (
            checkpoint.get("use_swa") is not False
            or checkpoint.get("selected_weights") != "raw_model"
        ):
            raise CausalValidationError(f"Fresh game {game_id} did not use raw non-SWA weights")
        if _valid_sha256(generator.get("source_sha256"), f"generator {game_id}") != generator_hash:
            raise CausalValidationError(f"Generator hash mismatch for fresh game {game_id}")
        common_hashes.add(
            _valid_sha256(
                generator.get("common_utils_source_sha256"),
                f"common_utils {game_id}",
            )
        )
        game_seed = int(rng.get("game_seed", -1))
        if game_seed < 0 or game_seed in seeds:
            raise CausalValidationError(f"Missing or duplicate fresh RNG seed for {game_id}")
        seeds.add(game_seed)
        timestamp = metadata.get("created_at_utc")
        if not isinstance(timestamp, str) or not timestamp:
            raise CausalValidationError(
                f"Fresh holdout game {game_id} lacks created_at_utc"
            )
        created_at.append(timestamp)

        for record, paths, description in (
            (protocol, protocol_paths, "protocol manifest"),
            (generator, generator_paths, "generator source"),
        ):
            key = "path" if description == "protocol manifest" else "source"
            path = Path(str(record.get(key, ""))).resolve()
            _reject_archive_path(path, f"Fresh {description}")
            if not path.is_file():
                raise FileNotFoundError(f"Fresh {description} does not exist: {path}")
            paths.add(str(path))
        common_path = Path(str(generator.get("common_utils_source", ""))).resolve()
        _reject_archive_path(common_path, "Fresh common_utils source")
        if not common_path.is_file():
            raise FileNotFoundError(f"Fresh common_utils source does not exist: {common_path}")
        common_paths.add(str(common_path))

    if len(common_hashes) != 1 or len(protocol_paths) != 1 or len(generator_paths) != 1 or len(common_paths) != 1:
        raise CausalValidationError(
            "Fresh holdout games do not share one protocol/generator/common_utils source"
        )
    common_hash = next(iter(common_hashes))
    if common_hash != declared_common_hash:
        raise CausalValidationError(
            "Fresh holdout common_utils hash differs from per-game metadata"
        )
    protocol_path = Path(next(iter(protocol_paths)))
    generator_path = Path(next(iter(generator_paths)))
    common_path = Path(next(iter(common_paths)))
    for path, expected, description in (
        (protocol_path, protocol_hash, "fresh protocol manifest"),
        (generator_path, generator_hash, "fresh generator source"),
        (common_path, common_hash, "fresh common_utils source"),
    ):
        if sha256_file(path) != expected:
            raise CausalValidationError(f"Current {description} differs from generation-time bytes")
    seed_digest = hashlib.sha256(
        ",".join(map(str, sorted(seeds))).encode("ascii")
    ).hexdigest()
    if seed_digest != fresh.get("rng_seed_set_sha256"):
        raise CausalValidationError("Fresh holdout RNG seed-set hash mismatch")
    protocol_document = _read_json(protocol_path, "fresh holdout protocol manifest")
    protocol_fresh = protocol_document.get("fresh_holdout") or {}
    protocol_sources = protocol_document.get("source_sha256") or {}
    current_causal_sources = _current_causal_source_hashes()
    declared_protocol_path = Path(str(fresh.get("protocol_path", ""))).resolve()
    if (
        protocol_document.get("status") != "frozen_before_fresh_data_generation"
        or protocol_fresh.get("cohort") != cohort
        or int(protocol_fresh.get("games", -1)) != len(holdout_games)
        or protocol_fresh.get("game_seed_set_sha256") != seed_digest
        or (protocol_document.get("checkpoint") or {}).get("sha256")
        != checkpoint_hash
        or protocol_sources.get("daniele_experiment/generate_games_dataset.py")
        != generator_hash
        or protocol_sources.get("daniele_experiment/common_utils.py") != common_hash
        or declared_protocol_path != protocol_path
        or int(fresh.get("protocol_split_seed", -1))
        != int(protocol_fresh.get("split_seed", -2))
        or fresh.get("created_at_utc_min") != min(created_at)
        or fresh.get("created_at_utc_max") != max(created_at)
    ):
        raise CausalValidationError(
            "Fresh holdout metadata does not reproduce its frozen protocol"
        )
    if fresh.get("protocol_source_sha256") != dict(protocol_sources):
        raise CausalValidationError(
            "Fresh holdout source map differs from its frozen protocol"
        )
    if any(
        protocol_sources.get(relative) != observed
        for relative, observed in current_causal_sources.items()
    ):
        raise CausalValidationError(
            "Current causal evaluator/control/contract sources differ from the "
            "prospectively frozen fresh-holdout protocol"
        )
    return {
        **dict(fresh),
        "cohort": cohort,
        "game_ids": holdout_games,
        "games": len(holdout_games),
        "checkpoint_sha256": checkpoint_hash,
        "protocol_manifest_sha256": protocol_hash,
        "generator_source_sha256": generator_hash,
        "common_utils_source_sha256": common_hash,
        "protocol_source_sha256": dict(protocol_sources),
        "protocol_manifest_path": str(protocol_path),
        "generator_source_path": str(generator_path),
        "common_utils_source_path": str(common_path),
        "holdout_game_ids_verified": holdout_games,
    }


def _fresh_provenance_record(run: "ValidatedRun") -> Dict[str, Any]:
    """Return the normalized fresh-cohort commitment embedded in outputs."""

    try:
        return {key: _json_safe(run.fresh_holdout[key]) for key in FRESH_PROVENANCE_KEYS}
    except KeyError as exc:  # Defensive for synthetic callers bypassing the loader.
        raise CausalValidationError(
            f"Fresh holdout verification did not produce required field {exc.args[0]!r}"
        ) from exc


def _verify_development_activation_fidelity(
    run_dir: Path,
    run_manifest: Mapping[str, Any],
    build_manifest: Mapping[str, Any],
    training_manifest: Mapping[str, Any],
    probe_metadata: Mapping[str, Any],
    splits: pd.DataFrame,
    fresh_holdout: Mapping[str, Any],
) -> Dict[str, Any]:
    """Revalidate the prospective checkpoint/saved-activation fidelity gate."""

    if not isinstance(run_manifest.get("fresh_holdout"), Mapping):
        raise CausalValidationError(
            "Activation-fidelity verification requires a fresh-holdout run"
        )
    if int(build_manifest.get("games", -1)) != len(splits):
        raise CausalValidationError(
            "Activation-fidelity-gated build game count differs from frozen splits"
        )

    training_record = training_manifest.get("checkpoint_activation_fidelity")
    probe_record = probe_metadata.get("checkpoint_activation_fidelity")
    if not isinstance(training_record, Mapping) or not isinstance(probe_record, Mapping):
        raise CausalValidationError(
            "Fresh-holdout probes require checkpoint_activation_fidelity provenance"
        )
    if dict(training_record) != dict(probe_record):
        raise CausalValidationError(
            "Probe and training manifests bind different activation-fidelity reports"
        )
    report_path = _inside(
        run_dir / str(training_record.get("path", "")),
        run_dir,
        "checkpoint activation-fidelity report",
    )
    _reject_archive_path(report_path, "Checkpoint activation-fidelity report")
    expected_report_hash = _valid_sha256(
        training_record.get("sha256"), "checkpoint activation-fidelity report"
    )
    _require_hash(
        report_path,
        expected_report_hash,
        "checkpoint activation-fidelity report",
    )
    report = _read_json(report_path, "checkpoint activation-fidelity report")
    protocol_path = Path(str(fresh_holdout.get("protocol_manifest_path", ""))).resolve()
    protocol = _read_json(protocol_path, "fresh holdout protocol manifest")
    gate = protocol.get("development_activation_fidelity_gate")
    if not isinstance(gate, Mapping) or gate.get("required_before_training") is not True:
        raise CausalValidationError(
            "Fresh protocol lacks a mandatory development activation-fidelity gate"
        )
    expected_checkpoint = _valid_sha256(
        fresh_holdout.get("checkpoint_sha256"), "fresh holdout checkpoint"
    )
    report_run = report.get("run") or {}
    report_checkpoint = report.get("checkpoint") or {}
    sampling = report.get("sampling") or {}
    aggregate = report.get("aggregate_errors") or {}
    tolerance = report.get("tolerance") or {}
    development_games = sorted(
        splits.loc[splits["split_role"].eq("development"), "game_id"].astype(str)
    )
    expected_games = int(gate.get("expected_games", -1))
    sampled_games = [str(item.get("game_id")) for item in report.get("samples") or ()]
    expected_validator_source = (protocol.get("source_sha256") or {}).get(
        "daniele_experiment/checkpoint_activation_fidelity.py"
    )
    observed_tolerance = float(tolerance.get("absolute_tolerance", math.inf))
    if report.get("status") != "passed" or report.get("validator") != (
        "checkpoint_activation_fidelity"
    ):
        raise CausalValidationError("Development activation-fidelity report did not pass")
    if report.get("validator_source_sha256") != expected_validator_source:
        raise CausalValidationError(
            "Activation-fidelity validator differs from the prospectively frozen source"
        )
    if report_run.get("manifest_sha256") != sha256_file(run_dir / "manifest.json"):
        raise CausalValidationError("Activation-fidelity report binds another run manifest")
    if report_run.get("build_manifest_sha256") != sha256_file(
        run_dir / "build_manifest.json"
    ):
        raise CausalValidationError("Activation-fidelity report binds another build manifest")
    if report_checkpoint.get("sha256") != expected_checkpoint:
        raise CausalValidationError(
            "Activation-fidelity checkpoint differs from the fresh-holdout checkpoint"
        )
    if (
        expected_games != len(development_games)
        or sampling.get("algorithm") != "one_deterministic_position_per_game_v1"
        or sampling.get("split_role_filter") != "development"
        or int(sampling.get("requested_sample_count", -1)) != expected_games
        or int(aggregate.get("sample_count", -1)) != expected_games
        or len(sampled_games) != expected_games
        or len(set(sampled_games)) != expected_games
        or sorted(sampled_games) != development_games
    ):
        raise CausalValidationError(
            "Activation-fidelity report does not cover every development game exactly once"
        )
    frozen_tolerance = float(gate.get("absolute_max_error_tolerance", -1))
    if not math.isfinite(observed_tolerance) or observed_tolerance > frozen_tolerance:
        raise CausalValidationError(
            "Activation-fidelity report used a looser tolerance than the frozen protocol"
        )
    normalized = {
        "path": str(report_path.relative_to(run_dir)),
        "sha256": expected_report_hash,
        "checkpoint_sha256": expected_checkpoint,
        "sample_count": expected_games,
        "sampling_algorithm": sampling.get("algorithm"),
        "absolute_tolerance": observed_tolerance,
        "observed_max_abs_error": aggregate.get("max_abs_error"),
        "claim_scope": report.get("claim_scope"),
    }
    if _json_safe(dict(training_record)) != _json_safe(normalized):
        raise CausalValidationError(
            "Activation-fidelity manifest record does not reproduce the immutable report"
        )
    return normalized


def flat_index_from_internal_loc(move_loc: Optional[int], board_size: int) -> Optional[int]:
    """Convert KataGo's padded internal location to row-major ``idx361``."""
    if move_loc is None or int(move_loc) == 0:
        return None
    loc = int(move_loc)
    stride = int(board_size) + 1
    x = (loc % stride) - 1
    y = (loc // stride) - 1
    if not (0 <= x < board_size and 0 <= y < board_size):
        raise CausalValidationError(
            f"Internal move_loc={loc} is outside a {board_size}x{board_size} board"
        )
    return y * board_size + x


def assert_disjoint_protocol(
    training_games: Iterable[str],
    calibration_games: Iterable[str],
    causal_test_games: Iterable[str],
) -> None:
    """Reject any game overlap among training, calibration, and final test."""
    groups = {
        "training": set(map(str, training_games)),
        "control_calibration": set(map(str, calibration_games)),
        "causal_test": set(map(str, causal_test_games)),
    }
    for left, right in (
        ("training", "control_calibration"),
        ("training", "causal_test"),
        ("control_calibration", "causal_test"),
    ):
        overlap = sorted(groups[left] & groups[right])
        if overlap:
            preview = ", ".join(overlap[:5])
            raise CausalValidationError(
                f"Split leakage: {left} and {right} overlap in {len(overlap)} games: {preview}"
            )


@dataclass(frozen=True)
class ValidatedRun:
    run_dir: Path
    manifest: Mapping[str, Any]
    build_manifest: Mapping[str, Any]
    training_manifest: Mapping[str, Any]
    splits: pd.DataFrame
    dataset: pd.DataFrame
    games_dir: Path
    labels_dir: Path
    concept: str
    representation: str
    probe_path: Path
    scaler_path: Path
    probe_metadata_path: Path
    probe_metadata: Mapping[str, Any]
    fresh_holdout: Mapping[str, Any]
    checkpoint_activation_fidelity: Mapping[str, Any]

    @property
    def seed(self) -> int:
        return int(self.manifest["seed"])

    @property
    def board_size(self) -> int:
        return int(self.build_manifest["board_size"])

    @property
    def channels(self) -> int:
        return int(self.build_manifest["trunk_channels"])


def load_validated_run(run_dir: Path, concept: str, representation: str) -> ValidatedRun:
    """Load and cryptographically verify one probe and all upstream inputs."""
    if representation not in REPRESENTATIONS:
        raise ValueError(f"representation must be one of {REPRESENTATIONS}")
    run_dir = run_dir.resolve()
    _reject_archive_path(run_dir, "Validated run")
    if any(
        part.lower().endswith(("_incomplete", ".incomplete", "-incomplete"))
        for part in run_dir.parts
    ):
        raise CausalValidationError(
            f"Causal evaluation rejects incomplete probe runs: {run_dir}"
        )
    manifest_path = run_dir / "manifest.json"
    manifest = _read_json(manifest_path, "probe run manifest")
    if manifest.get("schema_version") != 1 or manifest.get("pipeline") != PROBE_PIPELINE_NAME:
        raise CausalValidationError(
            f"{run_dir} is not a schema-1 {PROBE_PIPELINE_NAME} run"
        )
    current_sources = _current_probe_source_hashes()
    if manifest.get("source_code_sha256") != current_sources:
        raise CausalValidationError(
            "Probe run source code differs from the currently audited pipeline"
        )
    current_contract_source_hash = current_sources[
        "daniele_experiment/operational_definitions.py"
    ]
    if manifest.get("contract_implementation_sha256") != current_contract_source_hash:
        raise CausalValidationError(
            "Probe run contract implementation differs from current operational definitions"
        )

    artifacts = manifest.get("artifacts") or {}
    concepts_path = _inside(run_dir / "frozen_config" / "concepts.yaml", run_dir, "concepts")
    splits_path = _inside(run_dir / "splits.parquet", run_dir, "splits")
    _require_hash(concepts_path, artifacts.get("concepts_yaml_sha256"), "frozen concepts")
    _require_hash(splits_path, artifacts.get("splits_sha256"), "frozen game splits")
    contract = get_contract(concept)
    with concepts_path.open(encoding="utf-8") as handle:
        frozen_concepts = yaml.safe_load(handle) or {}
    frozen_spec = (frozen_concepts.get("concepts") or {}).get(concept)
    if not isinstance(frozen_spec, Mapping):
        raise CausalValidationError(f"Concept {concept!r} is absent from frozen concepts.yaml")
    if frozen_spec.get("contract_id") != contract.definition_id:
        raise CausalValidationError(
            f"Frozen concept {concept!r} does not bind exact contract {contract.definition_id}"
        )
    if frozen_spec.get("source") != contract.name:
        raise CausalValidationError(
            f"Frozen label source {frozen_spec.get('source')!r} does not reproduce "
            f"contract variable {contract.name!r}"
        )

    build_path = run_dir / "build_manifest.json"
    build = _read_json(build_path, "build manifest")
    if build.get("pipeline") != PROBE_PIPELINE_NAME or build.get("schema_version") != 1:
        raise CausalValidationError("Invalid build manifest producer or schema")
    if build.get("split_manifest_sha256") != artifacts.get("splits_sha256"):
        raise CausalValidationError("Build manifest is bound to a different split file")
    if build.get("concepts_yaml_sha256") != artifacts.get("concepts_yaml_sha256"):
        raise CausalValidationError("Build manifest is bound to a different concept config")
    if build.get("source_code_sha256") != current_sources:
        raise CausalValidationError("Build manifest source hashes differ from current pipeline")
    if build.get("contract_implementation_sha256") != current_contract_source_hash:
        raise CausalValidationError(
            "Build manifest contract implementation differs from current definitions"
        )
    dataset_path = _inside(run_dir / str(build.get("dataset", "dataset.parquet")), run_dir, "dataset")
    _require_hash(dataset_path, build.get("dataset_sha256"), "activation dataset")

    splits = pd.read_parquet(splits_path)
    required_split_columns = {"game_id", "split_role"}
    if not required_split_columns.issubset(splits.columns):
        raise CausalValidationError(f"Split file lacks {sorted(required_split_columns - set(splits))}")
    if splits["game_id"].astype(str).duplicated().any():
        raise CausalValidationError("Split file contains duplicate game IDs")
    unknown_roles = sorted(set(splits["split_role"].astype(str)) - {
        "development", "control_calibration", "causal_test"
    })
    if unknown_roles:
        raise CausalValidationError(f"Split file has unknown roles: {unknown_roles}")
    _verify_input_provenance_commitment(
        manifest,
        build,
        splits["game_id"].astype(str),
    )

    labels_manifest_rel = build.get("labels_manifest")
    if not isinstance(labels_manifest_rel, str):
        raise CausalValidationError("Build manifest does not identify its labels manifest")
    labels_manifest_path = _inside(
        run_dir / labels_manifest_rel, run_dir, "labels manifest"
    )
    _require_hash(
        labels_manifest_path,
        build.get("labels_manifest_sha256"),
        "labels manifest",
    )
    labels_manifest = _read_json(labels_manifest_path, "labels manifest")
    if (
        labels_manifest.get("pipeline") != "validated_label_builder"
        or labels_manifest.get("status") != "complete"
    ):
        raise CausalValidationError("Labels manifest is not a completed validated build")
    for key, expected in (
        ("run_manifest_sha256", sha256_file(manifest_path)),
        ("split_manifest_sha256", artifacts.get("splits_sha256")),
        ("concepts_yaml_sha256", artifacts.get("concepts_yaml_sha256")),
    ):
        if labels_manifest.get(key) != expected:
            raise CausalValidationError(f"Labels manifest provenance mismatch for {key}")

    labels_rel = build.get("labels_games_dir") or artifacts.get("labels_games_dir")
    if not isinstance(labels_rel, str):
        raise CausalValidationError("Build manifest does not identify run-scoped labels")
    labels_dir = _inside(run_dir / labels_rel, run_dir, "labels directory")
    _reject_archive_path(labels_dir, "Run-scoped labels")
    observed_labels_hash = _hash_run_labels(labels_dir, splits["game_id"].astype(str))
    if observed_labels_hash != build.get("labels_sha256"):
        raise CausalValidationError("Run-scoped label hash does not match build manifest")

    training_path = run_dir / "training_manifest.json"
    training = _read_json(training_path, "training manifest")
    if training.get("pipeline") != PROBE_PIPELINE_NAME or training.get("schema_version") != 1:
        raise CausalValidationError("Invalid training manifest producer or schema")
    if training.get("training_role") != "development":
        raise CausalValidationError("Probe was not explicitly fitted on development games")
    if training.get("dataset_sha256") != build.get("dataset_sha256"):
        raise CausalValidationError("Training manifest is bound to a different dataset")
    if training.get("build_manifest_sha256") != sha256_file(build_path):
        raise CausalValidationError("Training manifest is bound to a different build manifest")
    if training.get("labels_manifest_sha256") != build.get("labels_manifest_sha256"):
        raise CausalValidationError("Training manifest is bound to a different labels manifest")
    if training.get("source_code_sha256") != current_sources:
        raise CausalValidationError(
            "Training manifest source hashes differ from the current audited pipeline"
        )
    if training.get("contract_implementation_sha256") != current_contract_source_hash:
        raise CausalValidationError(
            "Training contract implementation differs from current operational definitions"
        )
    if training.get("split_manifest_sha256") != artifacts.get("splits_sha256"):
        raise CausalValidationError("Training manifest is bound to a different split file")
    if training.get("input_provenance_sha256") != build.get("input_provenance_sha256"):
        raise CausalValidationError(
            "Training manifest is bound to different activation input provenance"
        )
    if training.get("trunk_identity_bytes_sha256") != (
        build.get("input_provenance") or {}
    ).get("trunk_identity_bytes_sha256"):
        raise CausalValidationError(
            "Training manifest is bound to a different activation corpus commitment"
        )
    if concept not in set(map(str, training.get("concepts") or [])):
        raise CausalValidationError(f"Concept {concept!r} is absent from training manifest")
    if representation not in set(map(str, training.get("representations") or [])):
        raise CausalValidationError(
            f"Representation {representation!r} is absent from training manifest"
        )

    probe_rel = f"probes/{representation}/probe_{concept}.joblib"
    scaler_rel = f"probes/{representation}/scaler_{concept}.joblib"
    metadata_rel = f"probes/{representation}/probe_{concept}.meta.json"
    training_artifacts = training.get("artifacts")
    if not isinstance(training_artifacts, Mapping):
        raise CausalValidationError("Training manifest artifacts must be a path/hash mapping")
    for relative, expected_hash in training_artifacts.items():
        artifact_path = _inside(run_dir / str(relative), run_dir, "training artifact")
        _reject_archive_path(artifact_path, "Training artifact")
        _require_hash(artifact_path, expected_hash, f"training artifact {relative}")
    for relative in (probe_rel, scaler_rel, metadata_rel):
        if relative not in training_artifacts:
            raise CausalValidationError(f"Training manifest does not bind {relative}")
        # The complete artifact map was verified above.  Keep the explicit
        # membership checks here because these three files are executable
        # inputs rather than merely reports.

    metadata_path = run_dir / metadata_rel
    metadata = _read_json(metadata_path, "probe metadata")
    if metadata.get("representation") != representation:
        raise CausalValidationError("Probe metadata representation does not match requested variant")
    meta_concept = metadata.get("concept") or {}
    if meta_concept.get("name") != concept:
        raise CausalValidationError("Probe metadata concept does not match requested concept")
    if meta_concept.get("source") != contract.name:
        raise CausalValidationError("Probe label source differs from causal operational contract")
    if metadata.get("contract_id") != contract.definition_id:
        raise CausalValidationError("Probe metadata contract ID differs from causal contract")
    if metadata.get("contract_hash") != contract.contract_hash:
        raise CausalValidationError("Probe metadata contract hash differs from causal contract")
    if metadata.get("feature_mode") != "pre":
        raise CausalValidationError("Only pre-move probes can be causally intervened")
    if metadata.get("training_role") != "development":
        raise CausalValidationError("Probe metadata does not declare development-only fitting")
    if set(metadata.get("excluded_roles") or []) != {
        "control_calibration", "causal_test"
    }:
        raise CausalValidationError("Probe metadata does not exclude both causal holdout roles")
    for key, expected in (
        ("dataset_sha256", build.get("dataset_sha256")),
        ("split_manifest_sha256", artifacts.get("splits_sha256")),
        ("concepts_yaml_sha256", artifacts.get("concepts_yaml_sha256")),
        ("labels_manifest_sha256", build.get("labels_manifest_sha256")),
        ("input_provenance_sha256", build.get("input_provenance_sha256")),
        (
            "trunk_identity_bytes_sha256",
            (build.get("input_provenance") or {}).get(
                "trunk_identity_bytes_sha256"
            ),
        ),
    ):
        if metadata.get(key) != expected:
            raise CausalValidationError(f"Probe metadata {key} does not match upstream input")
    if metadata.get("source_code_sha256") != current_sources:
        raise CausalValidationError("Probe metadata source hashes differ from current pipeline")
    if metadata.get("contract_implementation_sha256") != current_contract_source_hash:
        raise CausalValidationError(
            "Probe metadata contract implementation differs from current definitions"
        )

    training_games = set(map(str, (metadata.get("final_fit") or {}).get("training_game_ids") or []))
    if not training_games:
        raise CausalValidationError("Probe metadata has no explicit training_game_ids")
    role_games = {
        role: set(splits.loc[splits["split_role"].eq(role), "game_id"].astype(str))
        for role in ("development", "control_calibration", "causal_test")
    }
    if not training_games.issubset(role_games["development"]):
        raise CausalValidationError("Probe training games are not a subset of development games")
    assert_disjoint_protocol(
        training_games, role_games["control_calibration"], role_games["causal_test"]
    )

    dataset = pd.read_parquet(dataset_path)
    required_dataset = {
        "row_id", "game_id", "move_number", "move_loc", "idx361", "split_role"
    }
    if not required_dataset.issubset(dataset.columns):
        raise CausalValidationError(
            f"Dataset lacks required columns: {sorted(required_dataset - set(dataset))}"
        )
    if dataset["row_id"].astype(str).duplicated().any():
        raise CausalValidationError("Dataset contains duplicate row IDs")
    role_by_game = splits.set_index(splits["game_id"].astype(str))["split_role"].to_dict()
    dataset_roles = dataset["game_id"].astype(str).map(role_by_game)
    if dataset_roles.isna().any() or not np.array_equal(
        dataset_roles.astype(str).to_numpy(), dataset["split_role"].astype(str).to_numpy()
    ):
        raise CausalValidationError("Dataset split_role values disagree with frozen game splits")
    board_size = int(build.get("board_size", 0))
    if board_size <= 1:
        raise CausalValidationError("Build manifest has an invalid board size")
    for row in dataset[["row_id", "move_loc", "idx361"]].itertuples(index=False):
        expected = flat_index_from_internal_loc(row.move_loc, board_size)
        observed = None if pd.isna(row.idx361) else int(row.idx361)
        if observed == board_size * board_size:
            observed = None
        if observed != expected:
            raise CausalValidationError(
                f"idx361 mismatch at {row.row_id}: stored {row.idx361!r}, expected {expected!r}"
            )

    games_dir = Path(str(manifest.get("source_games_dir", ""))).resolve()
    _reject_archive_path(games_dir, "Raw games directory")
    if not games_dir.is_dir():
        raise FileNotFoundError(f"Raw games directory does not exist: {games_dir}")
    fresh_holdout = _verify_fresh_holdout_manifest(
        manifest,
        build,
        splits,
        games_dir,
    )
    checkpoint_activation_fidelity = _verify_development_activation_fidelity(
        run_dir,
        manifest,
        build,
        training,
        metadata,
        splits,
        fresh_holdout,
    )
    return ValidatedRun(
        run_dir=run_dir,
        manifest=manifest,
        build_manifest=build,
        training_manifest=training,
        splits=splits,
        dataset=dataset,
        games_dir=games_dir,
        labels_dir=labels_dir,
        concept=concept,
        representation=representation,
        probe_path=run_dir / probe_rel,
        scaler_path=run_dir / scaler_rel,
        probe_metadata_path=metadata_path,
        probe_metadata=metadata,
        fresh_holdout=fresh_holdout,
        checkpoint_activation_fidelity=checkpoint_activation_fidelity,
    )


@dataclass(frozen=True)
class InterventionDirection:
    """Raw-space probe direction scaled to one saved linear-score unit."""

    representation: str
    channels: int
    global_delta: Optional[np.ndarray]
    local_delta: Optional[np.ndarray]
    raw_norm: float
    source: str = "trained"

    @classmethod
    def from_probe_objects(
        cls,
        probe: Any,
        scaler: Any,
        *,
        representation: str,
        channels: int,
        source: str = "trained",
    ) -> "InterventionDirection":
        if representation not in REPRESENTATIONS:
            raise ValueError(f"Unknown representation: {representation}")
        coef = np.asarray(probe.coef_, dtype=np.float64)
        if coef.ndim != 2 or coef.shape[0] != 1:
            raise CausalValidationError(f"Expected one binary probe coefficient row, got {coef.shape}")
        coef = coef[0]
        scale = np.asarray(scaler.scale_, dtype=np.float64)
        if coef.shape != scale.shape:
            raise CausalValidationError("Probe coefficient and scaler dimensions differ")
        expected_features = channels if representation != "combined" else 2 * channels
        if coef.size != expected_features:
            raise CausalValidationError(
                f"{representation} probe has {coef.size} features; expected {expected_features}"
            )
        if np.any(~np.isfinite(coef)) or np.any(~np.isfinite(scale)) or np.any(scale <= 0):
            raise CausalValidationError("Probe/scaler contains non-finite or non-positive scales")
        raw = coef / scale
        norm_sq = float(raw @ raw)
        if not math.isfinite(norm_sq) or norm_sq <= 0:
            raise CausalValidationError("Probe raw-space direction has zero or invalid norm")
        delta = (raw / norm_sq).astype(np.float32)
        if representation == "global":
            global_delta, local_delta = delta, None
        elif representation == "local":
            global_delta, local_delta = None, delta
        else:
            global_delta, local_delta = delta[:channels], delta[channels:]
        return cls(
            representation=representation,
            channels=int(channels),
            global_delta=global_delta,
            local_delta=local_delta,
            raw_norm=float(math.sqrt(norm_sq)),
            source=source,
        )

    @classmethod
    def load(cls, run: ValidatedRun) -> "InterventionDirection":
        probe = joblib.load(run.probe_path)
        scaler = joblib.load(run.scaler_path)
        direction = cls.from_probe_objects(
            probe,
            scaler,
            representation=run.representation,
            channels=run.channels,
        )
        if int(run.probe_metadata.get("n_features", -1)) != (
            run.channels if run.representation != "combined" else 2 * run.channels
        ):
            raise CausalValidationError("Probe metadata n_features is inconsistent")
        return direction

    def flattened(self) -> np.ndarray:
        pieces = []
        if self.global_delta is not None:
            pieces.append(np.asarray(self.global_delta, dtype=np.float64))
        if self.local_delta is not None:
            pieces.append(np.asarray(self.local_delta, dtype=np.float64))
        return np.concatenate(pieces)

    def metadata(self) -> Dict[str, Any]:
        """Reproducible math for translating a standardized probe into a hook."""
        flattened = self.flattened()
        return {
            "representation": self.representation,
            "channels": self.channels,
            "source": self.source,
            "standardized_probe": "s(x) = beta^T ((x - mu) / sigma) + intercept",
            "raw_space_gradient": "g = beta / sigma (elementwise)",
            "unit_score_direction": "delta = g / (g^T g)",
            "unit_score_identity": "g^T delta = 1",
            "combined_denominator": (
                "For combined probes, g^T g is computed once over concatenated "
                "[global, local] blocks before splitting delta into channel blocks."
            ),
            "hook": (
                "h'[c,y,x] = h[c,y,x] + dose * "
                "(delta_global[c] + mask[y,x] * delta_local[c]); absent blocks are zero."
            ),
            "mask_scaling": (
                "Aligned masks are positive on exact contract target actions, negative on "
                "all other legal board actions with equal total positive/negative mass, "
                "zero on ineligible points, then divided by RMS over the active legal support."
            ),
            "dose_caveat": (
                "A unit dose is one raw probe-score unit for an unmasked delta. For a local "
                "spatial intervention the score shift at a point is additionally scaled by "
                "that point's mask value; nominal and effective doses are both reported."
            ),
            "raw_gradient_l2": self.raw_norm,
            "delta_l2": float(np.linalg.norm(flattened)),
            "global_delta_l2": (
                None if self.global_delta is None else float(np.linalg.norm(self.global_delta))
            ),
            "local_delta_l2": (
                None if self.local_delta is None else float(np.linalg.norm(self.local_delta))
            ),
        }

    def random_control(self, control_id: str, *, seed: int) -> "InterventionDirection":
        """Create deterministic random channels with every active block norm matched."""

        rng = np.random.default_rng(
            stable_sha256_seed("random-channel-direction", int(seed), control_id)
        )

        def randomized_block(target: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if target is None:
                return None
            target = np.asarray(target, dtype=np.float64)
            target_norm = float(np.linalg.norm(target))
            if target_norm == 0.0:
                return np.zeros_like(target, dtype=np.float32)
            candidate = rng.normal(size=target.shape)
            candidate_norm = float(np.linalg.norm(candidate))
            if candidate_norm <= 0:
                raise RuntimeError("Random direction unexpectedly has zero norm")
            return (candidate * (target_norm / candidate_norm)).astype(np.float32)

        global_delta = randomized_block(self.global_delta)
        local_delta = randomized_block(self.local_delta)
        return InterventionDirection(
            self.representation,
            self.channels,
            global_delta,
            local_delta,
            self.raw_norm,
            source=control_id,
        )

    @contextmanager
    def apply(self, model: Any, dose: float, spatial_mask: Optional[np.ndarray]) -> Iterator[None]:
        """Hook ``act_trunkfinal`` for one or more model evaluations."""
        dose = float(dose)
        if dose == 0.0:
            yield
            return
        if self.local_delta is not None and spatial_mask is None:
            raise CausalValidationError(
                f"{self.representation} intervention requires an operational spatial mask"
            )
        if self.local_delta is None and spatial_mask is not None:
            # A global variant has no local component.  Silently consuming a
            # mask would make a claimed spatial control fictitious.
            raise CausalValidationError("A global-only intervention cannot consume a spatial mask")

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            import torch

            if output.ndim != 4 or int(output.shape[1]) != self.channels:
                raise CausalValidationError(
                    f"Expected trunk N,C,H,W with C={self.channels}, got {tuple(output.shape)}"
                )
            delta = torch.zeros_like(output)
            if self.global_delta is not None:
                global_delta = torch.as_tensor(
                    self.global_delta, dtype=output.dtype, device=output.device
                ).view(1, -1, 1, 1)
                delta = delta + global_delta
            if self.local_delta is not None:
                mask = np.asarray(spatial_mask, dtype=np.float32)
                if mask.ndim != 2 or tuple(mask.shape) != tuple(output.shape[-2:]):
                    raise CausalValidationError(
                        f"Spatial mask shape {mask.shape} does not match trunk {tuple(output.shape[-2:])}"
                    )
                local_delta = torch.as_tensor(
                    self.local_delta, dtype=output.dtype, device=output.device
                ).view(1, -1, 1, 1)
                torch_mask = torch.as_tensor(mask, dtype=output.dtype, device=output.device)
                delta = delta + local_delta * torch_mask.view(1, 1, *mask.shape)
            return output + dose * delta

        handle = model.act_trunkfinal.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()


def normalized_policy(
    outputs: Mapping[str, Any],
    board: Any,
    *,
    support: PolicySupport,
    time: PolicyTime = PolicyTime.PRE_MOVE,
) -> PolicyView:
    """Normalize a model output over its explicitly declared legal support."""
    return legal_policy_view(
        outputs,
        board,
        time=time,
        support=support,
        coordinate_system="board_loc",
    )


def policy_disruption(
    baseline: Mapping[str, Any], intervened: Mapping[str, Any], board: Any
) -> Dict[str, float]:
    """Downstream disruption over normalized legal board actions plus pass."""
    p = normalized_policy(baseline, board, support=PolicySupport.LEGAL_PLUS_PASS)
    q = normalized_policy(intervened, board, support=PolicySupport.LEGAL_PLUS_PASS)
    locations = sorted(set(p.probabilities) | set(q.probabilities))
    pv = np.asarray([p.probability(loc) for loc in locations], dtype=np.float64)
    qv = np.asarray([q.probability(loc) for loc in locations], dtype=np.float64)
    midpoint = 0.5 * (pv + qv)
    eps = 1e-15
    js = 0.5 * np.sum(pv * np.log((pv + eps) / (midpoint + eps)))
    js += 0.5 * np.sum(qv * np.log((qv + eps) / (midpoint + eps)))
    return {
        "policy_js": float(js),
        "policy_l1": float(np.abs(pv - qv).sum()),
        "top_move_flip": float(int(locations[int(np.argmax(pv))] != locations[int(np.argmax(qv))])),
    }


def _flatten_readouts(readouts: Mapping[str, Any]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for name, value in readouts.items():
        if isinstance(value, Mapping) or isinstance(value, str):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result[name] = number
    return result


def concept_policy_readouts(
    concept: str,
    outputs: Mapping[str, Any],
    board: Any,
    *,
    previous_move: Optional[int],
    candidate_reply_peaks: Optional[Mapping[int, float]] = None,
    anchor_region: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluate exactly the readout registered by the concept contract."""
    contract = get_contract(concept)
    if contract.name == "tenuki_distance6":
        policy = normalized_policy(
            outputs, board, support=PolicySupport.LEGAL_BOARD_CONDITIONAL
        )
        return _flatten_readouts(tenuki_policy_readouts(policy, board, previous_move))
    if contract.name == "reply_peak95":
        if candidate_reply_peaks is None:
            raise CausalValidationError(
                "reply_peak95 requires a frozen candidate post-move reply-peak map"
            )
        policy = normalized_policy(
            outputs, board, support=PolicySupport.LEGAL_BOARD_CONDITIONAL
        )
        return _flatten_readouts(
            reply_peak95_candidate_readouts(policy, candidate_reply_peaks)
        )
    if contract.name == "regional_policy_peak":
        policy = normalized_policy(
            outputs, board, support=PolicySupport.LEGAL_BOARD_CONDITIONAL
        )
        return _flatten_readouts(
            regional_policy_readouts(policy, board, anchor_region=anchor_region)
        )
    raise CausalValidationError(f"No causal readout is registered for {contract.definition_id}")


def concept_spatial_mask(
    concept: str,
    baseline: Mapping[str, Any],
    board: Any,
    *,
    previous_move: Optional[int],
    candidate_reply_peaks: Optional[Mapping[int, float]] = None,
    anchor_region: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[str]]:
    """Return the exact contract-aligned contrast mask and any fixed anchor."""
    contract = get_contract(concept)
    if contract.name == "tenuki_distance6":
        return tenuki_contrast_mask(board, previous_move), None
    if contract.name == "reply_peak95":
        if candidate_reply_peaks is None:
            raise CausalValidationError("reply_peak95 mask requires candidate reply peaks")
        return reply_peak95_contrast_mask(board, candidate_reply_peaks), None
    if contract.name == "regional_policy_peak":
        policy = normalized_policy(
            baseline, board, support=PolicySupport.LEGAL_BOARD_CONDITIONAL
        )
        if anchor_region is None:
            anchor_region = str(regional_policy_readouts(policy, board)["regional_policy_peak_region"])
        return regional_peak_contrast_mask(
            board, policy, anchor_region=anchor_region
        ), anchor_region
    raise CausalValidationError(f"No causal mask is registered for {contract.definition_id}")


def _load_moves(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing raw moves: {path}")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _quantile_labels_from_development_thresholds(
    run: ValidatedRun, frame: pd.DataFrame
) -> np.ndarray:
    """Apply the final development-only threshold without consulting holdouts."""

    spec = run.probe_metadata.get("concept") or {}
    if spec.get("type") != "quantile" and spec.get("kind") != "quantile":
        raise CausalValidationError(
            f"Dataset lacks label_{run.concept}, and probe metadata is not quantile"
        )
    raw_column = f"rawval_{run.concept}"
    if raw_column not in frame:
        raise CausalValidationError(f"Dataset lacks {raw_column}")
    thresholds = (run.probe_metadata.get("final_fit") or {}).get(
        "quantile_thresholds"
    )
    if thresholds is None:
        raise CausalValidationError(
            "Quantile probe metadata lacks final development-only thresholds"
        )

    allowed = np.ones(len(frame), dtype=bool)
    for rule in spec.get("filters") or ():
        column = str(rule["column"])
        if column not in frame:
            raise CausalValidationError(
                f"Quantile concept filter requires missing column {column!r}"
            )
        values = frame[column].to_numpy()
        target = rule["value"]
        operator = rule["operator"]
        if operator == "<=":
            current = values <= target
        elif operator == ">=":
            current = values >= target
        elif operator == "==":
            current = values == target
        elif operator == "!=":
            current = values != target
        elif operator == "<":
            current = values < target
        elif operator == ">":
            current = values > target
        else:
            raise CausalValidationError(f"Unsupported filter operator {operator!r}")
        allowed &= ~pd.isna(values) & np.asarray(current, dtype=bool)

    raw = frame[raw_column].to_numpy(dtype=float)
    values = np.abs(raw) if bool(spec.get("use_abs", False)) else raw
    phases = frame["game_phase"].astype(str).to_numpy()
    labels = np.full(len(frame), np.nan, dtype=float)
    if isinstance(thresholds, Mapping):
        threshold_rows = [thresholds.get(phase) for phase in phases]
    else:
        threshold_rows = [thresholds] * len(frame)
    direction = spec.get("direction")
    no_drop = bool(spec.get("no_drop", False))
    for index, pair in enumerate(threshold_rows):
        if (
            pair is None
            or not allowed[index]
            or not math.isfinite(values[index])
            or len(pair) != 2
        ):
            continue
        low, high = map(float, pair)
        if direction == "high":
            if values[index] >= high:
                labels[index] = 1.0
            elif values[index] <= low or no_drop:
                labels[index] = 0.0
        elif direction == "low":
            if values[index] <= low:
                labels[index] = 1.0
            elif values[index] >= high or no_drop:
                labels[index] = 0.0
        else:
            raise CausalValidationError(
                f"Quantile concept has invalid direction {direction!r}"
            )
    return labels


def select_positions(
    run: ValidatedRun,
    role: str,
    maximum: int,
    *,
    seed: int,
) -> pd.DataFrame:
    """Deterministically sample a label-stratified set from one causal role."""
    if role not in CAUSAL_ROLES:
        raise ValueError(f"role must be one of {CAUSAL_ROLES}")
    if maximum <= 0:
        raise ValueError("maximum positions must be positive")
    frame = run.dataset.loc[run.dataset["split_role"].eq(role)].copy()
    label_column = f"label_{run.concept}"
    if label_column not in frame:
        frame[label_column] = _quantile_labels_from_development_thresholds(run, frame)
    # Compute before dropping unlabeled rows.  The tenuki contract anchors to
    # the most recent observed non-pass move, not merely the prior eligible row
    # and not a pass that leaves no board coordinate.
    frame = frame.sort_values(["game_id", "move_number"])
    anchors = pd.Series(np.nan, index=frame.index, dtype=float)
    for _game_id, group in frame.groupby("game_id", sort=False):
        most_recent_nonpass: Optional[int] = None
        for index, move_loc in group["move_loc"].items():
            if most_recent_nonpass is not None:
                anchors.at[index] = float(most_recent_nonpass)
            if not pd.isna(move_loc) and int(move_loc) != 0:
                most_recent_nonpass = int(move_loc)
    frame["previous_move_loc"] = anchors
    frame = frame.loc[frame[label_column].notna()].copy()
    if run.representation in {"local", "combined"}:
        if "has_local" not in frame:
            raise CausalValidationError("Dataset lacks the fail-closed has_local indicator")
        frame = frame.loc[frame["has_local"].astype(bool)].copy()
    if get_contract(run.concept).name == "tenuki_distance6":
        frame = frame.loc[
            frame["previous_move_loc"].notna() & frame["previous_move_loc"].ne(0)
        ].copy()
    if frame.empty:
        raise CausalValidationError(f"No eligible {role} positions for {run.concept}")

    frame["_label_stratum"] = frame[label_column].astype(int)
    labels = sorted(frame["_label_stratum"].unique().tolist())
    if maximum < len(labels):
        raise CausalValidationError(
            f"Requested {maximum} {role} positions cannot represent all {len(labels)} "
            "label strata"
        )
    games = sorted(frame["game_id"].astype(str).unique().tolist())
    if len(games) < maximum:
        raise CausalValidationError(
            f"Need {maximum} independent {role} games for {run.concept}, but only "
            f"{len(games)} games contain eligible positions"
        )
    base, remainder = divmod(int(maximum), len(labels))
    quotas = {
        int(label): base + int(index < remainder)
        for index, label in enumerate(labels)
    }
    availability = {
        int(label): int(
            frame.loc[frame["_label_stratum"].eq(label), "game_id"]
            .astype(str)
            .nunique()
        )
        for label in labels
    }
    insufficient = {
        label: {"required": quotas[label], "available_games": availability[label]}
        for label in quotas
        if availability[label] < quotas[label]
    }
    if insufficient:
        raise CausalValidationError(
            f"Cannot satisfy balanced {role} label quotas without game reuse: {insufficient}"
        )

    # Assign one distinct game to every label-quota slot using a deterministic
    # minimum-cost bipartite matching.  This avoids a greedy allocation failing
    # when games contain eligible positions in more than one stratum.
    from scipy.optimize import linear_sum_assignment

    slots = [label for label in labels for _ in range(quotas[int(label)])]
    game_labels = {
        game_id: set(
            frame.loc[frame["game_id"].astype(str).eq(game_id), "_label_stratum"]
            .astype(int)
            .tolist()
        )
        for game_id in games
    }
    forbidden_cost = 1e30
    cost = np.full((len(slots), len(games)), forbidden_cost, dtype=np.float64)
    for slot_index, label in enumerate(slots):
        for game_index, game_id in enumerate(games):
            if int(label) not in game_labels[game_id]:
                continue
            rank = stable_sha256_seed(
                "independent-position-game", int(seed), role, int(label), game_id
            )
            cost[slot_index, game_index] = float(rank % (2**53)) / float(2**53)
    slot_indices, game_indices = linear_sum_assignment(cost)
    if len(slot_indices) != len(slots) or np.any(
        cost[slot_indices, game_indices] >= forbidden_cost
    ):
        raise CausalValidationError(
            f"Balanced {role} label quotas are infeasible without reusing a game: "
            f"quotas={quotas}, availability={availability}"
        )

    selected_rows: List[pd.Series] = []
    for slot_index, game_index in zip(slot_indices, game_indices):
        label = int(slots[int(slot_index)])
        game_id = games[int(game_index)]
        candidates = frame.loc[
            frame["game_id"].astype(str).eq(game_id)
            & frame["_label_stratum"].eq(label)
        ]
        if candidates.empty:
            raise RuntimeError("Bipartite position selection returned an ineligible game")
        chosen_index = min(
            candidates.index,
            key=lambda index: (
                stable_sha256_seed(
                    "independent-position-within-game",
                    int(seed),
                    role,
                    label,
                    game_id,
                    str(candidates.at[index, "row_id"]),
                ),
                str(candidates.at[index, "row_id"]),
            ),
        )
        selected_rows.append(frame.loc[chosen_index])
    selected = pd.DataFrame(selected_rows)
    if len(selected) != maximum or selected["game_id"].astype(str).duplicated().any():
        raise CausalValidationError(
            "Independent position selection did not produce exactly one row per requested game"
        )
    observed_quotas = selected["_label_stratum"].value_counts().to_dict()
    if any(int(observed_quotas.get(label, 0)) != quota for label, quota in quotas.items()):
        raise CausalValidationError(
            f"Position selection label counts differ from quotas: "
            f"expected={quotas}, observed={observed_quotas}"
        )
    selected["selection_stratum"] = selected["_label_stratum"].astype(int)
    selected["selection_quota"] = selected["selection_stratum"].map(quotas).astype(int)
    selected["selection_unit"] = "one_position_per_game"
    result = selected.sort_values(["game_id", "move_number"]).drop(columns="_label_stratum")
    result["position_id"] = result["game_id"].astype(str) + ":" + result["move_number"].astype(str)
    compact_columns = [
        "position_id", "row_id", "game_id", "move_number", "player", "move_loc",
        "idx361", "has_local", "split_role", "previous_move_loc", label_column,
        "selection_stratum", "selection_quota", "selection_unit",
    ]
    compact_columns = [column for column in compact_columns if column in result]
    return result[compact_columns].reset_index(drop=True)


def _clone_game_state(gs: Any) -> Any:
    """Clone the small mutable replay shell while preserving complete history."""
    from gamestate import GameState

    clone = GameState(gs.board_size, gs.rules)
    clone.board = gs.board.copy()
    clone.moves = list(gs.moves)
    clone.boards = [board.copy() for board in gs.boards]
    clone.redo_stack = []
    return clone


@dataclass
class ReplayPosition:
    position_id: str
    game_id: str
    move_number: int
    idx361: Optional[int]
    move_loc: int
    previous_move: Optional[int]
    split_role: str
    label: int
    game_state: Any
    baseline: Mapping[str, Any]
    spatial_mask: Optional[np.ndarray]
    anchor_region: Optional[str]
    candidate_reply_peaks: Optional[Mapping[int, float]]
    trunkfinal_path: Optional[Path] = None
    trunkfinal_sha256: Optional[str] = None
    trunkfinal: Optional[np.ndarray] = None
    model_board_mask: Optional[np.ndarray] = None

    @property
    def board(self) -> Any:
        return self.game_state.board


def _load_bound_trunkfinal(run: ValidatedRun, game_id: str, move_number: int) -> Tuple[Path, str, np.ndarray]:
    """Load one immutable causal input and bind its identity to its exact bytes."""

    game_root = _inside(run.games_dir / str(game_id), run.games_dir, "raw game directory")
    path = _inside(
        game_root / "trunkfinal" / f"move_{int(move_number):03d}.npy",
        game_root,
        "saved pre-move trunkfinal",
    )
    _reject_archive_path(path, "Saved pre-move trunkfinal")
    if not path.is_file():
        raise FileNotFoundError(f"Missing saved pre-move trunkfinal: {path}")
    provenance = run.build_manifest.get("input_provenance") or {}
    game_record = (provenance.get("games") or {}).get(str(game_id))
    filename = path.name
    leaf = (
        (game_record.get("files") or {}).get(filename)
        if isinstance(game_record, Mapping)
        else None
    )
    identity = f"{game_id}/trunkfinal/{filename}"
    if not isinstance(leaf, Mapping) or leaf.get("identity") != identity:
        raise CausalValidationError(
            f"Build provenance lacks the exact selected activation leaf {identity}"
        )
    if int(leaf.get("bytes", -1)) != int(path.stat().st_size):
        raise CausalValidationError(
            f"Selected activation size differs from build-time leaf: {identity}"
        )
    digest = sha256_file(path)
    if digest != leaf.get("sha256"):
        raise CausalValidationError(
            f"Selected activation bytes differ from build-time leaf: {identity}"
        )
    activation = np.load(path, allow_pickle=False)
    expected_shape = (run.channels, run.board_size, run.board_size)
    if activation.shape != expected_shape:
        raise CausalValidationError(
            f"Saved trunkfinal {path} has shape {activation.shape}; expected {expected_shape}"
        )
    if not np.issubdtype(activation.dtype, np.floating):
        raise CausalValidationError(f"Saved trunkfinal {path} is not floating point")
    if not np.all(np.isfinite(activation)):
        raise CausalValidationError(f"Saved trunkfinal {path} contains non-finite values")
    # The array is copied so a later filesystem mutation cannot change the
    # in-memory causal input.  The digest is persisted beside every result.
    frozen = np.asarray(activation, dtype=np.float32).copy()
    frozen.setflags(write=False)
    return path, digest, frozen


def _activation_binding_digest(positions: Sequence[ReplayPosition]) -> str:
    """Hash selected position identity, source-relative path, and file digest."""

    digest = hashlib.sha256()
    for position in sorted(positions, key=lambda item: item.position_id):
        if position.trunkfinal_path is None or position.trunkfinal_sha256 is None:
            raise CausalValidationError(
                f"Replay position {position.position_id} has no bound trunkfinal"
            )
        try:
            relative = position.trunkfinal_path.resolve().relative_to(
                position.trunkfinal_path.parents[2].resolve()
            )
        except (ValueError, IndexError):
            relative = position.trunkfinal_path.name
        record = (
            f"{position.position_id}\0{relative}\0{position.trunkfinal_sha256}\n"
        ).encode("utf-8")
        digest.update(record)
    return digest.hexdigest()


class PolicyHeadOnlyBackend:
    """Exact batched policy evaluation from saved ``act_trunkfinal`` outputs.

    ``trunkfinal`` files are outputs of ``model.act_trunkfinal``.  The next
    operation in KataGo's main network is ``model.policy_head``; therefore an
    intervention at this hook can be evaluated without replaying the trunk.
    The backend is not trusted until :meth:`validate_equivalence` compares a
    deterministic sample against complete ``GameState.get_model_outputs``
    evaluations from the same checkpoint.
    """

    def __init__(
        self,
        model: Any,
        *,
        channels: int,
        board_size: int,
        batch_size: int = DEFAULT_HEAD_BATCH_SIZE,
        policy_atol: float = DEFAULT_POLICY_EQUIVALENCE_ATOL,
        activation_atol: float = DEFAULT_ACTIVATION_EQUIVALENCE_ATOL,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("policy-head batch_size must be positive")
        if policy_atol <= 0 or activation_atol <= 0:
            raise ValueError("equivalence tolerances must be positive")
        if not hasattr(model, "policy_head"):
            raise CausalValidationError("Loaded model has no policy_head")
        if int(getattr(model, "pos_len", board_size)) != int(board_size):
            raise CausalValidationError(
                "Policy-head backend requires model.pos_len to equal the saved activation board size"
            )
        self.model = model
        self.channels = int(channels)
        self.board_size = int(board_size)
        self.batch_size = int(batch_size)
        self.policy_atol = float(policy_atol)
        self.activation_atol = float(activation_atol)
        self.validated = False
        self.equivalence_report: Optional[Dict[str, Any]] = None

    @property
    def device(self) -> Any:
        try:
            return self.model.device
        except AttributeError:
            import torch

            try:
                return next(self.model.policy_head.parameters()).device
            except StopIteration:
                return torch.device("cpu")

    def _validate_position_inputs(self, position: ReplayPosition) -> None:
        if position.trunkfinal is None or position.trunkfinal_sha256 is None:
            raise CausalValidationError(
                f"Position {position.position_id} lacks a hash-bound saved trunkfinal"
            )
        if position.trunkfinal.shape != (
            self.channels,
            self.board_size,
            self.board_size,
        ):
            raise CausalValidationError(
                f"Position {position.position_id} has invalid trunkfinal shape "
                f"{position.trunkfinal.shape}"
            )
        if position.model_board_mask is None:
            raise CausalValidationError(
                f"Position {position.position_id} lacks the policy-head board mask"
            )
        if position.model_board_mask.shape != (self.board_size, self.board_size):
            raise CausalValidationError(
                f"Position {position.position_id} has invalid board-mask shape"
            )

    def _perturbed_activation(
        self,
        position: ReplayPosition,
        direction: Optional[InterventionDirection],
        dose: float,
        spatial_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        self._validate_position_inputs(position)
        activation = np.asarray(position.trunkfinal, dtype=np.float32)
        if direction is None or float(dose) == 0.0:
            return activation
        if direction.channels != self.channels:
            raise CausalValidationError("Intervention direction and trunk channels differ")
        if direction.local_delta is not None and spatial_mask is None:
            raise CausalValidationError(
                f"{direction.representation} intervention requires a spatial mask"
            )
        if direction.local_delta is None and spatial_mask is not None:
            raise CausalValidationError(
                "A global-only intervention cannot consume a spatial mask"
            )
        delta = np.zeros_like(activation)
        if direction.global_delta is not None:
            delta += np.asarray(direction.global_delta, dtype=np.float32)[:, None, None]
        if direction.local_delta is not None:
            mask = np.asarray(spatial_mask, dtype=np.float32)
            if mask.shape != (self.board_size, self.board_size):
                raise CausalValidationError(
                    f"Spatial mask {mask.shape} does not match saved trunkfinal board"
                )
            if not np.all(np.isfinite(mask)):
                raise CausalValidationError("Spatial mask contains non-finite values")
            delta += (
                np.asarray(direction.local_delta, dtype=np.float32)[:, None, None]
                * mask[None, :, :]
            )
        return activation + np.float32(dose) * delta

    def _outputs_from_probabilities(
        self, position: ReplayPosition, probabilities: np.ndarray
    ) -> Dict[str, Any]:
        from board import Board

        expected = self.board_size * self.board_size + 1
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.shape != (expected,):
            raise CausalValidationError(
                f"Policy head returned {probabilities.shape}; expected {(expected,)}"
            )
        pairs: List[Tuple[int, float]] = []
        board = position.board
        for tensor_index in range(expected - 1):
            x = tensor_index % self.board_size
            y = tensor_index // self.board_size
            loc = board.loc(x, y)
            if board.would_be_legal(board.pla, loc):
                pairs.append((int(loc), float(probabilities[tensor_index])))
        pairs.append((int(Board.PASS_LOC), float(probabilities[-1])))
        return {
            "policy0": probabilities,
            "moves_and_probs0": pairs,
            "evaluation_backend": "saved_trunkfinal_policy_head",
        }

    def evaluate(
        self,
        positions: Sequence[ReplayPosition],
        *,
        direction: Optional[InterventionDirection] = None,
        dose: float = 0.0,
        spatial_masks: Optional[Sequence[Optional[np.ndarray]]] = None,
    ) -> List[Dict[str, Any]]:
        """Evaluate channel-0 policy softmax in exact policy-head batches."""

        import torch

        if not positions:
            return []
        if spatial_masks is None:
            spatial_masks = [None] * len(positions)
        if len(spatial_masks) != len(positions):
            raise ValueError("spatial_masks and positions must have equal length")
        result: List[Dict[str, Any]] = []
        self.model.eval()
        for start in range(0, len(positions), self.batch_size):
            batch_positions = positions[start : start + self.batch_size]
            batch_masks = spatial_masks[start : start + self.batch_size]
            activations = np.stack([
                self._perturbed_activation(position, direction, dose, mask)
                for position, mask in zip(batch_positions, batch_masks)
            ])
            board_masks = np.stack([
                np.asarray(position.model_board_mask, dtype=np.float32)
                for position in batch_positions
            ])[:, None, :, :]
            torch_activations = torch.as_tensor(
                activations, dtype=torch.float32, device=self.device
            )
            torch_masks = torch.as_tensor(
                board_masks, dtype=torch.float32, device=self.device
            )
            mask_sum_hw = torch.sum(torch_masks, dim=(2, 3), keepdim=True)
            mask_sum = torch.sum(torch_masks)
            with torch.no_grad():
                logits = self.model.policy_head(
                    torch_activations,
                    mask=torch_masks,
                    mask_sum_hw=mask_sum_hw,
                    mask_sum=mask_sum,
                    extra_outputs=None,
                )
                if logits.ndim != 3 or int(logits.shape[1]) < 1:
                    raise CausalValidationError(
                        f"policy_head returned invalid shape {tuple(logits.shape)}"
                    )
                probabilities = torch.nn.functional.softmax(
                    logits[:, 0, :], dim=1
                ).detach().cpu().numpy()
            result.extend(
                self._outputs_from_probabilities(position, probability)
                for position, probability in zip(batch_positions, probabilities)
            )
        return result

    def baseline_outputs(
        self, positions: Sequence[ReplayPosition]
    ) -> List[Dict[str, Any]]:
        return self.evaluate(positions)

    def validate_equivalence(
        self,
        positions: Sequence[ReplayPosition],
        *,
        seed: int,
        sample_size: int = DEFAULT_EQUIVALENCE_SAMPLE_SIZE,
    ) -> Dict[str, Any]:
        """Fail closed unless saved-activation head policies equal full replay."""

        if sample_size <= 0:
            raise ValueError("equivalence sample_size must be positive")
        if not positions:
            raise CausalValidationError("Cannot validate policy-head equivalence without positions")
        ranked = sorted(
            positions,
            key=lambda position: (
                stable_sha256_seed(
                    "policy-head-equivalence", int(seed), position.position_id
                ),
                position.position_id,
            ),
        )
        sample = ranked[: min(int(sample_size), len(ranked))]
        reused_cached_baselines = all(
            isinstance(position.baseline, Mapping) and "policy0" in position.baseline
            for position in sample
        )
        head_outputs = (
            [dict(position.baseline) for position in sample]
            if reused_cached_baselines
            else self.baseline_outputs(sample)
        )
        rows: List[Dict[str, Any]] = []
        for position, head_output in zip(sample, head_outputs):
            full_output = position.game_state.get_model_outputs(
                self.model, extra_output_names=["trunkfinal"]
            )
            head_policy = np.asarray(head_output["policy0"], dtype=np.float64)
            full_policy = np.asarray(full_output["policy0"], dtype=np.float64)
            if head_policy.shape != full_policy.shape:
                raise CausalValidationError(
                    f"Policy shape mismatch at {position.position_id}: "
                    f"head-only {head_policy.shape}, full {full_policy.shape}"
                )
            policy_abs = np.abs(head_policy - full_policy)
            replayed_trunk = np.asarray(full_output.get("trunkfinal"), dtype=np.float32)
            if replayed_trunk.shape != np.asarray(position.trunkfinal).shape:
                raise CausalValidationError(
                    f"Replayed trunkfinal shape mismatch at {position.position_id}"
                )
            activation_abs = np.abs(
                replayed_trunk - np.asarray(position.trunkfinal, dtype=np.float32)
            )
            row = {
                "position_id": position.position_id,
                "split_role": position.split_role,
                "saved_trunkfinal_sha256": position.trunkfinal_sha256,
                "policy_max_abs_error": float(policy_abs.max(initial=0.0)),
                "policy_l1_error": float(policy_abs.sum()),
                "activation_max_abs_error": float(activation_abs.max(initial=0.0)),
                "activation_mean_abs_error": float(activation_abs.mean()),
            }
            rows.append(row)
            if row["policy_max_abs_error"] > self.policy_atol:
                raise CausalValidationError(
                    "Saved-trunk policy-head equivalence failed at "
                    f"{position.position_id}: max policy error "
                    f"{row['policy_max_abs_error']:.3g} > {self.policy_atol:.3g}"
                )
            if row["activation_max_abs_error"] > self.activation_atol:
                raise CausalValidationError(
                    "Saved trunkfinal does not reproduce the requested checkpoint at "
                    f"{position.position_id}: max activation error "
                    f"{row['activation_max_abs_error']:.3g} > "
                    f"{self.activation_atol:.3g}"
                )
        report = {
            "status": "validated",
            "backend": "saved_trunkfinal_policy_head",
            "sample_selection": "lowest deterministic SHA-256-derived ranks",
            "seed": int(seed),
            "sample_size_requested": int(sample_size),
            "sample_size_evaluated": len(rows),
            "policy_max_abs_tolerance": self.policy_atol,
            "activation_max_abs_tolerance": self.activation_atol,
            "max_policy_abs_error": max(row["policy_max_abs_error"] for row in rows),
            "max_policy_l1_error": max(row["policy_l1_error"] for row in rows),
            "max_activation_abs_error": max(
                row["activation_max_abs_error"] for row in rows
            ),
            "positions": rows,
            "full_network_forwards": len(rows),
            "additional_policy_head_batches": (
                0 if reused_cached_baselines else math.ceil(len(rows) / self.batch_size)
            ),
            "cached_head_baselines_reused": reused_cached_baselines,
        }
        self.validated = True
        self.equivalence_report = report
        return report


def _candidate_maps_from_frame(frame: pd.DataFrame) -> Dict[str, Dict[int, float]]:
    required = {"position_id", "candidate_loc", "reply_peak"}
    if not required.issubset(frame.columns):
        raise CausalValidationError(f"Candidate cache lacks {sorted(required - set(frame))}")
    if frame.duplicated(["position_id", "candidate_loc"]).any():
        raise CausalValidationError("Candidate cache contains duplicate position/action rows")
    maps: Dict[str, Dict[int, float]] = {}
    for position_id, group in frame.groupby("position_id", sort=False):
        values = {int(row.candidate_loc): float(row.reply_peak) for row in group.itertuples()}
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values.values()):
            raise CausalValidationError(f"Candidate cache has invalid reply peak at {position_id}")
        maps[str(position_id)] = values
    return maps


def _load_forcing_cache(
    run: ValidatedRun,
    cache_dir: Path,
    *,
    checkpoint_sha256: str,
) -> Dict[str, Dict[int, float]]:
    cache_dir = cache_dir.resolve()
    _inside(cache_dir, run.run_dir, "forcing cache")
    _reject_archive_path(cache_dir, "Forcing cache")
    manifest = _read_json(cache_dir / "manifest.json", "forcing cache manifest")
    if manifest.get("pipeline") != PIPELINE_NAME or manifest.get("kind") != "reply_peak95_cache":
        raise CausalValidationError("Not a validated reply_peak95 candidate cache")
    if manifest.get("status") != "validated":
        raise CausalValidationError("Forcing cache status is not validated")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise CausalValidationError("Forcing cache has no immutable artifact records")
    for record in artifacts:
        if not isinstance(record, Mapping):
            raise CausalValidationError("Invalid forcing-cache artifact record")
        path = _inside(cache_dir / str(record.get("path", "")), cache_dir, "cache artifact")
        if path.stat().st_size != record.get("size_bytes"):
            raise CausalValidationError(f"Forcing-cache artifact size mismatch: {path.name}")
        _require_hash(path, record.get("sha256"), f"forcing-cache artifact {path.name}")
    contract = get_contract("reply_peak95")
    provenance = manifest.get("provenance") or {}
    current_sources = _current_causal_source_hashes()
    if provenance.get("source_code_sha256") != current_sources:
        raise CausalValidationError(
            "Forcing cache producer/control source hashes differ from current code"
        )
    if provenance.get("producer_source_sha256") != current_sources:
        raise CausalValidationError("Forcing cache producer source alias is inconsistent")
    if provenance.get("fresh_holdout") != _fresh_provenance_record(run):
        raise CausalValidationError(
            "Forcing cache fresh-holdout provenance differs from the validated run"
        )
    if provenance.get("checkpoint_activation_fidelity") != dict(
        run.checkpoint_activation_fidelity
    ):
        raise CausalValidationError(
            "Forcing cache activation-fidelity provenance differs from the validated run"
        )
    for key, expected in (
        ("checkpoint_sha256", checkpoint_sha256),
        ("probe_run_manifest_sha256", sha256_file(run.run_dir / "manifest.json")),
        ("build_manifest_sha256", sha256_file(run.run_dir / "build_manifest.json")),
        ("training_manifest_sha256", sha256_file(run.run_dir / "training_manifest.json")),
        ("dataset_sha256", run.build_manifest["dataset_sha256"]),
        (
            "input_provenance_sha256",
            run.build_manifest["input_provenance_sha256"],
        ),
        (
            "trunk_identity_bytes_sha256",
            run.build_manifest["input_provenance"]["trunk_identity_bytes_sha256"],
        ),
        ("splits_sha256", run.manifest["artifacts"]["splits_sha256"]),
        ("contract_id", contract.definition_id),
        ("contract_hash", contract.contract_hash),
    ):
        if provenance.get(key) != expected:
            raise CausalValidationError(f"Forcing cache provenance mismatch for {key}")
    relative = manifest.get("candidate_table", "candidate_reply_peaks.parquet")
    path = _inside(cache_dir / str(relative), cache_dir, "candidate table")
    _require_hash(path, manifest.get("candidate_table_sha256"), "candidate reply peaks")
    return _candidate_maps_from_frame(pd.read_parquet(path))


def _filter_forcing_mask_eligible(
    selected: pd.DataFrame,
    candidate_maps: Mapping[str, Mapping[int, float]],
) -> pd.DataFrame:
    """Keep positions with both forcing-proxy and comparison candidate actions."""
    keep = []
    for position_id in selected["position_id"].astype(str):
        scores = candidate_maps.get(position_id)
        if not scores:
            keep.append(False)
            continue
        flags = [float(value) > 0.95 for value in scores.values()]
        keep.append(any(flags) and not all(flags))
    return selected.loc[np.asarray(keep, dtype=bool)].reset_index(drop=True)


def prepare_replay_positions(
    model: Any,
    run: ValidatedRun,
    selected: pd.DataFrame,
    *,
    candidate_maps: Optional[Mapping[str, Mapping[int, float]]] = None,
    policy_backend: Optional[PolicyHeadOnlyBackend] = None,
) -> List[ReplayPosition]:
    """Replay and freeze selected positions, baseline outputs, masks, and anchors."""
    from board import Board
    from gamestate import GameState

    requested = {str(row.position_id): row for row in selected.itertuples(index=False)}
    dataset_by_row = run.dataset.set_index(run.dataset["row_id"].astype(str), drop=False)
    result: List[ReplayPosition] = []
    for game_id, group in selected.groupby("game_id", sort=False):
        wanted = {int(value) for value in group["move_number"]}
        moves = _load_moves(run.games_dir / str(game_id) / "moves.jsonl")
        if policy_backend is not None:
            _verify_current_game_sources(run, str(game_id), moves)
        gs = GameState(run.board_size, GameState.RULES_TT)
        previous_move: Optional[int] = None
        for move in moves:
            move_number = int(move["move_number"])
            if move_number in wanted:
                position_id = f"{game_id}:{move_number}"
                row = requested[position_id]
                raw_idx = move.get("idx361")
                stored_idx = None if pd.isna(row.idx361) else int(row.idx361)
                if stored_idx == run.board_size * run.board_size:
                    stored_idx = None
                if raw_idx is not None and int(raw_idx) == run.board_size * run.board_size:
                    raw_idx = None
                if raw_idx is not None:
                    raw_idx = int(raw_idx)
                if raw_idx != stored_idx:
                    raise CausalValidationError(
                        f"Raw and run-scoped idx361 disagree at {position_id}"
                    )
                if int(move["move_loc"]) != int(row.move_loc):
                    raise CausalValidationError(
                        f"Raw and run-scoped move_loc disagree at {position_id}"
                    )
                expected_player = Board.BLACK if move["player"] == "b" else Board.WHITE
                if int(gs.board.pla) != int(expected_player):
                    raise CausalValidationError(f"Replay player mismatch at {position_id}")
                selected_anchor = getattr(row, "previous_move_loc", np.nan)
                selected_anchor = (
                    None if pd.isna(selected_anchor) else int(selected_anchor)
                )
                if selected_anchor != previous_move:
                    raise CausalValidationError(
                        f"Most-recent non-pass anchor mismatch at {position_id}: "
                        f"selection={selected_anchor}, replay={previous_move}"
                    )
                frozen = _clone_game_state(gs)
                trunk_path: Optional[Path] = None
                trunk_sha: Optional[str] = None
                trunkfinal: Optional[np.ndarray] = None
                board_mask: Optional[np.ndarray] = None
                if policy_backend is not None:
                    trunk_path, trunk_sha, trunkfinal = _load_bound_trunkfinal(
                        run, str(game_id), move_number
                    )
                    if str(row.row_id) not in dataset_by_row.index:
                        raise CausalValidationError(
                            f"Dataset lacks selected row {row.row_id}"
                        )
                    source_row = dataset_by_row.loc[str(row.row_id)]
                    if isinstance(source_row, pd.DataFrame):
                        raise CausalValidationError(
                            f"Dataset has duplicate selected row {row.row_id}"
                        )
                    if "h_pre_global" in source_row.index:
                        stored_global = np.asarray(source_row["h_pre_global"], dtype=np.float32)
                        observed_global = np.asarray(trunkfinal).mean(axis=(1, 2))
                        if stored_global.shape != observed_global.shape or not np.allclose(
                            stored_global, observed_global, rtol=0.0, atol=1e-6
                        ):
                            raise CausalValidationError(
                                f"Saved trunkfinal no longer reproduces dataset global features "
                                f"at {position_id}"
                            )
                    if stored_idx is not None and "h_pre_local" in source_row.index:
                        stored_local = np.asarray(source_row["h_pre_local"], dtype=np.float32)
                        y, x = divmod(stored_idx, run.board_size)
                        observed_local = np.asarray(trunkfinal)[:, y, x]
                        if stored_local.shape != observed_local.shape or not np.allclose(
                            stored_local, observed_local, rtol=0.0, atol=1e-7
                        ):
                            raise CausalValidationError(
                                f"Saved trunkfinal no longer reproduces idx361 local features "
                                f"at {position_id}"
                            )
                    # With model.pos_len == run.board_size, KataGo feature plane
                    # zero is exactly one on every board point.  This is the
                    # policy head's geometric board mask, not a legality mask.
                    board_mask = np.ones(
                        (run.board_size, run.board_size), dtype=np.float32
                    )
                    board_mask.setflags(write=False)
                    baseline: Mapping[str, Any] = {}
                else:
                    baseline = frozen.get_model_outputs(model)
                candidate_map = None if candidate_maps is None else candidate_maps.get(position_id)
                if get_contract(run.concept).name == "reply_peak95" and candidate_map is None:
                    raise CausalValidationError(f"Forcing cache lacks selected position {position_id}")
                mask = None
                anchor = None
                if policy_backend is None and run.representation in {"local", "combined"}:
                    mask, anchor = concept_spatial_mask(
                        run.concept,
                        baseline,
                        frozen.board,
                        previous_move=previous_move,
                        candidate_reply_peaks=candidate_map,
                    )
                elif (
                    policy_backend is None
                    and get_contract(run.concept).name == "regional_policy_peak"
                ):
                    policy = normalized_policy(
                        baseline,
                        frozen.board,
                        support=PolicySupport.LEGAL_BOARD_CONDITIONAL,
                    )
                    anchor = str(regional_policy_readouts(policy, frozen.board)["regional_policy_peak_region"])
                result.append(ReplayPosition(
                    position_id=position_id,
                    game_id=str(game_id),
                    move_number=move_number,
                    idx361=stored_idx,
                    move_loc=int(row.move_loc),
                    previous_move=previous_move,
                    split_role=str(row.split_role),
                    label=int(getattr(row, f"label_{run.concept}")),
                    game_state=frozen,
                    baseline=baseline,
                    spatial_mask=mask,
                    anchor_region=anchor,
                    candidate_reply_peaks=candidate_map,
                    trunkfinal_path=trunk_path,
                    trunkfinal_sha256=trunk_sha,
                    trunkfinal=trunkfinal,
                    model_board_mask=board_mask,
                ))
            player = Board.BLACK if move["player"] == "b" else Board.WHITE
            gs.play(player, int(move["move_loc"]))
            if int(move["move_loc"]) != int(Board.PASS_LOC):
                previous_move = int(move["move_loc"])
            if wanted and move_number >= max(wanted):
                break
    if len(result) != len(selected):
        found = {position.position_id for position in result}
        missing = sorted(set(requested) - found)
        raise CausalValidationError(f"Could not replay {len(missing)} selected positions: {missing[:5]}")
    result = sorted(result, key=lambda position: (position.game_id, position.move_number))
    if policy_backend is not None:
        baselines = policy_backend.baseline_outputs(result)
        for position, baseline in zip(result, baselines):
            position.baseline = baseline
            if run.representation in {"local", "combined"}:
                position.spatial_mask, position.anchor_region = concept_spatial_mask(
                    run.concept,
                    baseline,
                    position.board,
                    previous_move=position.previous_move,
                    candidate_reply_peaks=position.candidate_reply_peaks,
                )
            elif get_contract(run.concept).name == "regional_policy_peak":
                policy = normalized_policy(
                    baseline,
                    position.board,
                    support=PolicySupport.LEGAL_BOARD_CONDITIONAL,
                )
                position.anchor_region = str(
                    regional_policy_readouts(policy, position.board)[
                        "regional_policy_peak_region"
                    ]
                )
    return result


def audit_operational_alignment(
    run: ValidatedRun,
    positions: Sequence[ReplayPosition],
    *,
    value_atol: float = DEFAULT_OPERATIONAL_ALIGNMENT_ATOL,
) -> Dict[str, Any]:
    """Reproduce selected labels from the exact causal baseline/candidate data.

    This is an all-selected-position gate, separate from the sampled policy-head
    equivalence audit.  It catches drift between the policies that generated
    run-scoped labels and the checkpoint/saved activations used for masks and
    causal readouts.
    """

    if value_atol <= 0 or not math.isfinite(value_atol):
        raise ValueError("operational alignment tolerance must be finite and positive")
    dataset = run.dataset.set_index(run.dataset["row_id"].astype(str), drop=False)
    label_analysis: Dict[str, Dict[int, Mapping[str, Any]]] = {}
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    contract = get_contract(run.concept)

    for position in sorted(positions, key=lambda item: item.position_id):
        record: Dict[str, Any] = {
            "position_id": position.position_id,
            "game_id": position.game_id,
            "move_number": position.move_number,
            "split_role": position.split_role,
            "contract_id": contract.definition_id,
            "value_absolute_tolerance": float(value_atol),
        }
        try:
            if position.position_id not in dataset.index:
                raise CausalValidationError(
                    f"Dataset lacks selected position {position.position_id}"
                )
            source_row = dataset.loc[position.position_id]
            if isinstance(source_row, pd.DataFrame):
                raise CausalValidationError(
                    f"Dataset duplicates selected position {position.position_id}"
                )
            if position.game_id not in label_analysis:
                label_rows = _load_moves(
                    run.labels_dir / position.game_id / "snorkel.jsonl"
                )
                label_analysis[position.game_id] = {
                    int(item["move_number"]): item.get("analysis", {})
                    for item in label_rows
                }
            analysis = label_analysis[position.game_id].get(position.move_number)
            if not isinstance(analysis, Mapping):
                raise CausalValidationError(
                    f"Run-scoped labels lack analysis at {position.position_id}"
                )

            expected_label = int(position.label)
            if contract.name == "regional_policy_peak":
                expected_value = float(source_row[f"rawval_{run.concept}"])
                readouts = concept_policy_readouts(
                    run.concept,
                    position.baseline,
                    position.board,
                    previous_move=position.previous_move,
                    anchor_region=position.anchor_region,
                )
                observed_value = float(readouts["regional_policy_peak"])
                threshold_frame = pd.DataFrame([source_row.to_dict()])
                threshold_frame.loc[:, f"rawval_{run.concept}"] = observed_value
                observed_labels = _quantile_labels_from_development_thresholds(
                    run, threshold_frame
                )
                if len(observed_labels) != 1 or not math.isfinite(observed_labels[0]):
                    raise CausalValidationError(
                        f"Causal urgency value became ineligible at {position.position_id}"
                    )
                observed_label = int(observed_labels[0])
                record["comparison"] = (
                    "saved-trunk baseline regional_policy_peak versus dataset raw value; "
                    "frozen development threshold reapplied"
                )
            elif contract.name == "reply_peak95":
                if position.candidate_reply_peaks is None:
                    raise CausalValidationError(
                        f"Forcing candidate map missing at {position.position_id}"
                    )
                selected_move = int(position.move_loc)
                if selected_move not in position.candidate_reply_peaks:
                    raise CausalValidationError(
                        f"Forcing candidate cache lacks recorded move at {position.position_id}"
                    )
                expected_value = float(analysis["reply_peak_value"])
                observed_value = float(position.candidate_reply_peaks[selected_move])
                observed_label = int(observed_value > 0.95)
                stored_analysis_label = int(bool(analysis["reply_peak95"]))
                stored_raw_label = int(bool(source_row[f"rawval_{run.concept}"]))
                if stored_analysis_label != expected_label or stored_raw_label != expected_label:
                    raise CausalValidationError(
                        f"Stored forcing labels disagree internally at {position.position_id}"
                    )
                record.update({
                    "recorded_move_loc": selected_move,
                    "strict_threshold": 0.95,
                    "comparison": (
                        "candidate-cache reply peak for recorded move versus run-scoped "
                        "reply_peak_value; strict >0.95 predicate reapplied"
                    ),
                })
            elif contract.name == "tenuki_distance6":
                expected_value = float(analysis["tenuki_manhattan_distance"])
                distance = manhattan_distance(
                    position.board, position.previous_move, position.move_loc
                )
                if distance is None:
                    raise CausalValidationError(
                        f"Tenuki selected move is ineligible at {position.position_id}"
                    )
                observed_value = float(distance)
                observed_label = int(observed_value >= 6.0)
                stored_analysis_label = int(bool(analysis["tenuki_distance6"]))
                stored_raw_label = int(bool(source_row[f"rawval_{run.concept}"]))
                if stored_analysis_label != expected_label or stored_raw_label != expected_label:
                    raise CausalValidationError(
                        f"Stored tenuki labels disagree internally at {position.position_id}"
                    )
                record.update({
                    "inclusive_threshold": 6.0,
                    "comparison": (
                        "replayed most-recent-nonpass Manhattan distance versus run-scoped "
                        "distance; >=6 predicate reapplied"
                    ),
                })
            else:
                raise CausalValidationError(
                    f"No operational alignment audit for {contract.definition_id}"
                )

            absolute_error = abs(observed_value - expected_value)
            value_agrees = absolute_error <= value_atol
            label_agrees = observed_label == expected_label
            record.update({
                "expected_raw_value": expected_value,
                "causal_observed_raw_value": observed_value,
                "absolute_value_error": absolute_error,
                "value_within_tolerance": value_agrees,
                "expected_label": expected_label,
                "causal_observed_label": observed_label,
                "threshold_label_agrees": label_agrees,
                "status": "validated" if value_agrees and label_agrees else "failed",
            })
            if not value_agrees or not label_agrees:
                errors.append(
                    f"{position.position_id}: value error={absolute_error:.6g}, "
                    f"labels expected/observed={expected_label}/{observed_label}"
                )
        except (KeyError, TypeError, ValueError, CausalValidationError) as exc:
            record.update({"status": "failed", "error": str(exc)})
            errors.append(f"{position.position_id}: {exc}")
        rows.append(record)

    if len(rows) != len(positions):
        raise RuntimeError("Operational alignment audit omitted selected positions")
    return {
        "status": "validated" if not errors else "failed",
        "concept": run.concept,
        "contract_id": contract.definition_id,
        "contract_hash": contract.contract_hash,
        "value_absolute_tolerance": float(value_atol),
        "positions_checked": len(rows),
        "games_checked": len({position.game_id for position in positions}),
        "failed_positions": len(errors),
        "errors": errors,
        "positions": rows,
    }


def _control_mask(
    position: ReplayPosition,
    control_kind: str,
    control_id: str,
    *,
    seed: int,
) -> Optional[np.ndarray]:
    if position.spatial_mask is None:
        return None
    if control_kind == "spatial_shuffle":
        return shuffled_position_mask(
            position.spatial_mask,
            base_seed=seed,
            repeat_id=control_id,
            position_id=position.position_id,
        )
    return position.spatial_mask


def _evaluate_intervention(
    model: Any,
    positions: Sequence[ReplayPosition],
    concept: str,
    direction: InterventionDirection,
    dose: float,
    *,
    control_kind: str,
    control_id: str,
    seed: int,
    include_behavior: bool,
) -> List[Dict[str, Any]]:
    masks = [
        _control_mask(position, control_kind, control_id, seed=seed)
        for position in positions
    ]
    if isinstance(model, PolicyHeadOnlyBackend):
        if not model.validated:
            raise CausalValidationError(
                "Policy-head backend was used before full-model equivalence validation"
            )
        if float(dose) == 0.0:
            steered_outputs = [position.baseline for position in positions]
        else:
            steered_outputs = model.evaluate(
                positions,
                direction=direction,
                dose=float(dose),
                spatial_masks=masks,
            )
    else:
        steered_outputs = []
        for position, mask in zip(positions, masks):
            if float(dose) == 0.0:
                steered = position.baseline
            else:
                with direction.apply(model, float(dose), mask):
                    steered = position.game_state.get_model_outputs(model)
            steered_outputs.append(steered)

    rows: List[Dict[str, Any]] = []
    for position, steered in zip(positions, steered_outputs):
        disruption = policy_disruption(position.baseline, steered, position.board)
        row: Dict[str, Any] = {
            "position_id": position.position_id,
            "game_id": position.game_id,
            "move_number": position.move_number,
            "split_role": position.split_role,
            "label": position.label,
            **disruption,
        }
        if include_behavior:
            baseline_readouts = concept_policy_readouts(
                concept,
                position.baseline,
                position.board,
                previous_move=position.previous_move,
                candidate_reply_peaks=position.candidate_reply_peaks,
                anchor_region=position.anchor_region,
            )
            steered_readouts = concept_policy_readouts(
                concept,
                steered,
                position.board,
                previous_move=position.previous_move,
                candidate_reply_peaks=position.candidate_reply_peaks,
                anchor_region=position.anchor_region,
            )
            for name, baseline_value in baseline_readouts.items():
                steered_value = steered_readouts[name]
                row[f"baseline_{name}"] = baseline_value
                row[f"steered_{name}"] = steered_value
                row[f"delta_{name}"] = steered_value - baseline_value
        rows.append(row)
    return rows


def _mean_disruption(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    if not rows:
        raise CausalValidationError("Cannot summarize an empty intervention evaluation")
    return {
        "mean_policy_js": float(np.mean([float(row["policy_js"]) for row in rows])),
        "mean_policy_l1": float(np.mean([float(row["policy_l1"]) for row in rows])),
    }


@dataclass(frozen=True)
class ControlSpec:
    control_id: str
    kind: str
    direction: InterventionDirection


def control_specs(
    learned: InterventionDirection,
    *,
    seed: int,
    spatial_shuffles: int = DEFAULT_SHUFFLES,
    random_directions: int = DEFAULT_RANDOM_DIRECTIONS,
) -> List[ControlSpec]:
    """Construct the mandatory deterministic control families."""
    if random_directions < DEFAULT_RANDOM_DIRECTIONS:
        raise CausalValidationError(
            f"At least {DEFAULT_RANDOM_DIRECTIONS} random directions are required"
        )
    specs: List[ControlSpec] = []
    if learned.local_delta is not None:
        if spatial_shuffles < MIN_CONTROL_REPEATS:
            raise CausalValidationError(
                f"At least {MIN_CONTROL_REPEATS} spatial shuffles are required"
            )
        specs.extend(
            ControlSpec(control_id, "spatial_shuffle", learned)
            for control_id in shuffle_control_ids(spatial_shuffles)
        )
    specs.extend(
        ControlSpec(
            control_id,
            "random_direction",
            learned.random_control(control_id, seed=seed),
        )
        for control_id in random_direction_control_ids(random_directions)
    )
    return specs


def _zero_match(nominal_dose: float) -> Dict[str, Any]:
    return {
        "target_mean_policy_js": 0.0,
        "dose_multiplier": 1.0,
        "nominal_dose": float(nominal_dose),
        "effective_dose": float(nominal_dose),
        "achieved_mean_policy_js": 0.0,
        "achieved_mean_policy_l1": 0.0,
        "matched": True,
        "status": "zero_dose_exact",
        "iterations": 0,
        "bracket_low": 1.0,
        "bracket_high": 1.0,
        "absolute_js_error": 0.0,
    }


def calibrate_controls(
    model: Any,
    concept: str,
    calibration_positions: Sequence[ReplayPosition],
    learned: InterventionDirection,
    controls: Sequence[ControlSpec],
    doses: Sequence[float],
    *,
    seed: int,
    require_match: bool = True,
) -> Tuple[Dict[Tuple[str, float], Dict[str, Any]], Dict[float, Dict[str, float]]]:
    """Match each control/dose to trained mean policy JS on calibration games."""
    targets: Dict[float, Dict[str, float]] = {}
    matches: Dict[Tuple[str, float], Dict[str, Any]] = {}
    for nominal_dose in doses:
        dose = float(nominal_dose)
        if dose == 0.0:
            targets[dose] = {"mean_policy_js": 0.0, "mean_policy_l1": 0.0}
        else:
            trained_rows = _evaluate_intervention(
                model,
                calibration_positions,
                concept,
                learned,
                dose,
                control_kind="trained",
                control_id="trained",
                seed=seed,
                include_behavior=False,
            )
            targets[dose] = _mean_disruption(trained_rows)

        for control in controls:
            if dose == 0.0:
                match = _zero_match(dose)
            else:
                def callback(multiplier: float) -> Mapping[str, float]:
                    rows = _evaluate_intervention(
                        model,
                        calibration_positions,
                        concept,
                        control.direction,
                        dose * float(multiplier),
                        control_kind=control.kind,
                        control_id=control.control_id,
                        seed=seed,
                        include_behavior=False,
                    )
                    return _mean_disruption(rows)

                match = match_mean_policy_js(
                    callback,
                    targets[dose]["mean_policy_js"],
                    nominal_dose=dose,
                ).to_dict()
            if require_match and not bool(match["matched"]):
                raise CausalValidationError(
                    f"Could not policy-match {control.control_id} at dose {dose}: "
                    f"{match['status']} (target={match['target_mean_policy_js']:.6g}, "
                    f"achieved={match['achieved_mean_policy_js']:.6g})"
                )
            matches[(control.control_id, dose)] = match
    return matches, targets


def evaluate_causal_test(
    model: Any,
    concept: str,
    test_positions: Sequence[ReplayPosition],
    learned: InterventionDirection,
    controls: Sequence[ControlSpec],
    doses: Sequence[float],
    matches: Mapping[Tuple[str, float], Mapping[str, Any]],
    *,
    calibration_targets: Mapping[float, Mapping[str, float]],
    seed: int,
) -> List[Dict[str, Any]]:
    """Apply frozen calibration multipliers to untouched causal-test games."""
    rows: List[Dict[str, Any]] = []
    for nominal in doses:
        dose = float(nominal)
        if dose not in calibration_targets:
            raise CausalValidationError(
                f"Missing trained calibration disruption target for dose {dose}"
            )
        trained_target = calibration_targets[dose]
        trained_rows = _evaluate_intervention(
            model,
            test_positions,
            concept,
            learned,
            dose,
            control_kind="trained",
            control_id="trained",
            seed=seed,
            include_behavior=True,
        )
        for row in trained_rows:
            row.update({
                "control_id": "trained",
                "control_kind": "trained",
                "nominal_dose": dose,
                "dose_multiplier": 1.0,
                "effective_dose": dose,
                "calibration_match_succeeded": True,
                "calibration_match_status": "trained_direction_defines_target",
                "calibration_target_mean_policy_js": float(
                    trained_target["mean_policy_js"]
                ),
                "calibration_achieved_mean_policy_js": float(
                    trained_target["mean_policy_js"]
                ),
            })
        rows.extend(trained_rows)

        for control in controls:
            match = matches[(control.control_id, dose)]
            multiplier = float(match["dose_multiplier"])
            effective = dose * multiplier
            control_rows = _evaluate_intervention(
                model,
                test_positions,
                concept,
                control.direction,
                effective,
                control_kind=control.kind,
                control_id=control.control_id,
                seed=seed,
                include_behavior=True,
            )
            for row in control_rows:
                row.update({
                    "control_id": control.control_id,
                    "control_kind": control.kind,
                    "nominal_dose": dose,
                    "dose_multiplier": multiplier,
                    "effective_dose": effective,
                    "calibration_match_succeeded": bool(match["matched"]),
                    "calibration_match_status": str(match["status"]),
                    "calibration_target_mean_policy_js": float(
                        match["target_mean_policy_js"]
                    ),
                    "calibration_achieved_mean_policy_js": float(
                        match["achieved_mean_policy_js"]
                    ),
                })
            rows.extend(control_rows)
    return rows


def summarize_causal_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise CausalValidationError("No causal-test rows were produced")
    delta_columns = sorted(column for column in frame if column.startswith("delta_"))
    summaries: List[Dict[str, Any]] = []
    for keys, group in frame.groupby(
        ["control_id", "control_kind", "nominal_dose", "dose_multiplier", "effective_dose"],
        sort=True,
        dropna=False,
    ):
        control_id, kind, nominal, multiplier, effective = keys
        item = {
            "control_id": str(control_id),
            "control_kind": str(kind),
            "n_positions": int(group["position_id"].nunique()),
            "n_games": int(group["game_id"].nunique()),
            "nominal_dose": float(nominal),
            "dose_multiplier": float(multiplier),
            "effective_dose": float(effective),
            "causal_test_observed_mean_policy_js": float(group["policy_js"].mean()),
            "mean_policy_l1": float(group["policy_l1"].mean()),
            "top_move_flip_rate": float(group["top_move_flip"].mean()),
            "calibration_match_succeeded": bool(
                group["calibration_match_succeeded"].all()
            ),
            "calibration_target_mean_policy_js": float(
                group["calibration_target_mean_policy_js"].iloc[0]
            ),
            "calibration_achieved_mean_policy_js": float(
                group["calibration_achieved_mean_policy_js"].iloc[0]
            ),
        }
        # Compatibility alias with an explicit observed-test meaning.  It is
        # not a claim that calibration matching transferred to the test set.
        item["mean_policy_js"] = item["causal_test_observed_mean_policy_js"]
        item["causal_test_js_residual_from_calibration_target"] = (
            item["causal_test_observed_mean_policy_js"]
            - item["calibration_target_mean_policy_js"]
        )
        item["causal_test_abs_js_residual_from_calibration_target"] = abs(
            item["causal_test_js_residual_from_calibration_target"]
        )
        item.update({f"mean_{column}": float(group[column].mean()) for column in delta_columns})
        summaries.append(item)
    trained_test_js = {
        float(item["nominal_dose"]): float(item["causal_test_observed_mean_policy_js"])
        for item in summaries
        if item["control_kind"] == "trained"
    }
    for item in summaries:
        dose = float(item["nominal_dose"])
        if dose not in trained_test_js:
            raise CausalValidationError(
                f"No trained causal-test disruption reference for dose {dose}"
            )
        reference = trained_test_js[dose]
        residual = float(item["causal_test_observed_mean_policy_js"]) - reference
        item["causal_test_trained_mean_policy_js"] = reference
        item["causal_test_js_residual_vs_trained"] = residual
        item["causal_test_abs_js_residual_vs_trained"] = abs(residual)
    return summaries


def _checkpoint_hash(model_path: Path) -> str:
    model_path = model_path.resolve()
    _reject_archive_path(model_path, "Model checkpoint")
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
    return sha256_file(model_path)


def _require_fresh_checkpoint(run: ValidatedRun, observed_sha256: str) -> None:
    expected = _valid_sha256(
        (run.fresh_holdout or {}).get("checkpoint_sha256"),
        "fresh holdout checkpoint",
    )
    if observed_sha256 != expected:
        raise CausalValidationError(
            "Supplied causal checkpoint does not match the checkpoint that generated "
            f"the fresh holdout: expected {expected}, got {observed_sha256}"
        )


def _require_complete_role_selection(
    run: ValidatedRun, selected: pd.DataFrame, role: str
) -> None:
    expected = set(
        run.splits.loc[run.splits["split_role"].eq(role), "game_id"].astype(str)
    )
    observed = set(selected["game_id"].astype(str))
    if observed != expected or len(selected) != len(expected):
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise CausalValidationError(
            f"Confirmatory {role} selection must contain exactly one position from every "
            f"fresh role game; expected={len(expected)}, observed={len(selected)}, "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )


def estimate_protocol_forwards(
    *,
    calibration_positions: int,
    causal_test_positions: int,
    representation: str,
    doses: Sequence[float],
    spatial_shuffles: int = DEFAULT_SHUFFLES,
    random_directions: int = DEFAULT_RANDOM_DIRECTIONS,
    typical_matching_callbacks: int = 8,
    maximum_matching_callbacks: int = 47,
    head_batch_size: int = DEFAULT_HEAD_BATCH_SIZE,
    equivalence_sample_size: int = DEFAULT_EQUIVALENCE_SAMPLE_SIZE,
) -> Dict[str, Any]:
    """Estimate full-network audits separately from policy-head evaluation.

    Matching is adaptive.  The lower/typical/upper totals therefore differ only
    in the number of non-zero callbacks needed by scalar bracketing/bisection.
    A multiplier-zero callback reuses the cached baseline and costs no forward.
    ``47`` is the maximum non-zero callback count under the current matcher
    defaults (initial endpoint, up to 16 bracket expansions, 30 bisections).

    A "policy-head position evaluation" is one activation passed through only
    ``model.policy_head``.  A "policy-head batch forward" can contain up to
    ``head_batch_size`` such positions.  Neither is a full KataGo trunk forward.
    """
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unknown representation: {representation}")
    n_cal = int(calibration_positions)
    n_test = int(causal_test_positions)
    if n_cal <= 0 or n_test <= 0:
        raise ValueError("Both causal protocol splits need at least one position")
    if int(head_batch_size) <= 0:
        raise ValueError("head_batch_size must be positive")
    if int(equivalence_sample_size) <= 0:
        raise ValueError("equivalence_sample_size must be positive")
    nonzero_doses = sum(float(dose) != 0.0 for dose in doses)
    shuffle_count = int(spatial_shuffles) if representation != "global" else 0
    if shuffle_count and shuffle_count < MIN_CONTROL_REPEATS:
        raise CausalValidationError(
            f"At least {MIN_CONTROL_REPEATS} spatial shuffles are required"
        )
    if int(random_directions) < DEFAULT_RANDOM_DIRECTIONS:
        raise CausalValidationError(
            f"At least {DEFAULT_RANDOM_DIRECTIONS} random directions are required"
        )
    controls = shuffle_count + int(random_directions)
    baseline = n_cal + n_test
    trained_calibration = nonzero_doses * n_cal
    final_test = nonzero_doses * (controls + 1) * n_test
    cal_batches = math.ceil(n_cal / int(head_batch_size))
    test_batches = math.ceil(n_test / int(head_batch_size))
    baseline_batches = cal_batches + test_batches
    trained_calibration_batches = nonzero_doses * cal_batches
    final_test_batches = nonzero_doses * (controls + 1) * test_batches

    def total_positions(callbacks: int) -> int:
        matching = controls * nonzero_doses * n_cal * int(callbacks)
        return baseline + trained_calibration + matching + final_test

    def total_batches(callbacks: int) -> int:
        matching = controls * nonzero_doses * cal_batches * int(callbacks)
        return (
            baseline_batches
            + trained_calibration_batches
            + matching
            + final_test_batches
        )

    audit_forwards = min(
        int(equivalence_sample_size), n_cal + n_test
    )

    return {
        "unit": "policy-head position evaluations (not full-network forwards)",
        "evaluation_backend": "saved_trunkfinal_policy_head",
        "head_batch_size": int(head_batch_size),
        "calibration_positions": n_cal,
        "causal_test_positions": n_test,
        "doses": [float(dose) for dose in doses],
        "nonzero_doses": nonzero_doses,
        "spatial_shuffle_controls": shuffle_count,
        "random_direction_controls": int(random_directions),
        "total_controls": controls,
        "fixed_components": {
            # Legacy key names are retained for readers of schema-1 estimates,
            # but their unit is now explicit: these are head-only positions.
            "baseline_forwards": baseline,
            "trained_calibration_forwards": trained_calibration,
            "final_causal_test_forwards": final_test,
            "baseline_policy_head_positions": baseline,
            "trained_calibration_policy_head_positions": trained_calibration,
            "final_causal_test_policy_head_positions": final_test,
            "baseline_policy_head_batches": baseline_batches,
            "trained_calibration_policy_head_batches": trained_calibration_batches,
            "final_causal_test_policy_head_batches": final_test_batches,
        },
        "full_network_forward_evaluations": {
            "equivalence_audit": audit_forwards,
            "causal_interventions_and_controls": 0,
            "forcing_candidate_cache": (
                "reported separately by build-cache because it depends on legal candidate counts"
            ),
        },
        "matching_callbacks_per_control_dose": {
            "lower_bound": 1,
            "planning_assumption": int(typical_matching_callbacks),
            "upper_bound": int(maximum_matching_callbacks),
        },
        "estimated_total_forwards": {
            "lower_bound": total_positions(1),
            "planning_assumption": total_positions(typical_matching_callbacks),
            "upper_bound": total_positions(maximum_matching_callbacks),
        },
        "estimated_policy_head_position_evaluations": {
            "lower_bound": total_positions(1),
            "planning_assumption": total_positions(typical_matching_callbacks),
            "upper_bound": total_positions(maximum_matching_callbacks),
        },
        "estimated_policy_head_batch_forwards": {
            "lower_bound": total_batches(1),
            "planning_assumption": total_batches(typical_matching_callbacks),
            "upper_bound": total_batches(maximum_matching_callbacks),
        },
        "warning": (
            "Control matching remains compute-intensive, but intervention evaluation runs "
            "only the policy head in batches. Full-network replay is limited to the small "
            "equivalence audit (and the separately built forcing candidate cache)."
        ),
    }


def _start_running_output(
    output_dir: Path,
    *,
    kind: str,
    provenance: Mapping[str, Any],
    estimate: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Path]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    manifest = new_run_manifest(output_dir.name, status="running", provenance=provenance)
    manifest.update({"pipeline": PIPELINE_NAME, "kind": kind})
    write_run_manifest(output_dir / "manifest.json", manifest)
    estimate_path = output_dir / "protocol_estimate.json"
    _write_new_json(estimate_path, estimate)
    return manifest, estimate_path


def _mark_running_output_failed(output_dir: Path) -> None:
    update_run_status(
        output_dir / "manifest.json",
        "failed",
        expected_current_status="running",
        verify_before_validation=False,
    )


def _candidate_action_rows(position_id: str, gs: Any, model: Any) -> List[Dict[str, Any]]:
    """Evaluate opponent reply peak after every legal non-pass current action."""
    board = gs.board
    player = int(board.pla)
    legal = [
        board.loc(x, y)
        for y in range(board.size)
        for x in range(board.size)
        if board.would_be_legal(player, board.loc(x, y))
    ]
    rows: List[Dict[str, Any]] = []
    for candidate in legal:
        x, y = board.loc_x(candidate), board.loc_y(candidate)
        gs.play(player, candidate)
        try:
            reply_outputs = gs.get_model_outputs(model)
            reply_policy = normalized_policy(
                reply_outputs,
                gs.board,
                support=PolicySupport.LEGAL_PLUS_PASS,
                time=PolicyTime.POST_MOVE_REPLY,
            )
            peak = reply_peak(reply_policy)
        finally:
            gs.undo()
        rows.append({
            "position_id": position_id,
            "candidate_loc": int(candidate),
            "candidate_idx361": int(y * board.size + x),
            "reply_peak": float(peak),
        })
    if not rows:
        raise CausalValidationError(f"No legal board candidates at {position_id}")
    return rows


def _count_candidate_forwards(run: ValidatedRun, selected: pd.DataFrame) -> int:
    """Count exact legal candidate forwards by replaying boards without the model."""
    from board import Board
    from gamestate import GameState

    count = 0
    requested = set(selected["position_id"].astype(str))
    for game_id, group in selected.groupby("game_id", sort=False):
        wanted = set(group["move_number"].astype(int))
        gs = GameState(run.board_size, GameState.RULES_TT)
        for move in _load_moves(run.games_dir / str(game_id) / "moves.jsonl"):
            move_number = int(move["move_number"])
            if move_number in wanted:
                count += sum(
                    gs.board.would_be_legal(gs.board.pla, gs.board.loc(x, y))
                    for y in range(gs.board.size)
                    for x in range(gs.board.size)
                )
                requested.discard(f"{game_id}:{move_number}")
            player = Board.BLACK if move["player"] == "b" else Board.WHITE
            gs.play(player, int(move["move_loc"]))
            if wanted and move_number >= max(wanted):
                break
    if requested:
        raise CausalValidationError(
            f"Could not replay positions while estimating cache: {sorted(requested)[:5]}"
        )
    return int(count)


def _populate_forcing_cache(
    model: Any,
    run: ValidatedRun,
    cache_dir: Path,
    selected: pd.DataFrame,
    manifest: Dict[str, Any],
    estimate_path: Path,
) -> Mapping[str, Any]:
    """Populate a started cache run; caller is responsible for failure status."""
    expected_sources = (manifest.get("provenance") or {}).get("source_code_sha256")
    if expected_sources != _current_causal_source_hashes():
        raise CausalValidationError("Causal/cache source changed before cache execution")
    selected_path = cache_dir / "positions.parquet"
    selected.to_parquet(selected_path, index=False)
    candidate_rows: List[Dict[str, Any]] = []

    from board import Board
    from gamestate import GameState

    requested = set(selected["position_id"].astype(str))
    role_by_position = selected.set_index("position_id")["split_role"].astype(str).to_dict()
    for game_id, group in selected.groupby("game_id", sort=False):
        wanted = set(group["move_number"].astype(int))
        gs = GameState(run.board_size, GameState.RULES_TT)
        for move in _load_moves(run.games_dir / str(game_id) / "moves.jsonl"):
            move_number = int(move["move_number"])
            if move_number in wanted:
                position_id = f"{game_id}:{move_number}"
                rows = _candidate_action_rows(position_id, gs, model)
                for row in rows:
                    row.update({
                        "game_id": str(game_id),
                        "move_number": move_number,
                        "split_role": role_by_position[position_id],
                    })
                candidate_rows.extend(rows)
                requested.discard(position_id)
            player = Board.BLACK if move["player"] == "b" else Board.WHITE
            gs.play(player, int(move["move_loc"]))
            if wanted and move_number >= max(wanted):
                break
    if requested:
        raise CausalValidationError(f"Could not replay cached positions: {sorted(requested)[:5]}")
    candidate_path = cache_dir / "candidate_reply_peaks.parquet"
    candidate_frame = pd.DataFrame(candidate_rows)
    _candidate_maps_from_frame(candidate_frame)
    candidate_frame.to_parquet(candidate_path, index=False)
    selected_path.chmod(0o444)
    candidate_path.chmod(0o444)

    manifest["positions"] = "positions.parquet"
    manifest["positions_sha256"] = sha256_file(selected_path)
    manifest["candidate_table"] = "candidate_reply_peaks.parquet"
    manifest["candidate_table_sha256"] = sha256_file(candidate_path)
    manifest["position_counts"] = {
        str(role): int(count)
        for role, count in selected.groupby("split_role")["position_id"].nunique().items()
    }
    manifest["selection_strata"] = {
        str(role): {
            str(int(label)): int(count)
            for label, count in group["selection_stratum"].value_counts().sort_index().items()
        }
        for role, group in selected.groupby("split_role", sort=True)
    }
    manifest["selection_unit"] = "exactly_one_position_per_fresh_holdout_game"
    manifest["candidate_rows"] = int(len(candidate_frame))
    manifest["artifacts"] = [
        artifact_record(estimate_path, run_dir=cache_dir),
        artifact_record(selected_path, run_dir=cache_dir),
        artifact_record(candidate_path, run_dir=cache_dir),
    ]
    if expected_sources != _current_causal_source_hashes():
        raise CausalValidationError("Causal/cache source changed during cache execution")
    write_run_manifest(cache_dir / "manifest.json", manifest)
    return update_run_status(
        cache_dir / "manifest.json", "validated", expected_current_status="running"
    )


def build_forcing_cache(
    model: Any,
    model_path: Path,
    run: ValidatedRun,
    cache_dir: Path,
    *,
    maximum_calibration_positions: int,
    maximum_test_positions: int,
    seed: int,
) -> Mapping[str, Any]:
    """Build an immutable, run-scoped candidate post-reply cache."""
    contract = get_contract(run.concept)
    if contract.name != "reply_peak95":
        raise ValueError("Candidate reply cache is only defined for reply_peak95")
    cache_dir = cache_dir.resolve()
    _inside(cache_dir, run.run_dir, "forcing cache output")
    _reject_archive_path(cache_dir, "Forcing cache output")
    checkpoint_sha = _checkpoint_hash(model_path)
    _require_fresh_checkpoint(run, checkpoint_sha)
    provenance = {
        "checkpoint": str(model_path.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "probe_run": str(run.run_dir),
        "probe_run_manifest_sha256": sha256_file(run.run_dir / "manifest.json"),
        "build_manifest_sha256": sha256_file(run.run_dir / "build_manifest.json"),
        "training_manifest_sha256": sha256_file(run.run_dir / "training_manifest.json"),
        "dataset_sha256": run.build_manifest["dataset_sha256"],
        "input_provenance_sha256": run.build_manifest["input_provenance_sha256"],
        "trunk_identity_bytes_sha256": run.build_manifest["input_provenance"][
            "trunk_identity_bytes_sha256"
        ],
        "splits_sha256": run.manifest["artifacts"]["splits_sha256"],
        "contract_id": contract.definition_id,
        "contract_hash": contract.contract_hash,
        "fresh_holdout": _fresh_provenance_record(run),
        "checkpoint_activation_fidelity": dict(
            run.checkpoint_activation_fidelity
        ),
        "source_code_sha256": _current_causal_source_hashes(),
        "producer_source_sha256": _current_causal_source_hashes(),
        "seed": int(seed),
    }
    calibration_selected = select_positions(
        run, "control_calibration", maximum_calibration_positions, seed=seed
    )
    test_selected = select_positions(
        run, "causal_test", maximum_test_positions, seed=seed
    )
    _require_complete_role_selection(run, calibration_selected, "control_calibration")
    _require_complete_role_selection(run, test_selected, "causal_test")
    selected = pd.concat([calibration_selected, test_selected], ignore_index=True)
    exact_forwards = _count_candidate_forwards(run, selected)
    estimate = {
        "unit": "single-position full-network forward evaluations",
        "evaluation_backend": "full_model_replay_required_for_post_move_reply",
        "kind": "reply_peak95_candidate_cache",
        "positions": int(len(selected)),
        "exact_candidate_reply_full_network_forwards": exact_forwards,
        # Schema-1 compatibility alias, with the unit clarified above.
        "exact_candidate_reply_forwards": exact_forwards,
        "note": (
            "Each legal non-pass current action is played once and requires one "
            "post-move opponent policy evaluation. Board replay itself uses no model forward."
        ),
    }
    manifest, estimate_path = _start_running_output(
        cache_dir,
        kind="reply_peak95_cache",
        provenance=provenance,
        estimate=estimate,
    )
    try:
        return _populate_forcing_cache(
            model, run, cache_dir, selected, manifest, estimate_path
        )
    except BaseException:
        _mark_running_output_failed(cache_dir)
        raise


def _execute_causal_protocol(
    model: Any,
    policy_backend: PolicyHeadOnlyBackend,
    run: ValidatedRun,
    output_dir: Path,
    *,
    calibration_selected: pd.DataFrame,
    test_selected: pd.DataFrame,
    candidate_maps: Optional[Mapping[str, Mapping[int, float]]],
    learned: InterventionDirection,
    controls: Sequence[ControlSpec],
    doses: Sequence[float],
    seed: int,
    require_control_match: bool,
    result_manifest: Dict[str, Any],
    estimate_path: Path,
    equivalence_sample_size: int,
) -> Mapping[str, Any]:
    """Execute a started protocol run and atomically advance it to validated."""
    expected_sources = (result_manifest.get("provenance") or {}).get(
        "source_code_sha256"
    )
    if expected_sources != _current_causal_source_hashes():
        raise CausalValidationError("Causal/control source changed before evaluation")
    calibration_positions = prepare_replay_positions(
        model,
        run,
        calibration_selected,
        candidate_maps=candidate_maps,
        policy_backend=policy_backend,
    )
    test_positions = prepare_replay_positions(
        model,
        run,
        test_selected,
        candidate_maps=candidate_maps,
        policy_backend=policy_backend,
    )
    all_positions = calibration_positions + test_positions
    equivalence_report = policy_backend.validate_equivalence(
        all_positions,
        seed=seed,
        sample_size=equivalence_sample_size,
    )
    binding_digest = _activation_binding_digest(all_positions)
    binding_rows: List[Dict[str, Any]] = []
    for position in sorted(all_positions, key=lambda item: item.position_id):
        if position.trunkfinal_path is None or position.trunkfinal_sha256 is None:
            raise CausalValidationError(
                f"Missing activation binding at {position.position_id}"
            )
        # Detect mutation between initial load and protocol execution.  The
        # in-memory copy is stable, but a changed source invalidates provenance.
        if sha256_file(position.trunkfinal_path) != position.trunkfinal_sha256:
            raise CausalValidationError(
                f"Saved trunkfinal changed during evaluation: {position.trunkfinal_path}"
            )
        binding_rows.append({
            "position_id": position.position_id,
            "game_id": position.game_id,
            "move_number": position.move_number,
            "split_role": position.split_role,
            "trunkfinal_path": str(
                position.trunkfinal_path.resolve().relative_to(run.games_dir.resolve())
            ),
            "trunkfinal_sha256": position.trunkfinal_sha256,
            "build_time_leaf_verified": True,
            "input_provenance_sha256": run.build_manifest[
                "input_provenance_sha256"
            ],
            "shape": "x".join(map(str, np.asarray(position.trunkfinal).shape)),
            "dtype": str(np.asarray(position.trunkfinal).dtype),
        })
    alignment_report = audit_operational_alignment(run, all_positions)
    bindings_path = output_dir / "activation_bindings.parquet"
    equivalence_path = output_dir / "policy_head_equivalence.json"
    alignment_path = output_dir / "operational_alignment.json"
    pd.DataFrame(binding_rows).to_parquet(bindings_path, index=False)
    bindings_path.chmod(0o444)
    _write_new_json(equivalence_path, equivalence_report)
    _write_new_json(alignment_path, alignment_report)
    if alignment_report["status"] != "validated":
        result_manifest["artifacts"] = [
            artifact_record(path, run_dir=output_dir)
            for path in (estimate_path, bindings_path, equivalence_path, alignment_path)
        ]
        result_manifest["operational_alignment"] = {
            "status": "failed",
            "report_sha256": sha256_file(alignment_path),
            "positions_checked": alignment_report["positions_checked"],
            "failed_positions": alignment_report["failed_positions"],
        }
        write_run_manifest(output_dir / "manifest.json", result_manifest)
        raise CausalValidationError(
            "Operational alignment failed for selected causal positions: "
            + "; ".join(alignment_report["errors"][:5])
        )
    matches, targets = calibrate_controls(
        policy_backend,
        run.concept,
        calibration_positions,
        learned,
        controls,
        doses,
        seed=seed,
        require_match=require_control_match,
    )
    rows = evaluate_causal_test(
        policy_backend,
        run.concept,
        test_positions,
        learned,
        controls,
        doses,
        matches,
        calibration_targets=targets,
        seed=seed,
    )
    summary_rows = summarize_causal_rows(rows)
    contract = get_contract(run.concept)

    positions_path = output_dir / "selected_positions.parquet"
    # The binding, equivalence, and operational-alignment gates were written
    # before any expensive control calibration so a failed run retains them.
    calibration_path = output_dir / "control_calibration.json"
    rows_path = output_dir / "causal_test_rows.parquet"
    summary_path = output_dir / "summary.json"
    pd.concat([
        calibration_selected.assign(causal_protocol_role="control_calibration"),
        test_selected.assign(causal_protocol_role="causal_test"),
    ], ignore_index=True).to_parquet(positions_path, index=False)
    control_by_id = {control.control_id: control for control in controls}
    calibration_records = []
    for (control_id, _nominal_dose), match in sorted(matches.items()):
        calibration_records.append({
            "control_id": control_id,
            "control_kind": control_by_id[control_id].kind,
            "direction_norms": {
                "total_delta_l2": float(
                    np.linalg.norm(control_by_id[control_id].direction.flattened())
                ),
                "global_delta_l2": (
                    None
                    if control_by_id[control_id].direction.global_delta is None
                    else float(
                        np.linalg.norm(
                            control_by_id[control_id].direction.global_delta
                        )
                    )
                ),
                "local_delta_l2": (
                    None
                    if control_by_id[control_id].direction.local_delta is None
                    else float(
                        np.linalg.norm(
                            control_by_id[control_id].direction.local_delta
                        )
                    )
                ),
            },
            **_json_safe(match),
        })
    _write_new_json(calibration_path, {
        "split_role": "control_calibration",
        "games": int(calibration_selected["game_id"].nunique()),
        "positions": int(len(calibration_selected)),
        "trained_targets_by_nominal_dose": {
            str(dose): target for dose, target in sorted(targets.items())
        },
        "matches": calibration_records,
    })
    pd.DataFrame(rows).to_parquet(rows_path, index=False)
    _write_new_json(summary_path, {
        "schema_version": SCHEMA_VERSION,
        "pipeline": PIPELINE_NAME,
        "source_code_sha256": expected_sources,
        "producer_source_sha256": expected_sources,
        "concept": run.concept,
        "contract": contract.metadata(),
        "contract_hash": contract.contract_hash,
        "representation": run.representation,
        "intervention_direction": learned.metadata(),
        "training_role": "development",
        "calibration_role": "control_calibration",
        "final_evaluation_role": "causal_test",
        "fresh_holdout": _fresh_provenance_record(run),
        "checkpoint_activation_fidelity": dict(
            run.checkpoint_activation_fidelity
        ),
        "position_sampling": "deterministic label-stratified sampling within each frozen role",
        "selection_unit": "exactly_one_position_per_fresh_holdout_game",
        "calibration_selection_strata": {
            str(int(label)): int(count)
            for label, count in calibration_selected["selection_stratum"]
            .value_counts()
            .sort_index()
            .items()
        },
        "causal_test_selection_strata": {
            str(int(label)): int(count)
            for label, count in test_selected["selection_stratum"]
            .value_counts()
            .sort_index()
            .items()
        },
        "calibration_games": int(calibration_selected["game_id"].nunique()),
        "causal_test_games": int(test_selected["game_id"].nunique()),
        "calibration_positions": int(len(calibration_selected)),
        "causal_test_positions": int(len(test_selected)),
        "doses": list(map(float, doses)),
        "spatial_controls_applicable": learned.local_delta is not None,
        "spatial_shuffle_repeats": sum(
            control.kind == "spatial_shuffle" for control in controls
        ),
        "random_direction_repeats": sum(
            control.kind == "random_direction" for control in controls
        ),
        "controls_policy_matched_on": (
            "control_calibration games: mean legal-plus-pass policy Jensen-Shannon divergence"
        ),
        "causal_test_disruption_reporting": (
            "Observed test JS and residuals versus both the calibration target and the "
            "trained direction on causal_test are reported; calibration success is not "
            "labelled as test-set matching."
        ),
        "random_direction_norm_matching": (
            "Every active representation block is L2-matched separately; combined "
            "controls preserve learned global and local block norms."
        ),
        "spatial_shuffle_null": (
            "Aligned mask values are permuted only within the exact active legal "
            "support; inactive/illegal points remain zero."
        ),
        "evaluation_backend": {
            "name": "saved_trunkfinal_policy_head",
            "batch_size": policy_backend.batch_size,
            "selected_activation_binding_sha256": binding_digest,
            "upstream_input_provenance_sha256": run.build_manifest[
                "input_provenance_sha256"
            ],
            "binding_semantics": (
                "Each selected current file SHA-256, byte size, and identity equals its "
                "individually addressable build-time leaf; the leaf map is bound by the "
                "upstream canonical input-provenance hash."
            ),
            "equivalence_status": equivalence_report["status"],
            "equivalence_sample_size": equivalence_report["sample_size_evaluated"],
            "equivalence_max_policy_abs_error": equivalence_report[
                "max_policy_abs_error"
            ],
            "equivalence_max_activation_abs_error": equivalence_report[
                "max_activation_abs_error"
            ],
        },
        "operational_alignment": {
            "status": alignment_report["status"],
            "positions_checked": alignment_report["positions_checked"],
            "failed_positions": alignment_report["failed_positions"],
            "value_absolute_tolerance": alignment_report[
                "value_absolute_tolerance"
            ],
            "report_sha256": sha256_file(alignment_path),
        },
        "forcing_readout": (
            "current legal-board policy mass on candidate actions whose frozen "
            "post-move opponent reply peak exceeds 0.95"
            if contract.name == "reply_peak95" else None
        ),
        "by_control_and_dose": summary_rows,
    })
    positions_path.chmod(0o444)
    rows_path.chmod(0o444)

    produced = [
        estimate_path,
        positions_path,
        bindings_path,
        equivalence_path,
        alignment_path,
        calibration_path,
        rows_path,
        summary_path,
    ]
    result_manifest["artifacts"] = [
        artifact_record(path, run_dir=output_dir) for path in produced
    ]
    result_manifest["counts"] = {
        "controls": len(controls),
        "rows": len(rows),
        "calibration_positions": len(calibration_positions),
        "causal_test_positions": len(test_positions),
        "calibration_games": len({position.game_id for position in calibration_positions}),
        "causal_test_games": len({position.game_id for position in test_positions}),
    }
    result_manifest["intervention_direction"] = learned.metadata()
    result_manifest["evaluation_backend"] = {
        "name": "saved_trunkfinal_policy_head",
        "batch_size": policy_backend.batch_size,
        "selected_activation_binding_sha256": binding_digest,
        "upstream_input_provenance_sha256": run.build_manifest[
            "input_provenance_sha256"
        ],
        "selected_leaves_verified_against_build": True,
        "equivalence_report_sha256": sha256_file(equivalence_path),
        "equivalence_status": equivalence_report["status"],
    }
    result_manifest["operational_alignment"] = {
        "status": alignment_report["status"],
        "report_sha256": sha256_file(alignment_path),
        "positions_checked": alignment_report["positions_checked"],
        "failed_positions": alignment_report["failed_positions"],
    }
    if expected_sources != _current_causal_source_hashes():
        raise CausalValidationError("Causal/control source changed during evaluation")
    write_run_manifest(output_dir / "manifest.json", result_manifest)
    return update_run_status(
        output_dir / "manifest.json", "validated", expected_current_status="running"
    )


def run_evaluation(
    model: Any,
    model_path: Path,
    run: ValidatedRun,
    output_dir: Path,
    *,
    doses: Sequence[float] = DEFAULT_DOSES,
    maximum_calibration_positions: int = DEFAULT_CALIBRATION_POSITIONS,
    maximum_test_positions: int = DEFAULT_CAUSAL_TEST_POSITIONS,
    spatial_shuffles: int = DEFAULT_SHUFFLES,
    random_directions: int = DEFAULT_RANDOM_DIRECTIONS,
    forcing_cache: Optional[Path] = None,
    seed: int = DEFAULT_SEED,
    require_control_match: bool = True,
    head_batch_size: int = DEFAULT_HEAD_BATCH_SIZE,
    equivalence_sample_size: int = DEFAULT_EQUIVALENCE_SAMPLE_SIZE,
    policy_equivalence_atol: float = DEFAULT_POLICY_EQUIVALENCE_ATOL,
    activation_equivalence_atol: float = DEFAULT_ACTIVATION_EQUIVALENCE_ATOL,
) -> Mapping[str, Any]:
    """Run matched controls on calibration games and evaluate final test games."""
    output_dir = output_dir.resolve()
    _inside(output_dir, run.run_dir, "causal result output")
    _reject_archive_path(output_dir, "Causal result output")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite causal result directory: {output_dir}")
    dose_values = tuple(float(dose) for dose in doses)
    if not dose_values or any(not math.isfinite(dose) for dose in dose_values):
        raise ValueError("doses must be a non-empty sequence of finite numbers")
    if len(set(dose_values)) != len(dose_values):
        raise ValueError("doses must be unique")
    checkpoint_sha = _checkpoint_hash(model_path)
    _require_fresh_checkpoint(run, checkpoint_sha)
    contract = get_contract(run.concept)

    candidate_maps: Optional[Mapping[str, Mapping[int, float]]] = None
    forcing_cache_manifest_sha = None
    if contract.name == "reply_peak95":
        if forcing_cache is None:
            raise CausalValidationError(
                "reply_peak95 evaluation requires --forcing-cache built by build-cache"
            )
        candidate_maps = _load_forcing_cache(
            run, forcing_cache, checkpoint_sha256=checkpoint_sha
        )
        forcing_cache_manifest_sha = sha256_file(Path(forcing_cache) / "manifest.json")
    elif forcing_cache is not None:
        raise CausalValidationError("A forcing cache is only valid for reply_peak95")

    calibration_selected = select_positions(
        run, "control_calibration", maximum_calibration_positions, seed=seed
    )
    test_selected = select_positions(run, "causal_test", maximum_test_positions, seed=seed)
    _require_complete_role_selection(run, calibration_selected, "control_calibration")
    _require_complete_role_selection(run, test_selected, "causal_test")
    assert_disjoint_protocol(
        (run.probe_metadata.get("final_fit") or {}).get("training_game_ids") or [],
        calibration_selected["game_id"].astype(str),
        test_selected["game_id"].astype(str),
    )
    if set(calibration_selected["game_id"].astype(str)) & set(test_selected["game_id"].astype(str)):
        raise CausalValidationError("Calibration and final causal position selections share games")
    if candidate_maps is not None:
        selected_ids = set(calibration_selected["position_id"]) | set(test_selected["position_id"])
        missing = sorted(selected_ids - set(candidate_maps))
        if missing:
            raise CausalValidationError(
                f"Forcing cache lacks {len(missing)} deterministically selected positions; "
                "rebuild it with the same seed/counts. Missing: " + ", ".join(missing[:5])
            )
        if run.representation in {"local", "combined"}:
            calibration_selected = _filter_forcing_mask_eligible(
                calibration_selected, candidate_maps
            )
            test_selected = _filter_forcing_mask_eligible(test_selected, candidate_maps)
            if calibration_selected.empty or test_selected.empty:
                raise CausalValidationError(
                    "No reply_peak95 positions have both target and comparison actions "
                    "in one of the causal protocol splits"
                )
            _require_complete_role_selection(
                run, calibration_selected, "control_calibration"
            )
            _require_complete_role_selection(run, test_selected, "causal_test")

    learned = InterventionDirection.load(run)
    controls = control_specs(
        learned,
        seed=seed,
        spatial_shuffles=spatial_shuffles,
        random_directions=random_directions,
    )
    policy_backend = PolicyHeadOnlyBackend(
        model,
        channels=run.channels,
        board_size=run.board_size,
        batch_size=head_batch_size,
        policy_atol=policy_equivalence_atol,
        activation_atol=activation_equivalence_atol,
    )
    estimate = estimate_protocol_forwards(
        calibration_positions=len(calibration_selected),
        causal_test_positions=len(test_selected),
        representation=run.representation,
        doses=dose_values,
        spatial_shuffles=spatial_shuffles,
        random_directions=random_directions,
        head_batch_size=head_batch_size,
        equivalence_sample_size=equivalence_sample_size,
    )
    estimate["intervention_direction"] = learned.metadata()
    provenance = {
        "checkpoint": str(model_path.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "probe_run": str(run.run_dir),
        "probe_run_manifest_sha256": sha256_file(run.run_dir / "manifest.json"),
        "build_manifest_sha256": sha256_file(run.run_dir / "build_manifest.json"),
        "training_manifest_sha256": sha256_file(run.run_dir / "training_manifest.json"),
        "dataset_sha256": run.build_manifest["dataset_sha256"],
        "input_provenance_sha256": run.build_manifest["input_provenance_sha256"],
        "trunk_identity_bytes_sha256": run.build_manifest["input_provenance"][
            "trunk_identity_bytes_sha256"
        ],
        "labels_sha256": run.build_manifest["labels_sha256"],
        "splits_sha256": run.manifest["artifacts"]["splits_sha256"],
        "probe_sha256": sha256_file(run.probe_path),
        "scaler_sha256": sha256_file(run.scaler_path),
        "probe_metadata_sha256": sha256_file(run.probe_metadata_path),
        "forcing_cache_manifest_sha256": forcing_cache_manifest_sha,
        "concept": run.concept,
        "contract_id": contract.definition_id,
        "contract_hash": contract.contract_hash,
        "representation": run.representation,
        "seed": int(seed),
        "evaluation_backend": "saved_trunkfinal_policy_head",
        "policy_head_batch_size": int(head_batch_size),
        "equivalence_sample_size": int(equivalence_sample_size),
        "policy_equivalence_atol": float(policy_equivalence_atol),
        "activation_equivalence_atol": float(activation_equivalence_atol),
        "fresh_holdout": _fresh_provenance_record(run),
        "checkpoint_activation_fidelity": dict(
            run.checkpoint_activation_fidelity
        ),
        "source_code_sha256": _current_causal_source_hashes(),
        "producer_source_sha256": _current_causal_source_hashes(),
    }
    result_manifest, estimate_path = _start_running_output(
        output_dir,
        kind="causal_evaluation",
        provenance=provenance,
        estimate=estimate,
    )
    try:
        return _execute_causal_protocol(
            model,
            policy_backend,
            run,
            output_dir,
            calibration_selected=calibration_selected,
            test_selected=test_selected,
            candidate_maps=candidate_maps,
            learned=learned,
            controls=controls,
            doses=dose_values,
            seed=seed,
            require_control_match=require_control_match,
            result_manifest=result_manifest,
            estimate_path=estimate_path,
            equivalence_sample_size=equivalence_sample_size,
        )
    except BaseException:
        _mark_running_output_failed(output_dir)
        raise


def _load_model(path: Path, device_name: str, board_size: int) -> Any:
    try:
        from .common_utils import get_device
    except ImportError:  # pragma: no cover - direct CLI execution.
        from common_utils import get_device
    from load_model import load_model

    device = get_device(device_name)
    model, _, _ = load_model(
        path, use_swa=False, device=device, pos_len=int(board_size), verbose=False
    )
    return model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache = subparsers.add_parser(
        "build-cache", help="Build immutable reply_peak95 candidate-action cache"
    )
    cache.add_argument("--run-dir", type=Path, required=True)
    cache.add_argument("--model", type=Path, required=True)
    cache.add_argument("--concept", default="forcing")
    cache.add_argument("--representation", choices=REPRESENTATIONS, default="local")
    cache.add_argument("--output-dir", type=Path, required=True)
    cache.add_argument(
        "--max-calibration-positions", type=int, default=DEFAULT_CALIBRATION_POSITIONS
    )
    cache.add_argument(
        "--max-test-positions", type=int, default=DEFAULT_CAUSAL_TEST_POSITIONS
    )
    cache.add_argument("--seed", type=int, default=DEFAULT_SEED)
    cache.add_argument("--device", default="auto")

    estimate = subparsers.add_parser(
        "estimate", help="Read-only forward-cost estimate for the causal protocol"
    )
    estimate.add_argument("--run-dir", type=Path, required=True)
    estimate.add_argument("--concept", required=True)
    estimate.add_argument("--representation", choices=REPRESENTATIONS, required=True)
    estimate.add_argument("--doses", type=float, nargs="+", default=list(DEFAULT_DOSES))
    estimate.add_argument(
        "--max-calibration-positions", type=int, default=DEFAULT_CALIBRATION_POSITIONS
    )
    estimate.add_argument(
        "--max-test-positions", type=int, default=DEFAULT_CAUSAL_TEST_POSITIONS
    )
    estimate.add_argument("--spatial-shuffles", type=int, default=DEFAULT_SHUFFLES)
    estimate.add_argument("--random-directions", type=int, default=DEFAULT_RANDOM_DIRECTIONS)
    estimate.add_argument("--head-batch-size", type=int, default=DEFAULT_HEAD_BATCH_SIZE)
    estimate.add_argument(
        "--equivalence-sample-size",
        type=int,
        default=DEFAULT_EQUIVALENCE_SAMPLE_SIZE,
    )
    estimate.add_argument("--seed", type=int, default=DEFAULT_SEED)

    evaluate = subparsers.add_parser(
        "evaluate", help="Calibrate controls and evaluate untouched causal-test games"
    )
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--concept", required=True)
    evaluate.add_argument("--representation", choices=REPRESENTATIONS, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--forcing-cache", type=Path)
    evaluate.add_argument("--doses", type=float, nargs="+", default=list(DEFAULT_DOSES))
    evaluate.add_argument(
        "--max-calibration-positions", type=int, default=DEFAULT_CALIBRATION_POSITIONS
    )
    evaluate.add_argument(
        "--max-test-positions", type=int, default=DEFAULT_CAUSAL_TEST_POSITIONS
    )
    evaluate.add_argument("--spatial-shuffles", type=int, default=DEFAULT_SHUFFLES)
    evaluate.add_argument("--random-directions", type=int, default=DEFAULT_RANDOM_DIRECTIONS)
    evaluate.add_argument("--head-batch-size", type=int, default=DEFAULT_HEAD_BATCH_SIZE)
    evaluate.add_argument(
        "--equivalence-sample-size",
        type=int,
        default=DEFAULT_EQUIVALENCE_SAMPLE_SIZE,
        help=(
            "Deterministic saved-activation/full-model baseline comparisons required "
            "before head-only evaluation"
        ),
    )
    evaluate.add_argument("--seed", type=int, default=DEFAULT_SEED)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument(
        "--allow-unmatched-controls",
        action="store_true",
        help="Persist explicitly flagged unmatched controls instead of failing closed",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    run = load_validated_run(args.run_dir, args.concept, args.representation)
    if args.command == "estimate":
        calibration = select_positions(
            run, "control_calibration", args.max_calibration_positions, seed=args.seed
        )
        causal_test = select_positions(
            run, "causal_test", args.max_test_positions, seed=args.seed
        )
        _require_complete_role_selection(run, calibration, "control_calibration")
        _require_complete_role_selection(run, causal_test, "causal_test")
        result = estimate_protocol_forwards(
            calibration_positions=len(calibration),
            causal_test_positions=len(causal_test),
            representation=run.representation,
            doses=args.doses,
            spatial_shuffles=args.spatial_shuffles,
            random_directions=args.random_directions,
            head_batch_size=args.head_batch_size,
            equivalence_sample_size=args.equivalence_sample_size,
        )
        result["intervention_direction"] = InterventionDirection.load(run).metadata()
        result["selection_note"] = (
            "Confirmatory counts must equal the frozen role sizes, with exactly one selected "
            "position per fresh game. A local/combined reply_peak95 run fails closed if any "
            "selected game lacks a position mask containing both target and comparison actions."
        )
        print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
        return
    model = _load_model(args.model, args.device, run.board_size)
    if args.command == "build-cache":
        result = build_forcing_cache(
            model,
            args.model,
            run,
            args.output_dir,
            maximum_calibration_positions=args.max_calibration_positions,
            maximum_test_positions=args.max_test_positions,
            seed=args.seed,
        )
    else:
        result = run_evaluation(
            model,
            args.model,
            run,
            args.output_dir,
            doses=args.doses,
            maximum_calibration_positions=args.max_calibration_positions,
            maximum_test_positions=args.max_test_positions,
            spatial_shuffles=args.spatial_shuffles,
            random_directions=args.random_directions,
            forcing_cache=args.forcing_cache,
            seed=args.seed,
            require_control_match=not args.allow_unmatched_controls,
            head_batch_size=args.head_batch_size,
            equivalence_sample_size=args.equivalence_sample_size,
        )
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
