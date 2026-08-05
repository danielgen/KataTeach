#!/usr/bin/env python3
"""Leakage-resistant, run-scoped linear-probe experiments.

This module intentionally does not reuse the legacy probe trainer.  A run has an
immutable game split and frozen concept configuration, a rebuilt feature table
whose local block is indexed by ``idx361``, and append-only build/training
manifests.  Existing outputs are never overwritten.

The default protocol reserves 350 games for development, 50 for calibrating
causal controls, and 100 for the final causal test.  Probe metrics are estimated
with five outer game folds.  C and the F1 threshold are selected entirely inside
each outer-training partition using four inner game folds.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

try:
    from .operational_definitions import get_contract
except ImportError:  # pragma: no cover - direct script execution
    from operational_definitions import get_contract


SCHEMA_VERSION = 1
PIPELINE_NAME = "validated_probe_pipeline"
DEFAULT_SEED = 20260730
DEFAULT_DEVELOPMENT_GAMES = 350
DEFAULT_CONTROL_CALIBRATION_GAMES = 50
DEFAULT_CAUSAL_TEST_GAMES = 100
DEFAULT_OUTER_FOLDS = 5
DEFAULT_INNER_FOLDS = 4
DEFAULT_C_VALUES = (0.001, 0.01, 0.1, 1.0, 10.0)
DEFAULT_MAX_ITER = 2000
REPRESENTATIONS = ("global", "local", "combined")
SPLIT_ROLES = ("development", "control_calibration", "causal_test")
SOURCE_FILES = (
    "validated_probe_pipeline.py",
    "operational_definitions.py",
)
CHECKPOINT_METADATA_KEYS = frozenset(
    {
        "checkpoint",
        "checkpoint_path",
        "checkpoint_sha256",
        "model_checkpoint",
        "model_path",
        "model_sha256",
    }
)
TRUNK_HASH_SCHEME = (
    "per-game sha256 over length-prefixed relative file identity, byte length, and "
    "the exact .npy bytes read by numpy, plus an individually addressable SHA-256 "
    "leaf for every file; overall sha256 over the resolved source root, game "
    "identity, per-game digest, count, and byte total"
)


@dataclass(frozen=True)
class ConceptSpec:
    name: str
    kind: str
    source: str
    contract_id: Optional[str] = None
    feature_mode: str = "pre"
    threshold: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    threshold_key: Optional[str] = None
    q: float = 0.1
    direction: Optional[str] = None
    use_abs: bool = False
    filters: Tuple[Mapping[str, Any], ...] = ()
    stratify_by_phase: bool = False
    no_drop: bool = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(_json_safe(value), sort_keys=True, indent=2) + "\n").encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_length_prefixed(digest: Any, value: bytes) -> None:
    """Add an unambiguous byte string to a streaming digest."""

    digest.update(len(value).to_bytes(8, "big", signed=False))
    digest.update(value)


def _update_identity_bytes_hash(
    digest: Any, identity: str, payload: bytes
) -> None:
    """Bind one relative identity and its exact bytes into ``digest``."""

    _update_length_prefixed(digest, identity.encode("utf-8"))
    digest.update(len(payload).to_bytes(8, "big", signed=False))
    digest.update(payload)


def _update_identity_stat_hash(digest: Any, identity: str, path: Path) -> int:
    """Bind a cheap filesystem change detector; return the observed byte size."""

    stat = path.stat()
    _update_length_prefixed(digest, identity.encode("utf-8"))
    digest.update(int(stat.st_size).to_bytes(8, "big", signed=False))
    digest.update(int(stat.st_mtime_ns).to_bytes(8, "big", signed=False))
    return int(stat.st_size)


def _nested_metadata_values(value: Any, prefix: str = "") -> List[Dict[str, str]]:
    """Collect literal checkpoint-related generator metadata without inference."""

    records: List[Dict[str, str]] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_prefix = f"{prefix}.{key}" if prefix else key
            parent_key = prefix.rsplit(".", 1)[-1].lower() if prefix else ""
            is_checkpoint_scalar = key.lower() in CHECKPOINT_METADATA_KEYS or (
                parent_key == "checkpoint" and key.lower() in {"path", "sha256"}
            )
            if (
                is_checkpoint_scalar
                and child is not None
                and not isinstance(child, (Mapping, list))
            ):
                records.append({"field": child_prefix, "declared_value": str(child)})
            records.extend(_nested_metadata_values(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            records.extend(_nested_metadata_values(child, f"{prefix}[{index}]"))
    return records


def _generator_metadata_provenance(
    games_dir: Path, game_ids: Iterable[str]
) -> Dict[str, Any]:
    """Describe whether game metadata actually attributes saved activations.

    This deliberately records declarations only.  A checkpoint mentioned in a
    different artifact (including the legacy-result archive) is not evidence
    that it generated these activation files.
    """

    games_dir = games_dir.resolve()
    game_ids = tuple(sorted(str(game) for game in game_ids))
    digest = hashlib.sha256()
    present = 0
    missing: List[str] = []
    declarations: List[Dict[str, str]] = []
    games_with_declarations: set[str] = set()
    for game_id in game_ids:
        meta_path = games_dir / game_id / "meta.json"
        if not meta_path.is_file():
            missing.append(game_id)
            continue
        payload = meta_path.read_bytes()
        _update_identity_bytes_hash(digest, f"{game_id}/meta.json", payload)
        present += 1
        try:
            metadata = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid generator metadata JSON: {meta_path}") from exc
        game_declarations = _nested_metadata_values(metadata)
        if game_declarations:
            games_with_declarations.add(game_id)
        for record in game_declarations:
            declarations.append({"game_id": game_id, **record})

    unique_declarations = sorted(
        {(record["field"], record["declared_value"]) for record in declarations}
    )
    if not declarations:
        checkpoint = {
            "status": "not_recorded_in_generator_metadata",
            "declarations": [],
            "checkpoint_sha256": None,
            "limitation": (
                "The saved activations cannot be cryptographically attributed to a "
                "checkpoint because no source game meta.json records one."
            ),
        }
    else:
        complete = len(games_with_declarations) == len(game_ids)
        checkpoint = {
            "status": (
                "declared_for_all_games_not_verified"
                if complete
                else "partially_declared_in_generator_metadata_not_verified"
            ),
            "declarations": [
                {"field": field, "declared_value": declared_value}
                for field, declared_value in unique_declarations
            ],
            "games_with_declarations": len(games_with_declarations),
            "games_without_declarations": sorted(
                set(game_ids) - games_with_declarations
            ),
            "checkpoint_sha256": None,
            "limitation": (
                "Generator metadata contains checkpoint declarations for some or all "
                "games, but this stage does not infer missing historical provenance or "
                "claim activation fidelity from declarations alone."
            ),
        }
    return {
        "meta_files_present": present,
        "meta_files_missing": missing,
        "meta_identity_bytes_sha256": digest.hexdigest(),
        "checkpoint_attribution": checkpoint,
        "sampled_checkpoint_activation_validation": {
            "status": "not_performed",
            "reason": (
                "Exact source checkpoint provenance is unavailable; replaying a sampled "
                "activation against an assumed checkpoint would create false fidelity evidence."
            ),
        },
    }


def _source_hashes() -> Dict[str, str]:
    source_dir = Path(__file__).resolve().parent
    return {
        f"daniele_experiment/{name}": _sha256(source_dir / name)
        for name in SOURCE_FILES
    }


def _derived_seed(base_seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{int(base_seed)}:{name}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        result = float(value)
        return None if not math.isfinite(result) else result
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_new_json(path: Path, value: Any, *, readonly: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_canonical_json_bytes(value))
    if readonly:
        path.chmod(0o444)


def _dump_new_joblib(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        joblib.dump(value, handle)


def _require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def _read_manifest(run_dir: Path) -> Dict[str, Any]:
    path = _require_file(run_dir / "manifest.json", "run manifest")
    with path.open() as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported run schema {manifest.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    if manifest.get("pipeline") != PIPELINE_NAME:
        raise ValueError(f"Run {run_dir} was not created by {PIPELINE_NAME}")
    return manifest


def _verify_frozen_inputs(run_dir: Path, manifest: Mapping[str, Any]) -> None:
    config = _require_file(run_dir / "frozen_config" / "concepts.yaml", "frozen concepts")
    splits = _require_file(run_dir / "splits.parquet", "split manifest")
    expected_config = manifest["artifacts"]["concepts_yaml_sha256"]
    expected_splits = manifest["artifacts"]["splits_sha256"]
    if _sha256(config) != expected_config:
        raise ValueError("Frozen concepts.yaml hash does not match manifest")
    if _sha256(splits) != expected_splits:
        raise ValueError("splits.parquet hash does not match manifest")
    expected_sources = manifest.get("source_code_sha256")
    if not isinstance(expected_sources, Mapping):
        raise ValueError("Run manifest does not freeze pipeline source hashes")
    observed_sources = _source_hashes()
    if dict(expected_sources) != observed_sources:
        raise ValueError(
            "Validated pipeline source changed after run preparation; start a new run"
        )
    if manifest.get("contract_implementation_sha256") != observed_sources[
        "daniele_experiment/operational_definitions.py"
    ]:
        raise ValueError("Frozen contract implementation hash does not match source")
    splits_frame = pd.read_parquet(splits)
    game_ids = splits_frame["game_id"].astype(str).tolist()
    games_dir = Path(str(manifest["source_games_dir"])).resolve()
    expected_generator = manifest.get("activation_provenance")
    if not isinstance(expected_generator, Mapping):
        raise ValueError("Run manifest lacks explicit activation checkpoint provenance")
    observed_generator = _generator_metadata_provenance(games_dir, game_ids)
    if dict(expected_generator) != observed_generator:
        raise ValueError(
            "Generator metadata changed after run preparation; start a new run"
        )
    _verify_fresh_protocol_sources(manifest)


def _verify_fresh_protocol_sources(
    manifest: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """Revalidate every prospectively frozen source at each pipeline stage."""

    fresh = manifest.get("fresh_holdout")
    if not isinstance(fresh, Mapping):
        return None
    protocol_path = Path(str(fresh.get("protocol_path", ""))).resolve()
    expected_protocol_hash = str(fresh.get("protocol_manifest_sha256", ""))
    if (
        not protocol_path.is_file()
        or len(expected_protocol_hash) != 64
        or _sha256(protocol_path) != expected_protocol_hash
    ):
        raise ValueError("Fresh holdout protocol is missing or changed")
    try:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid fresh holdout protocol: {protocol_path}") from exc
    sources = protocol.get("source_sha256")
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("Fresh protocol does not freeze its analysis sources")
    repo_root = Path(__file__).resolve().parent.parent
    for relative, expected in sources.items():
        source_path = (repo_root / str(relative)).resolve()
        try:
            source_path.relative_to(repo_root)
        except ValueError as exc:
            raise ValueError(
                f"Fresh protocol source escapes the repository: {relative}"
            ) from exc
        if (
            not source_path.is_file()
            or len(str(expected)) != 64
            or _sha256(source_path) != str(expected)
        ):
            raise ValueError(
                f"Current source differs from fresh protocol freeze: {relative}"
            )
    return {
        "protocol_path": str(protocol_path),
        "protocol_manifest_sha256": expected_protocol_hash,
        "source_sha256": dict(sources),
    }


def _fresh_protocol_document(manifest: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    fresh = manifest.get("fresh_holdout")
    if not isinstance(fresh, Mapping):
        return None
    protocol_path = Path(str(fresh.get("protocol_path", ""))).resolve()
    if not protocol_path.is_file() or _sha256(protocol_path) != fresh.get(
        "protocol_manifest_sha256"
    ):
        raise ValueError("Fresh holdout protocol is missing or changed")
    return json.loads(protocol_path.read_text(encoding="utf-8"))


def _verify_fresh_probe_protocol(
    manifest: Mapping[str, Any],
    *,
    specs: Sequence[ConceptSpec],
    representations: Sequence[str],
    C_values: Sequence[float],
    max_iter: int,
) -> Optional[Dict[str, Any]]:
    """Require training arguments to exactly reproduce the frozen probe design."""

    protocol = _fresh_protocol_document(manifest)
    if protocol is None:
        return None
    probes = protocol.get("probes")
    if not isinstance(probes, Mapping):
        raise ValueError("Fresh protocol lacks a probe design")
    observed = {
        "development_games_only": int(manifest["split_counts"]["development"]),
        "representations": list(representations),
        "concepts": [spec.name for spec in specs],
        "outer_group_folds": int(manifest["nested_cv"]["outer_folds"]),
        "inner_group_folds": int(manifest["nested_cv"]["inner_folds"]),
        "C_values": [float(value) for value in C_values],
        "selection_metric": "mean inner-fold average precision",
        "f1_threshold": "inner out-of-fold maximum F1",
        "probability_calibration": False,
        "quality_gate": None,
        "max_iter": int(max_iter),
    }
    for key, value in observed.items():
        expected = probes.get(key)
        if expected != value:
            raise ValueError(
                f"Probe setting {key!r} differs from the frozen protocol: "
                f"observed={value!r}, expected={expected!r}"
            )
    if probes.get("all_enabled_concepts_required") is not True:
        raise ValueError("Fresh protocol must require every enabled canonical concept")
    return observed


def discover_games(games_dir: Path) -> List[str]:
    """Return sorted raw-complete game directory names.

    Labels deliberately do not participate in discovery.  They are generated
    after ``prepare`` into the run-scoped ``labels/games`` namespace, so an old
    game-local ``snorkel.jsonl`` can never leak into a validated run.
    """
    games_dir = games_dir.resolve()
    if not games_dir.is_dir():
        raise FileNotFoundError(f"Games directory does not exist: {games_dir}")
    result = []
    for game_dir in games_dir.iterdir():
        if not game_dir.is_dir():
            continue
        if (
            (game_dir / "moves.jsonl").is_file()
            and (game_dir / "trunkfinal").is_dir()
        ):
            result.append(game_dir.name)
    return sorted(result)


def _fresh_holdout_metadata(
    games_dir: Path, game_ids: Sequence[str], cohort: str
) -> Dict[str, Any]:
    """Validate a newly generated cohort before assigning causal holdout roles."""

    selected: Dict[str, Mapping[str, Any]] = {}
    for game_id in game_ids:
        meta_path = games_dir / game_id / "meta.json"
        if not meta_path.is_file():
            continue
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid game metadata: {meta_path}") from exc
        if metadata.get("cohort") == cohort:
            selected[game_id] = metadata
    if not selected:
        raise ValueError(f"No games declare fresh holdout cohort {cohort!r}")

    protocol_hashes: set[str] = set()
    protocol_paths: set[str] = set()
    checkpoint_hashes: set[str] = set()
    generator_hashes: set[str] = set()
    common_utils_hashes: set[str] = set()
    seeds: set[int] = set()
    created_at: List[str] = []
    for game_id, metadata in selected.items():
        protocol = metadata.get("protocol_manifest")
        checkpoint = metadata.get("checkpoint")
        generator = metadata.get("generator")
        rng = metadata.get("rng")
        if not all(isinstance(item, Mapping) for item in (protocol, checkpoint, generator, rng)):
            raise ValueError(
                f"Fresh holdout game {game_id} lacks protocol/checkpoint/generator/RNG provenance"
            )
        verification = protocol.get("verification")
        if not isinstance(verification, Mapping):
            raise ValueError(
                f"Fresh holdout game {game_id} lacks pre-model-load protocol verification"
            )
        protocol_hashes.add(str(protocol.get("sha256", "")))
        protocol_paths.add(str(protocol.get("path", "")))
        checkpoint_hashes.add(str(checkpoint.get("sha256", "")))
        generator_hashes.add(str(generator.get("source_sha256", "")))
        common_utils_hashes.add(str(generator.get("common_utils_source_sha256", "")))
        game_seed = rng.get("game_seed")
        if game_seed is None or int(game_seed) in seeds:
            raise ValueError(f"Fresh holdout game {game_id} has a missing/duplicate RNG seed")
        seeds.add(int(game_seed))
        expected_game_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"katateach:{cohort}:{int(game_seed)}",
            )
        )
        if (
            game_id != expected_game_id
            or str(metadata.get("uuid", "")) != expected_game_id
            or rng.get("algorithm") != "numpy.default_rng/PCG64"
        ):
            raise ValueError(
                f"Fresh holdout game {game_id} does not reproduce its frozen "
                "UUIDv5/RNG identity"
            )
        timestamp = str(metadata.get("created_at_utc", ""))
        if not timestamp:
            raise ValueError(f"Fresh holdout game {game_id} lacks created_at_utc")
        created_at.append(timestamp)
        if checkpoint.get("use_swa") is not False or checkpoint.get("selected_weights") != "raw_model":
            raise ValueError(f"Fresh holdout game {game_id} was not generated from raw non-SWA weights")

    for description, values in (
        ("protocol manifest", protocol_hashes),
        ("checkpoint", checkpoint_hashes),
        ("generator source", generator_hashes),
        ("common-utils source", common_utils_hashes),
    ):
        if len(values) != 1 or len(next(iter(values))) != 64:
            raise ValueError(f"Fresh holdout games do not share one valid {description} hash")
    seed_digest = hashlib.sha256(
        ",".join(map(str, sorted(seeds))).encode("ascii")
    ).hexdigest()
    if len(protocol_paths) != 1:
        raise ValueError("Fresh holdout games do not share one protocol path")
    protocol_path = Path(next(iter(protocol_paths))).resolve()
    if not protocol_path.is_file() or _sha256(protocol_path) != next(iter(protocol_hashes)):
        raise ValueError("Fresh holdout protocol file is missing or its hash changed")
    try:
        protocol_document = json.loads(protocol_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid fresh holdout protocol JSON: {protocol_path}") from exc
    protocol_fresh = protocol_document.get("fresh_holdout") or {}
    protocol_sources = protocol_document.get("source_sha256") or {}
    generation = protocol_document.get("game_generation") or {}
    try:
        frozen_at = datetime.fromisoformat(str(protocol_document["frozen_at_utc"]))
        creation_times = [datetime.fromisoformat(value) for value in created_at]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Fresh protocol/game timestamps are missing or invalid") from exc
    if frozen_at.tzinfo is None or any(value.tzinfo is None for value in creation_times):
        raise ValueError("Fresh protocol/game timestamps must be timezone-aware")
    if any(value <= frozen_at for value in creation_times):
        raise ValueError("A fresh holdout game predates the protocol freeze")
    if (
        protocol_document.get("status") != "frozen_before_fresh_data_generation"
        or protocol_fresh.get("cohort") != cohort
        or int(protocol_fresh.get("games", -1)) != len(selected)
        or protocol_fresh.get("game_seed_set_sha256") != seed_digest
        or (protocol_document.get("checkpoint") or {}).get("sha256")
        != next(iter(checkpoint_hashes))
        or protocol_sources.get("daniele_experiment/generate_games_dataset.py")
        != next(iter(generator_hashes))
        or protocol_sources.get("daniele_experiment/common_utils.py")
        != next(iter(common_utils_hashes))
    ):
        raise ValueError(
            "Fresh holdout metadata does not reproduce its frozen protocol"
        )
    for game_id, metadata in selected.items():
        verification = (metadata.get("protocol_manifest") or {}).get("verification") or {}
        game_seed = int((metadata.get("rng") or {})["game_seed"])
        if (
            verification.get("status") != "passed_before_model_load"
            or verification.get("protocol_sha256") != next(iter(protocol_hashes))
            or dict(verification.get("verified_source_sha256") or {})
            != dict(protocol_sources)
            or not int(verification.get("shard_seed_first", game_seed))
            <= game_seed
            <= int(verification.get("shard_seed_last", game_seed))
            or int(verification.get("shard_game_count", -1)) < 1
        ):
            raise ValueError(
                f"Fresh game {game_id} was not generated under the frozen runtime bytes"
            )
    repo_root = Path(__file__).resolve().parent.parent
    for relative, expected_hash in protocol_sources.items():
        source_path = repo_root / str(relative)
        if not source_path.is_file() or _sha256(source_path) != expected_hash:
            raise ValueError(
                f"Current source differs from fresh protocol freeze: {relative}"
            )
    expected_generation = {
        "board_size": int(generation.get("board_size", -1)),
        "device": generation.get("device"),
        "torch_threads": int(generation.get("torch_threads", -1)),
        "initial_temperature": float(generation.get("initial_temperature", math.nan)),
        "final_temperature": float(generation.get("final_temperature", math.nan)),
        "transition_moves": int(generation.get("transition_moves", -1)),
        "min_prob": float(generation.get("minimum_raw_policy_probability", math.nan)),
        "top_k": int(generation.get("top_k", -1)),
        "resign_threshold": float(generation.get("resign_threshold", math.nan)),
        "resign_consec": int(generation.get("resign_consecutive_moves", -1)),
        "maximum_moves": int(generation.get("maximum_moves", -1)),
        "save_html": int(generation.get("save_html", -1)),
    }
    for game_id, metadata in selected.items():
        for key, expected in expected_generation.items():
            observed = metadata.get(key)
            if isinstance(expected, float):
                try:
                    agrees = math.isclose(
                        float(observed), expected, rel_tol=0.0, abs_tol=1e-12
                    )
                except (TypeError, ValueError):
                    agrees = False
            else:
                agrees = observed == expected
            if not agrees:
                raise ValueError(
                    f"Fresh game {game_id} generation parameter {key!r} differs "
                    "from the frozen protocol"
                )
        if (
            metadata.get("policy_source") != "direct_neural_policy_without_mcts"
            or metadata.get("immutable_outputs") is not True
        ):
            raise ValueError(
                f"Fresh game {game_id} lacks direct-policy/write-once provenance"
            )
        game_dir = games_dir / game_id
        forbidden = (game_dir / "snorkel.jsonl", game_dir / "viz.html")
        if any(path.exists() for path in forbidden):
            raise ValueError(
                f"Fresh game {game_id} contains forbidden derived artifacts"
            )
        immutable_paths = [
            game_dir,
            game_dir / "trunkfinal",
            game_dir / "meta.json",
            game_dir / "moves.jsonl",
            game_dir / "game.sgf",
            *(game_dir / "trunkfinal").glob("move_*.npy"),
        ]
        if any(
            not path.exists() or (path.stat().st_mode & 0o222)
            for path in immutable_paths
        ):
            raise ValueError(
                f"Fresh game {game_id} raw tree is missing or still writable"
            )
    return {
        "cohort": cohort,
        "game_ids": sorted(selected),
        "games": len(selected),
        "protocol_manifest_sha256": next(iter(protocol_hashes)),
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "generator_source_sha256": next(iter(generator_hashes)),
        "common_utils_source_sha256": next(iter(common_utils_hashes)),
        "rng_seed_set_sha256": seed_digest,
        "protocol_path": str(protocol_path),
        "protocol_source_sha256": dict(protocol_sources),
        "protocol_split_seed": int(protocol_fresh["split_seed"]),
        "created_at_utc_min": min(created_at),
        "created_at_utc_max": max(created_at),
    }


def assign_folds(game_ids: Sequence[str], n_folds: int, seed: int) -> Dict[str, int]:
    """Assign whole games to deterministic, approximately equal folds."""
    game_ids = sorted(set(str(game) for game in game_ids))
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if len(game_ids) < n_folds:
        raise ValueError(f"Need at least {n_folds} games, found {len(game_ids)}")
    shuffled = np.asarray(game_ids, dtype=object)
    rng = np.random.default_rng(seed)
    shuffled = shuffled[rng.permutation(len(shuffled))]
    return {str(game): int(index % n_folds) for index, game in enumerate(shuffled)}


def prepare_run(
    run_dir: Path,
    games_dir: Path,
    concepts_yaml: Path,
    *,
    seed: int = DEFAULT_SEED,
    development_games: int = DEFAULT_DEVELOPMENT_GAMES,
    control_calibration_games: int = DEFAULT_CONTROL_CALIBRATION_GAMES,
    causal_test_games: int = DEFAULT_CAUSAL_TEST_GAMES,
    outer_folds: int = DEFAULT_OUTER_FOLDS,
    inner_folds: int = DEFAULT_INNER_FOLDS,
    fresh_holdout_cohort: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a new run with a frozen configuration and game-level split.

    The destination must not exist.  This is deliberate: a run identifier can
    never silently acquire artifacts from an earlier experiment.
    """
    run_dir = run_dir.resolve()
    games_dir = games_dir.resolve()
    concepts_yaml = concepts_yaml.resolve()
    if run_dir.exists():
        raise FileExistsError(f"Refusing to reuse existing run directory: {run_dir}")
    _require_file(concepts_yaml, "concept configuration")

    counts = {
        "development": int(development_games),
        "control_calibration": int(control_calibration_games),
        "causal_test": int(causal_test_games),
    }
    if counts["development"] < 1 or any(
        counts[role] < 0 for role in ("control_calibration", "causal_test")
    ):
        raise ValueError(
            "Development must be non-empty and holdout counts non-negative: "
            f"{counts}"
        )
    if fresh_holdout_cohort is not None and any(
        counts[role] < 1 for role in ("control_calibration", "causal_test")
    ):
        raise ValueError(
            "Prospective causal runs require non-empty calibration and test splits"
        )
    games = discover_games(games_dir)
    expected = sum(counts.values())
    if len(games) != expected:
        raise ValueError(
            f"Split counts total {expected}, but found {len(games)} complete games in {games_dir}"
        )
    if development_games < max(outer_folds, inner_folds + 1):
        raise ValueError("Development split is too small for requested nested folds")

    run_dir.mkdir(parents=True)
    frozen_dir = run_dir / "frozen_config"
    frozen_dir.mkdir()
    (run_dir / "labels" / "games").mkdir(parents=True)
    frozen_config = frozen_dir / "concepts.yaml"
    with concepts_yaml.open("rb") as source, frozen_config.open("xb") as target:
        shutil.copyfileobj(source, target)

    rng = np.random.default_rng(seed)
    fresh_record = None
    if fresh_holdout_cohort is None:
        shuffled = np.asarray(games, dtype=object)[rng.permutation(len(games))]
        dev_end = counts["development"]
        calibration_end = dev_end + counts["control_calibration"]
        role_by_game = {
            **{str(game): "development" for game in shuffled[:dev_end]},
            **{
                str(game): "control_calibration"
                for game in shuffled[dev_end:calibration_end]
            },
            **{str(game): "causal_test" for game in shuffled[calibration_end:]},
        }
        split_assignment = "seeded_random_all_games"
    else:
        fresh_record = _fresh_holdout_metadata(
            games_dir, games, str(fresh_holdout_cohort)
        )
        if int(fresh_record["protocol_split_seed"]) != int(seed):
            raise ValueError(
                "Prepared run seed differs from the prospectively frozen holdout split seed"
            )
        protocol_document = _fresh_protocol_document(
            {"fresh_holdout": fresh_record}
        )
        assert protocol_document is not None
        protocol_probes = protocol_document.get("probes") or {}
        concepts_identity = protocol_probes.get("concepts_config")
        if (
            not isinstance(concepts_identity, str)
            or fresh_record["protocol_source_sha256"].get(concepts_identity)
            != _sha256(concepts_yaml)
        ):
            raise ValueError(
                "Requested concept configuration differs from the fresh protocol freeze"
            )
        fresh_games = list(fresh_record["game_ids"])
        development_pool = sorted(set(games) - set(fresh_games))
        if len(development_pool) != counts["development"]:
            raise ValueError(
                f"Fresh-cohort split requires exactly {counts['development']} "
                f"non-holdout games, found {len(development_pool)}"
            )
        historical = protocol_document.get("historical_data_scope") or {}
        development_identity_sha256 = hashlib.sha256(
            ",".join(development_pool).encode("utf-8")
        ).hexdigest()
        if (
            int(historical.get("development_games", -1))
            != counts["development"]
            or historical.get("development_game_ids_sha256")
            != development_identity_sha256
        ):
            raise ValueError(
                "Historical development game identities differ from the frozen protocol"
            )
        expected_fresh = counts["control_calibration"] + counts["causal_test"]
        if len(fresh_games) != expected_fresh:
            raise ValueError(
                f"Fresh cohort must contain exactly {expected_fresh} games, "
                f"found {len(fresh_games)}"
            )
        protocol_fresh = protocol_document.get("fresh_holdout") or {}
        if (
            int(protocol_fresh.get("control_calibration_games", -1))
            != counts["control_calibration"]
            or int(protocol_fresh.get("causal_test_games", -1))
            != counts["causal_test"]
        ):
            raise ValueError("Prepared holdout split counts differ from the protocol")
        shuffled_fresh = np.asarray(fresh_games, dtype=object)[
            rng.permutation(len(fresh_games))
        ]
        calibration_end = counts["control_calibration"]
        role_by_game = {
            **{game: "development" for game in development_pool},
            **{
                str(game): "control_calibration"
                for game in shuffled_fresh[:calibration_end]
            },
            **{
                str(game): "causal_test"
                for game in shuffled_fresh[calibration_end:]
            },
        }
        split_assignment = "legacy_development_fresh_cohort_holdouts"
    outer_by_game = assign_folds(
        [game for game, role in role_by_game.items() if role == "development"],
        outer_folds,
        seed + 1,
    )
    split_rows = [
        {
            "game_id": game,
            "split_role": role_by_game[game],
            "outer_fold": outer_by_game.get(game),
        }
        for game in games
    ]
    splits_path = run_dir / "splits.parquet"
    pd.DataFrame(split_rows).to_parquet(splits_path, index=False)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "pipeline": PIPELINE_NAME,
        "created_at_utc": _utc_now(),
        "seed": int(seed),
        "source_games_dir": str(games_dir),
        "source_concepts_yaml": str(concepts_yaml),
        "split_counts": counts,
        "split_assignment": split_assignment,
        "fresh_holdout": fresh_record,
        "nested_cv": {
            "outer_folds": int(outer_folds),
            "inner_folds": int(inner_folds),
            "selection_metric": "average_precision",
            "threshold_selection": "inner_oof_max_f1",
        },
        "source_code_sha256": _source_hashes(),
        "contract_implementation_sha256": _source_hashes()[
            "daniele_experiment/operational_definitions.py"
        ],
        "activation_provenance": _generator_metadata_provenance(games_dir, games),
        "artifacts": {
            "concepts_yaml": "frozen_config/concepts.yaml",
            "concepts_yaml_sha256": _sha256(frozen_config),
            "splits": "splits.parquet",
            "splits_sha256": _sha256(splits_path),
            "labels_games_dir": "labels/games",
        },
    }
    specs = load_concept_specs(frozen_config)
    validate_contract_specs(specs)
    _verify_fresh_probe_protocol(
        manifest,
        specs=specs,
        representations=REPRESENTATIONS,
        C_values=DEFAULT_C_VALUES,
        max_iter=DEFAULT_MAX_ITER,
    )
    _write_new_json(run_dir / "manifest.json", manifest)
    frozen_config.chmod(0o444)
    splits_path.chmod(0o444)
    return manifest


def load_concept_specs(path: Path, selected: Optional[Sequence[str]] = None) -> List[ConceptSpec]:
    with path.open() as handle:
        config = yaml.safe_load(handle) or {}
    raw_specs = config.get("concepts", {})
    wanted = list(selected) if selected else [
        name for name, raw in raw_specs.items() if raw.get("enabled", True)
    ]
    missing = sorted(set(wanted) - set(raw_specs))
    if missing:
        raise ValueError(f"Unknown concepts: {', '.join(missing)}")
    specs = []
    for name in wanted:
        raw = raw_specs[name]
        if not raw.get("enabled", True):
            raise ValueError(f"Concept {name!r} is disabled in the frozen configuration")
        mode = str(raw.get("feature_mode", "pre")).lower()
        if mode not in {"pre", "post", "delta"}:
            raise ValueError(f"Concept {name!r} has invalid feature_mode={mode!r}")
        kind = str(raw["type"])
        if kind not in {"binary", "threshold", "threshold_negative", "range", "quantile"}:
            raise ValueError(f"Concept {name!r} has unsupported type={kind!r}")
        specs.append(ConceptSpec(
            name=name,
            kind=kind,
            source=str(raw["source"]),
            contract_id=raw.get("contract_id"),
            feature_mode=mode,
            threshold=raw.get("threshold"),
            min_val=raw.get("min_val"),
            max_val=raw.get("max_val"),
            threshold_key=raw.get("threshold_key"),
            q=float(raw.get("q", 0.1)),
            direction=raw.get("direction"),
            use_abs=bool(raw.get("use_abs", False)),
            filters=tuple(raw.get("filters") or ()),
            stratify_by_phase=bool(raw.get("stratify_by_phase", False)),
            no_drop=bool(raw.get("no_drop", False)),
        ))
    return specs


def _expected_feature_mode_for_contract(contract: Any) -> str:
    if contract.representation_time.value == "pre_move":
        return "pre"
    if contract.representation_time.value == "post_move_reply":
        return "post"
    raise ValueError(
        f"No probe feature-mode mapping for policy time "
        f"{contract.representation_time.value!r}"
    )


def validate_contract_specs(specs: Sequence[ConceptSpec]) -> None:
    """Fail closed when YAML semantics drift from a versioned contract."""

    for spec in specs:
        if spec.contract_id is None:
            continue
        contract = get_contract(spec.contract_id)
        if contract.definition_id != spec.contract_id:
            raise ValueError(
                f"Concept {spec.name!r} must persist canonical contract ID "
                f"{contract.definition_id!r}, not alias {spec.contract_id!r}"
            )
        if spec.source != contract.name:
            raise ValueError(
                f"Concept {spec.name!r} source {spec.source!r} disagrees with "
                f"contract source {contract.name!r}"
            )
        expected_mode = _expected_feature_mode_for_contract(contract)
        if spec.feature_mode != expected_mode:
            raise ValueError(
                f"Concept {spec.name!r} feature_mode {spec.feature_mode!r} disagrees "
                f"with contract representation_time "
                f"{contract.representation_time.value!r} (expected {expected_mode!r})"
            )

        if contract.name in {"tenuki_distance6", "reply_peak95"} and spec.kind != "binary":
            raise ValueError(
                f"Contract concept {spec.name!r} must use a binary probe label"
            )
        if contract.name == "regional_policy_peak":
            positive_quantile = float(contract.parameters["positive_quantile"])
            expected_q = 1.0 - positive_quantile
            if spec.kind != "quantile" or spec.direction != "high":
                raise ValueError(
                    f"Contract concept {spec.name!r} must be a high-direction quantile"
                )
            if not math.isclose(spec.q, expected_q, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    f"Concept {spec.name!r} q={spec.q} disagrees with contract "
                    f"positive_quantile={positive_quantile} (expected q={expected_q})"
                )
            if not spec.no_drop:
                raise ValueError(
                    f"Concept {spec.name!r} must set no_drop=true to implement the "
                    "contract's top-quantile-vs-all-other-eligible-positions label"
                )


def game_phase(move_number: int) -> str:
    if move_number <= 60:
        return "early"
    if move_number <= 150:
        return "mid"
    return "end"


def _raw_analysis_value(analysis: Mapping[str, Any], spec: ConceptSpec) -> Optional[float]:
    value = analysis.get(spec.source)
    if isinstance(value, Mapping):
        values = list(value.values())
        if not values:
            value = 0.0
        elif spec.threshold_key == "sum":
            value = sum(values)
        elif spec.threshold_key == "mean":
            value = sum(values) / len(values)
        else:
            value = max(values)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _fixed_label(value: Optional[float], spec: ConceptSpec) -> Optional[int]:
    if value is None:
        return None
    if spec.kind == "binary":
        return int(bool(value))
    if spec.kind == "threshold":
        if spec.threshold is None:
            raise ValueError(f"Concept {spec.name!r} is missing threshold")
        return int(value >= spec.threshold)
    if spec.kind == "threshold_negative":
        if spec.threshold is None:
            raise ValueError(f"Concept {spec.name!r} is missing threshold")
        return int(value <= spec.threshold)
    if spec.kind == "range":
        if spec.min_val is None or spec.max_val is None:
            raise ValueError(f"Concept {spec.name!r} is missing range bounds")
        return int(spec.min_val <= value <= spec.max_val)
    return None


def activation_views(h: np.ndarray, idx361: Optional[int]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return global mean and the local vector at a flat board index.

    ``idx361`` is a row-major tensor index, unlike KataGo's padded internal
    ``move_loc``.  Passes are represented by ``None`` and have no local view.
    """
    h = np.asarray(h)
    if h.ndim != 3 or h.shape[1] != h.shape[2]:
        raise ValueError(f"Expected C×N×N activation, got shape {h.shape}")
    global_mean = np.asarray(h.mean(axis=(1, 2)), dtype=np.float32)
    if idx361 is None:
        return global_mean, None
    index = int(idx361)
    board_area = h.shape[1] * h.shape[2]
    if index == board_area:  # explicit flat pass sentinel, when present
        return global_mean, None
    if index < 0 or index >= board_area:
        raise ValueError(f"idx361={index} is outside flat board range [0, {board_area})")
    y, x = divmod(index, h.shape[2])
    return global_mean, np.asarray(h[:, y, x], dtype=np.float32)


def flat_index_from_internal_loc(move_loc: Optional[int], board_size: int) -> Optional[int]:
    """Convert KataGo's padded board location to a tensor-flat index.

    This helper is used only to validate the separately stored ``idx361``.  It
    prevents a future producer from silently changing either coordinate system.
    """
    if move_loc is None or int(move_loc) == 0:  # KataGo pass location
        return None
    loc = int(move_loc)
    stride = board_size + 1
    x = (loc % stride) - 1
    y = (loc // stride) - 1
    if x < 0 or x >= board_size or y < 0 or y >= board_size:
        raise ValueError(f"Internal move_loc={loc} is outside a {board_size}x{board_size} board")
    return y * board_size + x


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_jsonl_hashed(path: Path) -> Tuple[List[Dict[str, Any]], str, int]:
    """Parse and hash the same JSONL bytes."""

    payload = path.read_bytes()
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    return rows, hashlib.sha256(payload).hexdigest(), len(payload)


def _hash_label_files(labels_dir: Path, game_ids: Iterable[str]) -> str:
    """Hash label identities and bytes in a deterministic order."""
    digest = hashlib.sha256()
    for game_id in sorted(str(game) for game in game_ids):
        path = _require_file(labels_dir / game_id / "snorkel.jsonl", "run-scoped labels")
        digest.update(game_id.encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _verify_labels_manifest(
    run_dir: Path,
    manifest: Mapping[str, Any],
    specs: Sequence[ConceptSpec],
    game_ids: Iterable[str],
) -> Tuple[Path, str]:
    """Verify the completed label stage and every declared output hash."""

    path = _require_file(run_dir / "labels_manifest.json", "labels manifest")
    with path.open(encoding="utf-8") as handle:
        labels_manifest = json.load(handle)
    if labels_manifest.get("pipeline") != "validated_label_builder":
        raise ValueError("Labels were not produced by validated_label_builder")
    if labels_manifest.get("status") != "complete":
        raise ValueError("Labels manifest is not complete")
    for key, observed in (
        ("run_manifest_sha256", _sha256(run_dir / "manifest.json")),
        ("split_manifest_sha256", manifest["artifacts"]["splits_sha256"]),
        ("concepts_yaml_sha256", manifest["artifacts"]["concepts_yaml_sha256"]),
    ):
        if labels_manifest.get(key) != observed:
            raise ValueError(f"Labels manifest provenance mismatch for {key}")

    source_dir = Path(__file__).resolve().parent
    expected_builder = _sha256(source_dir / "build_validated_labels.py")
    expected_contracts = _sha256(source_dir / "operational_definitions.py")
    if labels_manifest.get("builder_source_sha256") != expected_builder:
        raise ValueError("Label-builder source changed after label generation")
    if labels_manifest.get("operational_definitions_source_sha256") != expected_contracts:
        raise ValueError("Operational-definition source changed after label generation")

    label_games = labels_manifest.get("games")
    if not isinstance(label_games, Mapping):
        raise ValueError("Labels manifest lacks per-game output records")
    labels_dir = run_dir / manifest["artifacts"]["labels_games_dir"]
    games_dir = Path(str(manifest["source_games_dir"])).resolve()
    expected_games = sorted(str(game_id) for game_id in game_ids)
    if sorted(map(str, label_games)) != expected_games:
        raise ValueError("Labels manifest game identities differ from the frozen split")
    for game_id in expected_games:
        output = _require_file(
            labels_dir / game_id / "snorkel.jsonl", "run-scoped labels"
        )
        record = label_games[game_id]
        if not isinstance(record, Mapping):
            raise ValueError(f"Labels manifest has an invalid game record for {game_id}")
        moves_path = _require_file(
            games_dir / game_id / "moves.jsonl", "raw moves file"
        )
        if _sha256(moves_path) != record.get("moves_sha256"):
            raise ValueError(
                f"Raw moves hash mismatch for {game_id}; labels no longer describe "
                "the current source transitions"
            )
        if _sha256(output) != record.get("output_sha256"):
            raise ValueError(f"Run-scoped label hash mismatch for {game_id}")

    contract_records = labels_manifest.get("contracts") or {}
    for spec in specs:
        if spec.contract_id is None:
            continue
        contract = get_contract(spec.contract_id)
        if contract.definition_id != spec.contract_id:
            raise ValueError(f"Concept {spec.name} does not use a canonical contract ID")
        record = contract_records.get(spec.contract_id)
        if not isinstance(record, Mapping):
            raise ValueError(f"Labels manifest lacks contract {spec.contract_id}")
        if record.get("contract_sha256") != contract.contract_hash:
            raise ValueError(f"Contract hash mismatch for {spec.contract_id}")
    return path, _sha256(path)


def _load_hashed_activation(
    path: Path,
    identity: str,
    identity_bytes_digest: Any,
    identity_stat_digest: Any,
) -> Tuple[np.ndarray, int, str]:
    """Read an activation once, hashing the same bytes passed to ``numpy``.

    Loading through ``BytesIO`` avoids a separate full-file provenance pass over
    the activation corpus.  The stat digest is a cheap later change detector;
    the content digest remains the cryptographic build-time binding.
    """

    _require_file(path, "trunk activation")
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        payload = handle.read()
        after = os.fstat(handle.fileno())
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or len(payload) != after.st_size
    ):
        raise RuntimeError(f"Activation changed while it was being read: {path}")
    _update_identity_bytes_hash(identity_bytes_digest, identity, payload)
    _update_length_prefixed(identity_stat_digest, identity.encode("utf-8"))
    identity_stat_digest.update(int(after.st_size).to_bytes(8, "big", signed=False))
    identity_stat_digest.update(
        int(after.st_mtime_ns).to_bytes(8, "big", signed=False)
    )
    try:
        activation = np.load(io.BytesIO(payload), allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"Invalid NPY activation: {path}") from exc
    return np.asarray(activation), int(len(payload)), hashlib.sha256(payload).hexdigest()


def _overall_trunk_identity_bytes_sha256(
    source_games_dir: str, game_records: Mapping[str, Mapping[str, Any]]
) -> str:
    """Return a Merkle-style digest that binds all per-game content digests."""

    digest = hashlib.sha256()
    _update_length_prefixed(digest, str(source_games_dir).encode("utf-8"))
    for game_id in sorted(map(str, game_records)):
        record = game_records[game_id]
        _update_length_prefixed(digest, game_id.encode("utf-8"))
        try:
            game_digest = bytes.fromhex(str(record["identity_bytes_sha256"]))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid per-game activation digest for {game_id}") from exc
        if len(game_digest) != 32:
            raise ValueError(f"Invalid per-game activation digest for {game_id}")
        digest.update(game_digest)
        digest.update(int(record["file_count"]).to_bytes(8, "big", signed=False))
        digest.update(int(record["total_bytes"]).to_bytes(8, "big", signed=False))
    return digest.hexdigest()


def _expected_activation_stat_record(
    games_dir: Path, game_id: str, moves: Sequence[Mapping[str, Any]]
) -> Tuple[str, int, int]:
    """Re-stat expected activation identities without rereading activation bytes."""

    digest = hashlib.sha256()
    total_bytes = 0
    seen: set[str] = set()
    for move in moves:
        move_number = int(move["move_number"])
        identity = f"{game_id}/trunkfinal/move_{move_number:03d}.npy"
        if identity in seen:
            raise ValueError(f"Duplicate activation identity in raw moves: {identity}")
        seen.add(identity)
        path = _require_file(games_dir / identity, "trunk activation")
        total_bytes += _update_identity_stat_hash(digest, identity, path)
    return digest.hexdigest(), len(seen), total_bytes


def _verify_build_input_provenance(
    run_dir: Path,
    run_manifest: Mapping[str, Any],
    build_manifest: Mapping[str, Any],
) -> None:
    """Verify build-time content commitments and cheap current-source identity.

    Training deliberately does not reread tens of gigabytes of ``.npy`` bytes.
    It verifies the immutable build's Merkle-style content commitment, then
    checks current paths, move hashes, file identities, sizes, and mtimes.  Any
    subsequent full-byte revalidation is a separate provenance audit, not part
    of probe fitting.
    """

    provenance = build_manifest.get("input_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Build manifest lacks cryptographic input provenance")
    observed_provenance_hash = hashlib.sha256(
        _canonical_json_bytes(provenance)
    ).hexdigest()
    if observed_provenance_hash != build_manifest.get("input_provenance_sha256"):
        raise ValueError("Build input-provenance record hash mismatch")

    games_dir = Path(str(run_manifest["source_games_dir"])).resolve()
    if provenance.get("source_games_dir") != str(games_dir):
        raise ValueError("Build source-games path disagrees with the frozen run")
    records = provenance.get("games")
    if not isinstance(records, Mapping):
        raise ValueError("Build input provenance lacks per-game records")
    splits = pd.read_parquet(run_dir / "splits.parquet")
    expected_games = sorted(splits["game_id"].astype(str).tolist())
    if sorted(map(str, records)) != expected_games:
        raise ValueError("Build input games disagree with the frozen split")

    overall = _overall_trunk_identity_bytes_sha256(str(games_dir), records)
    if overall != provenance.get("trunk_identity_bytes_sha256"):
        raise ValueError("Overall activation content commitment is inconsistent")
    total_files = 0
    total_bytes = 0
    for game_id in expected_games:
        record = records[game_id]
        if not isinstance(record, Mapping):
            raise ValueError(f"Invalid build input record for {game_id}")
        game_dir = (games_dir / game_id).resolve()
        moves_path = game_dir / "moves.jsonl"
        trunk_dir = (game_dir / "trunkfinal").resolve()
        if record.get("source_game_dir") != str(game_dir):
            raise ValueError(f"Source game path mismatch for {game_id}")
        if record.get("moves_path") != str(moves_path):
            raise ValueError(f"Moves path mismatch for {game_id}")
        if record.get("trunkfinal_dir") != str(trunk_dir):
            raise ValueError(f"Trunk path mismatch for {game_id}")
        _require_file(moves_path, "raw moves file")
        moves, moves_sha256, _ = _load_jsonl_hashed(moves_path)
        if moves_sha256 != record.get("moves_sha256"):
            raise ValueError(f"Raw moves changed after feature building for {game_id}")
        file_records = record.get("files")
        if not isinstance(file_records, Mapping):
            raise ValueError(f"Activation leaf hashes are missing for {game_id}")
        expected_names = {
            f"move_{int(move['move_number']):03d}.npy" for move in moves
        }
        if set(map(str, file_records)) != expected_names:
            raise ValueError(f"Activation leaf identities disagree for {game_id}")
        for filename in sorted(expected_names):
            leaf = file_records[filename]
            if not isinstance(leaf, Mapping):
                raise ValueError(f"Invalid activation leaf for {game_id}/{filename}")
            expected_identity = f"{game_id}/trunkfinal/{filename}"
            if leaf.get("identity") != expected_identity:
                raise ValueError(f"Activation leaf path mismatch for {expected_identity}")
            declared_hash = str(leaf.get("sha256", ""))
            if len(declared_hash) != 64:
                raise ValueError(f"Invalid activation leaf hash for {expected_identity}")
            try:
                bytes.fromhex(declared_hash)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid activation leaf hash for {expected_identity}"
                ) from exc
            source_path = games_dir / expected_identity
            if int(leaf.get("bytes", -1)) != int(source_path.stat().st_size):
                raise ValueError(
                    f"Activation leaf changed after build (size mismatch): "
                    f"{expected_identity}"
                )
        stat_digest, file_count, byte_count = _expected_activation_stat_record(
            games_dir, game_id, moves
        )
        if stat_digest != record.get("identity_stat_sha256"):
            raise ValueError(
                f"Activation identities, sizes, or mtimes changed after build for {game_id}"
            )
        if file_count != int(record.get("file_count", -1)):
            raise ValueError(f"Activation count mismatch for {game_id}")
        if byte_count != int(record.get("total_bytes", -1)):
            raise ValueError(f"Activation byte-count mismatch for {game_id}")

        meta_path = game_dir / "meta.json"
        expected_meta_path = record.get("meta_path")
        expected_meta_hash = record.get("meta_sha256")
        if meta_path.is_file():
            if expected_meta_path != str(meta_path) or _sha256(meta_path) != expected_meta_hash:
                raise ValueError(f"Generator metadata changed after build for {game_id}")
        elif expected_meta_path is not None or expected_meta_hash is not None:
            raise ValueError(f"Generator metadata disappeared after build for {game_id}")
        total_files += file_count
        total_bytes += byte_count

    if total_files != int(provenance.get("trunk_file_count", -1)):
        raise ValueError("Overall activation file count mismatch")
    if total_bytes != int(provenance.get("trunk_total_bytes", -1)):
        raise ValueError("Overall activation byte count mismatch")
    if provenance.get("generator_metadata") != run_manifest.get("activation_provenance"):
        raise ValueError("Build checkpoint-provenance limitation differs from frozen run")


def build_run(run_dir: Path, *, concepts: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Build the corrected, block-explicit activation dataset for a prepared run."""
    run_dir = run_dir.resolve()
    manifest = _read_manifest(run_dir)
    _verify_frozen_inputs(run_dir, manifest)
    dataset_path = run_dir / "dataset.parquet"
    build_manifest_path = run_dir / "build_manifest.json"
    if dataset_path.exists() or build_manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite build artifacts in {run_dir}")

    specs = load_concept_specs(run_dir / "frozen_config" / "concepts.yaml", concepts)
    validate_contract_specs(specs)
    splits = pd.read_parquet(run_dir / "splits.parquet")
    if splits["game_id"].duplicated().any():
        raise ValueError("Split manifest contains duplicate game IDs")
    split_by_game = splits.set_index("game_id").to_dict("index")
    games_dir = Path(manifest["source_games_dir"]).resolve()
    labels_dir = run_dir / manifest["artifacts"]["labels_games_dir"]
    labels_manifest_path, labels_manifest_sha256 = _verify_labels_manifest(
        run_dir, manifest, specs, split_by_game
    )
    labels_sha256 = _hash_label_files(labels_dir, split_by_game)
    rows: List[Dict[str, Any]] = []
    channels: Optional[int] = None
    board_size: Optional[int] = None
    trunk_input_games: Dict[str, Dict[str, Any]] = {}
    trunk_file_count = 0
    trunk_total_bytes = 0

    for game_id in sorted(split_by_game):
        game_dir = (games_dir / game_id).resolve()
        moves_path = _require_file(game_dir / "moves.jsonl", "moves file")
        moves, moves_sha256, moves_bytes = _load_jsonl_hashed(moves_path)
        move_numbers = [int(move["move_number"]) for move in moves]
        if len(move_numbers) != len(set(move_numbers)):
            raise ValueError(f"Raw moves contain duplicate move numbers for {game_id}")
        trunk_dir = (game_dir / "trunkfinal").resolve()
        if not trunk_dir.is_dir():
            raise FileNotFoundError(f"Missing trunk activation directory: {trunk_dir}")
        game_identity_bytes_digest = hashlib.sha256()
        game_identity_stat_digest = hashlib.sha256()
        consumed_activation_identities: set[str] = set()
        activation_files: Dict[str, Dict[str, Any]] = {}
        game_activation_bytes = 0

        def load_activation_once(path: Path) -> np.ndarray:
            nonlocal game_activation_bytes
            identity = f"{game_id}/trunkfinal/{path.name}"
            if identity in consumed_activation_identities:
                raise RuntimeError(f"Activation was unexpectedly reread: {identity}")
            activation, byte_count, content_sha256 = _load_hashed_activation(
                path,
                identity,
                game_identity_bytes_digest,
                game_identity_stat_digest,
            )
            consumed_activation_identities.add(identity)
            game_activation_bytes += byte_count
            activation_files[path.name] = {
                "identity": identity,
                "bytes": int(byte_count),
                "sha256": content_sha256,
            }
            return activation

        snorkel_rows = _load_jsonl(
            _require_file(labels_dir / game_id / "snorkel.jsonl", "run-scoped labels")
        )
        analysis_by_move = {
            int(item["move_number"]): item.get("analysis", {}) for item in snorkel_rows
        }
        cached_path: Optional[Path] = None
        cached_h: Optional[np.ndarray] = None

        for offset, move in enumerate(moves):
            move_number = int(move["move_number"])
            h_path = game_dir / "trunkfinal" / f"move_{move_number:03d}.npy"
            _require_file(h_path, "trunk activation")
            if cached_path == h_path and cached_h is not None:
                h = cached_h
            else:
                h = load_activation_once(h_path)
            if h.ndim != 3 or h.shape[1] != h.shape[2]:
                raise ValueError(f"Invalid activation shape {h.shape} in {h_path}")
            channels = channels or int(h.shape[0])
            board_size = board_size or int(h.shape[1])
            if h.shape[0] != channels or h.shape[1] != board_size:
                raise ValueError("Activation dimensions vary within the run")

            idx_value = move.get("idx361")
            move_loc = move.get("move_loc")
            expected_idx = flat_index_from_internal_loc(move_loc, int(board_size))
            if idx_value is None:
                if expected_idx is not None:
                    raise ValueError(
                        f"Non-pass move {game_id}:{move_number} is missing required idx361"
                    )
                idx361 = None
            else:
                idx361 = int(idx_value)
                normalized_idx = None if idx361 == int(board_size) ** 2 else idx361
                if normalized_idx != expected_idx:
                    raise ValueError(
                        f"Coordinate mismatch at {game_id}:{move_number}: "
                        f"idx361={idx361}, move_loc implies {expected_idx}"
                    )
            pre_global, pre_local = activation_views(h, idx361)

            next_h = None
            next_path = None
            if offset + 1 < len(moves):
                next_number = int(moves[offset + 1]["move_number"])
                next_path = game_dir / "trunkfinal" / f"move_{next_number:03d}.npy"
                if next_path.is_file():
                    next_h = load_activation_once(next_path)
            if next_h is None:
                post_global = post_local = None
            else:
                post_global, post_local = activation_views(next_h, idx361)
            cached_path, cached_h = next_path, next_h

            analysis = analysis_by_move.get(move_number, {})
            split = split_by_game[game_id]
            row: Dict[str, Any] = {
                "row_id": f"{game_id}:{move_number}",
                "game_id": game_id,
                "move_number": move_number,
                "player": move.get("player"),
                "move_loc": move_loc,
                "idx361": idx361,
                "game_phase": game_phase(move_number),
                "split_role": split["split_role"],
                "outer_fold": split.get("outer_fold"),
                "has_next": next_h is not None,
                "has_local": pre_local is not None,
                "h_pre_global": pre_global.tolist(),
                "h_pre_local": None if pre_local is None else pre_local.tolist(),
                "h_post_global": None if post_global is None else post_global.tolist(),
                "h_post_local": None if post_local is None else post_local.tolist(),
            }
            for spec in specs:
                raw_value = _raw_analysis_value(analysis, spec)
                row[f"rawval_{spec.name}"] = raw_value
                if spec.kind != "quantile":
                    row[f"label_{spec.name}"] = _fixed_label(raw_value, spec)
                for filt in spec.filters:
                    column = str(filt["column"])
                    if column not in row and column in analysis:
                        row[column] = analysis[column]
            rows.append(row)

        if len(consumed_activation_identities) != len(moves):
            raise RuntimeError(
                f"Expected one consumed activation per move for {game_id}; "
                f"observed {len(consumed_activation_identities)} for {len(moves)} moves"
            )
        meta_path = game_dir / "meta.json"
        meta_record = (
            {"meta_path": str(meta_path), "meta_sha256": _sha256(meta_path)}
            if meta_path.is_file()
            else {"meta_path": None, "meta_sha256": None}
        )
        game_record = {
            "source_game_dir": str(game_dir),
            "moves_path": str(moves_path),
            "moves_sha256": moves_sha256,
            "moves_bytes": int(moves_bytes),
            "trunkfinal_dir": str(trunk_dir),
            "file_count": int(len(consumed_activation_identities)),
            "total_bytes": int(game_activation_bytes),
            "identity_bytes_sha256": game_identity_bytes_digest.hexdigest(),
            "identity_stat_sha256": game_identity_stat_digest.hexdigest(),
            "files": activation_files,
            **meta_record,
        }
        trunk_input_games[game_id] = game_record
        trunk_file_count += int(game_record["file_count"])
        trunk_total_bytes += int(game_record["total_bytes"])

    if not rows:
        raise ValueError("No activation rows were built")
    frame = pd.DataFrame(rows).sort_values(["game_id", "move_number"]).reset_index(drop=True)
    if frame["row_id"].duplicated().any():
        raise ValueError("Built dataset contains duplicate game/move identities")
    for spec in specs:
        if not spec.filters or spec.kind == "quantile":
            continue
        mask = filter_mask(frame, spec)
        label_col = f"label_{spec.name}"
        frame.loc[~mask, label_col] = np.nan

    frame.to_parquet(dataset_path, index=False)
    input_provenance = {
        "schema_version": 1,
        "source_games_dir": str(games_dir),
        "source_path_binding": "resolved absolute root plus resolved per-game input paths",
        "trunk_hash_scheme": TRUNK_HASH_SCHEME,
        "trunk_identity_bytes_sha256": _overall_trunk_identity_bytes_sha256(
            str(games_dir), trunk_input_games
        ),
        "trunk_file_count": int(trunk_file_count),
        "trunk_total_bytes": int(trunk_total_bytes),
        "games": trunk_input_games,
        "generator_metadata": manifest["activation_provenance"],
        "training_verification": {
            "activation_content": (
                "Build-time exact-byte digests are verified internally through the "
                "overall commitment; training does not reread activation contents."
            ),
            "current_source": (
                "Training rehashes raw moves and generator metadata and re-stats every "
                "expected activation identity, size, and mtime."
            ),
        },
    }
    build_manifest = {
        "schema_version": SCHEMA_VERSION,
        "pipeline": PIPELINE_NAME,
        "created_at_utc": _utc_now(),
        "dataset": "dataset.parquet",
        "dataset_sha256": _sha256(dataset_path),
        "split_manifest_sha256": manifest["artifacts"]["splits_sha256"],
        "concepts_yaml_sha256": manifest["artifacts"]["concepts_yaml_sha256"],
        "labels_games_dir": str(labels_dir.relative_to(run_dir)),
        "labels_sha256": labels_sha256,
        "labels_manifest": str(labels_manifest_path.relative_to(run_dir)),
        "labels_manifest_sha256": labels_manifest_sha256,
        "source_code_sha256": _source_hashes(),
        "contract_implementation_sha256": _source_hashes()[
            "daniele_experiment/operational_definitions.py"
        ],
        "input_provenance": input_provenance,
        "input_provenance_sha256": hashlib.sha256(
            _canonical_json_bytes(input_provenance)
        ).hexdigest(),
        "concepts": [spec.name for spec in specs],
        "concept_contracts": {spec.name: spec.contract_id for spec in specs},
        "rows": int(len(frame)),
        "games": int(frame["game_id"].nunique()),
        "trunk_channels": int(channels or 0),
        "board_size": int(board_size or 0),
        "feature_blocks": {
            "global": "channel mean over h_pre/h_post",
            "local": "h_pre/h_post[:, idx361 // board_size, idx361 % board_size]",
            "combined": "global concatenated with local",
        },
        "missing_post_policy": "excluded from post and delta probes; never zero-imputed",
    }
    _write_new_json(build_manifest_path, build_manifest)
    dataset_path.chmod(0o444)
    return build_manifest


def filter_mask(frame: pd.DataFrame, spec: ConceptSpec) -> np.ndarray:
    mask = np.ones(len(frame), dtype=bool)
    for filt in spec.filters:
        column = str(filt["column"])
        if column not in frame.columns:
            raise ValueError(f"Concept {spec.name!r} requires missing filter column {column!r}")
        values = frame[column].to_numpy()
        value = filt["value"]
        operator = filt["operator"]
        valid = ~pd.isna(values)
        if operator == "<=":
            current = values <= value
        elif operator == ">=":
            current = values >= value
        elif operator == "==":
            current = values == value
        elif operator == "!=":
            current = values != value
        elif operator == "<":
            current = values < value
        elif operator == ">":
            current = values > value
        else:
            raise ValueError(f"Unsupported filter operator {operator!r}")
        mask &= valid & np.asarray(current, dtype=bool)
    return mask


def _stack_vectors(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    values = series.tolist()
    dimension = None
    for value in values:
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            dimension = len(value)
            break
    if dimension is None:
        raise ValueError(f"Feature column {series.name!r} has no vectors")
    matrix = np.zeros((len(values), dimension), dtype=np.float32)
    valid = np.zeros(len(values), dtype=bool)
    for index, value in enumerate(values):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        vector = np.asarray(value, dtype=np.float32)
        if vector.shape != (dimension,):
            raise ValueError(
                f"Feature column {series.name!r} has inconsistent shape {vector.shape}"
            )
        if not np.isfinite(vector).all():
            raise ValueError(f"Feature column {series.name!r} contains non-finite values")
        matrix[index] = vector
        valid[index] = True
    return matrix, valid


def feature_views(
    frame: pd.DataFrame, spec: ConceptSpec
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    pre_global, pre_global_valid = _stack_vectors(frame["h_pre_global"])
    pre_local, pre_local_valid = _stack_vectors(frame["h_pre_local"])
    if spec.feature_mode == "pre":
        global_matrix, global_valid = pre_global, pre_global_valid
        local_matrix, local_valid = pre_local, pre_local_valid
    else:
        post_global, post_global_valid = _stack_vectors(frame["h_post_global"])
        post_local, post_local_valid = _stack_vectors(frame["h_post_local"])
        next_valid = frame["has_next"].astype(bool).to_numpy()
        global_valid = pre_global_valid & post_global_valid & next_valid
        local_valid = pre_local_valid & post_local_valid & next_valid
        if spec.feature_mode == "post":
            global_matrix, local_matrix = post_global, post_local
        else:
            global_matrix = post_global - pre_global
            local_matrix = post_local - pre_local
    combined_valid = global_valid & local_valid
    return {
        "global": (global_matrix, global_valid),
        "local": (local_matrix, local_valid),
        "combined": (
            np.concatenate([global_matrix, local_matrix], axis=1),
            combined_valid,
        ),
    }


def _quantile_thresholds(
    frame: pd.DataFrame, spec: ConceptSpec, fit_indices: np.ndarray
) -> Any:
    raw = frame[f"rawval_{spec.name}"].to_numpy(dtype=float)
    allowed = filter_mask(frame, spec)
    fit_indices = np.asarray(fit_indices, dtype=int)

    def for_indices(indices: np.ndarray) -> Tuple[float, float]:
        values = raw[indices]
        values = values[allowed[indices] & np.isfinite(values)]
        if spec.use_abs:
            values = np.abs(values)
        if len(values) < 10:
            raise ValueError(
                f"Concept {spec.name!r} has fewer than 10 values for a quantile threshold"
            )
        return float(np.quantile(values, spec.q)), float(np.quantile(values, 1.0 - spec.q))

    if not spec.stratify_by_phase:
        return for_indices(fit_indices)
    phases = frame["game_phase"].to_numpy()
    result = {}
    for phase in ("early", "mid", "end"):
        phase_indices = fit_indices[phases[fit_indices] == phase]
        if len(phase_indices):
            try:
                result[phase] = for_indices(phase_indices)
            except ValueError:
                continue
    if not result:
        raise ValueError(f"Concept {spec.name!r} has no phase with enough quantile samples")
    return result


def labels_for(
    frame: pd.DataFrame,
    spec: ConceptSpec,
    target_indices: np.ndarray,
    *,
    threshold_fit_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Any]:
    """Label target rows, deriving quantile thresholds only from fit rows."""
    target_indices = np.asarray(target_indices, dtype=int)
    allowed = filter_mask(frame, spec)
    if spec.kind != "quantile":
        labels = frame[f"label_{spec.name}"].to_numpy(dtype=float)[target_indices]
        labels[~allowed[target_indices]] = np.nan
        return labels, None
    if threshold_fit_indices is None:
        raise ValueError("Quantile labels require threshold_fit_indices")
    thresholds = _quantile_thresholds(frame, spec, threshold_fit_indices)
    raw = frame[f"rawval_{spec.name}"].to_numpy(dtype=float)[target_indices]
    values = np.abs(raw) if spec.use_abs else raw
    phases = frame["game_phase"].to_numpy()[target_indices]
    labels = np.full(len(target_indices), np.nan, dtype=float)

    if isinstance(thresholds, Mapping):
        threshold_for_row = [thresholds.get(str(phase)) for phase in phases]
    else:
        threshold_for_row = [thresholds] * len(target_indices)
    for index, pair in enumerate(threshold_for_row):
        if pair is None or not allowed[target_indices[index]] or not np.isfinite(values[index]):
            continue
        low, high = pair
        if spec.direction == "high":
            if values[index] >= high:
                labels[index] = 1
            elif values[index] <= low or spec.no_drop:
                labels[index] = 0
        elif spec.direction == "low":
            if values[index] <= low:
                labels[index] = 1
            elif values[index] >= high or spec.no_drop:
                labels[index] = 0
        else:
            raise ValueError(f"Quantile concept {spec.name!r} needs direction high or low")
    return labels, thresholds


def _valid_xy(
    X: np.ndarray, indices: np.ndarray, labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(labels)
    selected = np.asarray(indices, dtype=int)[valid]
    y = labels[valid].astype(int)
    if len(y) < 4 or len(np.unique(y)) != 2:
        raise ValueError("Training/evaluation partition lacks enough samples from both classes")
    return X[selected], y, selected


def _require_converged(
    model: LogisticRegression, *, max_iter: int, context: str
) -> None:
    """Reject coefficients produced at the iteration cap."""

    iterations = np.asarray(getattr(model, "n_iter_", ()), dtype=int)
    if iterations.size == 0:
        raise RuntimeError(f"{context} did not expose a convergence iteration count")
    if np.any(iterations >= int(max_iter)):
        raise RuntimeError(
            f"{context} did not converge before max_iter={max_iter}; "
            f"n_iter={iterations.tolist()}"
        )


def _fit_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    C: float,
    seed: int,
    max_iter: int,
) -> Tuple[StandardScaler, LogisticRegression, np.ndarray]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(X_train)
    test_scaled = scaler.transform(X_test)
    model = LogisticRegression(
        C=float(C),
        class_weight="balanced",
        max_iter=int(max_iter),
        solver="lbfgs",
        random_state=int(seed),
    )
    model.fit(train_scaled, y_train)
    _require_converged(model, max_iter=max_iter, context="Outer-fold probe")
    return scaler, model, model.predict_proba(test_scaled)[:, 1]


def optimal_f1_threshold(y: np.ndarray, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    if not len(thresholds):
        return 0.5
    scores = 2.0 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    best = int(np.nanargmax(scores))
    return float(thresholds[best])


def _inner_selection(
    frame: pd.DataFrame,
    spec: ConceptSpec,
    X: np.ndarray,
    train_indices: np.ndarray,
    *,
    inner_folds: int,
    C_values: Sequence[float],
    seed: int,
    max_iter: int,
) -> Dict[str, Any]:
    games = frame["game_id"].astype(str).to_numpy()
    train_games = sorted(set(games[train_indices]))
    fold_by_game = assign_folds(train_games, inner_folds, seed)
    ap_by_c: Dict[float, List[float]] = {float(C): [] for C in C_values}
    predictions_by_c: Dict[float, List[np.ndarray]] = {float(C): [] for C in C_values}
    labels_by_c: Dict[float, List[np.ndarray]] = {float(C): [] for C in C_values}
    fold_records = []

    for inner_fold in range(inner_folds):
        validation_games = {game for game, fold in fold_by_game.items() if fold == inner_fold}
        is_validation = np.asarray([game in validation_games for game in games[train_indices]])
        fit_indices = train_indices[~is_validation]
        validation_indices = train_indices[is_validation]
        train_labels, thresholds = labels_for(
            frame, spec, fit_indices, threshold_fit_indices=fit_indices
        )
        validation_labels, _ = labels_for(
            frame, spec, validation_indices, threshold_fit_indices=fit_indices
        )
        X_train, y_train, _ = _valid_xy(X, fit_indices, train_labels)
        X_validation, y_validation, _ = _valid_xy(X, validation_indices, validation_labels)
        scaler = StandardScaler().fit(X_train)
        train_scaled = scaler.transform(X_train)
        validation_scaled = scaler.transform(X_validation)
        scores = {}
        for C in sorted(float(value) for value in C_values):
            model = LogisticRegression(
                C=C,
                class_weight="balanced",
                max_iter=max_iter,
                solver="lbfgs",
                random_state=seed,
            )
            model.fit(train_scaled, y_train)
            _require_converged(
                model,
                max_iter=max_iter,
                context=f"Inner-fold probe (fold={inner_fold}, C={C})",
            )
            probabilities = model.predict_proba(validation_scaled)[:, 1]
            score = float(average_precision_score(y_validation, probabilities))
            ap_by_c[C].append(score)
            predictions_by_c[C].append(probabilities)
            labels_by_c[C].append(y_validation)
            scores[str(C)] = score
        fold_records.append({
            "inner_fold": inner_fold,
            "train_games": sorted(set(games[fit_indices])),
            "validation_games": sorted(validation_games),
            "n_train": int(len(y_train)),
            "n_validation": int(len(y_validation)),
            "quantile_thresholds": thresholds,
            "average_precision_by_C": scores,
        })

    means = {C: float(np.mean(scores)) for C, scores in ap_by_c.items() if scores}
    if len(means) != len(C_values):
        raise ValueError("One or more C values were not scored in every inner fold")
    best_C = min(means)
    best_score = means[best_C]
    for C in sorted(means):
        if means[C] > best_score + 1e-12:
            best_C, best_score = C, means[C]
    oof_y = np.concatenate(labels_by_c[best_C])
    oof_probability = np.concatenate(predictions_by_c[best_C])
    threshold = optimal_f1_threshold(oof_y, oof_probability)
    return {
        "best_C": float(best_C),
        "mean_average_precision_by_C": {str(C): score for C, score in means.items()},
        "threshold": threshold,
        "inner_oof_f1": float(f1_score(oof_y, oof_probability >= threshold)),
        "folds": fold_records,
    }


def nested_group_evaluation(
    frame: pd.DataFrame,
    spec: ConceptSpec,
    X: np.ndarray,
    cohort_mask: np.ndarray,
    *,
    outer_folds: int,
    inner_folds: int,
    C_values: Sequence[float],
    seed: int,
    max_iter: int = 1000,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return honest outer-fold metrics and per-row outer predictions."""
    development = frame["split_role"].eq("development").to_numpy()
    outer_assignments = frame["outer_fold"].to_numpy(dtype=float)
    eligible = np.asarray(cohort_mask, dtype=bool) & development
    games = frame["game_id"].astype(str).to_numpy()
    fold_results: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []

    for outer_fold in range(outer_folds):
        test_indices = np.flatnonzero(eligible & (outer_assignments == outer_fold))
        train_indices = np.flatnonzero(eligible & (outer_assignments != outer_fold))
        if not len(test_indices) or not len(train_indices):
            raise ValueError(f"Outer fold {outer_fold} has an empty train or test partition")
        selection = _inner_selection(
            frame,
            spec,
            X,
            train_indices,
            inner_folds=inner_folds,
            C_values=C_values,
            seed=seed + 1000 + outer_fold,
            max_iter=max_iter,
        )
        train_labels, thresholds = labels_for(
            frame, spec, train_indices, threshold_fit_indices=train_indices
        )
        test_labels, _ = labels_for(
            frame, spec, test_indices, threshold_fit_indices=train_indices
        )
        X_train, y_train, _ = _valid_xy(X, train_indices, train_labels)
        X_test, y_test, selected_test = _valid_xy(X, test_indices, test_labels)
        scaler, model, probabilities = _fit_probe(
            X_train,
            y_train,
            X_test,
            C=selection["best_C"],
            seed=seed + outer_fold,
            max_iter=max_iter,
        )
        del scaler  # outer models are evaluation-only; final artifacts are fit below
        predictions = probabilities >= selection["threshold"]
        result = {
            "outer_fold": outer_fold,
            "train_games": sorted(set(games[train_indices])),
            "test_games": sorted(set(games[selected_test])),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "positive_train": int(y_train.sum()),
            "positive_test": int(y_test.sum()),
            "best_C": selection["best_C"],
            "f1_threshold": selection["threshold"],
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "average_precision": float(average_precision_score(y_test, probabilities)),
            "f1": float(f1_score(y_test, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
            "converged": bool(np.max(model.n_iter_) < max_iter),
            "quantile_thresholds": thresholds,
            "inner_selection": selection,
        }
        fold_results.append(result)
        for row_index, label, probability, prediction in zip(
            selected_test, y_test, probabilities, predictions
        ):
            prediction_rows.append({
                "row_id": frame.iloc[row_index]["row_id"],
                "game_id": games[row_index],
                "move_number": int(frame.iloc[row_index]["move_number"]),
                "outer_fold": outer_fold,
                "label": int(label),
                "probability": float(probability),
                "prediction": int(prediction),
                "f1_threshold": float(selection["threshold"]),
            })
    return fold_results, prediction_rows


def _aggregate_folds(folds: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"n_folds": len(folds)}
    for metric in ("roc_auc", "average_precision", "f1", "balanced_accuracy"):
        values = np.asarray([fold[metric] for fold in folds], dtype=float)
        result[f"mean_{metric}"] = float(values.mean())
        result[f"std_{metric}"] = float(values.std(ddof=0))
        result[f"min_{metric}"] = float(values.min())
        result[f"max_{metric}"] = float(values.max())
    return result


def _fit_final_probe(
    frame: pd.DataFrame,
    spec: ConceptSpec,
    X: np.ndarray,
    cohort_mask: np.ndarray,
    *,
    inner_folds: int,
    C_values: Sequence[float],
    seed: int,
    max_iter: int,
) -> Tuple[StandardScaler, LogisticRegression, Dict[str, Any]]:
    development = frame["split_role"].eq("development").to_numpy()
    train_indices = np.flatnonzero(np.asarray(cohort_mask, dtype=bool) & development)
    selection = _inner_selection(
        frame,
        spec,
        X,
        train_indices,
        inner_folds=inner_folds,
        C_values=C_values,
        seed=seed + 9000,
        max_iter=max_iter,
    )
    labels, thresholds = labels_for(
        frame, spec, train_indices, threshold_fit_indices=train_indices
    )
    X_train, y_train, selected = _valid_xy(X, train_indices, labels)
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(
        C=selection["best_C"],
        class_weight="balanced",
        max_iter=max_iter,
        solver="lbfgs",
        random_state=seed,
    )
    model.fit(scaler.transform(X_train), y_train)
    _require_converged(model, max_iter=max_iter, context="Final development probe")
    games = frame["game_id"].astype(str).to_numpy()
    metadata = {
        "best_C": selection["best_C"],
        "f1_threshold": selection["threshold"],
        "quantile_thresholds": thresholds,
        "selection": selection,
        "n_samples": int(len(y_train)),
        "positive_samples": int(y_train.sum()),
        "training_game_ids": sorted(set(games[selected])),
        "converged": bool(np.max(model.n_iter_) < max_iter),
    }
    return scaler, model, metadata


def _block_slices(representation: str, channels: int) -> Dict[str, List[int]]:
    if representation == "global":
        return {"global": [0, channels]}
    if representation == "local":
        return {"local": [0, channels]}
    if representation == "combined":
        return {"global": [0, channels], "local": [channels, 2 * channels]}
    raise ValueError(representation)


def _label_count_record(labels: np.ndarray, description: str) -> Dict[str, int]:
    finite = np.asarray(labels)[np.isfinite(labels)].astype(int)
    counts = {int(value): int((finite == value).sum()) for value in np.unique(finite)}
    if len(finite) < 4 or set(counts) != {0, 1}:
        raise ValueError(
            f"{description} lacks both classes or four samples: "
            f"n={len(finite)}, class_counts={counts}"
        )
    return {"samples": int(len(finite)), "negative": counts[0], "positive": counts[1]}


def audit_trainability(
    frame: pd.DataFrame,
    specs: Sequence[ConceptSpec],
    representations: Sequence[str],
    *,
    outer_folds: int,
    inner_folds: int,
    seed: int,
) -> Dict[str, Any]:
    """Fail before writing outputs if any requested nested-CV fold is invalid."""

    report: Dict[str, Any] = {}
    games = frame["game_id"].astype(str).to_numpy()
    development = frame["split_role"].eq("development").to_numpy()
    outer_assignment = frame["outer_fold"].to_numpy(dtype=float)
    for spec in specs:
        views = feature_views(frame, spec)
        common_mask = np.ones(len(frame), dtype=bool)
        for representation in representations:
            common_mask &= views[representation][1]
        eligible = common_mask & development
        development_indices = np.flatnonzero(eligible)
        labels, thresholds = labels_for(
            frame,
            spec,
            development_indices,
            threshold_fit_indices=development_indices,
        )
        concept_report: Dict[str, Any] = {
            "development": _label_count_record(labels, f"{spec.name} development"),
            "development_quantile_thresholds": thresholds,
            "outer_folds": [],
        }
        local_seed = _derived_seed(seed, spec.name)
        for outer_fold in range(outer_folds):
            train_indices = np.flatnonzero(eligible & (outer_assignment != outer_fold))
            test_indices = np.flatnonzero(eligible & (outer_assignment == outer_fold))
            train_labels, outer_thresholds = labels_for(
                frame, spec, train_indices, threshold_fit_indices=train_indices
            )
            test_labels, _ = labels_for(
                frame, spec, test_indices, threshold_fit_indices=train_indices
            )
            fold_record: Dict[str, Any] = {
                "outer_fold": outer_fold,
                "train": _label_count_record(
                    train_labels, f"{spec.name} outer {outer_fold} train"
                ),
                "test": _label_count_record(
                    test_labels, f"{spec.name} outer {outer_fold} test"
                ),
                "quantile_thresholds": outer_thresholds,
                "inner_folds": [],
            }
            train_games = sorted(set(games[train_indices]))
            inner_by_game = assign_folds(
                train_games, inner_folds, local_seed + 1000 + outer_fold
            )
            for inner_fold in range(inner_folds):
                validation_games = {
                    game for game, fold in inner_by_game.items() if fold == inner_fold
                }
                validation_mask = np.asarray(
                    [game in validation_games for game in games[train_indices]]
                )
                fit_indices = train_indices[~validation_mask]
                validation_indices = train_indices[validation_mask]
                fit_labels, _ = labels_for(
                    frame, spec, fit_indices, threshold_fit_indices=fit_indices
                )
                validation_labels, _ = labels_for(
                    frame, spec, validation_indices, threshold_fit_indices=fit_indices
                )
                fold_record["inner_folds"].append(
                    {
                        "inner_fold": inner_fold,
                        "fit": _label_count_record(
                            fit_labels,
                            f"{spec.name} outer {outer_fold} inner {inner_fold} fit",
                        ),
                        "validation": _label_count_record(
                            validation_labels,
                            f"{spec.name} outer {outer_fold} inner {inner_fold} validation",
                        ),
                    }
                )
            concept_report["outer_folds"].append(fold_record)

        final_inner_by_game = assign_folds(
            sorted(set(games[development_indices])), inner_folds, local_seed + 9000
        )
        final_inner = []
        for inner_fold in range(inner_folds):
            validation_games = {
                game for game, fold in final_inner_by_game.items() if fold == inner_fold
            }
            validation_mask = np.asarray(
                [game in validation_games for game in games[development_indices]]
            )
            fit_indices = development_indices[~validation_mask]
            validation_indices = development_indices[validation_mask]
            fit_labels, _ = labels_for(
                frame, spec, fit_indices, threshold_fit_indices=fit_indices
            )
            validation_labels, _ = labels_for(
                frame, spec, validation_indices, threshold_fit_indices=fit_indices
            )
            final_inner.append(
                {
                    "inner_fold": inner_fold,
                    "fit": _label_count_record(
                        fit_labels, f"{spec.name} final inner {inner_fold} fit"
                    ),
                    "validation": _label_count_record(
                        validation_labels,
                        f"{spec.name} final inner {inner_fold} validation",
                    ),
                }
            )
        concept_report["final_inner_folds"] = final_inner
        report[spec.name] = concept_report
    return report


def _verify_fresh_development_fidelity_gate(
    run_dir: Path,
    manifest: Mapping[str, Any],
    build_manifest: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """Require the prospectively frozen checkpoint-compatibility gate."""

    fresh = manifest.get("fresh_holdout")
    if not isinstance(fresh, Mapping):
        return None
    protocol_path = Path(str(fresh.get("protocol_path", ""))).resolve()
    if not protocol_path.is_file() or _sha256(protocol_path) != fresh.get(
        "protocol_manifest_sha256"
    ):
        raise ValueError("Fresh holdout protocol is missing or changed before training")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    gate = protocol.get("development_activation_fidelity_gate")
    if not isinstance(gate, Mapping) or gate.get("required_before_training") is not True:
        raise ValueError("Fresh protocol lacks a required development fidelity gate")

    report_path = _require_file(
        run_dir / "checkpoint_activation_fidelity.json",
        "development activation-fidelity report",
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "passed" or report.get("validator") != "checkpoint_activation_fidelity":
        raise ValueError("Development activation-fidelity gate did not pass")
    expected_validator_source = (protocol.get("source_sha256") or {}).get(
        "daniele_experiment/checkpoint_activation_fidelity.py"
    )
    if report.get("validator_source_sha256") != expected_validator_source:
        raise ValueError("Fidelity validator source differs from the frozen protocol")
    report_run = report.get("run") or {}
    if report_run.get("manifest_sha256") != _sha256(run_dir / "manifest.json"):
        raise ValueError("Fidelity report is bound to another run manifest")
    if report_run.get("build_manifest_sha256") != _sha256(
        run_dir / "build_manifest.json"
    ):
        raise ValueError("Fidelity report is bound to another build manifest")
    if (report.get("checkpoint") or {}).get("sha256") != fresh.get(
        "checkpoint_sha256"
    ):
        raise ValueError("Fidelity checkpoint differs from fresh holdout checkpoint")
    sampling = report.get("sampling") or {}
    expected_games = int(gate.get("expected_games", -1))
    if (
        sampling.get("algorithm") != "one_deterministic_position_per_game_v1"
        or sampling.get("split_role_filter") != "development"
        or int(sampling.get("requested_sample_count", -1)) != expected_games
        or int((report.get("aggregate_errors") or {}).get("sample_count", -1))
        != expected_games
    ):
        raise ValueError("Fidelity report does not cover one position per development game")
    tolerance = float((report.get("tolerance") or {}).get("absolute_tolerance", math.inf))
    if tolerance > float(gate.get("absolute_max_error_tolerance", -1)):
        raise ValueError("Fidelity report used a looser tolerance than the frozen protocol")
    splits = pd.read_parquet(run_dir / "splits.parquet")
    development_games = set(
        splits.loc[splits["split_role"].eq("development"), "game_id"].astype(str)
    )
    sampled_games = [str(sample.get("game_id")) for sample in report.get("samples") or ()]
    if (
        len(sampled_games) != expected_games
        or len(set(sampled_games)) != expected_games
        or set(sampled_games) != development_games
    ):
        raise ValueError("Fidelity samples do not cover every development game exactly once")
    if int(build_manifest.get("games", -1)) != len(splits):
        raise ValueError("Fidelity-gated build game count disagrees with frozen splits")
    return {
        "path": str(report_path.relative_to(run_dir)),
        "sha256": _sha256(report_path),
        "checkpoint_sha256": (report.get("checkpoint") or {}).get("sha256"),
        "sample_count": expected_games,
        "sampling_algorithm": sampling.get("algorithm"),
        "absolute_tolerance": tolerance,
        "observed_max_abs_error": (report.get("aggregate_errors") or {}).get(
            "max_abs_error"
        ),
        "claim_scope": report.get("claim_scope"),
    }


def train_run(
    run_dir: Path,
    *,
    concepts: Optional[Sequence[str]] = None,
    representations: Sequence[str] = REPRESENTATIONS,
    C_values: Sequence[float] = DEFAULT_C_VALUES,
    max_iter: int = DEFAULT_MAX_ITER,
) -> Dict[str, Any]:
    """Train nested-CV probes and development-only final artifacts."""
    run_dir = run_dir.resolve()
    manifest = _read_manifest(run_dir)
    _verify_frozen_inputs(run_dir, manifest)
    build_manifest_path = _require_file(run_dir / "build_manifest.json", "build manifest")
    with build_manifest_path.open() as handle:
        build_manifest = json.load(handle)
    dataset_path = _require_file(run_dir / "dataset.parquet", "rebuilt dataset")
    if _sha256(dataset_path) != build_manifest["dataset_sha256"]:
        raise ValueError("dataset.parquet hash does not match build manifest")
    if build_manifest["split_manifest_sha256"] != manifest["artifacts"]["splits_sha256"]:
        raise ValueError("Build and run split hashes disagree")
    if build_manifest.get("source_code_sha256") != _source_hashes():
        raise ValueError("Build manifest source hashes do not match the frozen pipeline")
    if build_manifest.get("contract_implementation_sha256") != _source_hashes()[
        "daniele_experiment/operational_definitions.py"
    ]:
        raise ValueError("Build contract implementation hash does not match source")
    labels_manifest_path = _require_file(run_dir / "labels_manifest.json", "labels manifest")
    if _sha256(labels_manifest_path) != build_manifest.get("labels_manifest_sha256"):
        raise ValueError("Labels manifest changed after feature building")
    _verify_build_input_provenance(run_dir, manifest, build_manifest)
    fidelity_gate = _verify_fresh_development_fidelity_gate(
        run_dir, manifest, build_manifest
    )

    probes_dir = run_dir / "probes"
    results_path = run_dir / "nested_cv_results.parquet"
    predictions_path = run_dir / "outer_predictions"
    summary_path = run_dir / "probe_results.json"
    training_manifest_path = run_dir / "training_manifest.json"
    stale = [
        path for path in (
            probes_dir, results_path, predictions_path, summary_path, training_manifest_path
        ) if path.exists()
    ]
    if stale:
        raise FileExistsError(
            "Refusing to train into a run containing training outputs: "
            + ", ".join(str(path) for path in stale)
        )
    requested_representations = tuple(dict.fromkeys(representations))
    invalid_representations = sorted(set(requested_representations) - set(REPRESENTATIONS))
    if invalid_representations:
        raise ValueError(f"Unknown representations: {', '.join(invalid_representations)}")
    if not requested_representations:
        raise ValueError("At least one representation is required")
    C_values = tuple(sorted(set(float(value) for value in C_values)))
    if not C_values or any(value <= 0 for value in C_values):
        raise ValueError("C values must be positive")
    if int(max_iter) < 1:
        raise ValueError("max_iter must be positive")

    specs = load_concept_specs(run_dir / "frozen_config" / "concepts.yaml", concepts)
    validate_contract_specs(specs)
    frozen_probe_design = _verify_fresh_probe_protocol(
        manifest,
        specs=specs,
        representations=requested_representations,
        C_values=C_values,
        max_iter=max_iter,
    )
    built = set(build_manifest["concepts"])
    not_built = sorted({spec.name for spec in specs} - built)
    if not_built:
        raise ValueError(f"Concepts absent from rebuilt dataset: {', '.join(not_built)}")
    frame = pd.read_parquet(dataset_path)
    nested_config = manifest["nested_cv"]
    outer_folds = int(nested_config["outer_folds"])
    inner_folds = int(nested_config["inner_folds"])
    seed = int(manifest["seed"])
    channels = int(build_manifest["trunk_channels"])

    eligibility = audit_trainability(
        frame,
        specs,
        requested_representations,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        seed=seed,
    )

    probes_dir.mkdir(parents=True)
    predictions_path.mkdir(parents=True)
    result_rows: List[Dict[str, Any]] = []
    summaries: Dict[str, Any] = {}
    artifact_paths: List[Path] = []

    for spec in specs:
        views = feature_views(frame, spec)
        common_mask = np.ones(len(frame), dtype=bool)
        for representation in requested_representations:
            common_mask &= views[representation][1]
        concept_summary: Dict[str, Any] = {}

        for representation in requested_representations:
            X, _ = views[representation]
            # Use identical inner game assignments across representations so
            # ablation differences are not split noise.
            local_seed = _derived_seed(seed, spec.name)
            folds, outer_predictions = nested_group_evaluation(
                frame,
                spec,
                X,
                common_mask,
                outer_folds=outer_folds,
                inner_folds=inner_folds,
                C_values=C_values,
                seed=local_seed,
                max_iter=max_iter,
            )
            scaler, model, final_metadata = _fit_final_probe(
                frame,
                spec,
                X,
                common_mask,
                inner_folds=inner_folds,
                C_values=C_values,
                seed=local_seed,
                max_iter=max_iter,
            )
            representation_dir = probes_dir / representation
            representation_dir.mkdir(exist_ok=True)
            probe_path = representation_dir / f"probe_{spec.name}.joblib"
            scaler_path = representation_dir / f"scaler_{spec.name}.joblib"
            metadata_path = representation_dir / f"probe_{spec.name}.meta.json"
            _dump_new_joblib(probe_path, model)
            _dump_new_joblib(scaler_path, scaler)
            artifact_metadata = {
                "schema_version": SCHEMA_VERSION,
                "pipeline": PIPELINE_NAME,
                "created_at_utc": _utc_now(),
                "concept": asdict(spec),
                "contract_id": spec.contract_id,
                "contract_hash": (
                    get_contract(spec.contract_id).contract_hash
                    if spec.contract_id is not None else None
                ),
                "representation": representation,
                "feature_mode": spec.feature_mode,
                "n_features": int(X.shape[1]),
                "trunk_channels": channels,
                "block_slices": _block_slices(representation, channels),
                "training_role": "development",
                "excluded_roles": ["control_calibration", "causal_test"],
                "dataset_sha256": build_manifest["dataset_sha256"],
                "split_manifest_sha256": manifest["artifacts"]["splits_sha256"],
                "concepts_yaml_sha256": manifest["artifacts"]["concepts_yaml_sha256"],
                "labels_manifest_sha256": build_manifest["labels_manifest_sha256"],
                "input_provenance_sha256": build_manifest["input_provenance_sha256"],
                "trunk_identity_bytes_sha256": build_manifest["input_provenance"][
                    "trunk_identity_bytes_sha256"
                ],
                "source_code_sha256": _source_hashes(),
                "checkpoint_activation_fidelity": fidelity_gate,
                "contract_implementation_sha256": build_manifest[
                    "contract_implementation_sha256"
                ],
                "seed": local_seed,
                "C_values": C_values,
                "max_iter": int(max_iter),
                "selection_metric": "average_precision",
                "threshold_selection": "inner_oof_max_f1",
                "frozen_probe_design": frozen_probe_design,
                "final_fit": final_metadata,
                "outer_metrics": _aggregate_folds(folds),
            }
            _write_new_json(metadata_path, artifact_metadata)
            artifact_paths.extend((probe_path, scaler_path, metadata_path))

            for fold in folds:
                flat = {
                    "concept": spec.name,
                    "representation": representation,
                    **{key: fold[key] for key in (
                        "outer_fold", "n_train", "n_test", "positive_train", "positive_test",
                        "best_C", "f1_threshold", "roc_auc", "average_precision", "f1",
                        "balanced_accuracy", "converged",
                    )},
                }
                result_rows.append(flat)
            prediction_path = predictions_path / f"{spec.name}__{representation}.parquet"
            if prediction_path.exists():
                raise FileExistsError(f"Refusing to overwrite {prediction_path}")
            pd.DataFrame([
                {
                    "concept": spec.name,
                    "representation": representation,
                    **prediction,
                }
                for prediction in outer_predictions
            ]).to_parquet(prediction_path, index=False)
            prediction_path.chmod(0o444)
            artifact_paths.append(prediction_path)
            concept_summary[representation] = {
                "outer_metrics": _aggregate_folds(folds),
                "outer_folds": folds,
                "final_fit": final_metadata,
                "artifact_metadata": str(metadata_path.relative_to(run_dir)),
            }
        summaries[spec.name] = concept_summary

    pd.DataFrame(result_rows).to_parquet(results_path, index=False)
    _write_new_json(summary_path, summaries)
    artifact_paths.extend((results_path, summary_path))
    for path in artifact_paths:
        path.chmod(0o444)
    training_manifest = {
        "schema_version": SCHEMA_VERSION,
        "pipeline": PIPELINE_NAME,
        "created_at_utc": _utc_now(),
        "concepts": [spec.name for spec in specs],
        "representations": list(requested_representations),
        "C_values": list(C_values),
        "max_iter": int(max_iter),
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "selection_metric": "mean inner-fold average precision",
        "threshold_selection": "inner out-of-fold maximum F1",
        "class_weight": "balanced",
        "frozen_probe_design": frozen_probe_design,
        "training_role": "development",
        "dataset_sha256": build_manifest["dataset_sha256"],
        "build_manifest_sha256": _sha256(build_manifest_path),
        "labels_manifest_sha256": build_manifest["labels_manifest_sha256"],
        "input_provenance_sha256": build_manifest["input_provenance_sha256"],
        "trunk_identity_bytes_sha256": build_manifest["input_provenance"][
            "trunk_identity_bytes_sha256"
        ],
        "split_manifest_sha256": manifest["artifacts"]["splits_sha256"],
        "source_code_sha256": _source_hashes(),
        "checkpoint_activation_fidelity": fidelity_gate,
        "contract_implementation_sha256": build_manifest[
            "contract_implementation_sha256"
        ],
        "trainability_audit": eligibility,
        "artifacts": {
            str(path.relative_to(run_dir)): _sha256(path) for path in artifact_paths
        },
    }
    _write_new_json(training_manifest_path, training_manifest)
    return training_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Create an immutable run and game split")
    prepare.add_argument("--run-dir", type=Path, required=True)
    prepare.add_argument("--games-dir", type=Path, default=Path("games"))
    prepare.add_argument(
        "--concepts-yaml", type=Path, default=Path("daniele_experiment/concepts.yaml")
    )
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prepare.add_argument("--development-games", type=int, default=DEFAULT_DEVELOPMENT_GAMES)
    prepare.add_argument(
        "--control-calibration-games", type=int, default=DEFAULT_CONTROL_CALIBRATION_GAMES
    )
    prepare.add_argument("--causal-test-games", type=int, default=DEFAULT_CAUSAL_TEST_GAMES)
    prepare.add_argument("--outer-folds", type=int, default=DEFAULT_OUTER_FOLDS)
    prepare.add_argument("--inner-folds", type=int, default=DEFAULT_INNER_FOLDS)
    prepare.add_argument(
        "--fresh-holdout-cohort",
        help=(
            "Require all non-development games to come from this prospectively "
            "generated metadata cohort"
        ),
    )

    build = subparsers.add_parser("build", help="Build corrected activation features")
    build.add_argument("--run-dir", type=Path, required=True)
    build.add_argument("--concepts", nargs="+")

    train = subparsers.add_parser("train", help="Run nested grouped CV and fit dev probes")
    train.add_argument("--run-dir", type=Path, required=True)
    train.add_argument("--concepts", nargs="+")
    train.add_argument(
        "--representations", nargs="+", choices=REPRESENTATIONS, default=list(REPRESENTATIONS)
    )
    train.add_argument("--C-values", type=float, nargs="+", default=list(DEFAULT_C_VALUES))
    train.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_run(
            args.run_dir,
            args.games_dir,
            args.concepts_yaml,
            seed=args.seed,
            development_games=args.development_games,
            control_calibration_games=args.control_calibration_games,
            causal_test_games=args.causal_test_games,
            outer_folds=args.outer_folds,
            inner_folds=args.inner_folds,
            fresh_holdout_cohort=args.fresh_holdout_cohort,
        )
    elif args.command == "build":
        result = build_run(args.run_dir, concepts=args.concepts)
    else:
        result = train_run(
            args.run_dir,
            concepts=args.concepts,
            representations=args.representations,
            C_values=args.C_values,
            max_iter=args.max_iter,
        )
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
