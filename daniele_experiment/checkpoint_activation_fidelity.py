#!/usr/bin/env python3
"""Empirically compare saved trunk activations with an explicit checkpoint.

This is an optional, run-scoped validation gate for runs prepared by
``validated_probe_pipeline.py``.  It deliberately makes a narrow claim: close
agreement on sampled, replayed positions is evidence that the supplied
checkpoint is empirically compatible with the saved activations.  It is not
proof that this checkpoint originally generated those files.

The replay mirrors ``generate_games_dataset.py``: a fresh ``GameState`` uses
``GameState.RULES_TT``; recorded moves are replayed in order; and
``trunkfinal`` is requested immediately before the recorded move is played.
The checkpoint is always loaded with ``use_swa=False``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1
VALIDATOR_NAME = "checkpoint_activation_fidelity"
RUN_PIPELINE = "validated_probe_pipeline"
RUN_SCHEMA_VERSION = 1
OUTPUT_NAME = "checkpoint_activation_fidelity.json"
PHASES = ("opening", "middle", "endgame")
SPLIT_ROLES = ("development", "control_calibration", "causal_test")
TRUNK_FILENAME = re.compile(r"move_(\d+)\.npy\Z")
CLAIM_SCOPE = (
    "Passing is empirical compatibility evidence for the supplied checkpoint "
    "and the sampled saved activations under this replay implementation. It is "
    "not proof that the checkpoint originally generated the activations, and it "
    "does not establish compatibility for unsampled positions."
)


class FidelityValidationError(RuntimeError):
    """Raised when the fidelity validation cannot be performed safely."""


class FidelityToleranceError(FidelityValidationError):
    """Raised after an immutable failed-tolerance report has been written."""

    def __init__(self, output_path: Path, report: Mapping[str, Any]):
        super().__init__(
            f"Checkpoint activation fidelity exceeded tolerance; report: {output_path}"
        )
        self.output_path = output_path
        self.report = report


@dataclass(frozen=True, order=True)
class PositionCandidate:
    """One saved pre-move activation eligible for sampling."""

    game_id: str
    move_number: int
    split_role: str
    game_phase: str
    board_size: int
    activation_path: Path

    @property
    def stratum(self) -> tuple[str, str]:
        return self.split_role, self.game_phase


@dataclass(frozen=True)
class ErrorComparison:
    """Absolute-error statistics for two same-shaped finite arrays."""

    element_count: int
    max_abs_error: float
    mean_abs_error: float
    rms_error: float
    sum_abs_error: float
    sum_squared_error: float


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Result returned by a checkpoint loader."""

    model: Any
    config: Any
    use_swa: bool = False
    selected_weights: str = "raw_model"


@dataclass(frozen=True)
class RunContext:
    run_dir: Path
    games_dir: Path
    manifest: Mapping[str, Any]
    manifest_path: Path
    splits_path: Path
    splits: pd.DataFrame
    stage: str
    build_manifest_path: Optional[Path]


ModelLoader = Callable[[Path, str, int], LoadedCheckpoint]
StateFactory = Callable[[int], Any]
ActivationEvaluator = Callable[[Any, Any], np.ndarray]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite value cannot be serialized in a fidelity report")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("Non-finite value cannot be serialized in a fidelity report")
        return result
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return str(value)


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(_json_safe(value), sort_keys=True, indent=2, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _write_immutable_json(path: Path, value: Any) -> None:
    """Create, never replace, a canonical read-only JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_canonical_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o444)


def _reject_archive_path(path: Path, description: str) -> None:
    parts = {part.casefold() for part in path.resolve().parts}
    if "archive" in parts:
        raise FidelityValidationError(
            f"Refusing {description} under an archive namespace: {path}"
        )


def _require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FidelityValidationError(f"Missing {description}: {path}")
    return path


def _artifact_inside_run(run_dir: Path, relative: str, description: str) -> Path:
    path = (run_dir / relative).resolve()
    if not path.is_relative_to(run_dir):
        raise FidelityValidationError(
            f"{description} escapes the run directory: {relative!r}"
        )
    _reject_archive_path(path, description)
    return path


def load_run_context(run_dir: Path) -> RunContext:
    """Validate the minimum immutable contract of a prepared or built run."""

    run_dir = run_dir.resolve()
    _reject_archive_path(run_dir, "run input")
    manifest_path = _require_file(run_dir / "manifest.json", "run manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FidelityValidationError(f"Invalid run manifest JSON: {manifest_path}") from exc

    if manifest.get("schema_version") != RUN_SCHEMA_VERSION:
        raise FidelityValidationError(
            f"Unsupported run schema {manifest.get('schema_version')!r}"
        )
    if manifest.get("pipeline") != RUN_PIPELINE:
        raise FidelityValidationError(
            f"Run was not prepared by {RUN_PIPELINE}: {run_dir}"
        )
    status = str(manifest.get("status", "")).casefold()
    if status in {"invalid", "invalid_do_not_use", "archived", "failed"}:
        raise FidelityValidationError(f"Run manifest has rejected status {status!r}")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise FidelityValidationError("Run manifest has no artifact mapping")
    splits_path = _artifact_inside_run(
        run_dir, str(artifacts.get("splits", "splits.parquet")), "split manifest"
    )
    _require_file(splits_path, "split manifest")
    expected_split_hash = artifacts.get("splits_sha256")
    if not expected_split_hash or sha256_file(splits_path) != expected_split_hash:
        raise FidelityValidationError("Split manifest SHA-256 does not match run manifest")
    splits = pd.read_parquet(splits_path)
    required_columns = {"game_id", "split_role"}
    if not required_columns.issubset(splits.columns):
        raise FidelityValidationError(
            f"Split manifest lacks columns {sorted(required_columns - set(splits.columns))}"
        )
    splits = splits.copy()
    splits["game_id"] = splits["game_id"].astype(str)
    splits["split_role"] = splits["split_role"].astype(str)
    if splits["game_id"].duplicated().any():
        raise FidelityValidationError("Split manifest contains duplicate game IDs")
    bad_roles = sorted(set(splits["split_role"]) - set(SPLIT_ROLES))
    if bad_roles:
        raise FidelityValidationError(f"Unknown split roles: {bad_roles}")

    games_dir = Path(str(manifest.get("source_games_dir", ""))).resolve()
    _reject_archive_path(games_dir, "source games input")
    if not games_dir.is_dir():
        raise FidelityValidationError(f"Source games directory is absent: {games_dir}")

    dataset_path = run_dir / "dataset.parquet"
    build_manifest_path = run_dir / "build_manifest.json"
    if dataset_path.exists() != build_manifest_path.exists():
        raise FidelityValidationError(
            "Run has only one of dataset.parquet and build_manifest.json"
        )
    stage = "prepared"
    build_path: Optional[Path] = None
    if dataset_path.is_file():
        build_path = build_manifest_path
        try:
            build_manifest = json.loads(build_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise FidelityValidationError(
                f"Invalid build manifest JSON: {build_manifest_path}"
            ) from exc
        if build_manifest.get("pipeline") != RUN_PIPELINE:
            raise FidelityValidationError("Build manifest belongs to another pipeline")
        expected_dataset_hash = build_manifest.get("dataset_sha256")
        if not expected_dataset_hash or sha256_file(dataset_path) != expected_dataset_hash:
            raise FidelityValidationError("Dataset SHA-256 does not match build manifest")
        expected_build_split = build_manifest.get("split_manifest_sha256")
        if expected_build_split and expected_build_split != expected_split_hash:
            raise FidelityValidationError("Build manifest used a different split manifest")
        stage = "built"
        if (run_dir / "training_manifest.json").is_file():
            stage = "trained"

    return RunContext(
        run_dir=run_dir,
        games_dir=games_dir,
        manifest=manifest,
        manifest_path=manifest_path,
        splits_path=splits_path,
        splits=splits,
        stage=stage,
        build_manifest_path=build_path,
    )


def game_phase(move_number: int) -> str:
    if move_number < 1:
        raise ValueError("move_number must be one-based and positive")
    if move_number <= 60:
        return "opening"
    if move_number <= 150:
        return "middle"
    return "endgame"


def discover_candidates(context: RunContext) -> list[PositionCandidate]:
    """Discover saved activations without parsing the large policy JSON fields."""

    candidates: list[PositionCandidate] = []
    for row in context.splits.sort_values("game_id").itertuples(index=False):
        game_id = str(row.game_id)
        if Path(game_id).name != game_id or game_id in {".", ".."}:
            raise FidelityValidationError(f"Unsafe game identity: {game_id!r}")
        game_dir = (context.games_dir / game_id).resolve()
        if not game_dir.is_relative_to(context.games_dir):
            raise FidelityValidationError(f"Game path escapes source root: {game_id!r}")
        _require_file(game_dir / "moves.jsonl", "recorded moves")
        meta_path = _require_file(game_dir / "meta.json", "game metadata")
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            board_size = int(metadata["board_size"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise FidelityValidationError(
                f"Invalid or missing board_size in {meta_path}"
            ) from exc
        if board_size < 2:
            raise FidelityValidationError(f"Invalid board size {board_size} in {meta_path}")
        trunk_dir = game_dir / "trunkfinal"
        if not trunk_dir.is_dir():
            raise FidelityValidationError(f"Missing trunkfinal directory: {trunk_dir}")
        seen_moves: set[int] = set()
        for activation_path in sorted(trunk_dir.glob("move_*.npy")):
            match = TRUNK_FILENAME.fullmatch(activation_path.name)
            if match is None:
                continue
            move_number = int(match.group(1))
            if move_number < 1 or move_number in seen_moves:
                raise FidelityValidationError(
                    f"Invalid or duplicate activation move number in {trunk_dir}"
                )
            seen_moves.add(move_number)
            candidates.append(
                PositionCandidate(
                    game_id=game_id,
                    move_number=move_number,
                    split_role=str(row.split_role),
                    game_phase=game_phase(move_number),
                    board_size=board_size,
                    activation_path=activation_path.resolve(),
                )
            )
    if not candidates:
        raise FidelityValidationError("No saved trunkfinal activations were discovered")
    return sorted(candidates)


def _derived_seed(seed: int, key: Sequence[str]) -> int:
    payload = json.dumps([int(seed), *map(str, key)], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def deterministic_stratified_sample(
    candidates: Sequence[PositionCandidate], sample_count: int, seed: int
) -> list[PositionCandidate]:
    """Select a reproducible, capacity-aware balanced sample of strata.

    Strata are the Cartesian labels ``split_role`` and ``game_phase`` that are
    actually present.  Selection cycles across nonempty strata, so every
    stratum is represented before a second item is drawn from one (provided the
    requested count is large enough).  Candidate order on disk is irrelevant.
    """

    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if sample_count > len(candidates):
        raise ValueError(
            f"Requested {sample_count} samples from only {len(candidates)} candidates"
        )
    grouped: dict[tuple[str, str], list[PositionCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.stratum, []).append(candidate)
    ordered_keys = sorted(grouped)
    key_rng = np.random.default_rng(_derived_seed(seed, ("strata",)))
    key_order = [ordered_keys[int(index)] for index in key_rng.permutation(len(ordered_keys))]
    shuffled: dict[tuple[str, str], list[PositionCandidate]] = {}
    for key in ordered_keys:
        items = sorted(grouped[key])
        rng = np.random.default_rng(_derived_seed(seed, key))
        shuffled[key] = [items[int(index)] for index in rng.permutation(len(items))]

    selected: list[PositionCandidate] = []
    offsets = {key: 0 for key in ordered_keys}
    while len(selected) < sample_count:
        progressed = False
        for key in key_order:
            offset = offsets[key]
            if offset >= len(shuffled[key]):
                continue
            selected.append(shuffled[key][offset])
            offsets[key] += 1
            progressed = True
            if len(selected) == sample_count:
                break
        if not progressed:  # defensive: the initial capacity check should prevent this
            raise RuntimeError("Stratified sampler exhausted candidates unexpectedly")
    return sorted(selected)


def deterministic_one_per_game_sample(
    candidates: Sequence[PositionCandidate], *, split_role: str, seed: int
) -> list[PositionCandidate]:
    """Select exactly one deterministic saved position from every game in a role."""

    if split_role not in SPLIT_ROLES:
        raise ValueError(f"Unknown split_role {split_role!r}")
    grouped: dict[str, list[PositionCandidate]] = {}
    for candidate in candidates:
        if candidate.split_role == split_role:
            grouped.setdefault(candidate.game_id, []).append(candidate)
    if not grouped:
        raise ValueError(f"No candidates exist for split_role={split_role!r}")
    selected = []
    for game_id in sorted(grouped):
        choices = sorted(grouped[game_id])
        rng = np.random.default_rng(_derived_seed(seed, (split_role, game_id)))
        selected.append(choices[int(rng.integers(0, len(choices)))])
    return sorted(selected)


def compare_activations(saved: np.ndarray, replayed: np.ndarray) -> ErrorComparison:
    """Return exact aggregate absolute-error components for two arrays."""

    saved_array = np.asarray(saved)
    replayed_array = np.asarray(replayed)
    if saved_array.shape != replayed_array.shape:
        raise FidelityValidationError(
            f"Activation shape mismatch: saved={saved_array.shape}, "
            f"replayed={replayed_array.shape}"
        )
    if saved_array.size == 0:
        raise FidelityValidationError("Cannot compare empty activations")
    if not np.isfinite(saved_array).all() or not np.isfinite(replayed_array).all():
        raise FidelityValidationError("Activations contain NaN or infinity")
    difference = np.abs(
        saved_array.astype(np.float64, copy=False)
        - replayed_array.astype(np.float64, copy=False)
    )
    sum_abs = float(np.sum(difference, dtype=np.float64))
    sum_squared = float(np.sum(np.square(difference), dtype=np.float64))
    count = int(difference.size)
    return ErrorComparison(
        element_count=count,
        max_abs_error=float(np.max(difference)),
        mean_abs_error=sum_abs / count,
        rms_error=math.sqrt(sum_squared / count),
        sum_abs_error=sum_abs,
        sum_squared_error=sum_squared,
    )


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    shape = json.dumps(list(contiguous.shape), separators=(",", ":")).encode()
    dtype = contiguous.dtype.str.encode()
    digest.update(len(shape).to_bytes(8, "big"))
    digest.update(shape)
    digest.update(len(dtype).to_bytes(8, "big"))
    digest.update(dtype)
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _default_model_loader(checkpoint: Path, device: str, board_size: int) -> LoadedCheckpoint:
    python_dir = Path(__file__).resolve().parent.parent / "python"
    if str(python_dir) not in sys.path:
        sys.path.insert(0, str(python_dir))
    from load_model import load_model

    model, swa_model, _ = load_model(
        str(checkpoint),
        use_swa=False,
        device=device,
        pos_len=board_size,
        verbose=False,
    )
    if swa_model is not None:
        raise FidelityValidationError(
            "Non-SWA loading unexpectedly returned an SWA model"
        )
    model.eval()
    return LoadedCheckpoint(
        model=model,
        config=model.config,
        use_swa=False,
        selected_weights="raw_model",
    )


def _default_state_factory(board_size: int) -> Any:
    python_dir = Path(__file__).resolve().parent.parent / "python"
    if str(python_dir) not in sys.path:
        sys.path.insert(0, str(python_dir))
    from gamestate import GameState

    return GameState(board_size, GameState.RULES_TT)


def _default_activation_evaluator(state: Any, model: Any) -> np.ndarray:
    outputs = state.get_model_outputs(model, extra_output_names=["trunkfinal"])
    if "trunkfinal" not in outputs:
        raise FidelityValidationError("Checkpoint evaluation did not expose trunkfinal")
    return np.asarray(outputs["trunkfinal"])


def _default_player_values() -> Mapping[str, int]:
    python_dir = Path(__file__).resolve().parent.parent / "python"
    if str(python_dir) not in sys.path:
        sys.path.insert(0, str(python_dir))
    from board import Board

    return {"b": int(Board.BLACK), "w": int(Board.WHITE)}


def _default_rules() -> Mapping[str, Any]:
    python_dir = Path(__file__).resolve().parent.parent / "python"
    if str(python_dir) not in sys.path:
        sys.path.insert(0, str(python_dir))
    from gamestate import GameState

    return dict(GameState.RULES_TT)


def _set_inference_seeds(seed: int) -> None:
    np.random.seed(int(seed) & 0xFFFFFFFF)
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except ImportError:  # pragma: no cover - real KataGo execution requires torch
        pass


def _load_moves_through(path: Path, maximum_move: int) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FidelityValidationError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from exc
            move_number = int(row.get("move_number", -1))
            expected = len(rows) + 1
            if move_number != expected:
                raise FidelityValidationError(
                    f"Non-consecutive move numbering in {path}: expected {expected}, "
                    f"found {move_number}"
                )
            rows.append(row)
            if move_number >= maximum_move:
                break
    if len(rows) < maximum_move:
        raise FidelityValidationError(
            f"Recorded moves end before selected move {maximum_move}: {path}"
        )
    return rows


def _replay_game(
    *,
    context: RunContext,
    selected: Sequence[PositionCandidate],
    loaded: LoadedCheckpoint,
    absolute_tolerance: float,
    state_factory: StateFactory,
    activation_evaluator: ActivationEvaluator,
    player_values: Mapping[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    game_id = selected[0].game_id
    if any(item.game_id != game_id for item in selected):
        raise ValueError("_replay_game received positions from multiple games")
    board_size = selected[0].board_size
    chosen = {item.move_number: item for item in selected}
    maximum_move = max(chosen)
    game_dir = context.games_dir / game_id
    moves_path = _require_file(game_dir / "moves.jsonl", "recorded moves")
    meta_path = _require_file(game_dir / "meta.json", "game metadata")
    rows = _load_moves_through(moves_path, maximum_move)
    state = state_factory(board_size)
    results: list[dict[str, Any]] = []

    for row in rows:
        move_number = int(row["move_number"])
        player_text = str(row.get("player", "")).casefold()
        if player_text not in player_values:
            raise FidelityValidationError(
                f"Unknown recorded player {player_text!r} at {game_id}:{move_number}"
            )
        player = int(player_values[player_text])
        board_player = int(state.board.pla)
        if player != board_player:
            raise FidelityValidationError(
                f"Player mismatch at {game_id}:{move_number}: "
                f"recorded={player_text}, replay_board={board_player}"
            )

        if move_number in chosen:
            candidate = chosen[move_number]
            saved = np.load(candidate.activation_path, allow_pickle=False)
            replayed = np.asarray(activation_evaluator(state, loaded.model))
            comparison = compare_activations(saved, replayed)
            results.append(
                {
                    "game_id": game_id,
                    "move_number": move_number,
                    "split_role": candidate.split_role,
                    "game_phase": candidate.game_phase,
                    "board_size": board_size,
                    "saved_activation_path": str(
                        candidate.activation_path.relative_to(context.games_dir)
                    ),
                    "saved_activation_file_sha256": sha256_file(candidate.activation_path),
                    "saved_activation_dtype": str(saved.dtype),
                    "replayed_activation_dtype": str(replayed.dtype),
                    "activation_shape": list(saved.shape),
                    "replayed_activation_array_sha256": _array_sha256(replayed),
                    "errors": {
                        "element_count": comparison.element_count,
                        "max_abs_error": comparison.max_abs_error,
                        "mean_abs_error": comparison.mean_abs_error,
                        "rms_error": comparison.rms_error,
                    },
                    "within_absolute_tolerance": bool(
                        comparison.max_abs_error <= absolute_tolerance
                    ),
                    "_sum_abs_error": comparison.sum_abs_error,
                    "_sum_squared_error": comparison.sum_squared_error,
                }
            )

        move_loc = int(row["move_loc"])
        try:
            state.play(player, move_loc)
        except Exception as exc:
            raise FidelityValidationError(
                f"Recorded move failed exact replay at {game_id}:{move_number}"
            ) from exc

    if len(results) != len(selected):
        raise FidelityValidationError(
            f"Replayed {len(results)} of {len(selected)} selected positions for {game_id}"
        )
    game_provenance = {
        "game_id": game_id,
        "moves_path": str(moves_path.relative_to(context.games_dir)),
        "moves_sha256": sha256_file(moves_path),
        "meta_path": str(meta_path.relative_to(context.games_dir)),
        "meta_sha256": sha256_file(meta_path),
        "replayed_through_move": maximum_move,
    }
    return results, game_provenance


def _stratum_counts(candidates: Iterable[PositionCandidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in candidates:
        key = f"{item.split_role}/{item.game_phase}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def validate_checkpoint_activation_fidelity(
    run_dir: Path,
    checkpoint: Path,
    *,
    sample_count: int,
    seed: int,
    device: str,
    absolute_tolerance: float,
    model_loader: ModelLoader = _default_model_loader,
    state_factory: StateFactory = _default_state_factory,
    activation_evaluator: ActivationEvaluator = _default_activation_evaluator,
    player_values: Optional[Mapping[str, int]] = None,
    replay_rules: Optional[Mapping[str, Any]] = None,
    output_path: Optional[Path] = None,
    sampling_mode: str = "stratified",
    split_role: Optional[str] = None,
) -> Mapping[str, Any]:
    """Run the immutable empirical checkpoint-compatibility validation."""

    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if not device.strip():
        raise ValueError("device must be explicit and nonempty")
    if not math.isfinite(absolute_tolerance) or absolute_tolerance < 0:
        raise ValueError("absolute_tolerance must be finite and nonnegative")

    context = load_run_context(Path(run_dir))
    checkpoint = Path(checkpoint).resolve()
    _reject_archive_path(checkpoint, "checkpoint input")
    _require_file(checkpoint, "checkpoint")
    destination = (
        context.run_dir / OUTPUT_NAME if output_path is None else Path(output_path).resolve()
    )
    if not destination.is_relative_to(context.run_dir):
        raise FidelityValidationError("Fidelity report must be stored inside the run")
    _reject_archive_path(destination, "fidelity output")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite fidelity report: {destination}")

    candidates = discover_candidates(context)
    if sampling_mode == "stratified":
        if split_role is not None:
            candidates = [
                candidate for candidate in candidates
                if candidate.split_role == split_role
            ]
        selected = deterministic_stratified_sample(candidates, sample_count, seed)
        sampling_algorithm = "balanced_cycle_over_split_role_x_game_phase_v1"
    elif sampling_mode == "one_per_game":
        if split_role is None:
            raise ValueError("one_per_game sampling requires split_role")
        selected = deterministic_one_per_game_sample(
            candidates, split_role=split_role, seed=seed
        )
        if sample_count != len(selected):
            raise ValueError(
                f"one_per_game selected {len(selected)} games, but sample_count="
                f"{sample_count} was requested"
            )
        sampling_algorithm = "one_deterministic_position_per_game_v1"
    else:
        raise ValueError(f"Unknown sampling_mode {sampling_mode!r}")
    board_sizes = sorted({candidate.board_size for candidate in selected})
    if len(board_sizes) != 1:
        raise FidelityValidationError(
            f"A single checkpoint replay requires one board size, selected {board_sizes}"
        )
    board_size = board_sizes[0]

    checkpoint_hash_before = sha256_file(checkpoint)
    checkpoint_bytes = int(checkpoint.stat().st_size)
    _set_inference_seeds(seed)
    loaded = model_loader(checkpoint, device, board_size)
    if loaded.use_swa or loaded.selected_weights != "raw_model":
        raise FidelityValidationError(
            "Validator requires explicit non-SWA raw-model checkpoint weights"
        )
    config = _json_safe(loaded.config)
    player_codec = dict(player_values or _default_player_values())
    rules = dict(replay_rules or _default_rules())

    by_game: dict[str, list[PositionCandidate]] = {}
    for candidate in selected:
        by_game.setdefault(candidate.game_id, []).append(candidate)
    sample_records: list[dict[str, Any]] = []
    replayed_games: list[dict[str, Any]] = []
    for game_id in sorted(by_game):
        records, game_record = _replay_game(
            context=context,
            selected=sorted(by_game[game_id]),
            loaded=loaded,
            absolute_tolerance=absolute_tolerance,
            state_factory=state_factory,
            activation_evaluator=activation_evaluator,
            player_values=player_codec,
        )
        sample_records.extend(records)
        replayed_games.append(game_record)

    checkpoint_hash_after = sha256_file(checkpoint)
    if checkpoint_hash_after != checkpoint_hash_before:
        raise FidelityValidationError("Checkpoint changed during validation")

    element_count = sum(int(item["errors"]["element_count"]) for item in sample_records)
    sum_abs = sum(float(item.pop("_sum_abs_error")) for item in sample_records)
    sum_squared = sum(float(item.pop("_sum_squared_error")) for item in sample_records)
    maximum = max(float(item["errors"]["max_abs_error"]) for item in sample_records)
    passed = maximum <= absolute_tolerance
    aggregate = {
        "sample_count": len(sample_records),
        "element_count": element_count,
        "max_abs_error": maximum,
        "mean_abs_error": sum_abs / element_count,
        "rms_error": math.sqrt(sum_squared / element_count),
        "positions_within_tolerance": sum(
            bool(item["within_absolute_tolerance"]) for item in sample_records
        ),
    }

    generation_source = Path(__file__).resolve().parent / "generate_games_dataset.py"
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "validator": VALIDATOR_NAME,
        "validator_source_sha256": sha256_file(Path(__file__).resolve()),
        "created_at_utc": _utc_now(),
        "status": "passed" if passed else "failed_tolerance",
        "claim_scope": CLAIM_SCOPE,
        "run": {
            "run_dir": str(context.run_dir),
            "stage_at_validation": context.stage,
            "manifest_sha256": sha256_file(context.manifest_path),
            "splits_sha256": sha256_file(context.splits_path),
            "build_manifest_sha256": (
                None
                if context.build_manifest_path is None
                else sha256_file(context.build_manifest_path)
            ),
            "source_games_dir": str(context.games_dir),
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_hash_before,
            "bytes": checkpoint_bytes,
            "model_config": config,
            "model_config_sha256": _canonical_sha256(config),
            "use_swa": False,
            "selected_weights": "raw_model",
            "device": device,
        },
        "sampling": {
            "algorithm": sampling_algorithm,
            "split_role_filter": split_role,
            "seed": int(seed),
            "requested_sample_count": int(sample_count),
            "candidate_count": len(candidates),
            "candidate_stratum_counts": _stratum_counts(candidates),
            "selected_stratum_counts": _stratum_counts(selected),
            "selected_game_count": len(by_game),
        },
        "replay_contract": {
            "saved_activation_semantics": (
                "trunkfinal from the pre-move forward pass for recorded move_number"
            ),
            "state": "GameState(board_size, GameState.RULES_TT)",
            "rules": _json_safe(rules),
            "evaluation": (
                "get_model_outputs(model, extra_output_names=['trunkfinal']) "
                "before applying each selected recorded move"
            ),
            "moves": "recorded player and internal move_loc replayed in order",
            "generation_implementation": str(generation_source),
            "generation_implementation_sha256": (
                sha256_file(generation_source) if generation_source.is_file() else None
            ),
        },
        "tolerance": {
            "criterion": "aggregate max_abs_error <= absolute_tolerance",
            "absolute_tolerance": float(absolute_tolerance),
        },
        "aggregate_errors": aggregate,
        "replayed_games": replayed_games,
        "samples": sorted(
            sample_records, key=lambda item: (item["game_id"], item["move_number"])
        ),
    }
    _write_immutable_json(destination, report)
    if not passed:
        raise FidelityToleranceError(destination, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Empirically validate a supplied checkpoint against a deterministic "
            "stratified sample of saved trunkfinal activations."
        )
    )
    parser.add_argument("run_dir", type=Path, help="Prepared/built validated run")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True, help="Explicit torch device, e.g. cpu")
    parser.add_argument(
        "--sampling-mode",
        choices=("stratified", "one_per_game"),
        default="stratified",
    )
    parser.add_argument("--split-role", choices=SPLIT_ROLES)
    parser.add_argument(
        "--absolute-tolerance",
        type=float,
        required=True,
        help="Maximum permitted absolute elementwise activation error",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        validate_checkpoint_activation_fidelity(
            args.run_dir,
            args.checkpoint,
            sample_count=args.sample_count,
            seed=args.seed,
            device=args.device,
            absolute_tolerance=args.absolute_tolerance,
            sampling_mode=args.sampling_mode,
            split_role=args.split_role,
        )
    except FidelityToleranceError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"Wrote immutable fidelity report: {Path(args.run_dir).resolve() / OUTPUT_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
