#!/usr/bin/env python3
"""Build run-scoped labels with explicit pre/post policy timing.

The raw game format stores the activation for the position *before* each
recorded action and the model outputs for the position *after* it.  Therefore:

* record ``n - 1`` supplies the pre-move policy for action ``n``;
* record ``n`` supplies the opponent's post-move reply policy for action ``n``.

The first action has no persisted pre-move policy and is missing for variables
that require one.  No policy is fabricated or shifted to fill that gap.

Older comprehensive board-analysis labels can be migrated only through an
explicit, field-level whitelist.  The three causal variables are always
recomputed from raw moves and policies and are never copied from the archive.
The destination is staged and atomically installed so a partial build cannot
look like a valid label set.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple
import uuid

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from board import Board  # noqa: E402

from daniele_experiment.operational_definitions import (  # noqa: E402
    CONTRACTS,
    PolicySupport,
    PolicyTime,
    TransitionContext,
    legal_policy_view,
    regional_policy_peak_observed_value,
    regional_policy_readouts,
    reply_peak95_observed_label,
    reply_peak95_observed_value,
    tenuki_observed_label,
    tenuki_observed_value,
)


SCHEMA_VERSION = 1
PIPELINE_NAME = "validated_label_builder"
CENTRAL_SOURCES = frozenset(
    {"tenuki_distance6", "reply_peak95", "regional_policy_peak"}
)
CENTRAL_LEGACY_FIELDS = frozenset(
    {
        "tenuki",
        "tenuki_distance6",
        "tenuki_manhattan_distance",
        "forcing",
        "reply_peak95",
        "reply_peak_value",
        "urgency",
        "urgency_intensity",
        "urgency_peak",
        "regional_policy_peak",
        "regional_policy_masses",
    }
)
BUILTIN_COLUMNS = frozenset(
    {
        "row_id",
        "game_id",
        "move_number",
        "player",
        "move_loc",
        "idx361",
        "game_phase",
        "split_role",
        "outer_fold",
        "has_next",
        "has_local",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _load_jsonl(path: Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _load_jsonl_with_sha256(path: Path) -> Tuple[list[Dict[str, Any]], str, int]:
    """Parse and hash the same raw bytes, avoiding a label/hash race."""

    payload = path.read_bytes()
    rows: list[Dict[str, Any]] = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected object at {path}:{line_number}")
        rows.append(value)
    return rows, hashlib.sha256(payload).hexdigest(), len(payload)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def required_migration_fields(concepts_yaml: Path) -> Tuple[str, ...]:
    """Return the exact non-central legacy analysis fields needed by YAML."""

    with concepts_yaml.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    concepts = config.get("concepts", {})
    fields: set[str] = set()
    enabled_names = {
        str(name) for name, raw in concepts.items() if raw.get("enabled", True)
    }
    for name, raw in concepts.items():
        if not raw.get("enabled", True):
            continue
        source = str(raw["source"])
        if source not in CENTRAL_SOURCES:
            fields.add(source)
        for rule in raw.get("filters") or ():
            column = str(rule["column"])
            # label_<concept> columns are produced by build_run from the
            # corresponding source; they are not fields in snorkel analysis.
            if column.startswith("label_") and column[6:] in enabled_names:
                continue
            if column not in BUILTIN_COLUMNS and column not in CENTRAL_SOURCES:
                fields.add(column)
    overlap = fields & CENTRAL_LEGACY_FIELDS
    if overlap:
        raise ValueError(
            "Legacy migration whitelist overlaps recomputed variables: "
            + ", ".join(sorted(overlap))
        )
    return tuple(sorted(fields))


def _archive_index(archive_dir: Path) -> Tuple[Dict[str, Mapping[str, Any]], Dict[str, Any]]:
    manifest = _load_json(archive_dir / "manifest.json")
    if manifest.get("status") != "invalid_do_not_use":
        raise ValueError(
            "Migration source must be the explicitly quarantined archive with "
            "status='invalid_do_not_use'"
        )
    by_original = {
        str(record["original_path"]): record for record in manifest.get("files", [])
    }
    return by_original, manifest


def _policy(
    source: Any,
    board: Board,
    *,
    time: PolicyTime,
    support: PolicySupport,
):
    return legal_policy_view(
        source,
        board,
        time=time,
        support=support,
        player_to_move=board.pla,
        coordinate_system="board_loc",
    )


def _validate_coordinate(move: Mapping[str, Any], board: Board) -> None:
    move_loc = int(move["move_loc"])
    idx361 = int(move["idx361"])
    if move_loc == Board.PASS_LOC:
        expected = board.size * board.size
    else:
        if not board.is_on_board(move_loc):
            raise ValueError(f"move_loc={move_loc} is not on board")
        expected = board.loc_y(move_loc) * board.size + board.loc_x(move_loc)
    if idx361 != expected:
        raise ValueError(
            f"Coordinate mismatch at move {move.get('move_number')}: "
            f"idx361={idx361}, move_loc implies {expected}"
        )


def rebuild_game_labels(
    moves: Sequence[Mapping[str, Any]],
    legacy_rows: Sequence[Mapping[str, Any]],
    migration_fields: Iterable[str],
    *,
    board_size: int = 19,
) -> Tuple[list[Dict[str, Any]], Dict[str, Any]]:
    """Rebuild one game and return output rows plus audit counts."""

    migration_fields = tuple(sorted(set(migration_fields)))
    legacy_by_move = {int(row["move_number"]): row for row in legacy_rows}
    if len(legacy_by_move) != len(legacy_rows):
        raise ValueError("Legacy labels contain duplicate move numbers")

    board = Board(int(board_size))
    previous_nonpass_move: Optional[int] = None
    previous_post_output: Optional[Mapping[str, Any]] = None
    legacy_alignment_valid = True
    missing_migrated = Counter()
    rows: list[Dict[str, Any]] = []
    counts = Counter()

    for offset, move in enumerate(moves):
        move_number = int(move["move_number"])
        if move_number != offset + 1:
            raise ValueError(
                f"Expected consecutive move number {offset + 1}, found {move_number}"
            )
        expected_player = Board.BLACK if move.get("player") == "b" else Board.WHITE
        if int(board.pla) != int(expected_player):
            raise ValueError(
                f"Player mismatch at move {move_number}: board={board.pla}, "
                f"record={expected_player}"
            )
        _validate_coordinate(move, board)
        selected_move = int(move["move_loc"])
        if not board.would_be_legal(expected_player, selected_move):
            raise ValueError(f"Illegal recorded move at {move_number}: {selected_move}")

        board_before = board.copy()
        pre_policy = None
        if previous_post_output is not None:
            pre_policy = _policy(
                previous_post_output,
                board_before,
                time=PolicyTime.PRE_MOVE,
                support=PolicySupport.LEGAL_BOARD_CONDITIONAL,
            )

        board.play(expected_player, selected_move)
        reply_policy = _policy(
            move,
            board,
            time=PolicyTime.POST_MOVE_REPLY,
            support=PolicySupport.LEGAL_PLUS_PASS,
        )
        context = TransitionContext(
            board_before=board_before,
            player=expected_player,
            previous_move=previous_nonpass_move,
            selected_move=selected_move,
            pre_policy=pre_policy,
            board_after=board.copy(),
            reply_policy=reply_policy,
        )

        analysis: Dict[str, Any] = {}
        legacy_row = legacy_by_move.get(move_number)
        legacy_analysis: Mapping[str, Any] = {}
        # The legacy replay omitted pass moves.  Do not propagate a field for a
        # pass or any later row whose board/player alignment may therefore be
        # stale.  The canonical variables below are rebuilt with correct replay.
        if migration_fields and legacy_alignment_valid and selected_move != Board.PASS_LOC:
            if not isinstance(legacy_row, Mapping):
                raise ValueError(
                    f"Missing archived label row for migration at move {move_number}"
                )
            legacy_player = str(legacy_row.get("player", "")).casefold()
            legacy_move_loc = legacy_row.get("move_loc")
            if (
                legacy_player != str(move.get("player", "")).casefold()
                or legacy_move_loc is None
                or int(legacy_move_loc) != selected_move
            ):
                raise ValueError(
                    f"Archived label alignment mismatch at move {move_number}: "
                    f"raw=({move.get('player')!r}, {selected_move}), "
                    f"archived=({legacy_row.get('player')!r}, {legacy_move_loc!r})"
                )
            legacy_analysis = legacy_row.get("analysis", {})
            if not isinstance(legacy_analysis, Mapping):
                raise ValueError(
                    f"Archived analysis is not an object at move {move_number}"
                )
            for field in migration_fields:
                if field in legacy_analysis:
                    analysis[field] = legacy_analysis[field]
                else:
                    missing_migrated[field] += 1
        elif migration_fields:
            counts["legacy_rows_excluded_for_pass_alignment"] += 1

        tenuki_value = tenuki_observed_value(context)
        tenuki_label = tenuki_observed_label(context)
        forcing_value = reply_peak95_observed_value(context)
        forcing_label = reply_peak95_observed_label(context)
        regional_value = regional_policy_peak_observed_value(context)

        analysis["tenuki_distance6"] = None if tenuki_label is None else bool(tenuki_label)
        analysis["tenuki_manhattan_distance"] = tenuki_value
        analysis["reply_peak95"] = None if forcing_label is None else bool(forcing_label)
        analysis["reply_peak_value"] = forcing_value
        analysis["regional_policy_peak"] = regional_value
        if pre_policy is None:
            analysis["regional_policy_masses"] = None
            counts["missing_pre_policy"] += 1
        else:
            analysis["regional_policy_masses"] = regional_policy_readouts(
                pre_policy, board_before
            )["regional_policy_masses"]

        if tenuki_label is None:
            counts["ineligible_tenuki"] += 1
        if forcing_label is None:
            counts["ineligible_forcing"] += 1
        if regional_value is None:
            counts["ineligible_regional_policy_peak"] += 1
        rows.append(
            {
                "move_number": move_number,
                "player": move.get("player"),
                "move_loc": selected_move,
                "idx361": int(move["idx361"]),
                "analysis": _json_safe(analysis),
            }
        )

        if selected_move != Board.PASS_LOC:
            previous_nonpass_move = selected_move
        previous_post_output = move
        if selected_move == Board.PASS_LOC:
            legacy_alignment_valid = False

    counts["moves"] = len(rows)
    counts["migrated_fields_missing"] = int(sum(missing_migrated.values()))
    return rows, {
        "counts": dict(counts),
        "missing_migrated_by_field": dict(sorted(missing_migrated.items())),
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("xb") as handle:
        for row in rows:
            handle.write(
                (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode(
                    "utf-8"
                )
            )


def build_labels(
    run_dir: Path, legacy_archive: Optional[Path] = None
) -> Dict[str, Any]:
    """Build all labels named by an immutable prepared run."""

    run_dir = run_dir.resolve()
    legacy_archive = None if legacy_archive is None else legacy_archive.resolve()
    run_manifest = _load_json(run_dir / "manifest.json")
    if run_manifest.get("pipeline") != "validated_probe_pipeline":
        raise ValueError("Run was not prepared by validated_probe_pipeline")
    fresh_protocol_revalidation = _verify_fresh_protocol_sources(run_manifest)
    labels_manifest_path = run_dir / "labels_manifest.json"
    labels_dir = run_dir / "labels" / "games"
    if labels_manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite {labels_manifest_path}")
    if not labels_dir.is_dir() or any(labels_dir.iterdir()):
        raise FileExistsError(f"Run label destination is absent or non-empty: {labels_dir}")

    concepts_yaml = run_dir / run_manifest["artifacts"]["concepts_yaml"]
    if _sha256(concepts_yaml) != run_manifest["artifacts"]["concepts_yaml_sha256"]:
        raise ValueError("Frozen concept configuration hash mismatch")
    splits_path = run_dir / run_manifest["artifacts"]["splits"]
    if _sha256(splits_path) != run_manifest["artifacts"]["splits_sha256"]:
        raise ValueError("Frozen split hash mismatch")

    migration_fields = required_migration_fields(concepts_yaml)
    if migration_fields:
        if legacy_archive is None:
            raise ValueError(
                "Legacy-derived concept fields require an explicit quarantined archive"
            )
        archive_by_original, archive_manifest = _archive_index(legacy_archive)
        archive_manifest_sha256 = _sha256(legacy_archive / "manifest.json")
    else:
        archive_by_original, archive_manifest = {}, {}
        archive_manifest_sha256 = None
    games_dir = Path(run_manifest["source_games_dir"])
    splits = pd.read_parquet(splits_path)
    game_ids = sorted(splits["game_id"].astype(str).tolist())
    fresh_record = run_manifest.get("fresh_holdout")
    fresh_game_ids = (
        set(map(str, fresh_record.get("game_ids") or ()))
        if isinstance(fresh_record, Mapping)
        else set()
    )
    staging = run_dir / "labels" / f".building-{uuid.uuid4().hex}"
    staging.mkdir()
    per_game: Dict[str, Any] = {}
    totals = Counter()

    try:
        for game_id in game_ids:
            moves_path = games_dir / game_id / "moves.jsonl"
            original_legacy = f"games/{game_id}/snorkel.jsonl"
            archive_record = archive_by_original.get(original_legacy)
            moves, moves_sha256, moves_bytes = _load_jsonl_with_sha256(moves_path)
            if game_id in fresh_game_ids:
                legacy_rows = []
                observed_legacy_hash = None
                game_migration_fields: Tuple[str, ...] = ()
                migration_status = "not_used_fresh_holdout"
                totals["fresh_holdout_games_without_legacy_migration"] += 1
            elif not migration_fields:
                legacy_rows = []
                observed_legacy_hash = None
                game_migration_fields = ()
                migration_status = "not_used_canonical_only"
                totals["canonical_only_games_without_legacy_migration"] += 1
            else:
                if archive_record is None:
                    raise FileNotFoundError(
                        f"Archive manifest has no explicit source record for {original_legacy}"
                    )
                legacy_path = legacy_archive / str(archive_record["archived_path"])
                observed_legacy_hash = _sha256(legacy_path)
                if observed_legacy_hash != archive_record["sha256"]:
                    raise ValueError(f"Archived source hash mismatch for {game_id}")
                legacy_rows = _load_jsonl(legacy_path)
                game_migration_fields = migration_fields
                migration_status = "hash_and_action_identity_verified"
            rows, audit = rebuild_game_labels(
                moves, legacy_rows, game_migration_fields
            )
            game_dir = staging / game_id
            game_dir.mkdir()
            output_path = game_dir / "snorkel.jsonl"
            _write_jsonl(output_path, rows)
            output_path.chmod(0o444)
            for key, value in audit["counts"].items():
                totals[key] += int(value)
            per_game[game_id] = {
                "moves_sha256": moves_sha256,
                "moves_bytes": moves_bytes,
                "legacy_source_sha256": observed_legacy_hash,
                "legacy_migration_status": migration_status,
                "migrated_fields": list(game_migration_fields),
                "output_sha256": _sha256(output_path),
                **audit,
            }

        labels_dir.rmdir()  # created empty by prepare_run
        staging.rename(labels_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    contract_metadata = {
        definition_id: {
            **contract.metadata(),
            "contract_sha256": contract.contract_hash,
        }
        for definition_id, contract in CONTRACTS.items()
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "pipeline": PIPELINE_NAME,
        "status": "complete",
        "created_at_utc": _utc_now(),
        "builder_source_sha256": _sha256(Path(__file__).resolve()),
        "operational_definitions_source_sha256": _sha256(
            Path(__file__).resolve().parent / "operational_definitions.py"
        ),
        "run_manifest_sha256": _sha256(run_dir / "manifest.json"),
        "split_manifest_sha256": _sha256(splits_path),
        "concepts_yaml_sha256": _sha256(concepts_yaml),
        "policy_timing": {
            "pre_move": "moves[n-1].moves_and_probs0; unavailable for move 1",
            "post_move_reply": "moves[n].moves_and_probs0",
        },
        "policy_normalization": {
            "regional_policy_peak": PolicySupport.LEGAL_BOARD_CONDITIONAL.value,
            "reply_peak95": PolicySupport.LEGAL_PLUS_PASS.value,
        },
        "contracts": contract_metadata,
        "fresh_protocol_revalidation": fresh_protocol_revalidation,
        "recomputed_fields": sorted(CENTRAL_SOURCES),
        "forbidden_legacy_fields": sorted(CENTRAL_LEGACY_FIELDS),
        "migrated_legacy_fields": list(migration_fields),
        "migration_source": {
            "archive_id": archive_manifest.get("archive_id"),
            "archive_manifest_sha256": archive_manifest_sha256,
            "source_status": archive_manifest.get("status"),
            "rule": (
                "No archive is opened when the frozen concept configuration requires "
                "only canonical fields. Otherwise, only whitelisted non-central "
                "board-analysis fields are copied; no legacy probe, causal output, "
                "or central concept field is read."
            ),
            "pass_alignment_rule": (
                "Whitelisted legacy fields are omitted for a pass and all later moves; "
                "canonical fields are rebuilt with correct pass replay."
            ),
            "fresh_holdout_rule": (
                "Games declared in run_manifest.fresh_holdout import zero legacy "
                "fields; only canonical central variables are recomputed from raw records."
            ),
            "fresh_holdout_games": len(fresh_game_ids),
        },
        "totals": dict(sorted(totals.items())),
        "games": per_game,
    }
    with labels_manifest_path.open("xb") as handle:
        handle.write(_canonical_json_bytes(manifest))
    labels_manifest_path.chmod(0o444)
    return manifest


def _verify_fresh_protocol_sources(
    run_manifest: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """Fail closed if any source changed after a prospective protocol freeze."""

    fresh = run_manifest.get("fresh_holdout")
    if not isinstance(fresh, Mapping):
        return None
    protocol_path = Path(str(fresh.get("protocol_path", ""))).resolve()
    expected_protocol_hash = str(fresh.get("protocol_manifest_sha256", ""))
    if (
        not protocol_path.is_file()
        or len(expected_protocol_hash) != 64
        or _sha256(protocol_path) != expected_protocol_hash
    ):
        raise ValueError("Fresh holdout protocol is missing or changed before labels")
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--legacy-archive",
        type=Path,
        help=(
            "Required only when the frozen concept configuration requests "
            "noncanonical legacy-derived fields"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    result = build_labels(args.run_dir, args.legacy_archive)
    print(
        json.dumps(
            {
                "status": result["status"],
                "games": len(result["games"]),
                "totals": result["totals"],
                "migrated_legacy_fields": result["migrated_legacy_fields"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
