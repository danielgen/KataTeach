#!/usr/bin/env python3
"""Run the canonical validity-v5 experiment as resumable, fail-closed stages.

Run this script inside the ``ml`` Conda environment. The default ``canonical``
stage performs the focused validated experiment, ``legacy`` retrains the
historical exploratory concepts, and ``all`` runs both. Existing completed
stages are verified and skipped; partial fresh-game or append-only analysis
stages stop for manual inspection rather than being silently replaced.
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid


REPO = Path(__file__).resolve().parents[1]
GAMES = REPO / "games"
CHECKPOINT = REPO / "daniele_experiment" / "model.ckpt"
CONCEPTS = REPO / "daniele_experiment" / "concepts_validated_v5.yaml"
LEGACY_CONCEPTS = REPO / "daniele_experiment" / "concepts.yaml"
ARTIFACTS = REPO / "daniele_experiment" / "artifacts"
PROTOCOL = ARTIFACTS / "protocols" / "validity_v5.json"
RUN = ARTIFACTS / "runs" / "validity_v5_canonical"
LEGACY_RUN = ARTIFACTS / "runs" / "validity_v5_legacy_exploratory"
CAUSAL = RUN / "causal" / "tenuki_local"
LOGS = ARTIFACTS / "logs" / "validity_v5"
AP_CORRECTION_MANIFEST = (
    Path("corrections")
    / "validated_results_report_apfix_v2"
    / "corrected_results_report_manifest.json"
)
LEGACY_ARCHIVE = (
    ARTIFACTS / "archive" / "20260730_pre_idx361_invalid_v1"
)

COHORT = "validity_v5_postfreeze_holdout"
FIRST_SEED = 202607300000
FRESH_GAMES = 150
CALIBRATION_GAMES = 50
TEST_GAMES = 100
ANALYSIS_SEED = 20260730
CHECKPOINT_TOLERANCE = 0.0001
DOSES = ("-2", "-1", "0", "1", "2")

FOCUSED_TESTS = (
    "daniele_experiment/tests/test_operational_definitions.py",
    "daniele_experiment/tests/test_build_validated_labels.py",
    "daniele_experiment/tests/test_validated_probe_pipeline.py",
    "daniele_experiment/tests/test_validated_results_report.py",
    "daniele_experiment/tests/test_validated_results_report_apfix_v2.py",
    "daniele_experiment/tests/test_validated_causal_results_report.py",
    "daniele_experiment/tests/test_validated_causal_eval.py",
    "daniele_experiment/tests/test_causal_controls.py",
    "daniele_experiment/tests/test_checkpoint_activation_fidelity.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_complete_games() -> list[Path]:
    return sorted(
        path
        for path in GAMES.iterdir()
        if path.is_dir()
        and (path / "moves.jsonl").is_file()
        and (path / "trunkfinal").is_dir()
    )


def _fresh_metadata() -> dict[str, dict]:
    selected: dict[str, dict] = {}
    for game in _raw_complete_games():
        meta = game / "meta.json"
        if not meta.is_file():
            continue
        document = json.loads(meta.read_text(encoding="utf-8"))
        if document.get("cohort") == COHORT:
            selected[game.name] = document
    return selected


def _tail(path: Path, lines: int = 40) -> str:
    if not path.is_file():
        return ""
    with path.open(encoding="utf-8", errors="replace") as handle:
        return "".join(deque(handle, maxlen=lines))


def _run_logged(
    name: str,
    args: list[str],
    *,
    env_overrides: dict[str, str] | None = None,
) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{name}.log"
    print(f"\n[{name}] running; log: {log.relative_to(REPO)}", flush=True)
    environment = os.environ.copy()
    if env_overrides:
        environment.update(env_overrides)
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            args,
            cwd=REPO,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode:
        excerpt = _tail(log)
        raise RuntimeError(
            f"Stage {name!r} failed with exit code {completed.returncode}.\n"
            f"Last log lines:\n{excerpt}"
        )
    print(f"[{name}] complete", flush=True)


def _archive_failed_causal_output() -> None:
    """Move a fail-closed causal attempt out of the validated run namespace."""

    manifest_path = CAUSAL / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Causal output exists without a manifest")
    failed = json.loads(manifest_path.read_text(encoding="utf-8"))
    if failed.get("status") != "failed":
        raise RuntimeError(
            f"Refusing to archive causal output with status={failed.get('status')!r}"
        )
    archive_id = (
        "20260731_validity_v5_tenuki_failed_"
        f"{_sha256(manifest_path)[:12]}"
    )
    archive = ARTIFACTS / "archive" / archive_id
    if archive.exists():
        raise FileExistsError(archive)
    files = sorted(path for path in CAUSAL.rglob("*") if path.is_file())
    entries = [
        {
            "original_path": str(path.relative_to(REPO)),
            "archived_path": str(
                Path("files") / path.relative_to(REPO)
            ),
            "bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
            "reason_codes": ["FAILED_CAUSAL_EQUIVALENCE_GATE"],
        }
        for path in files
    ]
    destination = archive / "files" / CAUSAL.relative_to(REPO)
    destination.parent.mkdir(parents=True)
    shutil.move(str(CAUSAL), str(destination))
    record = {
        "schema_version": 1,
        "archive_id": archive_id,
        "status": "invalid_do_not_use",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason_codes": ["FAILED_CAUSAL_EQUIVALENCE_GATE"],
        "reason": (
            "The fail-closed causal equivalence audit ran without the four CPU "
            "threads used for fresh generation. No causal result was produced."
        ),
        "file_count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "files": entries,
    }
    (archive / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    (archive / "checksums.sha256").write_text(
        "".join(
            f"{entry['sha256']}  {entry['archived_path']}\n"
            for entry in entries
        )
    )
    (archive / "INVALID.md").write_text(
        "# Failed causal attempt — do not use\n\n"
        "Status: `invalid_do_not_use`. No validated causal result was produced.\n"
    )
    print(f"[causal] archived failed attempt to {archive.relative_to(REPO)}")


def _verify_protocol() -> dict:
    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if (
        protocol.get("status") != "frozen_before_fresh_data_generation"
        or (protocol.get("fresh_holdout") or {}).get("cohort") != COHORT
        or (protocol.get("checkpoint") or {}).get("sha256")
        != _sha256(CHECKPOINT)
        or (protocol.get("probes") or {}).get("concepts")
        != ["tenuki", "forcing", "urgency_peak"]
    ):
        raise RuntimeError("Frozen protocol identity does not match this canonical runner")
    sources = protocol.get("source_sha256") or {}
    for identity, expected in sources.items():
        source = (REPO / identity).resolve()
        try:
            source.relative_to(REPO)
        except ValueError as exc:
            raise RuntimeError(f"Unsafe protocol source identity: {identity}") from exc
        if not source.is_file() or _sha256(source) != expected:
            raise RuntimeError(
                f"Source changed after protocol freeze: {identity}. Do not continue."
            )
    return protocol


def preflight(*, run_tests: bool = True) -> None:
    for path in (GAMES, CHECKPOINT, CONCEPTS, LEGACY_CONCEPTS, LEGACY_ARCHIVE):
        if not path.exists():
            raise FileNotFoundError(path)
    if "envs/ml" not in sys.executable:
        raise RuntimeError(
            "Use the ml environment: conda run -n ml python scripts/run_validity_v5.py ..."
        )
    fresh = _fresh_metadata()
    raw_count = len(_raw_complete_games())
    if len(fresh) not in (0, FRESH_GAMES):
        raise RuntimeError(
            f"Found a partial fresh cohort ({len(fresh)}/{FRESH_GAMES}). "
            "Do not refill it; inspect and quarantine it before refreezing."
        )
    expected_raw = 500 + len(fresh)
    if raw_count != expected_raw:
        raise RuntimeError(
            f"Expected {expected_raw} top-level raw-complete games, found {raw_count}"
        )
    if len(fresh) == FRESH_GAMES and not PROTOCOL.is_file():
        raise RuntimeError("Fresh games exist but their frozen protocol is missing")
    if PROTOCOL.exists():
        _verify_protocol()
    free_bytes = shutil.disk_usage(REPO).free
    if not fresh and free_bytes < 50 * 1024**3:
        raise RuntimeError("At least 50 GiB free space is required for fresh generation")
    if run_tests:
        _run_logged(
            "focused_tests",
            [sys.executable, "-m", "pytest", "-q", *FOCUSED_TESTS],
        )
    print(
        f"Preflight passed: {raw_count} raw games, {len(fresh)} fresh holdouts, "
        f"{free_bytes / 1024**3:.1f} GiB free.",
        flush=True,
    )


def freeze() -> None:
    if PROTOCOL.exists():
        _verify_protocol()
        print(f"[freeze] verified existing {PROTOCOL.relative_to(REPO)}")
        return
    if _fresh_metadata():
        raise RuntimeError("Refusing to freeze after fresh cohort generation")
    PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    _run_logged(
        "freeze_protocol",
        [
            sys.executable,
            "scripts/freeze_validity_v5_protocol.py",
            "--output",
            str(PROTOCOL),
            "--games-dir",
            str(GAMES),
            "--checkpoint",
            str(CHECKPOINT),
        ],
    )
    _verify_protocol()


def _verify_complete_fresh_cohort() -> None:
    fresh = _fresh_metadata()
    expected_seeds = set(range(FIRST_SEED, FIRST_SEED + FRESH_GAMES))
    observed_seeds = {
        int((metadata.get("rng") or {}).get("game_seed", -1))
        for metadata in fresh.values()
    }
    expected_ids = {
        str(uuid.uuid5(uuid.NAMESPACE_URL, f"katateach:{COHORT}:{seed}"))
        for seed in expected_seeds
    }
    if (
        len(fresh) != FRESH_GAMES
        or observed_seeds != expected_seeds
        or set(fresh) != expected_ids
    ):
        raise RuntimeError("Fresh cohort does not exactly reproduce frozen seeds/UUIDs")


def generate() -> None:
    _verify_protocol()
    fresh = _fresh_metadata()
    if len(fresh) == FRESH_GAMES:
        _verify_complete_fresh_cohort()
        print("[generate] verified existing complete fresh cohort")
        return
    if fresh:
        raise RuntimeError(
            "Partial fresh cohort detected. Stop: it must be quarantined, not refilled."
        )
    LOGS.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[subprocess.Popen, object, Path]] = []
    try:
        for shard, offset in enumerate((0, 50, 100), start=1):
            log = LOGS / f"generate_shard_{shard}.log"
            handle = log.open("w", encoding="utf-8")
            args = [
                sys.executable,
                "-u",
                "daniele_experiment/generate_games_dataset.py",
                "--model",
                str(CHECKPOINT),
                "--num-games",
                "50",
                "--output-dir",
                str(GAMES),
                "--device",
                "cpu",
                "--seed",
                str(FIRST_SEED + offset),
                "--cohort",
                COHORT,
                "--protocol-manifest",
                str(PROTOCOL),
                "--immutable",
                "--torch-threads",
                "4",
            ]
            print(f"[generate] starting shard {shard}; log: {log.relative_to(REPO)}")
            process = subprocess.Popen(
                args,
                cwd=REPO,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append((process, handle, log))
        failures = []
        for process, handle, log in processes:
            code = process.wait()
            handle.close()
            if code:
                failures.append((code, log))
        if failures:
            details = "\n".join(
                f"{log}: exit={code}\n{_tail(log)}" for code, log in failures
            )
            raise RuntimeError(
                "Fresh generation failed. The partial cohort must be quarantined; "
                f"do not rerun/refill it.\n{details}"
            )
    except KeyboardInterrupt:
        for process, handle, _log in processes:
            if process.poll() is None:
                process.terminate()
            handle.close()
        raise RuntimeError(
            "Generation was interrupted. Treat the partial cohort as invalid; do not resume it."
        ) from None
    _verify_complete_fresh_cohort()
    print("[generate] complete: 150 immutable fresh games")


def probes() -> None:
    _verify_complete_fresh_cohort()
    if not RUN.exists():
        _run_logged(
            "prepare_run",
            [
                sys.executable,
                "-m",
                "daniele_experiment.validated_probe_pipeline",
                "prepare",
                "--run-dir",
                str(RUN),
                "--games-dir",
                str(GAMES),
                "--concepts-yaml",
                str(CONCEPTS),
                "--seed",
                str(ANALYSIS_SEED),
                "--development-games",
                "500",
                "--control-calibration-games",
                str(CALIBRATION_GAMES),
                "--causal-test-games",
                str(TEST_GAMES),
                "--outer-folds",
                "5",
                "--inner-folds",
                "4",
                "--fresh-holdout-cohort",
                COHORT,
            ],
        )
    elif not (RUN / "manifest.json").is_file():
        raise RuntimeError("Run directory exists without a complete manifest")

    if not (RUN / "labels_manifest.json").is_file():
        _run_logged(
            "build_labels",
            [
                sys.executable,
                "-m",
                "daniele_experiment.build_validated_labels",
                "--run-dir",
                str(RUN),
            ],
        )
    if not (RUN / "build_manifest.json").is_file():
        _run_logged(
            "build_features",
            [
                sys.executable,
                "-m",
                "daniele_experiment.validated_probe_pipeline",
                "build",
                "--run-dir",
                str(RUN),
            ],
        )
    if not (RUN / "checkpoint_activation_fidelity.json").is_file():
        _run_logged(
            "checkpoint_fidelity",
            [
                sys.executable,
                "-m",
                "daniele_experiment.checkpoint_activation_fidelity",
                str(RUN),
                "--checkpoint",
                str(CHECKPOINT),
                "--sample-count",
                "500",
                "--seed",
                str(ANALYSIS_SEED),
                "--device",
                "cpu",
                "--sampling-mode",
                "one_per_game",
                "--split-role",
                "development",
                "--absolute-tolerance",
                str(CHECKPOINT_TOLERANCE),
            ],
        )
    if not (RUN / "training_manifest.json").is_file():
        _run_logged(
            "train_probes",
            [
                sys.executable,
                "-m",
                "daniele_experiment.validated_probe_pipeline",
                "train",
                "--run-dir",
                str(RUN),
                "--representations",
                "global",
                "local",
                "combined",
                "--C-values",
                "0.001",
                "0.01",
                "0.1",
                "1.0",
                "10.0",
                "--max-iter",
                "2000",
            ],
        )
    if not (RUN / "validated_results_report_manifest.json").is_file():
        _run_logged(
            "report_probes",
            [
                sys.executable,
                "-m",
                "daniele_experiment.validated_results_report",
                "--run-dir",
                str(RUN),
                "--bootstrap-replicates",
                "2000",
                "--bootstrap-seed",
                str(ANALYSIS_SEED),
                "--confidence-level",
                "0.95",
                "--write",
            ],
        )
    if not (RUN / AP_CORRECTION_MANIFEST).is_file():
        _run_logged(
            "report_probes_apfix_v2",
            [
                sys.executable,
                "-m",
                "daniele_experiment.validated_results_report_apfix_v2",
                "--run-dir",
                str(RUN),
                "--write",
            ],
        )
    print(f"[probes] complete: {RUN.relative_to(REPO)}")


def prepare_legacy() -> None:
    """Freeze the 500-game exploratory run before fresh games are added."""

    if LEGACY_RUN.exists():
        if not (LEGACY_RUN / "manifest.json").is_file():
            raise RuntimeError(
                "Legacy exploratory run directory exists without a complete manifest"
            )
        return
    if _fresh_metadata():
        raise RuntimeError(
            "Prepare the legacy exploratory split before generating fresh games"
        )
    _run_logged(
        "prepare_legacy_exploratory",
        [
            sys.executable,
            "-m",
            "daniele_experiment.validated_probe_pipeline",
            "prepare",
            "--run-dir",
            str(LEGACY_RUN),
            "--games-dir",
            str(GAMES),
            "--concepts-yaml",
            str(LEGACY_CONCEPTS),
            "--seed",
            str(ANALYSIS_SEED),
            "--development-games",
            "500",
            "--control-calibration-games",
            "0",
            "--causal-test-games",
            "0",
            "--outer-folds",
            "5",
            "--inner-folds",
            "4",
        ],
    )


def legacy() -> None:
    """Retrain all historical concepts in a physically separate exploratory run."""

    prepare_legacy()
    if not (LEGACY_RUN / "labels_manifest.json").is_file():
        _run_logged(
            "legacy_build_labels",
            [
                sys.executable,
                "-m",
                "daniele_experiment.build_validated_labels",
                "--run-dir",
                str(LEGACY_RUN),
                "--legacy-archive",
                str(LEGACY_ARCHIVE),
            ],
        )
    if not (LEGACY_RUN / "build_manifest.json").is_file():
        _run_logged(
            "legacy_build_features",
            [
                sys.executable,
                "-m",
                "daniele_experiment.validated_probe_pipeline",
                "build",
                "--run-dir",
                str(LEGACY_RUN),
            ],
        )
    if not (LEGACY_RUN / "training_manifest.json").is_file():
        _run_logged(
            "legacy_train_probes",
            [
                sys.executable,
                "-m",
                "daniele_experiment.validated_probe_pipeline",
                "train",
                "--run-dir",
                str(LEGACY_RUN),
                "--representations",
                "global",
                "local",
                "combined",
                "--C-values",
                "0.001",
                "0.01",
                "0.1",
                "1.0",
                "10.0",
                "--max-iter",
                "2000",
            ],
        )
    if not (LEGACY_RUN / "validated_results_report_manifest.json").is_file():
        _run_logged(
            "legacy_report_probes",
            [
                sys.executable,
                "-m",
                "daniele_experiment.validated_results_report",
                "--run-dir",
                str(LEGACY_RUN),
                "--bootstrap-replicates",
                "2000",
                "--bootstrap-seed",
                str(ANALYSIS_SEED),
                "--confidence-level",
                "0.95",
                "--write",
            ],
        )
    if not (LEGACY_RUN / AP_CORRECTION_MANIFEST).is_file():
        _run_logged(
            "legacy_report_probes_apfix_v2",
            [
                sys.executable,
                "-m",
                "daniele_experiment.validated_results_report_apfix_v2",
                "--run-dir",
                str(LEGACY_RUN),
                "--write",
            ],
        )
    print(
        "[legacy] complete (exploratory only; never merge with canonical tables): "
        f"{LEGACY_RUN.relative_to(REPO)}"
    )


def causal() -> None:
    if not (RUN / "training_manifest.json").is_file():
        raise RuntimeError("Complete canonical probes before causal evaluation")
    if CAUSAL.exists():
        causal_manifest = CAUSAL / "manifest.json"
        if not causal_manifest.is_file():
            raise RuntimeError("Causal output directory exists without a manifest")
        causal_status = json.loads(
            causal_manifest.read_text(encoding="utf-8")
        ).get("status")
        if causal_status == "failed":
            _archive_failed_causal_output()
        elif causal_status != "validated":
            raise RuntimeError(
                f"Causal output has nonterminal status {causal_status!r}"
            )
    if not CAUSAL.exists():
        CAUSAL.parent.mkdir(parents=True, exist_ok=True)
        common = [
            "--run-dir",
            str(RUN),
            "--concept",
            "tenuki",
            "--representation",
            "local",
            "--doses",
            *DOSES,
            "--max-calibration-positions",
            str(CALIBRATION_GAMES),
            "--max-test-positions",
            str(TEST_GAMES),
            "--spatial-shuffles",
            "50",
            "--random-directions",
            "100",
            "--head-batch-size",
            "64",
            "--equivalence-sample-size",
            "6",
            "--seed",
            str(ANALYSIS_SEED),
        ]
        _run_logged(
            "estimate_causal_cost",
            [
                sys.executable,
                "-m",
                "daniele_experiment.validated_causal_eval",
                "estimate",
                *common,
            ],
        )
        _run_logged(
            "evaluate_tenuki",
            [
                sys.executable,
                "-m",
                "daniele_experiment.validated_causal_eval",
                "evaluate",
                "--model",
                str(CHECKPOINT),
                "--output-dir",
                str(CAUSAL),
                "--device",
                "cpu",
                *common,
            ],
            env_overrides={"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"},
        )
    if not (CAUSAL / "validated_causal_results_report_manifest.json").is_file():
        _run_logged(
            "report_causal",
            [
                sys.executable,
                "-m",
                "daniele_experiment.validated_causal_results_report",
                "--causal-dir",
                str(CAUSAL),
                "--bootstrap-replicates",
                "2000",
                "--bootstrap-seed",
                str(ANALYSIS_SEED),
                "--confidence-level",
                "0.95",
                "--write",
            ],
        )
    print(f"[causal] complete: {CAUSAL.relative_to(REPO)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        nargs="?",
        default="canonical",
        choices=(
            "preflight",
            "prepare-legacy",
            "freeze",
            "generate",
            "probes",
            "causal",
            "canonical",
            "legacy",
            "all",
        ),
    )
    args = parser.parse_args()

    os.chdir(REPO)
    if args.stage in ("preflight", "canonical", "all"):
        preflight(run_tests=True)
    else:
        preflight(run_tests=False)
    if args.stage in ("prepare-legacy", "canonical", "all"):
        prepare_legacy()
    if args.stage in ("freeze", "canonical", "all"):
        freeze()
    if args.stage in ("generate", "canonical", "all"):
        if not PROTOCOL.exists():
            freeze()
        generate()
    if args.stage in ("probes", "canonical", "all"):
        probes()
    if args.stage in ("causal", "canonical", "all"):
        causal()
    if args.stage in ("legacy", "all"):
        legacy()
    if args.stage == "canonical":
        print(
            "\nCanonical analysis is complete. To run the slower, separately labelled "
            "historical concept probes later:\n"
            "  conda run -n ml python scripts/run_validity_v5.py legacy"
        )
    print("\nRequested validity-v5 stages completed successfully.")


if __name__ == "__main__":
    main()
