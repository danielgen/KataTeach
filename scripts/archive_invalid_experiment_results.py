#!/usr/bin/env python3
"""Quarantine invalid KataTeach experiment artifacts without deleting them.

This script is intentionally conservative: it only moves known derived files,
preserves their repository-relative paths, records SHA-256 checksums, and
refuses to reuse an existing archive directory. Raw games, model checkpoints,
SGFs, move records, metadata, and trunk activations are never selected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REASON_CODES = [
    "MOVE_LOC_PADDED_AS_FLAT",
    "LABEL_READOUT_MISMATCH",
    "NON_NESTED_CV",
    "IN_SAMPLE_CAUSAL",
    "CONTROLS_NOT_POLICY_MATCHED",
]

GAME_DERIVED_FILES = {
    "snorkel.jsonl",
    "viz.html",
    "concepts.json",
    "concepts_meta.json",
    "commentary.json",
    "commentary.jsonl",
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def git_output(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=False
    )
    return completed.stdout.strip()


def is_uuid_directory(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        uuid.UUID(path.name)
    except ValueError:
        return False
    return True


def iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
    elif path.is_dir():
        yield from sorted(candidate for candidate in path.rglob("*") if candidate.is_file())


def collect_targets(repo: Path) -> list[Path]:
    targets: list[Path] = []
    probes = repo / "daniele_experiment" / "linear_probes"
    if probes.exists():
        targets.append(probes)

    targets.extend(sorted(repo.glob("causal_*.json")))

    games = repo / "games"
    global_stats = games / "global_stats.json"
    if global_stats.exists():
        targets.append(global_stats)
    for game_dir in sorted(path for path in games.iterdir() if is_uuid_directory(path)):
        for name in sorted(GAME_DERIVED_FILES):
            candidate = game_dir / name
            if candidate.exists():
                targets.append(candidate)
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-id", default="20260730_pre_idx361_invalid_v1",
        help="New directory name below daniele_experiment/artifacts/archive",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    archive = repo / "daniele_experiment" / "artifacts" / "archive" / args.archive_id
    if archive.exists():
        raise FileExistsError(f"Archive already exists; refusing to merge or overwrite: {archive}")

    targets = collect_targets(repo)
    if not targets:
        print("No known invalid derived artifacts remain in active locations.")
        return 0

    files = [file for target in targets for file in iter_files(target)]
    total_bytes = sum(path.stat().st_size for path in files)
    print(f"Will archive {len(files):,} files ({total_bytes / 2**30:.2f} GiB) to {archive}")
    if args.dry_run:
        for target in targets:
            print(target.relative_to(repo))
        return 0

    archive.mkdir(parents=True, exist_ok=False)
    entries = []
    for index, source in enumerate(files, start=1):
        relative = source.relative_to(repo)
        destination = archive / "files" / relative
        entries.append({
            "original_path": str(relative),
            "archived_path": str(destination.relative_to(archive)),
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
            "reason_codes": REASON_CODES,
        })
        if index % 250 == 0 or index == len(files):
            print(f"Hashed {index:,}/{len(files):,} files")

    # Move top-level targets after every checksum has succeeded. Directory
    # moves preserve all nested relative paths and are cheap on the same disk.
    for source in targets:
        relative = source.relative_to(repo)
        destination = archive / "files" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.move(str(source), str(destination))
        else:
            source.replace(destination)

    status = git_output(repo, "status", "--porcelain=v1")
    diff = git_output(repo, "diff", "--binary", "HEAD")
    checkpoint = repo / "daniele_experiment" / "model.ckpt"
    provenance_files = [
        repo / "daniele_experiment" / "concepts.yaml",
        repo / "daniele_experiment" / "linear_probe_pipeline.py",
        repo / "daniele_experiment" / "generate_games_dataset.py",
        repo / "daniele_experiment" / "snorkel_board_positions.py",
        repo / "daniele_experiment" / "position_causal_eval.py",
        repo / "daniele_experiment" / "activation_manipulation.py",
    ]
    manifest = {
        "schema_version": 1,
        "archive_id": args.archive_id,
        "status": "invalid_do_not_use",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason_codes": REASON_CODES,
        "reason": (
            "Derived results are quarantined because local activations used padded KataGo "
            "locations as flat tensor indices; concept labels and causal readouts were not "
            "aligned; CV was not nested; causal positions overlapped probe fitting; and "
            "controls were not matched for downstream policy disruption."
        ),
        "git": {
            "head": git_output(repo, "rev-parse", "HEAD"),
            "status_porcelain": status.splitlines(),
            "tracked_diff_sha256": sha256_text(diff),
        },
        "checkpoint": {
            "path": str(checkpoint.relative_to(repo)),
            "bytes": checkpoint.stat().st_size if checkpoint.exists() else None,
            "sha256": sha256_file(checkpoint) if checkpoint.exists() else None,
        },
        "source_sha256": {
            str(path.relative_to(repo)): sha256_file(path)
            for path in provenance_files if path.exists()
        },
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "files": entries,
    }
    (archive / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    checksum_lines = [f"{entry['sha256']}  {entry['archived_path']}" for entry in entries]
    (archive / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n")
    (archive / "INVALID.md").write_text(
        "# Invalid experiment artifacts — do not use\n\n"
        "Status: `invalid_do_not_use`\n\n"
        "These files are preserved for auditability only. They must not be used in "
        "analysis, figures, prose, or model selection. See `manifest.json` for exact "
        "provenance, checksums, and failure reasons. Raw games and activations were not "
        "moved and remain the inputs to the corrected rebuild.\n"
    )
    print(f"Archived {len(entries):,} files. Active invalid paths were removed.")
    print(f"Manifest: {archive / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
