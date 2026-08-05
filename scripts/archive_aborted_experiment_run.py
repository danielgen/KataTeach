#!/usr/bin/env python3
"""Move one explicitly named incomplete experiment run into quarantine."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--archive-id", required=True)
    parser.add_argument("--reason-code", required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    runs_root = (repo / "daniele_experiment" / "artifacts" / "runs").resolve()
    run_dir = args.run_dir.resolve()
    if run_dir.parent != runs_root or not run_dir.name.endswith("_incomplete"):
        raise ValueError(
            "Refusing broad target: run must be a direct child of artifacts/runs "
            "whose name ends in _incomplete"
        )
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    archive = repo / "daniele_experiment" / "artifacts" / "archive" / args.archive_id
    if archive.exists():
        raise FileExistsError(archive)

    files = sorted(path for path in run_dir.rglob("*") if path.is_file())
    entries = [
        {
            "original_path": str(path.relative_to(repo)),
            "archived_path": str(
                Path("files") / path.relative_to(repo)
            ),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "reason_codes": [args.reason_code],
        }
        for path in files
    ]
    archive.mkdir(parents=True)
    destination = archive / "files" / run_dir.relative_to(repo)
    destination.parent.mkdir(parents=True)
    shutil.move(str(run_dir), str(destination))

    manifest = {
        "schema_version": 1,
        "archive_id": args.archive_id,
        "status": "invalid_do_not_use",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason_codes": [args.reason_code],
        "reason": args.reason,
        "file_count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "files": entries,
    }
    (archive / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (archive / "checksums.sha256").write_text(
        "".join(
            f"{entry['sha256']}  {entry['archived_path']}\n" for entry in entries
        )
    )
    (archive / "INVALID.md").write_text(
        "# Aborted experiment run — do not use\n\n"
        "Status: `invalid_do_not_use`\n\n"
        f"Reason: {args.reason}\n"
    )
    print(
        json.dumps(
            {
                "archive": str(archive),
                "files": len(entries),
                "bytes": manifest["total_bytes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
