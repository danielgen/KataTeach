#!/usr/bin/env python3
"""Quarantine the remaining derived legacy global statistics file."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


ARCHIVE_ID = "20260730_nested_old_gs_global_stats_invalid_v1"
SOURCE_RELATIVE = Path("games/old gs/global_stats.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    source = (repo / SOURCE_RELATIVE).resolve()
    expected = (repo / SOURCE_RELATIVE).resolve()
    if source != expected or not source.is_file():
        raise FileNotFoundError(f"Fixed legacy result is absent: {source}")
    archive = (
        repo / "daniele_experiment" / "artifacts" / "archive" / ARCHIVE_ID
    )
    if archive.exists():
        raise FileExistsError(f"Refusing to reuse archive: {archive}")
    digest = _sha256(source)
    archived_relative = Path("files") / SOURCE_RELATIVE
    destination = archive / archived_relative
    destination.parent.mkdir(parents=True)
    shutil.move(str(source), str(destination))
    entry = {
        "original_path": str(SOURCE_RELATIVE),
        "archived_path": str(archived_relative),
        "bytes": int(destination.stat().st_size),
        "sha256": digest,
        "reason_codes": ["DERIVED_FROM_INVALID_LEGACY_LABELS"],
    }
    manifest = {
        "schema_version": 1,
        "archive_id": ARCHIVE_ID,
        "status": "invalid_do_not_use",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": (
            "Global statistics derived from quarantined legacy labels are preserved "
            "for audit only and excluded from validated analysis."
        ),
        "raw_files_moved": False,
        "file_count": 1,
        "total_bytes": entry["bytes"],
        "files": [entry],
    }
    (archive / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (archive / "checksums.sha256").write_text(
        f"{digest}  {archived_relative}\n"
    )
    (archive / "INVALID.md").write_text(
        "# Invalid nested legacy global statistics — do not use\n\n"
        "Status: `invalid_do_not_use`. Raw game data was not moved.\n"
    )
    print(json.dumps({"archive": str(archive), **entry}, indent=2))


if __name__ == "__main__":
    main()
