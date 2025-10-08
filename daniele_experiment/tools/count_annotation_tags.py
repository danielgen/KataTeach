#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Union


def iter_files(root: Path, patterns: List[str]) -> Iterable[Path]:
    for pat in patterns:
        yield from root.rglob(pat)


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def read_json_any(path: Path) -> Iterable[dict]:
    # Yields dict samples regardless of top structure
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                yield item
    elif isinstance(obj, dict):
        # If looks like a container of records
        # try common keys: data, items, annotations
        for key in ("data", "items", "annotations", "records"):
            val = obj.get(key)
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        yield item
                return
        # Otherwise treat top-level dict as a single record
        yield obj


def extract_tags(record: dict, *, key_hints: Optional[List[str]] = None) -> Set[str]:
    tags: Set[str] = set()
    # Prefer hinted keys if provided
    keys_to_check = list(key_hints or [])
    # Heuristic keys
    keys_to_check += [
        "tags",
        "labels",
        "concepts",
        "concept_labels_names",
        "concept_names",
        "annotation_tags",
    ]

    for k in keys_to_check:
        if k in record:
            val = record[k]
            if isinstance(val, list):
                for t in val:
                    if isinstance(t, str) and t.strip():
                        tags.add(t.strip())

    # Fallback: scan shallow keys for any list of strings where key contains tag/label/concept
    for k, v in record.items():
        if not isinstance(v, list):
            continue
        lk = str(k).lower()
        if ("tag" in lk or "label" in lk or "concept" in lk):
            for t in v:
                if isinstance(t, str) and t.strip():
                    tags.add(t.strip())

    return tags


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count per-tag sample occurrences across annotation files (JSON/JSONL)."
    )
    parser.add_argument("annotations_dir", type=Path, help="Directory containing annotations")
    parser.add_argument(
        "--patterns",
        nargs="*",
        default=["*.jsonl", "*.json"],
        help="Glob patterns to include (default: *.jsonl *.json)",
    )
    parser.add_argument(
        "--keys",
        nargs="*",
        default=[],
        help="Optional keys to search for tags (e.g., --keys tags labels)",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Only show tags with at least this many samples (default: 1)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=0,
        help="Show only top-K tags by count (0 = all)",
    )
    args = parser.parse_args()

    root = args.annotations_dir
    if not root.exists():
        raise SystemExit(f"Annotations directory not found: {root}")

    per_tag_counts: Counter[str] = Counter()
    num_samples = 0

    for path in iter_files(root, args.patterns):
        if path.suffix.lower() == ".jsonl":
            iterator = read_jsonl(path)
        else:
            iterator = read_json_any(path)

        for rec in iterator:
            if not isinstance(rec, dict):
                continue
            num_samples += 1
            tags = extract_tags(rec, key_hints=args.keys)
            # Count a tag once per sample
            for t in tags:
                per_tag_counts[t] += 1

    # Filter and sort
    items = [(t, c) for t, c in per_tag_counts.items() if c >= args.min_count]
    items.sort(key=lambda x: (-x[1], x[0]))
    if args.topk > 0:
        items = items[: args.topk]

    print(f"Scanned samples: {num_samples}")
    print("tag,count")
    for tag, cnt in items:
        print(f"{tag},{cnt}")


if __name__ == "__main__":
    main()



