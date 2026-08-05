#!/usr/bin/env python3
"""
Analyze SGF games with KataGo and export slim JSON for the study app.

Keeps candidate moves whose winrate drop from the best move is <= 5%.
Requires a local KataGo binary, analysis config, and network weights.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple, Union

Color = str  # "b" | "w"
Move = Union[None, str, Tuple[int, int]]  # None/pass or (row, col) sgfmill


def _require_sgfmill():
    try:
        import sgfmill.sgf as sgf_mod
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "sgfmill is required. Install with: pip install sgfmill"
        ) from exc
    return sgf_mod

MAX_DROP_PCT = 5.0
GTP_COLS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"


def sgfmill_to_gtp(move: Move, board_size: int) -> str:
    if move is None or move == "pass":
        return "pass"
    row, col = move
    return GTP_COLS[col] + str(row + 1)


def gtp_to_sgf_coords(gtp: str, board_size: int) -> Optional[Tuple[int, int]]:
    """Return (row, col) in sgfmill coords, or None for pass."""
    if gtp.lower() == "pass":
        return None
    col = GTP_COLS.index(gtp[0].upper())
    row = int(gtp[1:]) - 1
    if not (0 <= row < board_size and 0 <= col < board_size):
        raise ValueError(f"Invalid GTP coord {gtp} for size {board_size}")
    return row, col


class KataGo:
    def __init__(
        self,
        katago_path: str,
        config_path: str,
        model_path: str,
        additional_args: Optional[List[str]] = None,
    ):
        self.query_counter = 0
        cmd = [
            katago_path,
            "analysis",
            "-config",
            config_path,
            "-model",
            model_path,
            *(additional_args or []),
        ]
        self.katago = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        def print_stderr() -> None:
            while self.katago.poll() is None:
                data = self.katago.stderr.readline()
                time.sleep(0)
                if data:
                    print("KataGo:", data.decode(), end="", file=sys.stderr)
            data = self.katago.stderr.read()
            if data:
                print("KataGo:", data.decode(), end="", file=sys.stderr)

        self._stderr_thread = Thread(target=print_stderr, daemon=True)
        self._stderr_thread.start()

    def close(self) -> None:
        if self.katago.stdin:
            self.katago.stdin.close()
        try:
            self.katago.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.katago.kill()

    def query_position(
        self,
        board_size: int,
        initial_stones: List[Tuple[Color, str]],
        moves: List[Tuple[Color, str]],
        komi: float,
        rules: str,
        max_visits: int,
    ) -> Dict[str, Any]:
        query: Dict[str, Any] = {
            "id": str(self.query_counter),
            "moves": [[c.upper(), m] for c, m in moves],
            "initialStones": [[c.upper(), m] for c, m in initial_stones],
            "rules": rules,
            "komi": komi,
            "boardXSize": board_size,
            "boardYSize": board_size,
            "maxVisits": max_visits,
            "includePolicy": False,
            "overrideSettings": {"reportAnalysisWinratesAs": "SIDETOMOVE"},
        }
        self.query_counter += 1
        assert self.katago.stdin and self.katago.stdout
        self.katago.stdin.write((json.dumps(query) + "\n").encode())
        self.katago.stdin.flush()

        line = ""
        while line == "":
            if self.katago.poll() is not None:
                time.sleep(0.2)
                raise RuntimeError("Unexpected KataGo exit")
            line = self.katago.stdout.readline().decode().strip()
        return json.loads(line)


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip()).strip("_").lower()
    return slug or "game"


def read_sgf_meta(game: Any) -> Dict[str, Any]:
    root = game.get_root()

    def prop(key: str, default: str = "") -> str:
        if root.has_property(key):
            val = root.get(key)
            if isinstance(val, bytes):
                return val.decode("utf-8", errors="replace")
            return str(val)
        return default

    size = game.get_size()
    komi = 7.5
    if root.has_property("KM"):
        try:
            komi = float(root.get("KM"))
        except (TypeError, ValueError):
            pass

    rules_raw = prop("RU", "tromp-taylor").lower()
    if "chinese" in rules_raw:
        rules = "chinese"
    elif "japanese" in rules_raw:
        rules = "japanese"
    else:
        rules = "tromp-taylor"

    black = prop("PB", "Black")
    white = prop("PW", "White")
    date = prop("DT", "")
    result = prop("RE", "")
    name = prop("GN", "") or f"{black} vs {white}"

    return {
        "boardSize": size,
        "komi": komi,
        "rules": rules,
        "black": black,
        "white": white,
        "date": date,
        "result": result,
        "name": name,
    }


def extract_setup_and_moves(
    game: Any,
) -> Tuple[List[Tuple[Color, Move]], List[Tuple[Color, Move]]]:
    """Return (initial_stones, mainline_moves) in sgfmill coords."""
    root = game.get_root()
    initial: List[Tuple[Color, Move]] = []
    for color, key in (("b", "AB"), ("w", "AW")):
        if root.has_property(key):
            for row, col in root.get(key):
                initial.append((color, (row, col)))

    moves: List[Tuple[Color, Move]] = []
    for node in game.get_main_sequence()[1:]:
        color, move = node.get_move()
        if color is None:
            continue
        moves.append((color, move))
    return initial, moves


def good_moves_from_response(response: Dict[str, Any]) -> Tuple[float, List[Dict[str, Any]]]:
    root_info = response.get("rootInfo") or {}
    root_wr = float(root_info.get("winrate", 0.5))
    move_infos = response.get("moveInfos") or []
    if not move_infos:
        return root_wr, []

    best_wr = max(float(m["winrate"]) for m in move_infos)
    good: List[Dict[str, Any]] = []
    for info in move_infos:
        wr = float(info["winrate"])
        drop_pct = round((best_wr - wr) * 100.0, 3)
        if drop_pct <= MAX_DROP_PCT + 1e-9:
            good.append(
                {
                    "move": info["move"],
                    "winrate": round(wr, 5),
                    "dropPct": drop_pct,
                    "visits": int(info.get("visits", 0)),
                    "order": int(info.get("order", 0)),
                }
            )
    good.sort(key=lambda m: (m["order"], m["dropPct"]))
    return root_wr, good


def mainline_drop(
    good_moves: List[Dict[str, Any]], mainline_gtp: str, best_wr: float, played_wr: Optional[float]
) -> float:
    for m in good_moves:
        if m["move"].upper() == mainline_gtp.upper():
            return float(m["dropPct"])
    if played_wr is None:
        return MAX_DROP_PCT
    return min(MAX_DROP_PCT, round((best_wr - played_wr) * 100.0, 3))


def analyze_game(
    sgf_path: Path,
    katago: Optional[KataGo],
    visits: int,
    game_id: str,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    sgf_mod = _require_sgfmill()
    raw = sgf_path.read_text(encoding="utf-8", errors="replace")
    game = sgf_mod.Sgf_game.from_string(raw)
    meta = read_sgf_meta(game)
    meta["tags"] = tags or ["interactive"]
    board_size = int(meta["boardSize"])
    initial, moves = extract_setup_and_moves(game)

    initial_gtp = [(c, sgfmill_to_gtp(m, board_size)) for c, m in initial]
    move_gtp = [(c, sgfmill_to_gtp(m, board_size)) for c, m in moves]

    positions: List[Dict[str, Any]] = []
    for turn in range(len(move_gtp) + 1):
        to_play = "B" if turn % 2 == 0 else "W"
        # Handicap / setup may leave Black to play still; approximate by parity of moves played.
        if turn < len(move_gtp):
            to_play = move_gtp[turn][0].upper()
        elif move_gtp:
            to_play = "W" if move_gtp[-1][0].lower() == "b" else "B"

        mainline = move_gtp[turn][1] if turn < len(move_gtp) else None
        prefix = move_gtp[:turn]

        if katago is None:
            # Synthetic placeholder for dry-run / tests without KataGo.
            root_wr = 0.5
            good = [
                {
                    "move": mainline or "pass",
                    "winrate": 0.5,
                    "dropPct": 0.0,
                    "visits": visits,
                    "order": 0,
                }
            ]
            if mainline and mainline != "pass":
                # Add a second nearby dummy candidate with small drop.
                good.append(
                    {
                        "move": "pass",
                        "winrate": 0.48,
                        "dropPct": 2.0,
                        "visits": max(1, visits // 4),
                        "order": 1,
                    }
                )
            mainline_drop_pct = 0.0 if mainline else None
        else:
            response = katago.query_position(
                board_size=board_size,
                initial_stones=initial_gtp,
                moves=prefix,
                komi=float(meta["komi"]),
                rules=str(meta["rules"]),
                max_visits=visits,
            )
            root_wr, good = good_moves_from_response(response)
            played_wr = None
            for info in response.get("moveInfos") or []:
                if mainline and info.get("move", "").upper() == mainline.upper():
                    played_wr = float(info["winrate"])
                    break
            best_wr = max((m["winrate"] for m in good), default=root_wr)
            mainline_drop_pct = (
                mainline_drop(good, mainline, best_wr, played_wr) if mainline else None
            )

        pos: Dict[str, Any] = {
            "turn": turn,
            "toPlay": to_play,
            "rootWinrate": round(float(root_wr), 5),
            "goodMoves": good,
            "mainline": mainline,
        }
        if mainline_drop_pct is not None:
            pos["mainlineDropPct"] = mainline_drop_pct
        positions.append(pos)

    return {
        "id": game_id,
        "sgf": raw.strip(),
        "meta": {
            "black": meta["black"],
            "white": meta["white"],
            "komi": meta["komi"],
            "result": meta["result"],
            "date": meta["date"],
            "name": meta["name"],
            "boardSize": board_size,
            "rules": meta["rules"],
            "tags": meta.get("tags") or ["interactive"],
        },
        "positions": positions,
    }


def normalize_tags(tags: Optional[List[str]], mode: str) -> List[str]:
    """Ensure exactly one of interactive / non-interactive is present."""
    mode = (mode or "interactive").strip().lower().replace("_", "-")
    if mode in ("non-interactive", "noninteractive", "review", "passive"):
        primary = "non-interactive"
    else:
        primary = "interactive"
    extras = []
    for t in tags or []:
        t = str(t).strip().lower().replace("_", "-")
        if t and t not in ("interactive", "non-interactive", "noninteractive", "review"):
            extras.append(t)
    return [primary, *extras]


def manifest_entry(game_json: Dict[str, Any]) -> Dict[str, Any]:
    meta = game_json["meta"]
    tags = meta.get("tags") or ["interactive"]
    return {
        "id": game_json["id"],
        "name": meta.get("name") or f"{meta['black']} vs {meta['white']}",
        "black": meta["black"],
        "white": meta["white"],
        "date": meta.get("date") or "",
        "result": meta.get("result") or "",
        "boardSize": meta["boardSize"],
        "numMoves": max(0, len(game_json["positions"]) - 1),
        "path": f"games/{game_json['id']}.json",
        "tags": tags,
    }


def default_tags_for_stem(stem: str, *, has_analysis: bool) -> List[str]:
    """Analyzed JSON → interactive study; SGF-only → non-interactive review."""
    if has_analysis:
        return ["interactive"]
    return ["non-interactive"]


def resolve_game_id(sgf_path: Path, games_dir: Path) -> str:
    """Map an SGF to its analysis id (slug). Prefer an existing JSON with that slug."""
    return slugify(sgf_path.stem)


def _sgf_prop(sgf_text: str, key: str, default: str = "") -> str:
    m = re.search(rf"{key}\[([^\]]*)\]", sgf_text)
    return m.group(1) if m else default


def _count_sgf_moves(sgf_text: str) -> int:
    return len(re.findall(r";[BW]\[[^\]]*\]", sgf_text, flags=re.IGNORECASE))


def peek_sgf_meta(sgf_path: Path) -> Dict[str, Any]:
    """Lightweight SGF root peek (no sgfmill) for manifest rows."""
    raw = sgf_path.read_text(encoding="utf-8", errors="replace")
    size_s = _sgf_prop(raw, "SZ", "19")
    try:
        board_size = int(size_s)
    except ValueError:
        board_size = 19
    black = _sgf_prop(raw, "PB", "Black")
    white = _sgf_prop(raw, "PW", "White")
    name = _sgf_prop(raw, "GN", "") or f"{black} vs {white}"
    return {
        "black": black,
        "white": white,
        "date": _sgf_prop(raw, "DT", ""),
        "result": _sgf_prop(raw, "RE", ""),
        "name": name,
        "boardSize": board_size,
        "numMoves": _count_sgf_moves(raw),
    }


def sync_manifest_from_sgfs(sgf_dir: Path, out_dir: Path) -> List[Dict[str, Any]]:
    """
    Rebuild manifest.json from every SGF in sgf_dir.

    - With analysis JSON: can be interactive (L*/M* by default) or as tagged in JSON.
    - Without analysis JSON: always non-interactive; app loads the .sgf for review.
    """
    games_dir = out_dir / "games"
    manifest_path = out_dir / "manifest.json"
    entries: List[Dict[str, Any]] = []

    sgf_files = sorted(sgf_dir.glob("*.sgf"))
    for sgf_path in sgf_files:
        game_id = resolve_game_id(sgf_path, games_dir)
        json_path = games_dir / f"{game_id}.json"
        sgf_rel = f"games/{sgf_path.name}"

        if json_path.exists():
            try:
                game_json = json.loads(json_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                print(f"Warning: skip broken {json_path.name}: {exc}", file=sys.stderr)
                game_json = None
            if game_json is not None:
                meta = game_json.setdefault("meta", {})
                # Analyzed games are always interactive for the study app.
                meta["tags"] = default_tags_for_stem(sgf_path.stem, has_analysis=True)
                json_path.write_text(
                    json.dumps(game_json, indent=2) + "\n", encoding="utf-8"
                )
                game_json["id"] = game_id
                entry = manifest_entry(game_json)
                # Prefer JSON for interactive study; keep sgfPath for reference.
                entry["sgfPath"] = sgf_rel
                entries.append(entry)
                continue

        # SGF-only review entry (no KataGo analysis required).
        peek = peek_sgf_meta(sgf_path)
        entries.append(
            {
                "id": game_id,
                "name": peek["name"],
                "black": peek["black"],
                "white": peek["white"],
                "date": peek["date"],
                "result": peek["result"],
                "boardSize": peek["boardSize"],
                "numMoves": peek["numMoves"],
                "path": sgf_rel,
                "tags": default_tags_for_stem(sgf_path.stem, has_analysis=False),
            }
        )

    manifest_path.write_text(
        json.dumps({"games": entries}, indent=2) + "\n", encoding="utf-8"
    )
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze SGFs for the KataTeach study app (good moves within 5% WR)."
    )
    parser.add_argument("--sgf-dir", type=Path, required=True, help="Directory of .sgf files")
    parser.add_argument("--out", type=Path, required=True, help="Output dir (study_app/data)")
    parser.add_argument("--katago", type=str, default="", help="Path to katago binary")
    parser.add_argument("--config", type=str, default="", help="Path to analysis config")
    parser.add_argument("--model", type=str, default="", help="Path to .bin.gz model")
    parser.add_argument("--visits", type=int, default=75, help="maxVisits (50-100 recommended)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Export structure with synthetic goodMoves (no KataGo)",
    )
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="Only rebuild manifest.json from all SGFs (JSON if analyzed, else SGF review)",
    )
    parser.add_argument(
        "--merge-manifest",
        action="store_true",
        help="(Deprecated) Manifest is always rebuilt from --sgf-dir + existing JSON",
    )
    parser.add_argument(
        "--mode",
        choices=("interactive", "non-interactive"),
        default=None,
        help="Tag for newly analyzed games. Default: L*/M* interactive, others non-interactive",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Extra tag(s) to attach (repeatable). Mode is set via --mode.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip SGFs that already have a matching games/<id>.json",
    )
    args = parser.parse_args()

    sgf_files = sorted(args.sgf_dir.glob("*.sgf"))
    if not sgf_files:
        raise SystemExit(f"No .sgf files found in {args.sgf_dir}")

    games_dir = args.out / "games"
    games_dir.mkdir(parents=True, exist_ok=True)

    if args.sync_only:
        entries = sync_manifest_from_sgfs(args.sgf_dir, args.out)
        with_json = sum(1 for e in entries if str(e.get("path", "")).endswith(".json"))
        print(
            f"Synced manifest: {len(entries)} SGFs "
            f"({with_json} with analysis, {len(entries) - with_json} review-only) "
            f"-> {args.out / 'manifest.json'}",
            file=sys.stderr,
        )
        return

    if args.visits < 1:
        raise SystemExit("--visits must be >= 1")

    katago: Optional[KataGo] = None
    if not args.dry_run:
        if not (args.katago and args.config and args.model):
            raise SystemExit("Provide --katago, --config, and --model (or use --dry-run / --sync-only)")
        katago = KataGo(args.katago, args.config, args.model)

    used_ids: set[str] = set()
    wrote = 0

    try:
        for sgf_path in sgf_files:
            game_id = resolve_game_id(sgf_path, games_dir)
            if game_id in used_ids:
                n = 2
                while f"{game_id}_{n}" in used_ids:
                    n += 1
                game_id = f"{game_id}_{n}"
            used_ids.add(game_id)

            out_path = games_dir / f"{game_id}.json"
            if args.skip_existing and out_path.exists():
                print(f"Skipping {sgf_path.name} (exists: {out_path.name})", file=sys.stderr)
                continue

            if args.mode:
                game_tags = normalize_tags(args.tag, args.mode)
            else:
                # New analyses are interactive by default.
                game_tags = normalize_tags(
                    args.tag, default_tags_for_stem(sgf_path.stem, has_analysis=True)[0]
                )

            print(
                f"Analyzing {sgf_path.name} -> {game_id} [{', '.join(game_tags)}] ...",
                file=sys.stderr,
            )
            game_json = analyze_game(
                sgf_path, katago, args.visits, game_id, tags=game_tags
            )
            out_path.write_text(json.dumps(game_json, indent=2) + "\n", encoding="utf-8")
            wrote += 1
            # Rebuild full manifest from SGFs so skipped/existing games stay listed.
            sync_manifest_from_sgfs(args.sgf_dir, args.out)
            print(f"  wrote {out_path.name} and synced manifest", file=sys.stderr)
    finally:
        if katago is not None:
            katago.close()

    entries = sync_manifest_from_sgfs(args.sgf_dir, args.out)
    print(
        f"Wrote {wrote} new analysis file(s). Manifest: {len(entries)} / {len(sgf_files)} SGFs.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
