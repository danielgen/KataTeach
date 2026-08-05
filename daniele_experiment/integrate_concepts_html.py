#!/usr/bin/env python3
"""
Integrate linear probe concept scores and LLM commentary into game HTML visualization.

This script adds concept annotations and a commentary box to existing viz.html
files in games/*/, updating as the move slider changes.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any


def load_game_concepts(concepts_dir: str, game_id: str) -> Optional[Dict]:
    """Load concept data for a game, preferring concepts_with_commentary.json."""
    base = Path(concepts_dir) / "html_data" / game_id
    for name in ("concepts_with_commentary.json", "concepts.json"):
        concepts_path = base / name
        if concepts_path.exists():
            with open(concepts_path, "r") as f:
                return json.load(f)
    return None


def load_commentary_map(concepts_dir: str, game_id: str, game_concepts: Optional[Dict]) -> Dict[int, Dict[str, Any]]:
    """
    Build move_number -> {comment, concepts_used} map.

    Loads embedded commentary and commentary.json, with the legacy JSONL cache
    taking precedence when present because it may contain fresher generations.
    """
    out: Dict[int, Dict[str, Any]] = {}

    if game_concepts:
        for move in game_concepts.get("moves", []):
            commentary = move.get("commentary")
            if isinstance(commentary, dict) and commentary.get("comment"):
                out[int(move["move_number"])] = {
                    "comment": commentary.get("comment", ""),
                    "concepts_used": commentary.get("concepts_used", []),
                }
            elif isinstance(commentary, str) and commentary.strip():
                out[int(move["move_number"])] = {
                    "comment": commentary.strip(),
                    "concepts_used": [],
                }

    base = Path(concepts_dir) / "html_data" / game_id
    json_path = base / "commentary.json"
    if json_path.exists():
        with open(json_path, "r") as f:
            payload = json.load(f)
        entries = payload.get("moves", []) if isinstance(payload, dict) else payload
        for entry in entries:
            try:
                move_num = int(entry["move_number"])
                out[move_num] = {
                    "comment": entry.get("comment", ""),
                    "concepts_used": entry.get("concepts_used", []),
                }
            except (KeyError, TypeError, ValueError):
                continue

    jsonl_path = base / "commentary.jsonl"
    if jsonl_path.exists():
        with open(jsonl_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    move_num = int(entry["move_number"])
                    # Prefer JSONL if present (freshest generation cache)
                    out[move_num] = {
                        "comment": entry.get("comment", ""),
                        "concepts_used": entry.get("concepts_used", []),
                    }
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue

    return out


def commentary_document(game_id: str, commentary_by_move: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Return the canonical, portable commentary.json representation."""
    return {
        "game_id": game_id,
        "moves": [
            {"move_number": move_number, **entry}
            for move_number, entry in sorted(commentary_by_move.items())
        ],
    }


def export_game_artifacts(
    game_dir: Path,
    game_concepts: Dict[str, Any],
    concepts_meta: Dict[str, Any],
    commentary_by_move: Dict[int, Dict[str, Any]],
) -> None:
    """Package visualization inputs beside viz.html so a game is self-contained."""
    with open(game_dir / "concepts.json", "w") as f:
        json.dump(game_concepts, f, indent=2)
    with open(game_dir / "commentary.json", "w") as f:
        json.dump(commentary_document(game_dir.name, commentary_by_move), f, indent=2)
    with open(game_dir / "concepts_meta.json", "w") as f:
        json.dump(concepts_meta, f, indent=2)


def load_concepts_meta(concepts_dir: str) -> Dict:
    """Load concept metadata."""
    meta_path = Path(concepts_dir) / "concepts_meta.json"
    if not meta_path.exists():
        return {}

    with open(meta_path, "r") as f:
        return json.load(f)


def generate_concepts_js(
    game_concepts: Dict,
    concepts_meta: Dict,
    commentary_by_move: Dict[int, Dict[str, Any]],
) -> str:
    """Generate JavaScript code for concept + commentary display."""

    moves_data = json.dumps(game_concepts["moves"])
    meta_data = json.dumps(concepts_meta)
    concept_names = json.dumps(game_concepts.get("concept_names", []))
    commentary_data = json.dumps({str(k): v for k, v in commentary_by_move.items()})

    return f'''
<script>
// Concept data injected by linear probe pipeline
const conceptData = {{
    moves: {moves_data},
    meta: {meta_data},
    conceptNames: {concept_names},
    commentaryByMove: {commentary_data}
}};

function getMoveConcepts(moveNumber) {{
    const move = conceptData.moves.find(m => m.move_number === moveNumber);
    if (!move) return null;
    return move;
}}

function getMoveCommentary(moveNumber) {{
    if (!conceptData.commentaryByMove) return null;
    return conceptData.commentaryByMove[String(moveNumber)] || null;
}}

function escapeHtml(text) {{
    if (text === null || text === undefined) return '';
    return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}}

function formatConceptBadges(move, showDeltas = true) {{
    let html = '<div class="concept-badges">';

    if (move.top_concepts && move.top_concepts.length > 0) {{
        const sortedConcepts = move.top_concepts
            .map(concept => [concept, move.scores[concept]])
            .filter(([concept, score]) => score !== null && score !== undefined);

        if (sortedConcepts.length > 0) {{
            html += '<div class="active-concepts"><strong>Top concepts:</strong> ';
            for (const [concept, score] of sortedConcepts) {{
                const percentile = move.score_percentiles[concept];
                const probability = move.probs ? move.probs[concept] : null;
                const value = probability !== null && probability !== undefined
                    ? `${{(probability * 100).toFixed(0)}}%`
                    : `p${{Math.round(percentile * 100)}}`;
                html += `<span class="concept-badge" style="background: #2196F3; color: white; padding: 2px 6px; border-radius: 3px; margin: 2px; font-size: 11px;">${{escapeHtml(concept)}}: ${{value}}</span>`;
            }}
            html += '</div>';
        }}
    }}

    if (showDeltas && move.top_delta_concepts && move.top_delta_concepts.length > 0) {{
        html += '<div class="delta-concepts"><strong>Changes:</strong> ';
        for (const item of move.top_delta_concepts) {{
            const color = item.delta > 0 ? '#4CAF50' : '#f44336';
            const sign = item.delta > 0 ? '+' : '';
            html += `<span class="concept-badge delta" style="background: ${{color}}; color: white; padding: 2px 6px; border-radius: 3px; margin: 2px; font-size: 11px;">${{escapeHtml(item.concept)}}: ${{sign}}${{item.delta.toFixed(2)}}</span>`;
        }}
        html += '</div>';
    }}

    html += '</div>';
    return html;
}}

function addCommentarySection() {{
    const existing = document.getElementById('commentary-section');
    if (existing) existing.remove();

    const section = document.createElement('div');
    section.className = 'sticky-section';
    section.id = 'commentary-section';
    section.innerHTML = `
        <h3 style="margin-top: 0; color: #1a5f2a;">Move Commentary</h3>
        <div id="commentary-box" style="
            background: #f4faf5;
            border: 1px solid #b7d7be;
            border-radius: 8px;
            padding: 12px 14px;
            min-height: 64px;
            font-size: 14px;
            line-height: 1.45;
            color: #1f2a1f;
        ">
            <div id="commentary-text" style="margin-bottom: 8px;">No commentary for this move.</div>
            <div id="commentary-concepts" style="font-size: 12px; color: #456;"></div>
        </div>
    `;

    const container = document.getElementById('stickyValueContainer');
    if (!container) {{
        document.body.appendChild(section);
        return;
    }}

    const snorkel = document.getElementById('snorkel-section');
    if (snorkel && snorkel.parentNode === container) {{
        container.insertBefore(section, snorkel);
    }} else {{
        // Place after value section if present, else at top
        const value = document.getElementById('value-section');
        if (value && value.nextSibling) {{
            container.insertBefore(section, value.nextSibling);
        }} else if (value) {{
            container.appendChild(section);
        }} else {{
            container.insertBefore(section, container.firstChild);
        }}
    }}
}}

function addConceptsPanel() {{
    const existing = document.getElementById('concepts-panel');
    if (existing) existing.remove();

    const panel = document.createElement('div');
    panel.id = 'concepts-panel';
    panel.style.cssText = 'position: fixed; right: 10px; top: 10px; width: 300px; max-height: 80vh; overflow-y: auto; background: white; border: 2px solid #2196F3; border-radius: 8px; padding: 15px; box-shadow: 0 4px 20px rgba(0,0,0,0.3); z-index: 10000; font-family: Arial, sans-serif;';
    panel.innerHTML = `
        <h3 style="margin-top: 0; color: #2196F3;">Concept Analysis</h3>
        <div id="concept-content">
            <p style="font-size: 12px; color: #666;">
                Navigate moves with the slider. Concepts and commentary update automatically.
            </p>
            <p id="concept-status" style="font-size: 11px; color: #999; margin-top: 5px;">Loading...</p>
        </div>
        <div id="concept-legend" style="margin-top: 10px; font-size: 11px; border-top: 1px solid #eee; padding-top: 10px;">
            <strong>Legend:</strong><br>
            <span style="color: #2196F3;">●</span> Active concept (positive)<br>
            <span style="color: #FF9800;">●</span> Active concept (negative)<br>
            <span style="color: #4CAF50;">●</span> Increased after move<br>
            <span style="color: #f44336;">●</span> Decreased after move
        </div>
    `;
    document.body.appendChild(panel);
}}

function updateCommentaryBox(moveNumber) {{
    const textEl = document.getElementById('commentary-text');
    const conceptsEl = document.getElementById('commentary-concepts');
    if (!textEl || !conceptsEl) return;

    if (moveNumber === null || moveNumber === 0) {{
        textEl.textContent = 'Initial position — no move commentary.';
        conceptsEl.innerHTML = '';
        return;
    }}

    const entry = getMoveCommentary(moveNumber);
    if (!entry || !entry.comment) {{
        textEl.textContent = `No commentary for move ${{moveNumber}}.`;
        conceptsEl.innerHTML = '';
        return;
    }}

    textEl.textContent = entry.comment;
    const used = entry.concepts_used || [];
    if (used.length) {{
        conceptsEl.innerHTML = '<strong>Concepts:</strong> ' + used.map(c =>
            `<span style="display:inline-block;background:#e8f5e9;border:1px solid #a5d6a7;border-radius:3px;padding:1px 6px;margin:2px;font-size:11px;">${{escapeHtml(c)}}</span>`
        ).join(' ');
    }} else {{
        conceptsEl.innerHTML = '';
    }}
}}

function updateConceptsPanel(moveNumber) {{
    updateCommentaryBox(moveNumber);

    const content = document.getElementById('concept-content');
    if (!content) return;

    if (!conceptData || !conceptData.moves) {{
        content.innerHTML = '<p style="color: red;">Concept data not loaded.</p>';
        return;
    }}

    const move = getMoveConcepts(moveNumber);
    if (!move) {{
        content.innerHTML = `<p style="color: #666;">No concept data for move ${{moveNumber}}.</p>`;
        return;
    }}

    let html = `<p><strong>Move ${{moveNumber}}</strong> (${{move.player ? move.player.toUpperCase() : '?'}})</p>`;
    html += formatConceptBadges(move);

    html += `
        <details style="margin-top: 10px;">
            <summary style="cursor: pointer;">All Scores</summary>
            <table style="width: 100%; font-size: 11px; margin-top: 5px;">
                <tr><th>Concept</th><th>Probability</th><th>Score</th><th>Delta</th><th>AUC</th></tr>
    `;

    for (const concept of conceptData.conceptNames) {{
        const score = move.scores[concept];
        const delta = move.deltas[concept];
        const probability = move.probs ? move.probs[concept] : null;
        const quality = conceptData.meta[concept] || {{}};
        const scoreStr = score !== null && score !== undefined ? score.toFixed(3) : '-';
        const deltaStr = delta !== null && delta !== undefined ? (delta > 0 ? '+' : '') + delta.toFixed(3) : '-';
        const probabilityStr = probability !== null && probability !== undefined ? (probability * 100).toFixed(1) + '%' : '-';
        const aucStr = quality.auc !== null && quality.auc !== undefined ? quality.auc.toFixed(3) : '-';
        const warning = quality.auc !== null && quality.auc !== undefined && quality.auc < 0.8 ? ' title="Lower-confidence probe" style="color:#b26a00"' : '';
        html += `<tr><td${{warning}}>${{escapeHtml(concept)}}</td><td>${{probabilityStr}}</td><td>${{scoreStr}}</td><td>${{deltaStr}}</td><td>${{aucStr}}</td></tr>`;
    }}

    html += '</table></details>';
    content.innerHTML = html;
}}

window.selectMove = function(moveNumber) {{
    updateConceptsPanel(moveNumber);
}};

function getCurrentMoveNumber() {{
    const moveInfo = document.getElementById('moveInfo');
    if (moveInfo) {{
        const text = moveInfo.textContent || moveInfo.innerText;
        const match = text.match(/Move (\\d+)/);
        if (match) return parseInt(match[1]);
    }}

    try {{
        if (typeof gameData !== 'undefined' && typeof currentMove !== 'undefined') {{
            const moveData = gameData[currentMove];
            if (moveData && moveData.move_number !== undefined) {{
                return moveData.move_number;
            }}
        }}
    }} catch(e) {{}}

    if (typeof window.gameData !== 'undefined' && typeof window.currentMove !== 'undefined') {{
        const moveData = window.gameData[window.currentMove];
        if (moveData && moveData.move_number !== undefined) {{
            return moveData.move_number;
        }}
    }}

    return null;
}}

document.addEventListener('DOMContentLoaded', function() {{
    // Wait briefly so snorkel sticky sections exist, then insert commentary above them
    setTimeout(function() {{
        addCommentarySection();
        addConceptsPanel();

        const slider = document.getElementById('moveSlider');
        if (slider) {{
            slider.addEventListener('input', function() {{
                setTimeout(function() {{
                    const moveNum = getCurrentMoveNumber();
                    if (moveNum !== null) updateConceptsPanel(moveNum);
                }}, 50);
            }}, true);
        }}

        let lastMoveNumber = null;
        setInterval(function() {{
            const moveNum = getCurrentMoveNumber();
            if (moveNum !== null && moveNum !== lastMoveNumber) {{
                lastMoveNumber = moveNum;
                updateConceptsPanel(moveNum);
            }}
        }}, 200);

        setTimeout(function() {{
            const statusEl = document.getElementById('concept-status');
            const initialMoveNum = getCurrentMoveNumber();
            if (initialMoveNum !== null) {{
                updateConceptsPanel(initialMoveNum);
                lastMoveNumber = initialMoveNum;
                if (statusEl) statusEl.textContent = `Loaded move ${{initialMoveNum}}`;
            }} else if (statusEl) {{
                statusEl.innerHTML = '<span style="color: orange;">Waiting for move data...</span>';
            }}
        }}, 400);
    }}, 600);
}});
</script>

<style>
.concept-badges {{
    margin: 5px 0;
}}
.concept-badge {{
    display: inline-block;
    margin: 2px;
}}
.active-concepts, .delta-concepts {{
    margin: 5px 0;
}}
#commentary-section h3 {{
    color: #1a5f2a;
}}
#concepts-panel details summary::-webkit-details-marker {{
    display: none;
}}
#concepts-panel details summary {{
    list-style: none;
}}
#concepts-panel details summary::before {{
    content: '▶ ';
}}
#concepts-panel details[open] summary::before {{
    content: '▼ ';
}}
</style>
'''


def inject_concepts_into_html(
    html_path: str,
    game_concepts: Dict,
    concepts_meta: Dict,
    commentary_by_move: Optional[Dict[int, Dict[str, Any]]] = None,
) -> str:
    """Inject concept + commentary JavaScript into existing HTML file."""
    with open(html_path, "r") as f:
        html = f.read()

    # Remove existing injected block (from // Concept data injected to </style>)
    concept_script_pattern = r'<script>\s*// Concept data injected by linear probe pipeline.*?</style>\s*'
    html = re.sub(concept_script_pattern, "", html, flags=re.DOTALL)

    concepts_js = generate_concepts_js(
        game_concepts,
        concepts_meta,
        commentary_by_move or {},
    )

    if "</body>" in html:
        html = html.replace("</body>", f"{concepts_js}\n</body>")
    else:
        html += concepts_js

    return html


def process_game(
    game_dir: Path,
    concepts_dir: str,
    concepts_meta: Dict,
    overwrite: bool = True,
    output_suffix: str = "_with_concepts",
) -> bool:
    """Process a single game. Returns True if processed."""
    game_id = game_dir.name
    html_path = game_dir / "viz.html"
    if not html_path.exists():
        if overwrite:
            return False
        existing_output = game_dir / f"viz{output_suffix}.html"
        if existing_output.exists():
            html_path = existing_output
        else:
            return False

    game_concepts = load_game_concepts(concepts_dir, game_id)
    if game_concepts is None:
        return False

    commentary_by_move = load_commentary_map(concepts_dir, game_id, game_concepts)
    canonical_commentary_path = Path(concepts_dir) / "html_data" / game_id / "commentary.json"
    with open(canonical_commentary_path, "w") as f:
        json.dump(commentary_document(game_id, commentary_by_move), f, indent=2)
    new_html = inject_concepts_into_html(
        str(html_path),
        game_concepts,
        concepts_meta,
        commentary_by_move,
    )

    output_path = html_path if overwrite else (game_dir / f"viz{output_suffix}.html")
    with open(output_path, "w") as f:
        f.write(new_html)
    export_game_artifacts(
        game_dir,
        game_concepts,
        concepts_meta,
        commentary_by_move,
    )
    return True


def process_all_games(
    games_dir: str,
    concepts_dir: str,
    output_suffix: str = "_with_concepts",
    overwrite: bool = True,
    game_id: Optional[str] = None,
):
    """
    Process games and create HTML files with concepts + commentary.

    Args:
        games_dir: Path to games directory
        concepts_dir: Path to linear_probes output directory
        output_suffix: Suffix for output HTML files (only used if overwrite=False)
        overwrite: If True, overwrite original viz.html (default: True)
        game_id: If set, process only this game
    """
    games_path = Path(games_dir)
    concepts_meta = load_concepts_meta(concepts_dir)

    processed = 0
    skipped = 0

    if game_id:
        game_dirs = [games_path / game_id]
        if not game_dirs[0].is_dir():
            print(f"Game not found: {game_dirs[0]}")
            return
    else:
        game_dirs = sorted(d for d in games_path.iterdir() if d.is_dir())

    for game_dir in game_dirs:
        ok = process_game(
            game_dir,
            concepts_dir,
            concepts_meta,
            overwrite=overwrite,
            output_suffix=output_suffix,
        )
        if ok:
            processed += 1
            commentary_n = len(load_commentary_map(
                concepts_dir,
                game_dir.name,
                load_game_concepts(concepts_dir, game_dir.name),
            ))
            print(f"  {game_dir.name}: injected concepts + {commentary_n} commentary moves")
        else:
            skipped += 1

    print(f"Processed {processed} games, skipped {skipped}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Integrate concepts and commentary into game HTML"
    )
    parser.add_argument(
        "--games-dir",
        type=str,
        default="../games",
        help="Path to games directory",
    )
    parser.add_argument(
        "--concepts-dir",
        type=str,
        default="linear_probes",
        help="Path to linear probes output directory",
    )
    parser.add_argument(
        "--game-id",
        type=str,
        default=None,
        help="Process only this game ID",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Create new files with suffix instead of overwriting viz.html",
    )

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    games_dir = script_dir / args.games_dir
    concepts_dir = script_dir / args.concepts_dir

    process_all_games(
        str(games_dir),
        str(concepts_dir),
        overwrite=not args.no_overwrite,
        game_id=args.game_id,
    )


if __name__ == "__main__":
    main()
