#!/usr/bin/env python3
"""
Integrate linear probe concept scores into game HTML visualization.

This script adds concept annotations to existing viz.html files in games/*/,
allowing users to see which concepts are activated at each move.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional


def load_game_concepts(concepts_dir: str, game_id: str) -> Optional[Dict]:
    """Load concept data for a game."""
    concepts_path = Path(concepts_dir) / "html_data" / game_id / "concepts.json"
    if not concepts_path.exists():
        return None
    
    with open(concepts_path, 'r') as f:
        return json.load(f)


def load_concepts_meta(concepts_dir: str) -> Dict:
    """Load concept metadata."""
    meta_path = Path(concepts_dir) / "concepts_meta.json"
    if not meta_path.exists():
        return {}
    
    with open(meta_path, 'r') as f:
        return json.load(f)


def format_concept_badge(concept: str, score: float, is_delta: bool = False) -> str:
    """Format a concept as an HTML badge."""
    if is_delta:
        if score > 0:
            color = "#4CAF50"  # Green for positive delta
            sign = "+"
        else:
            color = "#f44336"  # Red for negative delta
            sign = ""
        return f'<span class="concept-badge delta" style="background: {color}; color: white; padding: 2px 6px; border-radius: 3px; margin: 2px; font-size: 11px;">{concept}: {sign}{score:.2f}</span>'
    else:
        # Score badge (activation strength)
        intensity = min(abs(score) / 2, 1)  # Normalize
        if score > 0:
            color = f"rgba(33, 150, 243, {0.3 + intensity * 0.7})"  # Blue
        else:
            color = f"rgba(255, 152, 0, {0.3 + intensity * 0.7})"  # Orange
        return f'<span class="concept-badge score" style="background: {color}; padding: 2px 6px; border-radius: 3px; margin: 2px; font-size: 11px;">{concept}: {score:.2f}</span>'


def generate_concepts_js(game_concepts: Dict, concepts_meta: Dict) -> str:
    """Generate JavaScript code for concept display."""
    
    moves_data = json.dumps(game_concepts['moves'])
    meta_data = json.dumps(concepts_meta)
    concept_names = json.dumps(game_concepts['concept_names'])
    
    return f'''
<script>
// Concept data injected by linear probe pipeline
const conceptData = {{
    moves: {moves_data},
    meta: {meta_data},
    conceptNames: {concept_names}
}};

// Get concepts for a move
function getMoveConcepts(moveNumber) {{
    const move = conceptData.moves.find(m => m.move_number === moveNumber);
    if (!move) return null;
    return move;
}}

// Format concept badges
function formatConceptBadges(move, showDeltas = true) {{
    let html = '<div class="concept-badges">';
    
    // Top activated concepts
    if (move.top_concepts && move.top_concepts.length > 0) {{
        html += '<div class="active-concepts"><strong>Active:</strong> ';
        for (const concept of move.top_concepts) {{
            const score = move.scores[concept];
            const color = score > 0 ? '#2196F3' : '#FF9800';
            html += `<span class="concept-badge" style="background: ${{color}}; color: white; padding: 2px 6px; border-radius: 3px; margin: 2px; font-size: 11px;">${{concept}}: ${{score.toFixed(2)}}</span>`;
        }}
        html += '</div>';
    }}
    
    // Top delta concepts (what changed)
    if (showDeltas && move.top_delta_concepts && move.top_delta_concepts.length > 0) {{
        html += '<div class="delta-concepts"><strong>Changes:</strong> ';
        for (const item of move.top_delta_concepts) {{
            const color = item.delta > 0 ? '#4CAF50' : '#f44336';
            const sign = item.delta > 0 ? '+' : '';
            html += `<span class="concept-badge delta" style="background: ${{color}}; color: white; padding: 2px 6px; border-radius: 3px; margin: 2px; font-size: 11px;">${{item.concept}}: ${{sign}}${{item.delta.toFixed(2)}}</span>`;
        }}
        html += '</div>';
    }}
    
    html += '</div>';
    return html;
}}

// Add concepts panel to page
function addConceptsPanel() {{
    // Remove existing panel if it exists
    const existing = document.getElementById('concepts-panel');
    if (existing) {{
        existing.remove();
    }}
    
    const panel = document.createElement('div');
    panel.id = 'concepts-panel';
    panel.style.cssText = 'position: fixed; right: 10px; top: 10px; width: 300px; max-height: 80vh; overflow-y: auto; background: white; border: 2px solid #2196F3; border-radius: 8px; padding: 15px; box-shadow: 0 4px 20px rgba(0,0,0,0.3); z-index: 10000; font-family: Arial, sans-serif;';
    panel.innerHTML = `
        <h3 style="margin-top: 0; color: #2196F3;">Concept Analysis</h3>
        <div id="concept-content">
            <p style="font-size: 12px; color: #666;">
                <strong>How to use:</strong><br>
                Use the move slider or navigation buttons at the top to navigate through moves. 
                The concept panel will automatically update to show activations for the current move.
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
    console.log('Concept panel added to page');
}}

// Update concepts panel for current move
function updateConceptsPanel(moveNumber) {{
    const content = document.getElementById('concept-content');
    if (!content) {{
        console.warn('Concept panel content element not found');
        return;
    }}
    
    if (!conceptData || !conceptData.moves) {{
        content.innerHTML = '<p style="color: red;">Concept data not loaded.</p>';
        return;
    }}
    
    const move = getMoveConcepts(moveNumber);
    if (!move) {{
        content.innerHTML = `<p style="color: #666;">No concept data for move ${{moveNumber}}.</p><p style="font-size: 11px; color: #999;">Available moves: ${{conceptData.moves.length > 0 ? conceptData.moves.map(m => m.move_number).join(', ') : 'none'}}</p>`;
        return;
    }}
    
    let html = `<p><strong>Move ${{moveNumber}}</strong> (${{move.player ? move.player.toUpperCase() : '?'}})</p>`;
    html += formatConceptBadges(move);
    
    // Show all scores (collapsible)
    html += `
        <details style="margin-top: 10px;">
            <summary style="cursor: pointer;">All Scores</summary>
            <table style="width: 100%; font-size: 11px; margin-top: 5px;">
                <tr><th>Concept</th><th>Score</th><th>Delta</th></tr>
    `;
    
    for (const concept of conceptData.conceptNames) {{
        const score = move.scores[concept];
        const delta = move.deltas[concept];
        const scoreStr = score !== null && score !== undefined ? score.toFixed(3) : '-';
        const deltaStr = delta !== null && delta !== undefined ? (delta > 0 ? '+' : '') + delta.toFixed(3) : '-';
        html += `<tr><td>${{concept}}</td><td>${{scoreStr}}</td><td>${{deltaStr}}</td></tr>`;
    }}
    
    html += '</table></details>';
    content.innerHTML = html;
}}

// Expose selectMove function for external use
window.selectMove = function(moveNumber) {{
    updateConceptsPanel(moveNumber);
}};

// Helper function to get current move number from visualization
function getCurrentMoveNumber() {{
    // Method 1: Try to read from the moveInfo element (most reliable)
    const moveInfo = document.getElementById('moveInfo');
    if (moveInfo) {{
        const text = moveInfo.textContent || moveInfo.innerText;
        const match = text.match(/Move (\\d+)/);
        if (match) {{
            return parseInt(match[1]);
        }}
    }}
    
    // Method 2: Try to access gameData and currentMove if they're in global scope
    try {{
        if (typeof gameData !== 'undefined' && typeof currentMove !== 'undefined') {{
            const moveData = gameData[currentMove];
            if (moveData && moveData.move_number !== undefined) {{
                return moveData.move_number;
            }}
        }}
    }} catch(e) {{
        // Variables not accessible
    }}
    
    // Method 3: Try window properties (in case they were exposed)
    if (typeof window.gameData !== 'undefined' && typeof window.currentMove !== 'undefined') {{
        const moveData = window.gameData[window.currentMove];
        if (moveData && moveData.move_number !== undefined) {{
            return moveData.move_number;
        }}
    }}
    
    return null;
}}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {{
    addConceptsPanel();
    
    // Hook into existing visualization after it loads
    setTimeout(function() {{
        // Method 1: Monitor the slider value directly (most reliable)
        const slider = document.getElementById('moveSlider');
        if (slider) {{
            // Add our own listener that runs after the existing one
            slider.addEventListener('input', function() {{
                setTimeout(function() {{
                    const moveNum = getCurrentMoveNumber();
                    if (moveNum !== null) {{
                        updateConceptsPanel(moveNum);
                    }}
                }}, 50); // Small delay to let updateDisplay finish
            }}, true); // Use capture phase to run after existing listeners
        }}
        
        // Method 2: Monitor moveInfo text changes (reliable fallback)
        let lastMoveNumber = null;
        
        const checkMoveChange = setInterval(function() {{
            const moveNum = getCurrentMoveNumber();
            if (moveNum !== null && moveNum !== lastMoveNumber) {{
                lastMoveNumber = moveNum;
                updateConceptsPanel(moveNum);
            }}
        }}, 200); // Check every 200ms (smooth but not too frequent)
        
        // Update immediately if we can get the move number
        setTimeout(function() {{
            const statusEl = document.getElementById('concept-status');
            const initialMoveNum = getCurrentMoveNumber();
            if (initialMoveNum !== null) {{
                updateConceptsPanel(initialMoveNum);
                lastMoveNumber = initialMoveNum;
                if (statusEl) statusEl.textContent = `Loaded move ${{initialMoveNum}}`;
                console.log('Concept panel initialized with move', initialMoveNum);
            }} else {{
                // Debug: show what we found
                const moveInfo = document.getElementById('moveInfo');
                console.log('Could not get initial move number.');
                console.log('moveInfo element:', moveInfo);
                console.log('moveInfo text:', moveInfo ? moveInfo.textContent : 'not found');
                console.log('conceptData:', conceptData);
                console.log('conceptData.moves length:', conceptData ? conceptData.moves.length : 'N/A');
                if (statusEl) statusEl.innerHTML = '<span style="color: orange;">Waiting for move data...</span>';
            }}
        }}, 1000); // Wait 1 second for visualization to fully initialize
        
        // Keep polling active (it's lightweight and ensures updates work)
        // The interval will continue running but only updates when move actually changes
    }}, 500); // Wait 500ms for visualization to initialize
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


def inject_concepts_into_html(html_path: str, game_concepts: Dict, concepts_meta: Dict) -> str:
    """Inject concept JavaScript into existing HTML file."""
    with open(html_path, 'r') as f:
        html = f.read()
    
    # Remove existing concept script if present (to avoid duplicates)
    # Pattern to match the concept script block (from // Concept data injected to </style>)
    concept_script_pattern = r'<script>\s*// Concept data injected by linear probe pipeline.*?</style>\s*'
    html = re.sub(concept_script_pattern, '', html, flags=re.DOTALL)
    
    # Generate concepts JS
    concepts_js = generate_concepts_js(game_concepts, concepts_meta)
    
    # Find insertion point (before </body> or at end)
    if '</body>' in html:
        html = html.replace('</body>', f'{concepts_js}\n</body>')
    else:
        html += concepts_js
    
    return html


def process_all_games(
    games_dir: str,
    concepts_dir: str,
    output_suffix: str = "_with_concepts",
    overwrite: bool = True,
):
    """
    Process all games and create HTML files with concepts.
    
    Args:
        games_dir: Path to games directory
        concepts_dir: Path to linear_probes output directory
        output_suffix: Suffix for output HTML files (only used if overwrite=False)
        overwrite: If True, overwrite original viz.html (default: True)
    """
    games_path = Path(games_dir)
    concepts_meta = load_concepts_meta(concepts_dir)
    
    processed = 0
    skipped = 0
    
    for game_dir in sorted(games_path.iterdir()):
        if not game_dir.is_dir():
            continue
        
        game_id = game_dir.name
        
        # Find source HTML file
        html_path = game_dir / "viz.html"
        if not html_path.exists():
            # If overwriting and viz.html doesn't exist, skip (can't overwrite what doesn't exist)
            if overwrite:
                skipped += 1
                continue
            # If not overwriting, try to use existing viz_with_concepts.html as source
            existing_output = game_dir / f"viz{output_suffix}.html"
            if existing_output.exists():
                html_path = existing_output
            else:
                skipped += 1
                continue
        
        # Load concepts for this game
        game_concepts = load_game_concepts(concepts_dir, game_id)
        if game_concepts is None:
            skipped += 1
            continue
        
        # Inject concepts
        new_html = inject_concepts_into_html(str(html_path), game_concepts, concepts_meta)
        
        # Save
        if overwrite:
            output_path = html_path
        else:
            output_path = game_dir / f"viz{output_suffix}.html"
        
        with open(output_path, 'w') as f:
            f.write(new_html)
        
        processed += 1
    
    print(f"Processed {processed} games, skipped {skipped}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Integrate concepts into game HTML")
    parser.add_argument('--games-dir', type=str, default='../games',
                        help='Path to games directory')
    parser.add_argument('--concepts-dir', type=str, default='linear_probes',
                        help='Path to linear probes output directory')
    parser.add_argument('--no-overwrite', action='store_true',
                        help='Create new files with suffix instead of overwriting viz.html')
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    games_dir = script_dir / args.games_dir
    concepts_dir = script_dir / args.concepts_dir
    
    process_all_games(
        str(games_dir),
        str(concepts_dir),
        overwrite=not args.no_overwrite,
    )


if __name__ == "__main__":
    main()

