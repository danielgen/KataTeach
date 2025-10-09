from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Add the parent directory to Python path to find cbm module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cbm.move_candidate_dataset import MoveCandidateDataset, slate_group_collate
from cbm.concept_utils import load_model_with_concept_learning


def sgf_to_coord(move_string: str) -> Tuple[int, int]:
    """Convert SGF move string to board coordinates."""
    if not move_string or move_string == 'pass':
        return None, None
    
    # Handle SGF coordinates like 'pd', 'dp', etc. (lowercase letters)
    if len(move_string) == 2 and move_string.islower() and move_string.isalpha():
        x = ord(move_string[0]) - ord('a')
        y = ord(move_string[1]) - ord('a')
        return x, y
    
    # Handle human coordinates like 'C16', 'D4', 'P16', etc.
    if len(move_string) >= 2 and move_string[0].isupper():
        letter = move_string[0]
        number = int(move_string[1:])
        
        # Convert letter to x coordinate (A=0, B=1, ..., H=7, J=8, ..., T=18)
        x = ord(letter) - ord('A')
        if x >= 8:  # Skip 'I' in Go notation
            x -= 1
        
        # Convert number to y coordinate (1=18, 2=17, ..., 19=0)
        y = 19 - number
        
        if 0 <= x < 19 and 0 <= y < 19:
            return x, y
    
    return None, None


def coord_to_sgf(x: int, y: int) -> str:
    """Convert board coordinates to SGF format."""
    if x < 0 or x >= 19 or y < 0 or y >= 19:
        return "pass"
    return chr(ord('a') + x) + chr(ord('a') + y)


def coord_to_human(x: int, y: int) -> str:
    """Convert board coordinates to human format."""
    if x < 0 or x >= 19 or y < 0 or y >= 19:
        return "pass"
    
    # Convert x to letter (A=0, B=1, ..., H=7, J=8, ..., T=18)
    if x >= 8:  # Skip 'I' in Go notation
        letter = chr(ord('A') + x + 1)
    else:
        letter = chr(ord('A') + x)
    
    # Convert y to number (0=19, 1=18, ..., 18=1)
    number = 19 - y
    
    return f"{letter}{number}"


def load_concept_names(ontology_path: Path) -> List[str]:
    """Load concept names from ontology.yaml file."""
    if not ontology_path.exists():
        return []
    
    try:
        with ontology_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        concept_names = []
        tags_data = data.get("tags", {})
        
        # Collect all concept names from all categories
        for category, concepts in tags_data.items():
            if isinstance(concepts, list):
                for concept in concepts:
                    if isinstance(concept, dict) and "name" in concept:
                        concept_names.append(concept["name"])
        
        return concept_names
    except Exception as e:
        print(f"Warning: Could not load concept names from {ontology_path}: {e}")
        return []


def create_concept_visualization_html(
    model,
    dataset,
    device,
    output_path: Path,
    num_positions: int = 10,
    concept_threshold: float = 0.3,
    concept_names: List[str] = None,
    games_dir: Path = None
):
    """Create an HTML page visualizing concept activations on Go positions."""
    
    model.eval()
    
    # Load SGF and policy data if available
    sgf_data = {}
    policy_data = {}
    if games_dir and games_dir.exists():
        # Find SGF files
        sgf_files = list(games_dir.glob("*.sgf"))
        policy_files = list((games_dir / "policy").glob("*.json"))
        
        for sgf_file in sgf_files:
            game_id = sgf_file.stem
            try:
                with sgf_file.open("r", encoding="utf-8") as f:
                    sgf_data[game_id] = f.read().strip()
            except Exception as e:
                print(f"Warning: Could not load SGF file {sgf_file}: {e}")
        
        for policy_file in policy_files:
            game_id = policy_file.stem
            try:
                with policy_file.open("r", encoding="utf-8") as f:
                    policy_data[game_id] = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load policy file {policy_file}: {e}")
    
    # Collect data for visualization
    positions_data = []
    
    with torch.no_grad():
        for i in range(min(num_positions, len(dataset))):
            slate = dataset[i]
            x = slate["x"].to(device)
            moves = slate["moves"].to(device)
            slate_probs = slate["slate_probs"].to(device)
            
            # Get model predictions
            model_scores = model.score_candidates(x, moves)
            model_probs = torch.softmax(model_scores, dim=0)
            
            # Try to find corresponding game data
            game_id = None
            sgf_text = None
            policy_info = None
            
            # Look for game data that matches this slate
            if hasattr(slate, 'get') and 'slate_id' in slate:
                slate_id = slate['slate_id']
                # Try to find matching game data
                for gid in sgf_data.keys():
                    if gid in slate_id or slate_id in gid:
                        game_id = gid
                        sgf_text = sgf_data[gid]
                        policy_info = policy_data.get(gid, {})
                        break
            
            # Get concepts for each move
            move_concepts = []
            for j, move in enumerate(moves):
                move_idx = move.item()
                concepts = model.concepts_for_move(x, move_idx, return_probs=True)
                
                # Find active concepts
                active_concepts = []
                for concept_idx, activation in enumerate(concepts[0]):
                    if activation > concept_threshold:
                        # Get concept name if available
                        concept_name = concept_names[concept_idx] if concept_names and concept_idx < len(concept_names) else f"concept_{concept_idx}"
                        active_concepts.append({
                            'concept_id': concept_idx,
                            'concept_name': concept_name,
                            'activation': activation.item(),
                            'is_labeled': concept_idx < model.num_labeled_concepts
                        })
                
                # Sort by activation strength
                active_concepts.sort(key=lambda x: x['activation'], reverse=True)
                
                # Convert move index to coordinates
                x_coord, y_coord = sgf_to_coord(coord_to_sgf(move_idx // 19, move_idx % 19))
                
                move_concepts.append({
                    'move_idx': move_idx,
                    'coordinates': {'x': x_coord, 'y': y_coord},
                    'human_coord': coord_to_human(x_coord, y_coord) if x_coord is not None else "pass",
                    'kata_go_prob': slate_probs[j].item(),
                    'model_prob': model_probs[j].item(),
                    'active_concepts': active_concepts
                })
            
            positions_data.append({
                'position_id': i,
                'game_id': game_id,
                'sgf_text': sgf_text,
                'policy_info': policy_info,
                'moves': move_concepts
            })
    
    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8' />
    <title>Concept Learning CBM Visualization</title>
    <script src='../../web/wgo.min.js'></script>
    <style>
        body {{ 
            margin: 20px; 
            font-family: Arial, sans-serif; 
            background-color: #f5f5f5;
        }}
        .container {{ 
            display: flex; 
            gap: 20px; 
            max-width: 1600px; 
            margin: 0 auto;
        }}
        .board-section {{ 
            flex: 0 0 520px; 
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .info-section {{ 
            flex: 1; 
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        #board {{ 
            width: 480px; 
            height: 480px; 
            border: 2px solid #333; 
            border-radius: 5px; 
        }}
        .game-info {{ 
            background: #f9f9f9; 
            padding: 8px; 
            border-radius: 5px; 
            margin-bottom: 10px;
            font-size: 14px;
            border: 1px solid #ddd;
        }}
        .player-names {{ 
            display: flex; 
            justify-content: space-between; 
            margin-bottom: 5px;
        }}
        .player-black {{ font-weight: bold; }}
        .player-white {{ font-weight: bold; }}
        .position-selector {{
            margin-bottom: 20px;
            text-align: center;
        }}
        .position-selector select {{
            padding: 10px;
            font-size: 16px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .navigation {{ 
            margin: 10px 0; 
            text-align: center;
            padding: 10px;
            background: #f5f5f5;
            border-radius: 5px;
        }}
        .nav-button {{ 
            padding: 8px 16px; 
            margin: 0 5px; 
            background: #007bff; 
            color: white; 
            border: none; 
            border-radius: 3px; 
            cursor: pointer;
        }}
        .nav-button:hover {{ background: #0056b3; }}
        .move-counter {{ 
            display: inline-block; 
            margin: 0 10px; 
            font-weight: bold; 
            font-size: 16px;
        }}
        .move-info {{
            margin-bottom: 15px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #007bff;
        }}
        .move-header {{
            font-weight: bold;
            font-size: 18px;
            margin-bottom: 10px;
            color: #333;
        }}
        .move-details {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 10px;
        }}
        .detail-item {{
            background: white;
            padding: 8px;
            border-radius: 4px;
            border: 1px solid #e0e0e0;
        }}
        .detail-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }}
        .detail-value {{
            font-weight: bold;
            color: #333;
        }}
        .concepts-list {{
            margin-top: 10px;
        }}
        .concept-item {{
            display: inline-block;
            margin: 2px;
            padding: 4px 8px;
            background: #e3f2fd;
            border-radius: 12px;
            font-size: 12px;
            border: 1px solid #2196f3;
        }}
        .concept-item.labeled {{
            background: #e8f5e8;
            border-color: #4caf50;
        }}
        .concept-item.latent {{
            background: #fff3e0;
            border-color: #ff9800;
        }}
        .concept-activation {{
            font-weight: bold;
            margin-left: 5px;
        }}
        .legend {{
            margin-top: 20px;
            padding: 15px;
            background: #f0f0f0;
            border-radius: 8px;
        }}
        .legend h4 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        .legend-item {{
            display: inline-block;
            margin: 5px 10px 5px 0;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 12px;
        }}
        .stats {{
            margin-top: 20px;
            padding: 15px;
            background: #e8f4fd;
            border-radius: 8px;
        }}
        .stats h4 {{
            margin: 0 0 10px 0;
            color: #1976d2;
        }}
        .stat-item {{
            margin: 5px 0;
            display: flex;
            justify-content: space-between;
        }}
        
        /* Hide WGo.js default controls */
        .wgo-player-wrapper .wgo-player-control {{ display: none !important; }}
        .wgo-player-wrapper .wgo-comments {{ display: none !important; }}
        .wgo-player-wrapper .wgo-info {{ display: none !important; }}
        .wgo-player-wrapper .wgo-infobox {{ display: none !important; }}
        .wgo-player__box.wgo-player__player-tag {{ display: none !important; }}
        .wgo-player__player-tag {{ display: none !important; }}
        
        /* Style policy labels */
        .policy-label {{
            fill: red !important;
            color: red !important;
            font-weight: bold !important;
            font-size: 0.8px !important;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="board-section">
            <div class="game-info">
                <div class="player-names">
                    <span class="player-black">⚫ <span id='black-player'>Black</span></span>
                    <span class="player-white">⚪ <span id='white-player'>White</span></span>
                </div>
                <div>Game: <span id='game-info'>Loading...</span></div>
            </div>
            <div class="position-selector">
                <label for="position-select">Position: </label>
                <select id="position-select">
                    {chr(10).join([f'<option value="{i}">Position {i+1}</option>' for i in range(len(positions_data))])}
                </select>
            </div>
            <div id="board"></div>
            <div class="navigation">
                <button id='prev' class='nav-button'>← Prev</button>
                <span class='move-counter'>Move <span id='move_idx'>0</span></span>
                <button id='next' class='nav-button'>Next →</button>
            </div>
        </div>
        <div class="info-section">
            <div id="position-info"></div>
            <div class="legend">
                <h4>Legend</h4>
                <span class="legend-item" style="background: #e8f5e8; border: 1px solid #4caf50;">Labeled Concepts</span>
                <span class="legend-item" style="background: #fff3e0; border: 1px solid #ff9800;">Latent Concepts</span>
                <span class="legend-item" style="background: #e3f2fd; border: 1px solid #2196f3;">All Concepts</span>
            </div>
            <div class="stats">
                <h4>Model Statistics</h4>
                <div class="stat-item">
                    <span>Total Concepts:</span>
                    <span>{model.num_concepts}</span>
                </div>
                <div class="stat-item">
                    <span>Labeled Concepts:</span>
                    <span>{model.num_labeled_concepts}</span>
                </div>
                <div class="stat-item">
                    <span>Latent Concepts:</span>
                    <span>{model.num_latent_concepts}</span>
                </div>
                <div class="stat-item">
                    <span>Concept Threshold:</span>
                    <span>{concept_threshold}</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        const positionsData = {json.dumps(positions_data)};
        let player = null;
        let currentPosition = 0;
        let currentMove = 0;

        function initBoard() {{
            const boardElement = document.getElementById('board');
            boardElement.innerHTML = '';
            
            // Initialize WGo.js player if we have SGF data
            const position = positionsData[currentPosition];
            if (position.sgf_text) {{
                try {{
                    player = new WGo.SimplePlayer(boardElement, {{ sgf: position.sgf_text }});
                    initGameInfo(position);
                }} catch (e) {{
                    console.error('Error initializing WGo player:', e);
                    createSimpleBoard();
                }}
            }} else {{
                createSimpleBoard();
            }}
        }}

        function createSimpleBoard() {{
            const boardElement = document.getElementById('board');
            boardElement.innerHTML = '';
            
            // Create SVG board
            const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            svg.setAttribute('width', '480');
            svg.setAttribute('height', '480');
            svg.setAttribute('viewBox', '0 0 480 480');
            
            // Draw board lines
            for (let i = 0; i < 19; i++) {{
                // Vertical lines
                const vLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                vLine.setAttribute('x1', 20 + i * 23);
                vLine.setAttribute('y1', 20);
                vLine.setAttribute('x2', 20 + i * 23);
                vLine.setAttribute('y2', 460);
                vLine.setAttribute('stroke', '#000');
                vLine.setAttribute('stroke-width', '1');
                svg.appendChild(vLine);
                
                // Horizontal lines
                const hLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                hLine.setAttribute('x1', 20);
                hLine.setAttribute('y1', 20 + i * 23);
                hLine.setAttribute('x2', 460);
                hLine.setAttribute('y2', 20 + i * 23);
                hLine.setAttribute('stroke', '#000');
                hLine.setAttribute('stroke-width', '1');
                svg.appendChild(hLine);
            }}
            
            // Add star points
            const starPoints = [[3,3], [3,9], [3,15], [9,3], [9,9], [9,15], [15,3], [15,9], [15,15]];
            starPoints.forEach(([x, y]) => {{
                const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                circle.setAttribute('cx', 20 + x * 23);
                circle.setAttribute('cy', 20 + y * 23);
                circle.setAttribute('r', '3');
                circle.setAttribute('fill', '#000');
                svg.appendChild(circle);
            }});
            
            boardElement.appendChild(svg);
        }}

        function initGameInfo(position) {{
            if (!position.sgf_text) return;
            
            try {{
                // Extract game info from SGF
                const sgf = position.sgf_text;
                const pbMatch = sgf.match(/PB\\[([^\\]]*)\\]/);
                const pwMatch = sgf.match(/PW\\[([^\\]]*)\\]/);
                const dtMatch = sgf.match(/DT\\[([^\\]]*)\\]/);
                const reMatch = sgf.match(/RE\\[([^\\]]*)\\]/);
                
                const blackPlayer = pbMatch ? pbMatch[1] : 'Black';
                const whitePlayer = pwMatch ? pwMatch[1] : 'White';
                const gameDate = dtMatch ? dtMatch[1] : '';
                const result = reMatch ? reMatch[1] : '';
                
                document.getElementById('black-player').textContent = blackPlayer;
                document.getElementById('white-player').textContent = whitePlayer;
                document.getElementById('game-info').textContent = gameDate + (result ? ' • ' + result : '');
            }} catch (e) {{
                console.error('Error extracting game info:', e);
            }}
        }}

        function renderPosition(positionIndex) {{
            currentPosition = positionIndex;
            const position = positionsData[positionIndex];
            
            // Reinitialize board for new position
            initBoard();
            
            // Wait a bit for WGo.js to initialize, then add policy markers
            setTimeout(() => {{
                renderPolicyMarkers(position);
            }}, 100);
            
            // Update position info
            updatePositionInfo(position);
        }}

        function renderPolicyMarkers(position) {{
            if (!player || !player.board) {{
                // Fallback to simple board rendering
                renderSimpleMarkers(position);
                return;
            }}
            
            const board = player.board;
            
            // Remove existing policy markers
            if (board.objects) {{
                const objectsToRemove = board.objects.filter(obj => 
                    obj.type === 'LB' || obj.type === 'MA'
                );
                objectsToRemove.forEach(obj => board.removeObject(obj));
            }}
            
            // Add policy markers for each move
            position.moves.forEach((move, index) => {{
                if (move.coordinates.x !== null && move.coordinates.y !== null) {{
                    const x = move.coordinates.x;
                    const y = move.coordinates.y;
                    
                    // Create policy label
                    const winrateText = (move.kata_go_prob * 100).toFixed(0);
                    const labelObj = new WGo.LabelBoardObject(winrateText, x, y);
                    labelObj.isPolicyLabel = true;
                    board.addObject(labelObj);
                    
                    // Style the label
                    setTimeout(() => {{
                        const boardElement = document.getElementById('board');
                        const textElements = boardElement.querySelectorAll('text');
                        
                        textElements.forEach(textEl => {{
                            if (textEl.textContent === winrateText) {{
                                textEl.style.setProperty('fill', 'red', 'important');
                                textEl.style.setProperty('color', 'red', 'important');
                                textEl.style.setProperty('font-weight', 'bold', 'important');
                                textEl.style.setProperty('font-size', '0.8px', 'important');
                                textEl.classList.add('policy-label');
                            }}
                        }});
                    }}, 50);
                }}
            }});
            
            // Refresh the board
            if (board.redraw) {{
                board.redraw();
            }}
        }}

        function renderSimpleMarkers(position) {{
            const boardElement = document.getElementById('board');
            const svg = boardElement.querySelector('svg');
            if (!svg) return;
            
            // Clear existing markers
            const existingMarkers = svg.querySelectorAll('.move-marker');
            existingMarkers.forEach(marker => marker.remove());
            
            // Add move markers
            position.moves.forEach((move, index) => {{
                if (move.coordinates.x !== null && move.coordinates.y !== null) {{
                    const x = 20 + move.coordinates.x * 23;
                    const y = 20 + move.coordinates.y * 23;
                    
                    // Create move marker
                    const marker = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                    marker.setAttribute('cx', x);
                    marker.setAttribute('cy', y);
                    marker.setAttribute('r', '8');
                    marker.setAttribute('fill', index < 3 ? '#ff4444' : '#4444ff');
                    marker.setAttribute('stroke', '#fff');
                    marker.setAttribute('stroke-width', '2');
                    marker.classList.add('move-marker');
                    
                    // Add move number
                    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                    text.setAttribute('x', x);
                    text.setAttribute('y', y + 3);
                    text.setAttribute('text-anchor', 'middle');
                    text.setAttribute('font-size', '10');
                    text.setAttribute('font-weight', 'bold');
                    text.setAttribute('fill', '#fff');
                    text.textContent = index + 1;
                    text.classList.add('move-marker');
                    
                    svg.appendChild(marker);
                    svg.appendChild(text);
                }}
            }});
        }}

        function updatePositionInfo(position) {{
            const infoDiv = document.getElementById('position-info');
            
            let html = `<h3>Position ${{position.position_id + 1}} - Move Analysis</h3>`;
            
            if (position.game_id) {{
                html += `<p><strong>Game ID:</strong> ${{position.game_id}}</p>`;
            }}
            
            position.moves.forEach((move, index) => {{
                const isTopMove = index < 3;
                const moveClass = isTopMove ? 'move-info' : 'move-info';
                const borderColor = isTopMove ? '#ff4444' : '#4444ff';
                
                html += `
                    <div class="${{moveClass}}" style="border-left-color: ${{borderColor}};">
                        <div class="move-header">
                            Move ${{index + 1}}: ${{move.human_coord}} ${{isTopMove ? '(Top Move)' : ''}}
                        </div>
                        <div class="move-details">
                            <div class="detail-item">
                                <div class="detail-label">KataGo Prob</div>
                                <div class="detail-value">${{(move.kata_go_prob * 100).toFixed(1)}}%</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Model Prob</div>
                                <div class="detail-value">${{(move.model_prob * 100).toFixed(1)}}%</div>
                            </div>
                        </div>
                        <div class="concepts-list">
                            <strong>Active Concepts:</strong><br>
                            ${{move.active_concepts.map(concept => `
                                <span class="concept-item ${{concept.is_labeled ? 'labeled' : 'latent'}}">
                                    ${{concept.concept_name}}
                                    <span class="concept-activation">${{concept.activation.toFixed(2)}}</span>
                                </span>
                            `).join('')}}
                        </div>
                    </div>
                `;
            }});
            
            infoDiv.innerHTML = html;
        }}

        function gotoMove(idx) {{
            if (!player || !player.next || !player.previous) return;
            
            if (idx > currentMove) {{
                for (let i = currentMove; i < idx; i++) {{
                    player.next();
                }}
            }} else if (idx < currentMove) {{
                for (let i = currentMove; i > idx; i--) {{
                    player.previous();
                }}
            }}
            
            currentMove = idx;
            document.getElementById('move_idx').textContent = currentMove;
        }}

        // Event listeners
        document.getElementById('position-select').addEventListener('change', (e) => {{
            currentPosition = parseInt(e.target.value);
            currentMove = 0;
            renderPosition(currentPosition);
        }});

        document.getElementById('next').onclick = () => {{
            if (player && player.next) {{
                player.next();
                currentMove++;
                document.getElementById('move_idx').textContent = currentMove;
            }}
        }};

        document.getElementById('prev').onclick = () => {{
            if (player && player.previous) {{
                player.previous();
                currentMove--;
                document.getElementById('move_idx').textContent = currentMove;
            }}
        }};

        // Initialize
        renderPosition(0);
    </script>
</body>
</html>"""

    output_path.write_text(html_content, encoding='utf-8')
    print(f"Concept visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create HTML visualization of learned concepts")
    parser.add_argument("--model-path", type=Path, default=Path("cbm_output/concept_learning_cbm.pt"), 
                       help="Path to trained model")
    parser.add_argument("--slates-path", type=Path, default=Path("games/slates.jsonl"), 
                       help="Path to slates.jsonl")
    parser.add_argument("--output", type=Path, default=Path("cbm_output/concept_visualization.html"),
                       help="Output HTML file")
    parser.add_argument("--num-positions", type=int, default=10,
                       help="Number of positions to visualize")
    parser.add_argument("--concept-threshold", type=float, default=0.3,
                       help="Threshold for concept activation")
    parser.add_argument("--ontology", type=Path, 
                       default=Path(__file__).parent.parent.parent / "configs" / "ontology.yaml",
                       help="Path to ontology.yaml file")
    parser.add_argument("--games-dir", type=Path, 
                       default=Path(__file__).parent.parent.parent / "games",
                       help="Path to games directory containing SGF and policy files")
    
    args = parser.parse_args()
    
    # Load model
    print("Loading trained model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_with_concept_learning(args.model_path, device)
    print(f"Model loaded on {device}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = MoveCandidateDataset(
        slates_path=args.slates_path,
        num_labeled_concepts=0,
        num_latent_concepts=model.num_latent_concepts,
    )
    print(f"Dataset loaded: {len(dataset)} slates")
    
    # Load concept names from ontology
    print("Loading concept names from ontology...")
    concept_names = load_concept_names(args.ontology)
    print(f"Loaded {len(concept_names)} concept names from ontology")
    
    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate visualization
    print("Generating concept visualization...")
    create_concept_visualization_html(
        model, dataset, device, args.output, 
        args.num_positions, args.concept_threshold, concept_names, args.games_dir
    )
    
    print(f"Visualization complete! Open {args.output} in your browser to view the concepts.")


if __name__ == "__main__":
    main()
