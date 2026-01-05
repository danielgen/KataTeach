#!/usr/bin/env python3
"""
Simple script to visualize ownership values from moves.jsonl in HTML format.
"""

import json
import numpy as np
from pathlib import Path
import argparse


def load_moves_from_jsonl(jsonl_path: Path):
    """Load all moves from a JSONL file."""
    moves = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                moves.append(json.loads(line))
    return moves


def loc_to_xy(loc, size=19):
    """Convert board location to (x,y) coordinates."""
    if loc == 0:  # Pass
        return None, None
    return (loc % (size + 1)) - 1, (loc // (size + 1)) - 1


def normalize_ownership_to_player_frame(ownership: np.ndarray, ownership_frame_player: str, target_player: str) -> np.ndarray:
    """
    Normalize ownership to target player's perspective.
    
    This is consistent with snorkel_board_positions.py which expects ownership
    from the current player's perspective (positive = current player's territory).
    
    Args:
        ownership: Ownership map (19x19) from ownership_frame_player's perspective
        ownership_frame_player: 'b' (Black) or 'w' (White) - who the ownership is currently positive for
        target_player: 'b' (Black) or 'w' (White) - who we want ownership to be positive for
    
    Returns:
        Ownership map from target_player's perspective (positive = target_player territory)
    """
    if ownership_frame_player == target_player:
        return ownership
    else:
        return -ownership


def generate_html(moves, output_path: Path, normalize_ownership: bool = True):
    """Generate HTML visualization of ownership values."""
    
    # Convert moves data to JavaScript format
    moves_js = []
    for move in moves:
        move_num = move.get('move_number', 0)
        player = move.get('player', 'b')
        move_loc = move.get('move_loc', 0)
        
        # Get ownership and reshape to 19x19
        ownership_raw = move.get('ownership', [0.0] * 361)
        ownership = np.array(ownership_raw).reshape(19, 19)
        
        # Normalize ownership to current player's perspective if requested
        # NOTE: This assumes ownership in moves.jsonl is from KataGo's perspective
        # (Black's perspective: positive = Black territory). If your data is already
        # normalized to current player's perspective, disable normalization.
        # This normalization is consistent with snorkel_board_positions.py which expects
        # ownership from current player's perspective (positive = current player territory).
        if normalize_ownership:
            ownership_normalized = normalize_ownership_to_player_frame(
                ownership, 
                ownership_frame_player='b',  # KataGo outputs from Black's perspective
                target_player=player  # Current player - normalize to their perspective
            )
            ownership = ownership_normalized.tolist()
        else:
            ownership = ownership.tolist()
        
        # Get board state if available
        board_state = move.get('board_state', None)
        
        moves_js.append({
            'move_number': move_num,
            'player': 'Black' if player == 'b' else 'White',
            'move_loc': move_loc,
            'ownership': ownership,
            'board_state': board_state
        })
    
    # Generate move location string
    def get_move_string(move_loc):
        if move_loc == 0:
            return 'Pass'
        x, y = loc_to_xy(move_loc)
        if x is None or y is None:
            return f'Loc {move_loc}'
        return f"{chr(97 + x)}{chr(97 + y)}"
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Ownership Visualization</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        
        h1 {{
            text-align: center;
            color: #333;
        }}
        
        .controls {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        
        .move-info {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }}
        
        .move-slider {{
            width: 80%;
            margin: 10px 0;
        }}
        
        .control-buttons {{
            margin-top: 15px;
        }}
        
        .control-buttons button {{
            padding: 10px 20px;
            margin: 0 5px;
            font-size: 14px;
            cursor: pointer;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
        }}
        
        .control-buttons button:hover {{
            background: #45a049;
        }}
        
        .board-container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: flex;
            justify-content: center;
        }}
        
        .go-board {{
            position: relative;
            width: 400px;
            height: 400px;
            background: #DCB35C;
            border: 2px solid #8B4513;
            border-radius: 4px;
        }}
        
        .grid-line {{
            position: absolute;
            background: #000;
        }}
        
        .grid-line.horizontal {{
            width: 360px;
            height: 1px;
            left: 20px;
        }}
        
        .grid-line.vertical {{
            width: 1px;
            height: 360px;
            top: 20px;
        }}
        
        .stone {{
            position: absolute;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            z-index: 10;
        }}
        
        .stone.black {{
            background: #000;
        }}
        
        .stone.white {{
            background: #fff;
            border: 1px solid #000;
        }}
        
        .ownership-cell {{
            position: absolute;
            width: 20px;
            height: 20px;
            z-index: 5;
        }}
        
        .ownership-label {{
            position: absolute;
            font-size: 10px;
            font-weight: bold;
            z-index: 15;
            text-align: center;
            width: 20px;
            line-height: 20px;
        }}
        
        .legend {{
            margin-top: 20px;
            text-align: center;
            font-size: 14px;
        }}
        
        .legend-item {{
            display: inline-block;
            margin: 0 10px;
            padding: 5px 10px;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <h1>Ownership Visualization</h1>
    
    <div class="controls">
        <div class="move-info" id="moveInfo">Move 0</div>
        <input type="range" class="move-slider" id="moveSlider" min="0" max="{len(moves_js)-1}" value="0">
        <div class="control-buttons">
            <button onclick="previousMove()">← Previous</button>
            <button onclick="nextMove()">Next →</button>
            <button onclick="toggleAutoPlay()" id="autoPlayBtn">Auto Play</button>
        </div>
    </div>
    
    <div class="board-container">
        <div class="go-board" id="board"></div>
    </div>
    
    <div class="legend">
        <div class="legend-item" style="background: rgba(255,0,0,0.3);">
            <strong>Red:</strong> Positive ownership (Black territory)
        </div>
        <div class="legend-item" style="background: rgba(0,0,255,0.3);">
            <strong>Blue:</strong> Negative ownership (White territory)
        </div>
        <div class="legend-item">
            <strong>Values:</strong> Ownership × 10 (rounded)
        </div>
    </div>
    
    <script>
        const movesData = {json.dumps(moves_js)};
        let currentMove = 0;
        let autoPlayInterval = null;
        
        function getMoveString(moveLoc) {{
            if (moveLoc === 0) return 'Pass';
            const size = 19;
            const dy = size + 1;
            const x = (moveLoc % dy) - 1;
            const y = Math.floor(moveLoc / dy) - 1;
            if (x < 0 || x >= size || y < 0 || y >= size) return `Loc ${{moveLoc}}`;
            return String.fromCharCode(97 + x) + String.fromCharCode(97 + y);
        }}
        
        function updateDisplay() {{
            const move = movesData[currentMove];
            document.getElementById('moveInfo').textContent = 
                `Move ${{move.move_number}}: ${{move.player}} ${{getMoveString(move.move_loc)}}`;
            document.getElementById('moveSlider').value = currentMove;
            
            updateBoard(move);
        }}
        
        function drawGridLines(board) {{
            // Draw horizontal lines
            for (let i = 0; i < 19; i++) {{
                const line = document.createElement('div');
                line.className = 'grid-line horizontal';
                line.style.top = `${{20 + i * 20}}px`;
                board.appendChild(line);
            }}
            
            // Draw vertical lines
            for (let i = 0; i < 19; i++) {{
                const line = document.createElement('div');
                line.className = 'grid-line vertical';
                line.style.left = `${{20 + i * 20}}px`;
                board.appendChild(line);
            }}
            
            // Draw star points
            const starPoints = [3, 9, 15];
            for (const x of starPoints) {{
                for (const y of starPoints) {{
                    const star = document.createElement('div');
                    star.style.position = 'absolute';
                    star.style.width = '4px';
                    star.style.height = '4px';
                    star.style.background = '#000';
                    star.style.borderRadius = '50%';
                    star.style.left = `${{18 + x * 20}}px`;
                    star.style.top = `${{18 + y * 20}}px`;
                    star.style.zIndex = '5';
                    board.appendChild(star);
                }}
            }}
        }}
        
        function updateBoard(move) {{
            const board = document.getElementById('board');
            board.innerHTML = '';
            drawGridLines(board);
            
            const ownership = move.ownership;
            const boardState = move.board_state;
            
            // Find min/max ownership for normalization
            let minOwn = 1, maxOwn = -1;
            for (let y = 0; y < 19; y++) {{
                for (let x = 0; x < 19; x++) {{
                    const own = ownership[y][x];
                    if (own < minOwn) minOwn = own;
                    if (own > maxOwn) maxOwn = own;
                }}
            }}
            const range = Math.max(Math.abs(minOwn), Math.abs(maxOwn));
            
            // Draw ownership heatmap and stones
            for (let y = 0; y < 19; y++) {{
                for (let x = 0; x < 19; x++) {{
                    const pixelX = 20 + x * 20;
                    const pixelY = 20 + y * 20;
                    
                    const own = ownership[y][x];
                    
                    // Draw ownership background
                    if (Math.abs(own) > 0.01) {{
                        const cell = document.createElement('div');
                        cell.className = 'ownership-cell';
                        cell.style.left = `${{pixelX - 10}}px`;
                        cell.style.top = `${{pixelY - 10}}px`;
                        
                        // Color based on ownership value
                        const intensity = Math.abs(own) / range;
                        if (own > 0) {{
                            // Positive = Black territory (red)
                            cell.style.background = `rgba(255, 0, 0, ${{intensity * 0.5}})`;
                        }} else {{
                            // Negative = White territory (blue)
                            cell.style.background = `rgba(0, 0, 255, ${{intensity * 0.5}})`;
                        }}
                        board.appendChild(cell);
                    }}
                    
                    // Draw stones if board state is available
                    if (boardState) {{
                        const dy = 20;
                        const loc = (x + 1) + dy * (y + 1);
                        const stone = boardState[loc];
                        
                        if (stone === 1 || stone === -1) {{
                            const stoneEl = document.createElement('div');
                            stoneEl.className = `stone ${{stone === 1 ? 'black' : 'white'}}`;
                            stoneEl.style.left = `${{pixelX - 9}}px`;
                            stoneEl.style.top = `${{pixelY - 9}}px`;
                            board.appendChild(stoneEl);
                        }}
                    }}
                    
                    // Draw ownership value label
                    if (Math.abs(own) > 0.1) {{
                        const label = document.createElement('div');
                        label.className = 'ownership-label';
                        label.textContent = Math.round(Math.abs(own) * 10);
                        label.style.left = `${{pixelX - 10}}px`;
                        label.style.top = `${{pixelY - 10}}px`;
                        label.style.color = own > 0 ? '#000' : '#fff';
                        label.style.textShadow = own > 0 
                            ? '1px 1px 1px rgba(255,255,255,0.8)' 
                            : '1px 1px 1px rgba(0,0,0,0.8)';
                        board.appendChild(label);
                    }}
                }}
            }}
            
            // Highlight the move location
            if (move.move_loc && move.move_loc !== 0) {{
                const dy = 20;
                const x = (move.move_loc % dy) - 1;
                const y = Math.floor(move.move_loc / dy) - 1;
                if (x >= 0 && x < 19 && y >= 0 && y < 19) {{
                    const pixelX = 20 + x * 20;
                    const pixelY = 20 + y * 20;
                    
                    const highlight = document.createElement('div');
                    highlight.style.position = 'absolute';
                    highlight.style.width = '24px';
                    highlight.style.height = '24px';
                    highlight.style.border = '3px solid yellow';
                    highlight.style.borderRadius = '50%';
                    highlight.style.left = `${{pixelX - 12}}px`;
                    highlight.style.top = `${{pixelY - 12}}px`;
                    highlight.style.zIndex = '20';
                    highlight.style.pointerEvents = 'none';
                    board.appendChild(highlight);
                }}
            }}
        }}
        
        function previousMove() {{
            if (currentMove > 0) {{
                currentMove--;
                updateDisplay();
            }}
        }}
        
        function nextMove() {{
            if (currentMove < movesData.length - 1) {{
                currentMove++;
                updateDisplay();
            }}
        }}
        
        function toggleAutoPlay() {{
            if (autoPlayInterval) {{
                clearInterval(autoPlayInterval);
                autoPlayInterval = null;
                document.getElementById('autoPlayBtn').textContent = 'Auto Play';
            }} else {{
                autoPlayInterval = setInterval(() => {{
                    if (currentMove < movesData.length - 1) {{
                        currentMove++;
                        updateDisplay();
                    }} else {{
                        toggleAutoPlay();
                    }}
                }}, 1000);
                document.getElementById('autoPlayBtn').textContent = 'Stop';
            }}
        }}
        
        // Slider control
        document.getElementById('moveSlider').addEventListener('input', (e) => {{
            currentMove = parseInt(e.target.value);
            updateDisplay();
        }});
        
        // Initialize
        updateDisplay();
    </script>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"HTML visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize ownership values from moves.jsonl')
    parser.add_argument('jsonl_path', type=Path, help='Path to moves.jsonl file')
    parser.add_argument('-o', '--output', type=Path, default=None, 
                       help='Output HTML file path (default: ownership_visualization.html)')
    parser.add_argument('--no-normalize', action='store_true',
                       help='Disable ownership normalization (use if ownership is already from current player perspective)')
    
    args = parser.parse_args()
    
    if not args.jsonl_path.exists():
        print(f"Error: File not found: {args.jsonl_path}")
        return
    
    output_path = args.output or args.jsonl_path.parent / 'ownership_visualization.html'
    
    print(f"Loading moves from {args.jsonl_path}...")
    moves = load_moves_from_jsonl(args.jsonl_path)
    print(f"Loaded {len(moves)} moves")
    
    normalize = not args.no_normalize
    if normalize:
        print("Normalizing ownership to current player's perspective (consistent with snorkel_board_positions.py)")
    else:
        print("Using ownership as-is (no normalization)")
    
    print(f"Generating HTML visualization...")
    generate_html(moves, output_path, normalize_ownership=normalize)
    print(f"Done! Open {output_path} in your browser to view the visualization.")


if __name__ == '__main__':
    main()

