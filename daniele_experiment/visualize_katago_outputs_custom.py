#!/usr/bin/env python3
"""
KataGo Outputs Visualization with Custom Board Rendering

This script plays a short game and generates an HTML visualization
showing all KataGo model outputs at each move using custom board rendering.
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np

# Add the python directory to the path so we can import KataGo modules
sys.path.append(str(Path(__file__).parent.parent / "python"))

from board import Board
from gamestate import GameState
from model_pytorch import Model
from load_model import load_model
import torch

def get_device():
    """Get the appropriate device for model inference."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def convert_numpy_to_python(obj):
    """Convert numpy arrays and other non-JSON-serializable objects to Python equivalents."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_to_python(item) for item in obj)
    else:
        return obj

def play_short_game(model, max_moves=10):
    """Play a short game and capture KataGo outputs at each step."""
    print(f"Playing a short game with max {max_moves} moves...")
    
    # Initialize game state
    board_size = 19
    game_state = GameState(board_size, GameState.RULES_TT)
    
    game_data = []
    moves = []
    
    # Add initial position
    initial_outputs = game_state.get_model_outputs(model)
    converted_initial_outputs = convert_numpy_to_python(initial_outputs)
    game_data.append({
        "move_number": 0,
        "player": "Initial",
        "last_move": None,
        "board_state": [0] * game_state.board.arrsize,  # Empty board with full size
        **converted_initial_outputs
    })
    
    # Play moves
    for move_num in range(1, max_moves + 1):
        # Get current player
        current_player = game_state.board.pla
        player_str = "Black" if current_player == Board.BLACK else "White"
        
        # Get model outputs for current position
        outputs = game_state.get_model_outputs(model)
        
        # Always use the main policy (policy0) for move selection
        # policy0 = current player's move distribution
        # policy1 = opponent's predicted next move distribution (not used for selection)
        moves_and_probs = outputs["moves_and_probs0"]
        
        if not moves_and_probs:
            print(f"No legal moves available for {player_str} at move {move_num}")
            break
        
        # Use sampling to get more realistic moves (like in play_games.py)
        import random
        
        # Find the best probability
        best_prob = max(prob for _, prob in moves_and_probs)
        
        # Find all moves within 1% of the best move
        prob_threshold = 0.01
        candidate_moves = []
        for move, prob in moves_and_probs:
            if prob >= best_prob - prob_threshold:
                candidate_moves.append((move, prob))
        
        # If only one candidate, return it (no sampling)
        if len(candidate_moves) == 1:
            best_move, best_prob = candidate_moves[0]
        else:
            # Sample from candidates using their probabilities as weights
            candidate_moves_list, candidate_probs = zip(*candidate_moves)
            
            # Normalize probabilities within the candidate set
            total_prob = sum(candidate_probs)
            normalized_probs = [p / total_prob for p in candidate_probs]
            
            # Sample based on normalized probabilities
            selected_idx = random.choices(range(len(candidate_moves_list)), weights=normalized_probs)[0]
            best_move = candidate_moves_list[selected_idx]
            best_prob = candidate_probs[selected_idx]
        
        # Debug: print top moves
        print(f"  Top 5 moves: {moves_and_probs[:5]}")
        print(f"  Selected move: {best_move} (prob: {best_prob:.6f})")
        
        # Play the move
        game_state.play(current_player, best_move)
        moves.append((current_player, best_move))
        
        # Get board state AFTER playing the move
        # Use the full board size including walls and padding (arrsize = (19+1)*(19+2)+1 = 420)
        board_state = [0] * game_state.board.arrsize
        for i in range(game_state.board.arrsize):
            if game_state.board.board[i] == 1:
                board_state[i] = 1  # Black
            elif game_state.board.board[i] == 2:
                board_state[i] = -1  # White
        
        # Get outputs after the move
        post_move_outputs = game_state.get_model_outputs(model)
        
        # Convert numpy arrays to Python objects for JSON serialization
        converted_outputs = convert_numpy_to_python(post_move_outputs)
        
        # Store game data
        game_data.append({
            "move_number": move_num,
            "player": player_str,
            "last_move": (current_player, best_move),
            "board_state": board_state,
            **converted_outputs
        })
        
        # Print move information
        if best_move == Board.PASS_LOC:
            print(f"Move {move_num}: {player_str} passes (prob: {best_prob:.3f})")
        else:
            # Convert to SGF coordinates for display
            x = game_state.board.loc_x(best_move)
            y = game_state.board.loc_y(best_move)
            sgf_coord = f"{chr(ord('a') + x)}{chr(ord('a') + y)}"
            print(f"Move {move_num}: {player_str} plays at {best_move} ({sgf_coord}) (prob: {best_prob:.3f})")
        
        # Debug: print board state
        black_count = sum(1 for x in board_state if x == 1)
        white_count = sum(1 for x in board_state if x == -1)
        print(f"  Board state: {black_count} black, {white_count} white stones")
        
        # Debug: check what values are actually in the board
        unique_values = set(game_state.board.board)
        print(f"  Board values: {unique_values}")
        
        # Debug: check the specific location that was just played
        if best_move != Board.PASS_LOC:
            print(f"  Location {best_move} has value: {game_state.board.board[best_move]}")
        
        # Debug: print all non-zero board positions
        non_zero_positions = []
        for i in range(game_state.board.arrsize):
            if game_state.board.board[i] != 0:
                x = game_state.board.loc_x(i)
                y = game_state.board.loc_y(i)
                if 0 <= x < 19 and 0 <= y < 19:  # Only show valid board positions
                    sgf_coord = f"{chr(ord('a') + x)}{chr(ord('a') + y)}"
                    non_zero_positions.append(f"{i}({sgf_coord})={game_state.board.board[i]}")
        print(f"  Non-zero positions: {non_zero_positions}")
        
        # Debug: check bottom row specifically
        bottom_row_stones = []
        for x in range(19):
            loc = game_state.board.loc(x, 0)  # Bottom row (y=0)
            if game_state.board.board[loc] != 0:
                bottom_row_stones.append(f"x={x}, loc={loc}, value={game_state.board.board[loc]}")
        if bottom_row_stones:
            print(f"  Bottom row stones: {bottom_row_stones}")
        else:
            print("  No stones on bottom row")
    
    # Generate SGF content
    sgf_content = generate_sgf(moves)
    
    return game_data, sgf_content

def generate_sgf(moves):
    """Generate SGF content from moves."""
    sgf_lines = [
        "(;FF[4]CA[UTF-8]AP[KataGo:1.0]ST[2]",
        "RU[Chinese]SZ[19]KM[7.5]",
        "PW[White]PB[Black]"
    ]
    
    # Create a temporary board for coordinate conversion
    temp_board = Board(19)
    
    for player, move in moves:
        if move == Board.PASS_LOC:
            move_str = "[]"
        else:
            # Convert to SGF format
            x = temp_board.loc_x(move)
            y = temp_board.loc_y(move)
            move_str = f"[{chr(ord('a') + x)}{chr(ord('a') + y)}]"
        
        color = "B" if player == Board.BLACK else "W"
        sgf_lines.append(f";{color}{move_str}")
    
    sgf_lines.append(")")
    return "\n".join(sgf_lines)


def generate_html_visualization(game_data, sgf_content, output_file):
    """Generate HTML visualization with custom board rendering."""
    
    # Convert game data to JSON
    game_data_js = json.dumps(game_data)
    sgf_js = json.dumps(sgf_content)
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KataGo Outputs Visualization</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            padding-top: 120px;
            background-color: #f5f5f5;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        h1 {{
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }}
        
        .controls {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            text-align: center;
        }}
        
        .move-info {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }}
        
        .move-slider {{
            width: 100%;
            max-width: 600px;
            margin: 10px 0;
        }}
        
        .control-buttons {{
            margin-top: 15px;
        }}
        
        .control-buttons button {{
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            margin: 0 5px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }}
        
        .control-buttons button:hover {{
            background: #0056b3;
        }}
        
        .control-buttons button:disabled {{
            background: #ccc;
            cursor: not-allowed;
        }}
        
        .boards-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}
        
        .board-section {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 20px;
        }}
        
        .board-section h3 {{
            margin-top: 0;
            color: #333;
            text-align: center;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }}
        
        .go-board {{
            width: 380px;
            height: 380px;
            margin: 0 auto;
            border: 2px solid #8B4513;
            border-radius: 4px;
            background: #DEB887;
            position: relative;
        }}
        
        
        .board-cell {{
            position: absolute;
            width: 20px;
            height: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: bold;
        }}
        
        .stone {{
            width: 18px;
            height: 18px;
            border-radius: 50%;
            border: 1px solid #333;
        }}
        
        .stone.black {{
            background: #000;
        }}
        
        .stone.white {{
            background: #fff;
            border: 1px solid #333;
        }}
        
        .label {{
            position: absolute;
            font-size: 10px;
            font-weight: bold;
            text-shadow: 1px 1px 1px rgba(255,255,255,0.8);
            z-index: 10;
        }}
        
        .info-panel {{
            margin-top: 15px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 4px;
            font-size: 14px;
        }}
        
        .heatmap-legend {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            justify-content: center;
        }}
        
        .heatmap-legend span {{
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            color: white;
            font-weight: bold;
        }}
        
        .value-display {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }}
        
        .value-item {{
            background: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            text-align: center;
        }}
        
        .value-item strong {{
            display: block;
            margin-bottom: 5px;
            color: #333;
        }}
        
        .sticky-value-section {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 1000;
            background: white;
            border-radius: 0 0 8px 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 15px 20px;
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        #stickyValueContainer {{
            position: relative;
            width: 100%;
        }}
        
        .sticky-value-section h3 {{
            margin: 0 0 10px 0;
            text-align: center;
            font-size: 16px;
            color: #333;
        }}
        
        .sticky-value-display {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }}
        
        .sticky-value-item {{
            background: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            text-align: center;
        }}
        
        .sticky-value-item strong {{
            display: block;
            margin-bottom: 5px;
            color: #333;
        }}
        
        .policy-moves {{
            max-height: 300px;
            overflow-y: auto;
        }}
        
        .move-item {{
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #eee;
        }}
        
        .move-item:last-child {{
            border-bottom: none;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>KataGo Model Outputs Visualization</h1>
        <div class="controls">
            <div class="move-info" id="moveInfo">Move 0: Initial Position</div>
            <input type="range" class="move-slider" id="moveSlider" min="0" max="{len(game_data)-1}" value="0">
            <div class="control-buttons">
                <button onclick="previousMove()">← Previous</button>
                <button onclick="nextMove()">Next →</button>
                <button onclick="toggleAutoPlay()" id="autoPlayBtn">Auto Play</button>
                <button onclick="resetToStart()">Reset</button>
            </div>
        </div>
        <div id="stickyValueContainer">
            <!-- Sticky value section will be generated by JavaScript -->
        </div>
        <div class="boards-container" id="boardsContainer">
            <!-- Boards will be generated by JavaScript -->
        </div>
    </div>
    
    <script>
        const gameData = {game_data_js};
        let currentMove = 0;
        let autoPlayInterval = null;
        
        function updateDisplay() {{
            const data = gameData[currentMove];
            document.getElementById('moveInfo').textContent = 
                `Move ${{data.move_number}}: ${{data.player}} ${{data.last_move ? getMoveString(data.last_move) : 'Initial Position'}}`;
            document.getElementById('moveSlider').value = currentMove;
            
            updateBoards(data);
        }}
        
        function getMoveString(move) {{
            if (!move) return '';
            if (move[1] === 361) return 'Pass';
            // Convert KataGo location to x,y coordinates using the same logic as KataGo's loc_x/loc_y
            const dy = 20; // size + 1 = 19 + 1 = 20
            const x = (move[1] % dy) - 1;
            const y = Math.floor(move[1] / dy) - 1;
            return String.fromCharCode(97 + x) + String.fromCharCode(97 + y);
        }}
        
        function getMoveLocation(move) {{
            if (!move) return null;
            if (move[1] === 361) return null;
            // Convert KataGo location to x,y coordinates using the same logic as KataGo's loc_x/loc_y
            const dy = 20; // size + 1 = 19 + 1 = 20
            const x = (move[1] % dy) - 1;
            const y = Math.floor(move[1] / dy) - 1;
            return {{x: x, y: y}};
        }}
        
        function initializeBoards() {{
            const container = document.getElementById('boardsContainer');
            container.innerHTML = '';
            
            // Board state
            addBoardSection('Board State', 'board_state');
            
            // Policy outputs
            addBoardSection('Policy 0 (Current Player)', 'policy0');
            addBoardSection('Policy 1 (Next Player)', 'policy1');
            
            // Ownership
            addBoardSection('Ownership', 'ownership');
            
            // Scoring
            addBoardSection('Scoring', 'scoring');
            
            // Future position
            addBoardSection('Future Position 0', 'futurepos0');
            addBoardSection('Future Position 1', 'futurepos1');
            
            // Seki
            addBoardSection('Seki', 'seki');
            
            // Policy moves
            addPolicyMovesSection();
        }}
        
        function addBoardSection(title, type) {{
            const container = document.getElementById('boardsContainer');
            const section = document.createElement('div');
            section.className = 'board-section';
            
            section.innerHTML = `
                <h3>${{title}}</h3>
                <div class="go-board" id="board-${{type}}"></div>
                <div class="info-panel" id="info-${{type}}"></div>
            `;
            
            container.appendChild(section);
        }}
        
        function addValueSection() {{
            const container = document.getElementById('stickyValueContainer');
            const section = document.createElement('div');
            section.className = 'sticky-value-section';
            section.id = 'value-section';
            
            section.innerHTML = `
                <h3>Value & Score Predictions</h3>
                <div class="sticky-value-display">
                    <div class="sticky-value-item">
                        <strong>Current Player</strong>
                        <span id="current-player">-</span>
                    </div>
                    <div class="sticky-value-item">
                        <strong>Win Rate</strong>
                        <span id="value-winrate">-</span>%
                    </div>
                    <div class="sticky-value-item">
                        <strong>Score Mean</strong>
                        <span id="score-mean">-</span>
                    </div>
                    <div class="sticky-value-item">
                        <strong>Lead</strong>
                        <span id="lead">-</span>
                    </div>
                    <div class="sticky-value-item">
                        <strong>Score StdDev</strong>
                        <span id="score-stdev">-</span>
                    </div>
                    <div class="sticky-value-item">
                        <strong>Variance Time</strong>
                        <span id="vtime">-</span>
                    </div>
                    <div class="sticky-value-item">
                        <strong>Value Error</strong>
                        <span id="estv">-</span>
                    </div>
                    <div class="sticky-value-item">
                        <strong>TD Value (Long)</strong>
                        <span id="value-td-long">-</span>%
                    </div>
                    <div class="sticky-value-item">
                        <strong>TD Value (Mid)</strong>
                        <span id="value-td-mid">-</span>%
                    </div>
                    <div class="sticky-value-item">
                        <strong>TD Value (Short)</strong>
                        <span id="value-td-short">-</span>%
                    </div>
                </div>
            `;
            
            container.appendChild(section);
        }}
        
        function addPolicyMovesSection() {{
            const container = document.getElementById('boardsContainer');
            const section = document.createElement('div');
            section.className = 'board-section';
            section.id = 'policy-moves-section';
            
            section.innerHTML = `
                <h3>Top Policy Moves</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                    <div>
                        <h4>Policy 0 (Current Player)</h4>
                        <div class="policy-moves" id="policy-moves-0">
                            <!-- Will be populated by updatePolicyMoves -->
                        </div>
                    </div>
                    <div>
                        <h4>Policy 1 (Next Player)</h4>
                        <div class="policy-moves" id="policy-moves-1">
                            <!-- Will be populated by updatePolicyMoves -->
                        </div>
                    </div>
                </div>
            `;
            
            container.appendChild(section);
        }}
        
        function updateBoards(data) {{
            // Update each board
            updateBoardState(data);
            updatePolicy(data, 'policy0');
            updatePolicy(data, 'policy1');
            updateOwnership(data);
            updateScoring(data);
            updateFuturePos(data, 'futurepos0', 0);
            updateFuturePos(data, 'futurepos1', 1);
            updateSeki(data);
            
            // Update value and policy info
            updateValueInfo(data);
            updatePolicyMoves(data);
        }}
        
        function drawGridLines(board) {{
            // Draw horizontal lines
            for (let i = 0; i < 19; i++) {{
                const line = document.createElement('div');
                line.style.position = 'absolute';
                line.style.left = '10px';
                line.style.right = '10px';
                line.style.height = '1px';
                line.style.top = `${{10 + i * 20}}px`;
                line.style.backgroundColor = '#8B4513';
                line.style.zIndex = '1';
                board.appendChild(line);
            }}
            
            // Draw vertical lines
            for (let i = 0; i < 19; i++) {{
                const line = document.createElement('div');
                line.style.position = 'absolute';
                line.style.top = '10px';
                line.style.bottom = '10px';
                line.style.width = '1px';
                line.style.left = `${{10 + i * 20}}px`;
                line.style.backgroundColor = '#8B4513';
                line.style.zIndex = '1';
                board.appendChild(line);
            }}
        }}

        function updateBoardState(data) {{
            const board = document.getElementById('board-board_state');
            if (!board) return;
            
            board.innerHTML = '';
            drawGridLines(board);
            
            // Create stones and labels at proper Go board coordinates
            for (let y = 0; y < 19; y++) {{
                for (let x = 0; x < 19; x++) {{
                    // Convert x,y to KataGo location using the same logic as KataGo's loc() function
                    const dy = 20; // size + 1 = 19 + 1 = 20
                    const loc = (x + 1) + dy * (y + 1);
                    const stone = data.board_state[loc] || 0; // Default to 0 if out of bounds
                    
                    // Debug: log bottom row stones
                    if (y === 0 && stone !== 0) {{
                        console.log(`Bottom row stone at (x=${{x}}, y=${{y}}), loc=${{loc}}, stone=${{stone}}`);
                    }}
                    
                    // Calculate pixel position (Go board coordinates)
                    const pixelX = 10 + x * 20;  // 10px margin + 20px spacing
                    const pixelY = 10 + y * 20;  // 10px margin + 20px spacing
                    
                    if (stone === 1 || stone === -1) {{
                        const stoneEl = document.createElement('div');
                        stoneEl.className = `stone ${{stone === 1 ? 'black' : 'white'}}`;
                        stoneEl.style.position = 'absolute';
                        stoneEl.style.left = `${{pixelX - 9}}px`;  // Center the stone (18px wide)
                        stoneEl.style.top = `${{pixelY - 9}}px`;   // Center the stone (18px high)
                        stoneEl.style.zIndex = '10';
                        board.appendChild(stoneEl);
                    }}
                    
                    // Add move number for last move
                    if (data.last_move) {{
                        const moveLoc = getMoveLocation(data.last_move);
                        if (moveLoc && moveLoc.x === x && moveLoc.y === y) {{
                            const label = document.createElement('div');
                            label.className = 'label';
                            label.textContent = data.move_number;
                            label.style.position = 'absolute';
                            label.style.left = `${{pixelX - 5}}px`;
                            label.style.top = `${{pixelY - 5}}px`;
                            label.style.color = 'red';
                            label.style.zIndex = '15';
                            board.appendChild(label);
                        }}
                    }}
                }}
            }}
        }}
        
        function updatePolicy(data, type) {{
            const board = document.getElementById(`board-${{type}}`);
            if (!board || !data[type]) return;
            
            board.innerHTML = '';
            drawGridLines(board);
            
            const policy = data[type];
            const maxProb = Math.max(...policy);
            const threshold = maxProb * 0.05;
            
            // Create stones and policy values at proper Go board coordinates
            for (let y = 0; y < 19; y++) {{
                for (let x = 0; x < 19; x++) {{
                    // Policy arrays are 1D with simple row-major indexing: y * 19 + x
                    const policyIndex = y * 19 + x;
                    const prob = policy[policyIndex];
                    
                    // Convert x,y to KataGo location for board state lookup
                    const dy = 20; // size + 1 = 19 + 1 = 20
                    const loc = (x + 1) + dy * (y + 1);
                    
                    // Calculate pixel position (Go board coordinates)
                    const pixelX = 10 + x * 20;  // 10px margin + 20px spacing
                    const pixelY = 10 + y * 20;  // 10px margin + 20px spacing
                    
                    // Add stones
                    const stone = data.board_state[loc];
                    if (stone === 1 || stone === -1) {{
                        const stoneEl = document.createElement('div');
                        stoneEl.className = `stone ${{stone === 1 ? 'black' : 'white'}}`;
                        stoneEl.style.position = 'absolute';
                        stoneEl.style.left = `${{pixelX - 9}}px`;  // Center the stone (18px wide)
                        stoneEl.style.top = `${{pixelY - 9}}px`;   // Center the stone (18px high)
                        stoneEl.style.zIndex = '10';
                        board.appendChild(stoneEl);
                    }}
                    
                    // Add policy probability
                    if (prob > threshold) {{
                        const label = document.createElement('div');
                        label.className = 'label';
                        label.textContent = (prob * 100).toFixed(1);
                        label.style.position = 'absolute';
                        label.style.left = `${{pixelX - 5}}px`;
                        label.style.top = `${{pixelY - 5}}px`;
                        label.style.color = 'red';
                        label.style.zIndex = '15';
                        board.appendChild(label);
                    }}
                }}
            }}
            
            // Show pass probability
            const passProb = policy[361];
            if (passProb > threshold) {{
                const info = document.getElementById(`info-${{type}}`);
                if (info) {{
                    info.innerHTML = `Pass: ${{(passProb * 100).toFixed(1)}}%`;
                }}
            }}
        }}
        
        function updateOwnership(data) {{
            const board = document.getElementById('board-ownership');
            if (!board || !data.ownership) return;
            
            board.innerHTML = '';
            drawGridLines(board);
            
            // Create stones and ownership values at proper Go board coordinates
            for (let y = 0; y < 19; y++) {{
                for (let x = 0; x < 19; x++) {{
                    // Ownership is 2D array with [y][x] indexing
                    const ownership = data.ownership[0][y][x];
                    
                    // Calculate pixel position (Go board coordinates)
                    const pixelX = 10 + x * 20;  // 10px margin + 20px spacing
                    const pixelY = 10 + y * 20;  // 10px margin + 20px spacing
                    
                    // Add stones
                    // Convert x,y to KataGo location for board state lookup
                    const dy = 20; // size + 1 = 19 + 1 = 20
                    const loc = (x + 1) + dy * (y + 1);
                    const stone = data.board_state[loc] || 0; // Default to 0 if out of bounds
                    if (stone === 1 || stone === -1) {{
                        const stoneEl = document.createElement('div');
                        stoneEl.className = `stone ${{stone === 1 ? 'black' : 'white'}}`;
                        stoneEl.style.position = 'absolute';
                        stoneEl.style.left = `${{pixelX - 9}}px`;  // Center the stone (18px wide)
                        stoneEl.style.top = `${{pixelY - 9}}px`;   // Center the stone (18px high)
                        stoneEl.style.zIndex = '10';
                        board.appendChild(stoneEl);
                    }}
                    
                    // Add ownership
                    if (Math.abs(ownership) > 0.1) {{
                        const label = document.createElement('div');
                        label.className = 'label';
                        label.textContent = Math.abs(ownership).toFixed(1);
                        label.style.position = 'absolute';
                        label.style.left = `${{pixelX - 5}}px`;
                        label.style.top = `${{pixelY - 5}}px`;
                        // Color based on who the value is good for:
                        // Black text = good for Black (positive ownership)
                        // White text = good for White (negative ownership)
                        // KataGo flips ownership based on current player, so we need to flip it back
                        const isBlackPlayer = data.player === 'Black';
                        const adjustedOwnership = isBlackPlayer ? ownership : -ownership;
                        const isGoodForBlack = adjustedOwnership > 0;
                        const shouldUseBlackText = isGoodForBlack;
                        
                        label.style.color = shouldUseBlackText ? 'yellow' : 'grey';
                        label.style.textShadow = shouldUseBlackText ? '1px 1px 1px rgba(255,255,255,0.8)' : '1px 1px 1px rgba(0,0,0,0.8)';
                        label.style.zIndex = '15';
                        board.appendChild(label);
                    }}
                }}
            }}
            
            const info = document.getElementById('info-ownership');
            if (info) {{
                info.innerHTML = `
                    <div class="heatmap-legend">
                        <span style="color: yellow; background: rgba(255,255,255,0.8);">Yellow Text: Good for Black</span>
                        <span style="color: grey; background: rgba(0,0,0,0.8);">Grey Text: Good for White</span>
                    </div>
                `;
            }}
        }}
        
        function updateScoring(data) {{
            const board = document.getElementById('board-scoring');
            if (!board || !data.scoring) return;
            
            board.innerHTML = '';
            drawGridLines(board);
            
            // Create stones and scoring values at proper Go board coordinates
            for (let y = 0; y < 19; y++) {{
                for (let x = 0; x < 19; x++) {{
                    // Scoring is 2D array with [y][x] indexing
                    const scoring = data.scoring[0][y][x];
                    
                    // Calculate pixel position (Go board coordinates)
                    const pixelX = 10 + x * 20;  // 10px margin + 20px spacing
                    const pixelY = 10 + y * 20;  // 10px margin + 20px spacing
                    
                    // Add stones
                    // Convert x,y to KataGo location for board state lookup
                    const dy = 20; // size + 1 = 19 + 1 = 20
                    const loc = (x + 1) + dy * (y + 1);
                    const stone = data.board_state[loc] || 0; // Default to 0 if out of bounds
                    if (stone === 1 || stone === -1) {{
                        const stoneEl = document.createElement('div');
                        stoneEl.className = `stone ${{stone === 1 ? 'black' : 'white'}}`;
                        stoneEl.style.position = 'absolute';
                        stoneEl.style.left = `${{pixelX - 9}}px`;  // Center the stone (18px wide)
                        stoneEl.style.top = `${{pixelY - 9}}px`;   // Center the stone (18px high)
                        stoneEl.style.zIndex = '10';
                        board.appendChild(stoneEl);
                    }}
                    
                    // Add scoring
                    if (Math.abs(scoring) > 0.1) {{
                        const label = document.createElement('div');
                        label.className = 'label';
                        label.textContent = Math.abs(scoring).toFixed(1);
                        label.style.position = 'absolute';
                        label.style.left = `${{pixelX - 5}}px`;
                        label.style.top = `${{pixelY - 5}}px`;
                        // Color based on who the value is good for:
                        // Black text = good for Black (positive scoring)
                        // White text = good for White (negative scoring)
                        // KataGo flips scoring based on current player, so we need to flip it back
                        const isBlackPlayer = data.player === 'Black';
                        const adjustedScoring = isBlackPlayer ? scoring : -scoring;
                        const isGoodForBlack = adjustedScoring > 0;
                        const shouldUseBlackText = isGoodForBlack;
                        label.style.color = shouldUseBlackText ? 'yellow' : 'grey';
                        label.style.textShadow = shouldUseBlackText ? '1px 1px 1px rgba(255,255,255,0.8)' : '1px 1px 1px rgba(0,0,0,0.8)';
                        label.style.zIndex = '15';
                        board.appendChild(label);
                    }}
                }}
            }}
            
            const info = document.getElementById('info-scoring');
            if (info) {{
                info.innerHTML = `
                    <div class="heatmap-legend">
                        <span style="color: yellow; background: rgba(255,255,255,0.8);">Yellow Text: Good for Black</span>
                        <span style="color: grey; background: rgba(0,0,0,0.8);">Grey Text: Good for White</span>
                    </div>
                `;
            }}
        }}
        
        function updateFuturePos(data, type, channel) {{
            const board = document.getElementById(`board-${{type}}`);
            if (!board || !data.futurepos) return;
            
            board.innerHTML = '';
            drawGridLines(board);
            
            // Create stones and future position values at proper Go board coordinates
            for (let y = 0; y < 19; y++) {{
                for (let x = 0; x < 19; x++) {{
                    // Future position is 2D array with [y][x] indexing
                    const futurepos = data.futurepos[channel][y][x];
                    
                    // Calculate pixel position (Go board coordinates)
                    const pixelX = 10 + x * 20;  // 10px margin + 20px spacing
                    const pixelY = 10 + y * 20;  // 10px margin + 20px spacing
                    
                    // Add stones
                    // Convert x,y to KataGo location for board state lookup
                    const dy = 20; // size + 1 = 19 + 1 = 20
                    const loc = (x + 1) + dy * (y + 1);
                    const stone = data.board_state[loc] || 0; // Default to 0 if out of bounds
                    if (stone === 1 || stone === -1) {{
                        const stoneEl = document.createElement('div');
                        stoneEl.className = `stone ${{stone === 1 ? 'black' : 'white'}}`;
                        stoneEl.style.position = 'absolute';
                        stoneEl.style.left = `${{pixelX - 9}}px`;  // Center the stone (18px wide)
                        stoneEl.style.top = `${{pixelY - 9}}px`;   // Center the stone (18px high)
                        stoneEl.style.zIndex = '10';
                        board.appendChild(stoneEl);
                    }}
                    
                    // Add future position
                    if (Math.abs(futurepos) > 0.1) {{
                        const label = document.createElement('div');
                        label.className = 'label';
                        label.textContent = Math.abs(futurepos).toFixed(1);
                        label.style.position = 'absolute';
                        label.style.left = `${{pixelX - 5}}px`;
                        label.style.top = `${{pixelY - 5}}px`;
                        // Color based on who the value is good for:
                        // Black text = good for Black (positive future position)
                        // White text = good for White (negative future position)
                        // KataGo flips future position based on current player, so we need to flip it back
                        const isBlackPlayer = data.player === 'Black';
                        const adjustedFuturepos = isBlackPlayer ? futurepos : -futurepos;
                        const isGoodForBlack = adjustedFuturepos > 0;
                        const shouldUseBlackText = isGoodForBlack;
                        
                        label.style.color = shouldUseBlackText ? 'yellow' : 'grey';
                        label.style.textShadow = shouldUseBlackText ? '1px 1px 1px rgba(255,255,255,0.8)' : '1px 1px 1px rgba(0,0,0,0.8)';
                        label.style.zIndex = '15';
                        board.appendChild(label);
                    }}
                }}
            }}
            
            const info = document.getElementById(`info-${{type}}`);
            if (info) {{
                info.innerHTML = `
                    <div class="heatmap-legend">
                        <span style="color: yellow; background: rgba(255,255,255,0.8);">Yellow Text: Good for Black</span>
                        <span style="color: grey; background: rgba(0,0,0,0.8);">Grey Text: Good for White</span>
                    </div>
                `;
            }}
        }}
        
        function updateSeki(data) {{
            const board = document.getElementById('board-seki');
            if (!board || !data.seki) return;
            
            board.innerHTML = '';
            drawGridLines(board);
            
            // Create stones and seki values at proper Go board coordinates
            for (let y = 0; y < 19; y++) {{
                for (let x = 0; x < 19; x++) {{
                    // Seki is 2D array with [y][x] indexing
                    const seki = data.seki[y][x];
                    
                    // Calculate pixel position (Go board coordinates)
                    const pixelX = 10 + x * 20;  // 10px margin + 20px spacing
                    const pixelY = 10 + y * 20;  // 10px margin + 20px spacing
                    
                    // Add stones
                    // Convert x,y to KataGo location for board state lookup
                    const dy = 20; // size + 1 = 19 + 1 = 20
                    const loc = (x + 1) + dy * (y + 1);
                    const stone = data.board_state[loc] || 0; // Default to 0 if out of bounds
                    if (stone === 1 || stone === -1) {{
                        const stoneEl = document.createElement('div');
                        stoneEl.className = `stone ${{stone === 1 ? 'black' : 'white'}}`;
                        stoneEl.style.position = 'absolute';
                        stoneEl.style.left = `${{pixelX - 9}}px`;  // Center the stone (18px wide)
                        stoneEl.style.top = `${{pixelY - 9}}px`;   // Center the stone (18px high)
                        stoneEl.style.zIndex = '10';
                        board.appendChild(stoneEl);
                    }}
                    
                    // Add seki
                    if (Math.abs(seki) > 0.1) {{
                        const label = document.createElement('div');
                        label.className = 'label';
                        label.textContent = 'S';
                        label.style.position = 'absolute';
                        label.style.left = `${{pixelX - 5}}px`;
                        label.style.top = `${{pixelY - 5}}px`;
                        label.style.color = 'purple';
                        label.style.zIndex = '15';
                        board.appendChild(label);
                    }}
                }}
            }}
            
            const info = document.getElementById('info-seki');
            if (info) {{
                info.innerHTML = `
                    <div class="heatmap-legend">
                        <span style="background: rgba(128,0,128,0.3);">Purple: Seki Probability</span>
                    </div>
                `;
            }}
        }}
        
        function updateValueInfo(data) {{
            if (!data) return;
            
            // Determine current player (opposite of who just played)
            let currentPlayer = 'Black';
            if (data.player === 'Black') {{
                currentPlayer = 'White';
            }} else if (data.player === 'White') {{
                currentPlayer = 'Black';
            }} else if (data.player === 'Initial') {{
                currentPlayer = 'Black'; // Black starts first
            }}
            
            const elements = {{
                'value-winrate': (data.value[0] * 100).toFixed(1),
                'score-mean': data.scoremean.toFixed(1),
                'lead': data.lead.toFixed(1),
                'score-stdev': data.scorestdev.toFixed(1),
                'vtime': data.vtime.toFixed(1),
                'estv': data.estv.toFixed(3),
                'value-td-long': (data.td_value[0] * 100).toFixed(1),
                'value-td-mid': (data.td_value2[0] * 100).toFixed(1),
                'value-td-short': (data.td_value3[0] * 100).toFixed(1),
                'current-player': currentPlayer
            }};
            
            for (const [id, value] of Object.entries(elements)) {{
                const element = document.getElementById(id);
                if (element) {{
                    element.textContent = value;
                }}
            }}
        }}
        
        function updatePolicyMoves(data) {{
            if (!data) return;
            
            const topMoves0 = data.moves_and_probs0.slice(0, 8);
            const topMoves1 = data.moves_and_probs1.slice(0, 8);
            
            const moves0Element = document.getElementById('policy-moves-0');
            if (moves0Element) {{
                moves0Element.innerHTML = topMoves0.map((move, i) => `
                    <div class="move-item">
                        <span>${{i+1}}. ${{getMoveString([0, move[0]])}}</span>
                        <span>${{(move[1] * 100).toFixed(1)}}%</span>
                    </div>
                `).join('');
            }}
            
            const moves1Element = document.getElementById('policy-moves-1');
            if (moves1Element) {{
                moves1Element.innerHTML = topMoves1.map((move, i) => `
                    <div class="move-item">
                        <span>${{i+1}}. ${{getMoveString([0, move[0]])}}</span>
                        <span>${{(move[1] * 100).toFixed(1)}}%</span>
                    </div>
                `).join('');
            }}
        }}
        
        // Control functions
        function previousMove() {{
            if (currentMove > 0) {{
                currentMove--;
                updateDisplay();
            }}
        }}
        
        function nextMove() {{
            if (currentMove < gameData.length - 1) {{
                currentMove++;
                updateDisplay();
            }}
        }}
        
        function resetToStart() {{
            currentMove = 0;
            updateDisplay();
        }}
        
        function toggleAutoPlay() {{
            const btn = document.getElementById('autoPlayBtn');
            if (autoPlayInterval) {{
                clearInterval(autoPlayInterval);
                autoPlayInterval = null;
                btn.textContent = 'Auto Play';
                btn.style.background = '#007bff';
            }} else {{
                autoPlayInterval = setInterval(() => {{
                    if (currentMove < gameData.length - 1) {{
                        currentMove++;
                        updateDisplay();
                    }} else {{
                        clearInterval(autoPlayInterval);
                        autoPlayInterval = null;
                        btn.textContent = 'Auto Play';
                        btn.style.background = '#007bff';
                    }}
                }}, 1000);
                btn.textContent = 'Stop';
                btn.style.background = '#dc3545';
            }}
        }}
        
        // Event listeners
        document.getElementById('moveSlider').addEventListener('input', (e) => {{
            currentMove = parseInt(e.target.value);
            updateDisplay();
        }});
        
        // Keyboard controls
        document.addEventListener('keydown', (e) => {{
            switch(e.key) {{
                case 'ArrowLeft':
                    previousMove();
                    break;
                case 'ArrowRight':
                    nextMove();
                    break;
                case ' ':
                    e.preventDefault();
                    toggleAutoPlay();
                    break;
                case 'Home':
                    resetToStart();
                    break;
            }}
        }});
        
        // Initialize display
        addValueSection();
        initializeBoards();
        updateDisplay();
    </script>
</body>
</html>
"""
    
    # Write the HTML file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"Visualization saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Visualize KataGo model outputs')
    parser.add_argument('model_path', help='Path to the KataGo model file')
    parser.add_argument('--max-moves', type=int, default=10, help='Maximum number of moves to play')
    parser.add_argument('--output', default='katago_visualization.html', help='Output HTML file')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    model, swa_model, _ = load_model(args.model_path, use_swa=False, device=get_device(), pos_len=19, verbose=True)
    if swa_model is not None:
        model = swa_model
    model.eval()
    
    # Play game and generate visualization
    game_data, sgf_content = play_short_game(model, args.max_moves)
    
    # Generate HTML visualization
    generate_html_visualization(game_data, sgf_content, args.output)
    
    print(f"Generated visualization with {len(game_data)} positions")

if __name__ == "__main__":
    main()
