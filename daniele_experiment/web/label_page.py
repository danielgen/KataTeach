"""Generate a simple HTML page to label Go board positions from an SGF file.

This script parses an SGF game record and emits a self-contained web page
that lets a human annotator step through each position and assign the
experiment's concept tags.  Tags are defined in ``configs/ontology.yaml``.

The resulting page uses the WGo.js library (loaded from a CDN) to render the
board.  Global tags are presented as checkboxes, while spatial tags can be
annotated by selecting the tag and clicking points on the board.  Labels can
be downloaded as a JSON file for further processing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence, Mapping

import yaml


def load_tags(ontology_path: Path) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Return dictionaries of grouped tag names from the ontology."""
    with ontology_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    # Handle new ontology structure with categories
    tags_data = data.get("tags", {})
    
    # Global tags grouped by category
    global_groups = {
        "global": [t["name"] for t in tags_data.get("global", [])],
        "initiative": [t["name"] for t in tags_data.get("initiative", [])]
    }
    
    # Spatial tags grouped by category
    spatial_groups = {
        "strategic": [t["name"] for t in tags_data.get("strategic", [])],
        "tactical": [t["name"] for t in tags_data.get("tactical", [])],
        "stress_area": [t["name"] for t in tags_data.get("stress_area", [])],
        "shape": [t["name"] for t in tags_data.get("shape", [])]
    }
    
    return global_groups, spatial_groups


def build_label_page(
    combined_data_path: Path,
    html_path: Path,
    ontology_path: Path,
) -> None:
    """Create an interactive labeling web page from combined SGF+policy data.

    Parameters
    ----------
    combined_data_path:
        JSON file containing both SGF data and policy suggestions.
        Expected format: {"sgf": "...", "policy": {...}}
    html_path:
        Destination HTML file.
    ontology_path:
        Path to the ontology YAML used for tag definitions.
    """

    # Load combined data
    with combined_data_path.open("r", encoding="utf-8") as f:
        combined_data = json.load(f)
    
    sgf_text = combined_data.get("sgf", "")
    policy: Mapping[str, object] = combined_data.get("policy", {})
    global_groups, spatial_groups = load_tags(ontology_path)

    sgf_js = json.dumps(sgf_text)
    global_groups_js = json.dumps(global_groups)
    spatial_groups_js = json.dumps(spatial_groups)
    policy_js = json.dumps(policy)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset='utf-8' />
<title>Go Position Labeler</title>
<script src='web/wgo.min.js'></script>
<style>
  body {{ margin: 20px; font-family: Arial, sans-serif; }}
  .container {{ display: flex; gap: 20px; max-width: 100%; }}
  .board-section {{ flex: 0 0 520px; }}
  .controls-section {{ flex: 1; min-width: 500px; }}
  .labels-container {{ 
    display: grid; 
    grid-template-columns: 1fr 1fr; 
    gap: 15px; 
    margin-bottom: 20px;
    max-width: 800px;
  }}
  .label-column {{ 
    background: #f8f9fa; 
    padding: 6px; 
    border-radius: 6px; 
    border: 1px solid #dee2e6;
    max-width: 380px;
  }}
  .label-group {{ 
    margin: 14px 0 10px; 
  }}
  .label-group h4 {{ 
    margin: 8px 0 8px; 
    color: #495057; 
    font-size: 12px; 
    font-weight: bold; 
    text-transform: uppercase; 
    border-bottom: 2px solid #007bff; 
    padding-bottom: 0;
  }}
  .label-column .label-group:first-child h4 {{
    margin-top: 12px;
  }}
  .tag-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    row-gap: 5px;
    column-gap: 6px;
    align-items: start;
  }}
  .tag-block {{ 
    display: flex;
    align-items: center;
    gap: 6px;
    margin: 0;
    font-size: 16px;
  }}
  .tag-block label {{ 
    display: flex; 
    align-items: center; 
    padding: 0;
    margin: 0;
    line-height: 1.05;
    height: 16px;
    min-height: 16px;
  }}
  .tag-block input {{ 
    margin-right: 4px; 
    flex-shrink: 0;
    transform: scale(0.85);
    vertical-align: middle;
  }}
  .labels-container .tag-block {{ 
    margin: 0 !important; 
  }}
  #board {{ width: 480px; height: 480px; }}
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
  .tag-block {{ margin: 5px 0; }}
  .policy-info {{ 
    background: #f5f5f5; 
    padding: 10px; 
    border-radius: 5px; 
    margin: 10px 0;
    max-height: 200px;
    overflow-y: auto;
    border: 1px solid #ddd;
  }}
  .policy-move {{ 
    margin: 2px 0; 
    padding: 4px 8px; 
    background: white; 
    border-radius: 3px;
    font-size: 12px;
    border-left: 3px solid #007bff;
    cursor: pointer;
    transition: background-color 0.2s;
  }}
  .policy-move:hover {{
    background: #f8f9fa;
  }}
  .policy-move.selected {{
    background: #e3f2fd;
    border-left-color: #2196f3;
    font-weight: bold;
  }}
  .policy-move:has(★) {{
    background: #fff3cd;
    border-left-color: #ffc107;
    font-weight: bold;
  }}
  .policy-move.selected:has(★) {{
    background: #fff8e1;
    border-left-color: #ff9800;
  }}
  h3 {{ margin-top: 20px; margin-bottom: 10px; }}
  h4 {{ margin: 10px 0 5px 0; color: #333; }}
  
  /* Hide WGo.js default controls that appear at bottom */
  .wgo-player-wrapper .wgo-player-control {{ display: none !important; }}
  .wgo-player-wrapper .wgo-comments {{ display: none !important; }}
  .wgo-player-wrapper .wgo-info {{ display: none !important; }}
  .wgo-player-wrapper .wgo-infobox {{ display: none !important; }}
  
  /* Hide WGo.js player name boxes that push the board down */
  .wgo-player__box.wgo-player__player-tag {{ display: none !important; }}
  .wgo-player__player-tag {{ display: none !important; }}
  
  /* Style our custom board container */
  #board {{ 
    border: 2px solid #333; 
    border-radius: 5px; 
    overflow: hidden;
    box-sizing: border-box;
  }}
  
  /* Ensure board container doesn't overflow */
  .board-section {{
    overflow: hidden;
  }}
  
  /* Style only policy move labels to be red - more targeted approach */
  .policy-label {{
    fill: red !important;
    color: red !important;
    font-weight: bold !important;
    font-size: 1px !important;
  }}
</style>
</head>
<body>
<div class='container'>
  <div class='board-section'>
    <div class='game-info'>
      <div class='player-names'>
        <span class='player-black'>⚫ <span id='black-player'></span></span>
        <span class='player-white'>⚪ <span id='white-player'></span></span>
      </div>
      <div>Game: <span id='game-info'>Loading...</span></div>
    </div>
    <div id='board'></div>
    <div class='navigation'>
      <button id='prev' class='nav-button'>← Prev</button>
      <span class='move-counter'>Move <span id='move_idx'>0</span></span>
      <button id='next' class='nav-button'>Next →</button>
    </div>
    <div class='policy-info'>
      <h4>AI Suggestions</h4>
      <div id='policy_suggestions'></div>
    </div>
  </div>
  <div class='controls-section'>
    <div id='selected-move-info' style='background: #e3f2fd; padding: 10px; border-radius: 5px; margin-bottom: 15px; font-weight: bold; text-align: center; display: none;'>
      Annotating move: <span id='selected-move-text'></span>
    </div>
    <div class='labels-container'>
      <div class='label-column' id='global_column'></div>
      <div class='label-column' id='spatial_column'></div>
    </div>
    <button id='export'>Export Labels</button>
  </div>
</div>
<script>
const SGF = {sgf_js};
const GLOBAL_GROUPS = {global_groups_js};
const SPATIAL_GROUPS = {spatial_groups_js};
const POLICY = {policy_js};
let player = new WGo.SimplePlayer(document.getElementById('board'), {{ sgf: SGF }});
let currentMove = 0;
let labels = {{}};  // move index -> move_id -> tag mapping  
let globalLabels = {{}};  // move index -> global tag mapping (shared across all moves)
let selectedMoveId = null;  // Currently selected candidate move for annotation

// Try to sync with WGo.js player state changes
function syncPlayerState() {{
  // Calculate current move from player state
  let moveCount = 0;
  if(player.currentNode && player.rootNode) {{
    let node = player.rootNode;
    while(node !== player.currentNode && node.children && node.children.length > 0) {{
      moveCount++;
      node = node.children[0];
    }}
  }}
  
  // Only update if move actually changed
  if(moveCount !== currentMove) {{
    console.log('Move changed from', currentMove, 'to', moveCount);
    currentMove = moveCount;
    
    // Reset selected move when navigating to a different position
    selectedMoveId = null;
    
    // Update our custom UI
    updateMoveDisplay();
    renderForm();
    renderPolicy();
    
    // Delay marker rendering to let WGo.js finish updating
    setTimeout(() => {{
      renderMarkers();
    }}, 10);
  }}
}}

// Check for player state changes periodically
setInterval(syncPlayerState, 100);

// Also try to hook into WGo.js events if available
if(player.on && typeof player.on === 'function') {{
  console.log('Using player.on for events');
  player.on('update', syncPlayerState);
  player.on('change', syncPlayerState);
}} else if(player.addEventListener && typeof player.addEventListener === 'function') {{
  console.log('Using player.addEventListener for events');
  player.addEventListener('update', syncPlayerState);
  player.addEventListener('change', syncPlayerState);
}} else {{
  console.log('Using polling method for player state sync');
}}

// Extract game info from SGF
function initGameInfo() {{
  try {{
    let sgfRoot = null;
    
    // Try different ways to access SGF info - WGo SimplePlayer structure
    if(player.rootNode && player.rootNode.properties) {{
      sgfRoot = player.rootNode.properties;
    }} else if(player.game && player.game.root && player.game.root.properties) {{
      sgfRoot = player.game.root.properties;
    }} else if(player.rootNode) {{
      sgfRoot = player.rootNode;
    }} else {{
      console.log('Could not find SGF root info, using defaults');
    }}
    
    console.log('SGF Root found:', sgfRoot);
    
    // Extract player names - try multiple formats
    let blackPlayer = 'Black';
    let whitePlayer = 'White';
    let gameDate = '';
    let result = '';
    
    if(sgfRoot) {{
      // Try array format first [SGF standard]
      if(sgfRoot.PB && Array.isArray(sgfRoot.PB)) blackPlayer = sgfRoot.PB[0];
      else if(sgfRoot.PB) blackPlayer = sgfRoot.PB;
      
      if(sgfRoot.PW && Array.isArray(sgfRoot.PW)) whitePlayer = sgfRoot.PW[0];
      else if(sgfRoot.PW) whitePlayer = sgfRoot.PW;
      
      if(sgfRoot.DT && Array.isArray(sgfRoot.DT)) gameDate = sgfRoot.DT[0];
      else if(sgfRoot.DT) gameDate = sgfRoot.DT;
      
      if(sgfRoot.RE && Array.isArray(sgfRoot.RE)) result = sgfRoot.RE[0];
      else if(sgfRoot.RE) result = sgfRoot.RE;
    }}
    
    // Fallback: extract from raw SGF string if properties didn't work
    if(blackPlayer === 'Black' || whitePlayer === 'White') {{
      const pbMatch = SGF.match(/PB\\[([^\\]]*)\\]/);
      const pwMatch = SGF.match(/PW\\[([^\\]]*)\\]/);
      if(pbMatch && pbMatch[1]) blackPlayer = pbMatch[1];
      if(pwMatch && pwMatch[1]) whitePlayer = pwMatch[1];
    }}
    
    console.log('Final extracted names:', blackPlayer, whitePlayer);
    
    document.getElementById('black-player').textContent = blackPlayer || 'Black';
    document.getElementById('white-player').textContent = whitePlayer || 'White';
    document.getElementById('game-info').textContent = gameDate + (result ? ' • ' + result : '');
  }} catch(e) {{
    console.error('Error initializing game info:', e);
    document.getElementById('black-player').textContent = 'Black';
    document.getElementById('white-player').textContent = 'White';
    document.getElementById('game-info').textContent = 'Game loaded';
  }}
}}

function renderForm() {{
  const globalColumn = document.getElementById('global_column');
  const spatialColumn = document.getElementById('spatial_column');
  
  // Clear both columns
  globalColumn.innerHTML = '';
  spatialColumn.innerHTML = '';
  
  // Clear any previous form state
  
  // If no move is selected, select the first available move (actual move played or first candidate)
  if (!selectedMoveId) {{
    const positionKey = currentMove > 0 ? (currentMove - 1).toString() : null;
    const positionData = positionKey ? POLICY[positionKey] : null;
    const suggestions = positionData && positionData.suggestions ? positionData.suggestions : [];
    
    if (suggestions.length > 0) {{
      // Prefer the actual move played, otherwise first suggestion
      const actualMove = suggestions.find(s => s.is_actual_move);
      selectedMoveId = actualMove ? actualMove.move : suggestions[0].move;
    }}
  }}
  
  // Update selected move indicator
  const selectedMoveInfo = document.getElementById('selected-move-info');
  const selectedMoveText = document.getElementById('selected-move-text');
  if (selectedMoveId) {{
    selectedMoveText.textContent = selectedMoveId;
    selectedMoveInfo.style.display = 'block';
  }} else {{
    selectedMoveInfo.style.display = 'none';
  }}
  
  // Render global groups (global, initiative, location, shape)
  Object.entries(GLOBAL_GROUPS).forEach(([groupName, tags]) => {{
    if (tags.length === 0) return;
    
    const groupDiv = document.createElement('div');
    groupDiv.className = 'label-group';
    groupDiv.innerHTML = `<h4>${{groupName}}</h4><div class="tag-grid"></div>`;
    
    const tagGrid = groupDiv.querySelector('.tag-grid');
    tags.forEach(tag => {{
      // Global tags are stored at position level, not per-move
      const positionGlobalLabels = globalLabels[currentMove] || {{}};
      const checked = positionGlobalLabels[tag] ? 'checked' : '';
      const tagBlock = document.createElement('div');
      tagBlock.className = 'tag-block';
      tagBlock.innerHTML = `<label><input type="checkbox" data-tag="${{tag}}" data-is-global="true" ${{checked}}/> ${{tag}}</label>`;
      tagGrid.appendChild(tagBlock);
    }});
    
    globalColumn.appendChild(groupDiv);
  }});
  
  // Add stress_area and shape to the left column
  ['stress_area', 'shape'].forEach(groupName => {{
    const tags = SPATIAL_GROUPS[groupName] || [];
    if (tags.length === 0) return;
    
    const groupDiv = document.createElement('div');
    groupDiv.className = 'label-group';
    groupDiv.innerHTML = `<h4>${{groupName}}</h4><div class="tag-grid"></div>`;
    
    const tagGrid = groupDiv.querySelector('.tag-grid');
    tags.forEach(tag => {{
      const moveLabels = labels[currentMove] && labels[currentMove][selectedMoveId] ? labels[currentMove][selectedMoveId] : {{}};
      const checked = moveLabels[tag] ? 'checked' : '';
      
      const tagBlock = document.createElement('div');
      tagBlock.className = 'tag-block';
      tagBlock.innerHTML = `<label><input type="checkbox" data-tag="${{tag}}" ${{checked}}/> ${{tag}}</label>`;
      tagGrid.appendChild(tagBlock);
    }});
    
    globalColumn.appendChild(groupDiv);
  }});
  
  // Render strategic and tactical groups only
  ['strategic', 'tactical'].forEach(groupName => {{
    const tags = SPATIAL_GROUPS[groupName] || [];
    if (tags.length === 0) return;
    
    const groupDiv = document.createElement('div');
    groupDiv.className = 'label-group';
    groupDiv.innerHTML = `<h4>${{groupName}}</h4><div class="tag-grid"></div>`;
    
    const tagGrid = groupDiv.querySelector('.tag-grid');
    tags.forEach(tag => {{
      const moveLabels = labels[currentMove] && labels[currentMove][selectedMoveId] ? labels[currentMove][selectedMoveId] : {{}};
      const checked = moveLabels[tag] ? 'checked' : '';
      
      const tagBlock = document.createElement('div');
      tagBlock.className = 'tag-block';
      tagBlock.innerHTML = `<label><input type="checkbox" data-tag="${{tag}}" ${{checked}}/> ${{tag}}</label>`;
      tagGrid.appendChild(tagBlock);
    }});
    
    spatialColumn.appendChild(groupDiv);
  }});
}}

function renderMarkers() {{
  // Access board through SimplePlayer structure - try multiple ways
  let board = null;
  
  console.log('=== RENDER MARKERS DEBUG ===');
  
  // Try different ways to access the board
  if(player.components && player.components.get) {{
    // components is a Map, get the board container
    const boardElement = document.getElementById('board');
    const container = player.components.get(boardElement);
    console.log('Board from components Map:', container);
    
    // Based on the structure we can see, try to get SVGBoardComponent directly
    if(container && container.children && container.children[1] && container.children[1].children) {{
      const svgBoardComponent = container.children[1].children[0];
      console.log('Direct access to SVGBoardComponent:', svgBoardComponent);
      if(svgBoardComponent && svgBoardComponent.board) {{
        console.log('Direct access to SVGBoard:', svgBoardComponent.board);
        board = svgBoardComponent.board;
      }}
    }}
    
    if(!board) {{
      board = container;
    }}
  }}
  
  if(!board && player.board) {{
    board = player.board;
    console.log('Board from player.board:', board);
  }}
  
  if(!board && player.components && player.components.board) {{
    board = player.components.board;
    console.log('Board from player.components.board:', board);
  }}
  
  console.log('Final board object:', board);
  console.log('Board methods available:', board ? Object.getOwnPropertyNames(board) : 'No board');
  
  // If we found a Container, search through its children for the board component
  if(board && board.children && Array.isArray(board.children)) {{
    console.log('Container children:', board.children);
    
    // Look for a child that has board methods
    for(let child of board.children) {{
      console.log('Checking child:', child);
      console.log('Child methods:', child ? Object.getOwnPropertyNames(child) : 'No child');
      
      if(child && child.removeAllObjects) {{
        console.log('Found board in child:', child);
        board = child;
        break;
      }}
      
      // Also check if child has a board property
      if(child && child.board && child.board.removeAllObjects) {{
        console.log('Found board in child.board:', child.board);
        board = child.board;
        break;
      }}
      
      // Check if this is a ResponsiveComponent with a component property
      if(child && child.component) {{
        console.log('Checking child.component:', child.component);
        if(child.component.removeAllObjects) {{
          console.log('Found board in child.component:', child.component);
          board = child.component;
          break;
        }}
        
        // If component is a Container, check its children too
        if(child.component.children && Array.isArray(child.component.children)) {{
          console.log('Checking nested container children:', child.component.children);
          for(let nestedChild of child.component.children) {{
            console.log('Nested child:', nestedChild);
            console.log('Nested child type:', nestedChild.constructor.name);
            
            if(nestedChild && nestedChild.removeAllObjects) {{
              console.log('Found board in nested child:', nestedChild);
              board = nestedChild;
              break;
            }}
            
            // Check for SVGBoardComponent which has a board property
            if(nestedChild && nestedChild.constructor && nestedChild.constructor.name === 'SVGBoardComponent') {{
              console.log('Found SVGBoardComponent, checking its board property:', nestedChild.board);
              if(nestedChild.board) {{
                console.log('SVGBoard object:', nestedChild.board);
                console.log('SVGBoard methods:', Object.getOwnPropertyNames(nestedChild.board));
                board = nestedChild.board;
                break;
              }}
            }}
            
            // Also check nested child's board property
            if(nestedChild && nestedChild.board && nestedChild.board.removeAllObjects) {{
              console.log('Found board in nested child.board:', nestedChild.board);
              board = nestedChild.board;
              break;
            }}
          }}
          if(board && board.removeAllObjects) break; // Exit outer loop if found
        }}
      }}
    }}
  }}
  
  // If we found a Container, try to access its board
  if(board && !board.removeAllObjects && board.board) {{
    console.log('Trying board.board:', board.board);
    board = board.board;
  }}
  
  // Also try accessing through getBoard method if it exists
  if(board && !board.removeAllObjects && board.getBoard) {{
    console.log('Trying board.getBoard():', board.getBoard());
    board = board.getBoard();
  }}
  
  // If we still haven't found the board, do a recursive search
  if(!board || !board.removeAllObjects) {{
    console.log('Board not found yet, doing recursive search...');
    
    function findBoardRecursively(obj, path = '') {{
      if(!obj) return null;
      
      // Check if this object is the board we want
      if(obj.removeAllObjects && obj.addObject) {{
        console.log('Found board at path:', path, obj);
        return obj;
      }}
      
      // Check if this is SVGBoardComponent
      if(obj.constructor && obj.constructor.name === 'SVGBoardComponent') {{
        console.log('Found SVGBoardComponent at path:', path, obj);
        if(obj.board && obj.board.removeAllObjects) {{
          console.log('Found board in SVGBoardComponent.board:', obj.board);
          return obj.board;
        }}
      }}
      
      // Recursively search common properties
      const propsToSearch = ['board', 'component', 'children', 'components'];
      for(let prop of propsToSearch) {{
        if(obj[prop]) {{
          if(Array.isArray(obj[prop])) {{
            for(let i = 0; i < obj[prop].length; i++) {{
              const result = findBoardRecursively(obj[prop][i], path + '.' + prop + '[' + i + ']');
              if(result) return result;
            }}
          }} else {{
            const result = findBoardRecursively(obj[prop], path + '.' + prop);
            if(result) return result;
          }}
        }}
      }}
      
      return null;
    }}
    
    const foundBoard = findBoardRecursively(player, 'player');
    if(foundBoard) {{
      board = foundBoard;
      console.log('Recursively found board:', board);
    }}
  }}
  
  console.log('Final resolved board:', board);
  if(board) {{
    console.log('Board methods:', Object.getOwnPropertyNames(board));
    console.log('Board config:', board.config);
    console.log('Board size/dimensions:', {{ width: board.width, height: board.height, size: board.size }});
    console.log('Existing board objects:', board.objects);
    
    // Check if there are existing objects and their coordinate format
    if(board.objects && board.objects.length > 0) {{
      console.log('Sample existing object:', board.objects[0]);
      console.log('Existing object coordinates:', {{ x: board.objects[0].x, y: board.objects[0].y }});
    }}
  }}
  
  if(board && board.objects) {{
    // Remove only our custom objects (AI labels and user tags), keep game stones
    const objectsToRemove = board.objects.filter(obj => 
      obj.type === 'LB' || obj.type === 'MA' || obj.type === 'CR' || obj.type === 'SQ'
    );
    objectsToRemove.forEach(obj => board.removeObject(obj));
    
    console.log('Removed', objectsToRemove.length, 'custom objects, keeping game stones');
    
    // No user markers to render - all tags are now simple booleans
    
         // Render policy suggestions for the position that was just played (not the upcoming move)
     // Show suggestions for the previous move (what AI suggested before this move was played)
     const positionKey = currentMove > 0 ? (currentMove - 1).toString() : null;
     const positionData = positionKey ? POLICY[positionKey] : null;
     const policyMoves = positionData && positionData.suggestions ? positionData.suggestions : [];
     console.log('Rendering', policyMoves.length, 'policy moves for position', positionKey, 'Position data:', positionData);
    
    if(policyMoves.length > 0) {{
      const playerToMove = currentMove % 2 === 0 ? 'black' : 'white';
      console.log('Player to move:', playerToMove);
      
      policyMoves.forEach((move, index) => {{
        const coord = sgfToCoord(move.move);
        
        if(coord && typeof coord.x === 'number' && typeof coord.y === 'number' && 
           coord.x >= 0 && coord.x < 19 && coord.y >= 0 && coord.y < 19) {{
          try {{
            
            // Add policy move marker with red styling
            const winrateText = (move.winrate * 100).toFixed(0);
            console.log('Adding policy label at:', coord.x, coord.y, 'with text:', winrateText);
            
            // Create label object with unique identifier
            const uniqueId = 'policy-' + coord.x + '-' + coord.y + '-' + index;
            const labelObj = new WGo.LabelBoardObject(winrateText, coord.x, coord.y);
            
            // Mark this as a policy label so we can style it differently
            labelObj.isPolicyLabel = true;
            labelObj.policyId = uniqueId;
            
            board.addObject(labelObj);
            
                        // Apply red styling after the object is added to the DOM
            setTimeout(() => {{
              const boardElement = document.getElementById('board');
              const textElements = boardElement.querySelectorAll('text');
              
              let matchingElements = [];
              
              textElements.forEach((textEl, index) => {{
                // Only style text elements that contain our winrate numbers AND are not board coordinates
                // Board coordinates are typically single/double digit numbers or letters, winrates are percentages
                if(textEl.textContent === winrateText) {{
                  // Additional check: make sure this isn't a coordinate label
                  // Coordinates are usually positioned at board edges, policy labels are on intersections
                  const rect = textEl.getBoundingClientRect();
                  const boardRect = boardElement.getBoundingClientRect();
                  
                  // Check if the text is within the main board area (not on edges where coordinates would be)
                  const isInMainBoardArea = rect.left > boardRect.left + 20 && 
                                          rect.right < boardRect.right - 20 &&
                                          rect.top > boardRect.top + 10 && 
                                          rect.bottom < boardRect.bottom - 10;
                  
                  if(isInMainBoardArea) {{
                    matchingElements.push(textEl);
                  }}
                }}
              }});
              
              // Style only the elements that are actually policy labels
              matchingElements.forEach((elem) => {{
                // Apply red styling with smaller font size
                elem.style.setProperty('fill', 'red', 'important');
                elem.style.setProperty('color', 'red', 'important');
                elem.style.setProperty('font-weight', 'bold', 'important');
                elem.style.setProperty('font-size', '0.8px', 'important');
                elem.setAttribute('fill', 'red');
                elem.classList.add('policy-label');
              }});
            }}, 100);
            
            console.log('Successfully added red label at', coord.x, coord.y);
            

            
            console.log('Added policy move at', coord.x, coord.y, 'with winrate', winrateText + '%');
          }} catch(e) {{
            console.error('Error adding board object:', e);
            console.error('Board object at error:', board);
            console.error('Board addObject method:', typeof board.addObject);
          }}
        }} else {{
          console.error('Invalid coordinates for move:', move.move, coord);
        }}
      }});
    }} else {{
      console.log('No policy moves to render for position', currentMove);
    }}
    
    // Refresh only the objects layer, not the entire board
    if(board && board.redraw) {{
      console.log('Calling board.redraw() to refresh custom objects');
      board.redraw();
    }}
    
    console.log('Total objects on board after rendering:', board.objects.length);
    console.log('Objects:', board.objects.map(obj => obj.type || 'unknown'));
  }} else {{
    console.error('Board object not found or missing removeAllObjects method');
    console.error('Player structure:', player);
  }}
}}

function sgfToCoord(moveString) {{
  if(!moveString || moveString === 'pass') {{
    console.log('Skipping move:', moveString);
    return null;
  }}
  
  console.log('Converting move string:', moveString);
  
  // Handle SGF coordinates like 'pd', 'dp', etc. (lowercase letters)
  if(moveString.length === 2 && /^[a-s][a-s]$/.test(moveString)) {{
    const x = moveString.charCodeAt(0) - 'a'.charCodeAt(0);
    const y = moveString.charCodeAt(1) - 'a'.charCodeAt(0);
    console.log(`SGF format ${{moveString}}: x=${{x}}, y=${{y}}`);
    return {{ x: x, y: y }};
  }}
  
  // Handle human coordinates like 'C16', 'D4', 'P16', etc.
  const match = moveString.match(/^([A-HJ-T])(\\d+)$/);
  if(match) {{
    const letter = match[1];
    const number = parseInt(match[2], 10);
    
    // Convert letter to x coordinate (A=0, B=1, ..., H=7, J=8, ..., T=18)
    let x = letter.charCodeAt(0) - 'A'.charCodeAt(0);
    if(x >= 8) x--; // Skip 'I' in Go notation
    
    // Convert number to y coordinate (1=18, 2=17, ..., 19=0)
    const y = 19 - number;
    
    console.log(`Human format ${{moveString}}: letter=${{letter}}, number=${{number}}, x=${{x}}, y=${{y}}`);
    
    // Validate the coordinates
    if(isNaN(x) || isNaN(y) || x < 0 || x >= 19 || y < 0 || y >= 19) {{
      console.error(`Invalid coordinates generated: x=${{x}}, y=${{y}} from ${{moveString}}`);
      return null;
    }}
    
    return {{ x: x, y: y }};
  }}
  
  console.error('Failed to parse move:', moveString);
  return null;
}}

  function renderPolicy() {{
    const div = document.getElementById('policy_suggestions');
    // Show policy suggestions for the position that was just played (not the upcoming move)
    const positionKey = currentMove > 0 ? (currentMove - 1).toString() : null;
    const positionData = positionKey ? POLICY[positionKey] : null;
    const opts = positionData && positionData.suggestions ? positionData.suggestions : [];
    
    console.log('Current move:', currentMove, 'Policy data for position:', positionKey, 'Position data:', positionData, 'Suggestions:', opts);
  
     if(currentMove === 0) {{
     div.innerHTML = '<em>No move played yet</em>';
     return;
   }}
   
   if(!positionData || !opts || opts.length === 0) {{
     div.innerHTML = '<em>No AI suggestions available for this position</em>';
     return;
   }}
   
       // The player who was to move at the previous position (before current move was played)
    const playerWhoMoved = (currentMove - 1) % 2 === 0 ? 'Black' : 'White';
    const lines = opts.map(o => {{
      const winrateText = `${{(o.winrate * 100).toFixed(1)}}%`;
      const policyText = o.policy_prob ? `${{(o.policy_prob * 100).toFixed(1)}}%` : 'N/A';
      const actualMoveMarker = o.is_actual_move ? ' ★' : '';
      const selectedClass = selectedMoveId === o.move ? ' selected' : '';
      return `<div class="policy-move${{selectedClass}}" data-move-id="${{o.move}}">${{o.move}}: ${{winrateText}} win, ${{policyText}} prob${{actualMoveMarker}}</div>`;
    }});
    // Display move number as 1-indexed (currentMove is already the human-readable move number)
    div.innerHTML = `<strong>AI suggestions for ${{playerWhoMoved}} at move ${{currentMove}} (click to select move for annotation)</strong><br/>${{lines.join('')}}`;
}}

// Set up event delegation for both columns
document.getElementById('global_column').addEventListener('change', e => {{
  const tag = e.target.getAttribute('data-tag');
  const isGlobal = e.target.getAttribute('data-is-global') === 'true';
  
  if(tag) {{
    if(isGlobal) {{
      // Global tags are stored at position level
      globalLabels[currentMove] = globalLabels[currentMove] || {{}};
      globalLabels[currentMove][tag] = e.target.checked;
    }} else if(selectedMoveId) {{
      // Non-global tags are stored per-move
      labels[currentMove] = labels[currentMove] || {{}};
      labels[currentMove][selectedMoveId] = labels[currentMove][selectedMoveId] || {{}};
      labels[currentMove][selectedMoveId][tag] = e.target.checked;
    }}
  }}
}});

document.getElementById('spatial_column').addEventListener('change', e => {{
  const tag = e.target.getAttribute('data-tag');
  if(tag && selectedMoveId) {{
    labels[currentMove] = labels[currentMove] || {{}};
    labels[currentMove][selectedMoveId] = labels[currentMove][selectedMoveId] || {{}};
    
    // All tags are now simple boolean values
    labels[currentMove][selectedMoveId][tag] = e.target.checked;
  }}
}});

// Board clicking is no longer needed - all tags are simple checkboxes

function updateMoveDisplay() {{
  document.getElementById('move_idx').textContent = currentMove;
}}

function gotoMove(idx) {{
  try {{
    console.log('Manual navigation to move', idx, 'from current move', currentMove);
    
    if(idx < 0) {{
      console.log('Move index too low');
      return;
    }}
    
    // Use WGo.js navigation methods that properly maintain board state
    if(idx > currentMove) {{
      // Move forward one step at a time
      for(let i = currentMove; i < idx; i++) {{
        if(player.next && typeof player.next === 'function') {{
          player.next();
        }} else {{
          break;
        }}
      }}
    }} else if(idx < currentMove) {{
      // Move backward one step at a time
      for(let i = currentMove; i > idx; i--) {{
        if(player.previous && typeof player.previous === 'function') {{
          player.previous();
        }} else {{
          break;
        }}
      }}
    }}
    
    // The 'update' event listener will handle UI updates
    console.log('Navigation completed');
  }} catch(e) {{
    console.error('Error in gotoMove:', e);
  }}
}}

document.getElementById('next').onclick = () => gotoMove(currentMove + 1);
document.getElementById('prev').onclick = () => gotoMove(currentMove - 1);

// Set up click handler for move selection in policy suggestions
document.getElementById('policy_suggestions').addEventListener('click', e => {{
  if(e.target.classList.contains('policy-move')) {{
    const moveId = e.target.getAttribute('data-move-id');
    if(moveId) {{
      selectedMoveId = moveId;
      console.log('Selected move for annotation:', selectedMoveId);
      
      // Re-render the form and policy display to show selection
      renderForm();
      renderPolicy();
      renderMarkers();
    }}
  }}
}});

document.getElementById('export').onclick = () => {{
  // Combine global labels and per-move labels in export
  const exportData = {{
    perMoveLabels: labels,
    globalLabels: globalLabels
  }};
  const data = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(exportData));
  const a = document.createElement('a');
  a.setAttribute('href', data);
  a.setAttribute('download', 'labels.json');
  a.click();
}};

console.log('Player object:', player);
console.log('Player kifu:', player.kifu);
console.log('Player available methods:', Object.getOwnPropertyNames(player));
if(player.kifu) {{
  console.log('Kifu structure:', Object.getOwnPropertyNames(player.kifu));
}}
initGameInfo();
renderForm();
updateMoveDisplay();
renderPolicy();
renderMarkers();
</script>
</body>
</html>
"""

    html_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SGF labeling web page")
    parser.add_argument("combined_data", type=Path, help="Combined JSON file with SGF and policy data")
    parser.add_argument("html", type=Path, help="Output HTML file")
    parser.add_argument(
        "--ontology",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "ontology.yaml",
        help="Ontology YAML path",
    )
    args = parser.parse_args()
    build_label_page(args.combined_data, args.html, args.ontology)


if __name__ == "__main__":
    main()
    #python web\label_page.py D:\KataGo\daniele_experiment\games\policy\3212f8f3-c7be-4fc7-80c8-7a9e87f8be9c.json test_out.html
    