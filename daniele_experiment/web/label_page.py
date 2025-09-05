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


def load_tags(ontology_path: Path) -> tuple[Sequence[str], Sequence[str]]:
    """Return lists of global and spatial tag names from the ontology."""
    with ontology_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    global_tags = [t["name"] for t in data.get("tags", []) if t.get("type") == "global"]
    spatial_tags = [t["name"] for t in data.get("tags", []) if t.get("type") == "spatial"]
    return global_tags, spatial_tags


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
    global_tags, spatial_tags = load_tags(ontology_path)

    sgf_js = json.dumps(sgf_text)
    globals_js = json.dumps(list(global_tags))
    spatial_js = json.dumps(list(spatial_tags))
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
  .controls-section {{ flex: 1; min-width: 300px; }}
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
    font-size: 13px;
    border-left: 3px solid #007bff;
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
    <form id='tag_form'></form>
    <button id='export'>Export Labels</button>
  </div>
</div>
<script>
const SGF = {sgf_js};
const GLOBAL_TAGS = {globals_js};
const SPATIAL_TAGS = {spatial_js};
const POLICY = {policy_js};
let player = new WGo.SimplePlayer(document.getElementById('board'), {{ sgf: SGF }});
let currentMove = 0;
let labels = {{}};  // move index -> tag mapping
let selectedSpatial = null;

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
  const form = document.getElementById('tag_form');
  form.innerHTML = '';
  form.innerHTML += '<h3>Global tags</h3>';
  GLOBAL_TAGS.forEach(tag => {{
    const checked = labels[currentMove] && labels[currentMove][tag] ? 'checked' : '';
    form.innerHTML += `<div class="tag-block"><label><input type="checkbox" data-tag="${{tag}}" ${{checked}}/> ${{tag}}</label></div>`;
  }});
  form.innerHTML += '<h3>Spatial tags</h3>';
  SPATIAL_TAGS.forEach(tag => {{
    const pts = (labels[currentMove] && labels[currentMove][tag]) || [];
    const checked = selectedSpatial === tag ? 'checked' : '';
    form.innerHTML += `<div class="tag-block"><label><input type="radio" name="spatial" value="${{tag}}" ${{checked}}/> ${{tag}}</label> <span id="${{tag}}_pts">${{pts.map(p => '(' + p[0] + ',' + p[1] + ')').join(' ')}}</span></div>`;
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
    
    // Render user tags
    const tags = labels[currentMove] || {{}};
    Object.keys(tags).forEach(tag => {{
      const pts = tags[tag];
      if(Array.isArray(pts)) {{
        pts.forEach(pt => board.addObject({{ x: pt[0], y: pt[1], type: 'MA' }}));
      }}
    }});
    
         // Render policy suggestions for the position that was just played (not the upcoming move)
     // Show suggestions for the previous move (what AI suggested before this move was played)
     const policyMoves = currentMove > 0 ? (POLICY[currentMove - 1] || []) : [];
     console.log('Rendering', policyMoves.length, 'policy moves for position', currentMove > 0 ? currentMove - 1 : 'none');
    
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
            
            // Create label object
            const labelObj = new WGo.LabelBoardObject(winrateText, coord.x, coord.y);
            
            // Mark this as a policy label so we can style it differently
            labelObj.isPolicyLabel = true;
            
            board.addObject(labelObj);
            
                         // Apply red styling after the object is added to the DOM
             setTimeout(() => {{
               const boardElement = document.getElementById('board');
               const textElements = boardElement.querySelectorAll('text');
               textElements.forEach(textEl => {{
                 // Only style text elements that contain our winrate numbers
                 if(textEl.textContent === winrateText) {{
                   textEl.style.fill = 'red';
                   textEl.style.fontWeight = 'bold';
                   textEl.style.fontSize = '10px';
                   textEl.classList.add('policy-label');
                 }}
               }});
             }}, 50);
            
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
    const opts = currentMove > 0 ? (POLICY[currentMove - 1] || []) : [];
    
    console.log('Current move:', currentMove, 'Policy data for position:', currentMove > 0 ? currentMove - 1 : 'none', opts);
  
     if(currentMove === 0) {{
     div.innerHTML = '<em>No move played yet</em>';
     return;
   }}
   
   if(opts.length === 0) {{
     div.innerHTML = '<em>No AI suggestions available for this position</em>';
     return;
   }}
   
       // The player who was to move at the previous position (before current move was played)
    const playerWhoMoved = (currentMove - 1) % 2 === 0 ? 'Black' : 'White';
    const lines = opts.map(o => 
      `<div class="policy-move">${{o.move}}: ${{(o.winrate * 100).toFixed(1)}}%</div>`
    );
    // Display move number as 1-indexed (currentMove is already the human-readable move number)
    div.innerHTML = `<strong>AI suggestions for ${{playerWhoMoved}} at move ${{currentMove}}</strong><br/>${{lines.join('')}}`;
}}

document.getElementById('tag_form').addEventListener('change', e => {{
  const tag = e.target.getAttribute('data-tag');
  if(tag) {{
    labels[currentMove] = labels[currentMove] || {{}};
    labels[currentMove][tag] = e.target.checked;
  }} else if(e.target.name === 'spatial') {{
    selectedSpatial = e.target.value;
  }}
}});

// Set up click handler for the board
const board = player.board || (player.components && player.components.board);
if(board && board.addEventListener) {{
  board.addEventListener('click', (x, y) => {{
    if(selectedSpatial === null) return;
    labels[currentMove] = labels[currentMove] || {{}};
    labels[currentMove][selectedSpatial] = labels[currentMove][selectedSpatial] || [];
    labels[currentMove][selectedSpatial].push([x, y]);
    renderForm();
    renderMarkers();
  }});
}}

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

document.getElementById('export').onclick = () => {{
  const data = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(labels));
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
    #python label_page.py D:\KataGo\daniele_experiment\games\policy\[a531233903]vs[danielgen]1737762562030002806.json test_out.html
    