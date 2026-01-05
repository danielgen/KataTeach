#!/usr/bin/env python3
"""
Example script showing how to use extract_features.py programmatically.

BOARD COORDINATE SYSTEM:
- x, y coordinates: (0, 0) is top-left corner, (18, 18) is bottom-right
- Go notation: (0, 0) = A19, (18, 0) = S19, (0, 18) = A1, (18, 18) = S1
- Policy index: idx = y * 19 + x for position (x, y)

EDITING BOARD POSITIONS:
There are three ways to define board positions:

1. VISUAL BOARD REPRESENTATION (RECOMMENDED - easiest to edit!):
   Use the visual_board() function with a multi-line string:
   
   board_moves = visual_board('''
       . . . . . . . . . . . . . . . . . . .
       . . . B . . . . . . . . . . . . . . .
       . . . . . . . . . . . . . . . W . . .
       ...
   ''')
   
   Characters: 'B' or 'X' = Black, 'W' or 'O' = White, '.' or ' ' = empty

2. List format:
   board_moves = [(x, y, 'B'), (x, y, 'W'), ...]
   where 'B' = Black, 'W' = White

3. Interactive editor:
   Use --editor flag or set use_editor=True in example functions
"""

import sys
from pathlib import Path
import numpy as np


# Add the python directory to the path
sys.path.append(str(Path(__file__).parent.parent / "python"))

from board import Board
from extract_features import extract_features, load_ownership, load_board_from_moves
import json


def create_board_from_moves(moves, size=19):
    """
    Helper function to create a Board from a list of moves.
    
    Args:
        moves: List of (x, y, color) tuples where color is 'B' or 'W'
        size: Board size (default 19)
    
    Returns:
        Board object with moves played
    """
    board = Board(size)
    for x, y, color in moves:
        player = Board.BLACK if color == 'B' else Board.WHITE
        board.play(player, board.loc(x, y))
    return board


def parse_visual_board(board_str, size=19):
    """
    Parse a visual board representation (ASCII grid) into a list of moves.
    
    Args:
        board_str: Multi-line string representing the board.
                   Use 'B' or 'X' for Black, 'W' or 'O' for White, '.' or ' ' for empty.
                   Can include coordinate labels (A-S, 1-19) which will be ignored.
        size: Board size (default 19)
    
    Returns:
        List of (x, y, color) tuples
    
    Example:
        board_str = '''
            A B C D E F G H J K L M N O P Q R S T
        19  . . . . . . . . . . . . . . . . . . .  19
        18  . . . . . . . . . . . . . . . . . . .  18
        17  . . . B . . . . . . . . . . . . . . .  17
        16  . . . . . . . . . . . . . . . W . . .  16
        ...
        '''
    """
    lines = board_str.strip().split('\n')
    moves = []
    coord_chars = "ABCDEFGHJKLMNOPQRST"
    row_index = 0  # Track actual board row (0-18)
    
    for line in lines:
        line_stripped = line.strip()
        # Skip empty lines
        if not line_stripped:
            continue
        
        # Skip lines that are just coordinate labels (all coordinates or numbers)
        if all(c in coord_chars + '1234567890 ' for c in line_stripped):
            continue
        
        # Extract the actual board row (filter to only board characters, ignore spaces)
        row_chars = []
        for char in line:
            char_upper = char.upper()
            if char_upper in 'BWOX.':
                row_chars.append(char_upper)
            # Skip spaces - they're just separators
        
        # Process each character in the row
        x = 0
        for char in row_chars:
            if x >= size or row_index >= size:
                break
            if char == 'B' or char == 'X':
                moves.append((x, row_index, 'B'))
            elif char == 'W' or char == 'O':
                moves.append((x, row_index, 'W'))
            # '.' means empty, but we still increment x for the position
            x += 1
        
        # Only increment row_index if we found board characters
        if row_chars:
            row_index += 1
    
    return moves


def board_to_visual(moves, size=19, show_coords=True, highlight_move=None):
    """
    Convert a list of moves to a visual ASCII board representation.
    
    Args:
        moves: List of (x, y, color) tuples
        size: Board size (default 19)
        show_coords: Whether to show coordinate labels
        highlight_move: Optional (x, y) tuple to highlight with '*'
    
    Returns:
        Multi-line string representing the board
    
    Example output:
        A B C D E F G H J K L M N O P Q R S T
    19  . . . . . . . . . . . . . . . . . . .  19
    18  . . . . . . . . . . . . . . . . . . .  18
    17  . . . B . . . . . . . . . . . . . . .  17
    ...
    """
    # Create board state: 0 = empty, 1 = Black, 2 = White
    board_state = [[0 for _ in range(size)] for _ in range(size)]
    for x, y, color in moves:
        if 0 <= x < size and 0 <= y < size:
            board_state[y][x] = 1 if color == 'B' else 2
    
    coord_chars = "ABCDEFGHJKLMNOPQRST"
    lines = []
    
    if show_coords:
        # Top coordinate row
        top_row = "   " + " ".join(coord_chars[:size]) + "  "
        lines.append(top_row)
    
    for y in range(size):
        row_parts = []
        if show_coords:
            row_parts.append(f"{size-y:2d} ")
        
        for x in range(size):
            if highlight_move and highlight_move == (x, y):
                row_parts.append('*')
            elif board_state[y][x] == 1:
                row_parts.append('B')
            elif board_state[y][x] == 2:
                row_parts.append('W')
            else:
                row_parts.append('.')
        
        if show_coords:
            row_parts.append(f" {size-y:2d}")
        lines.append(" ".join(row_parts))
    
    if show_coords:
        # Bottom coordinate row
        bottom_row = "   " + " ".join(coord_chars[:size]) + "  "
        lines.append(bottom_row)
    
    return "\n".join(lines)


def visual_board(board_str):
    """
    Convenience function to create board_moves from a visual board string.
    Use this as a decorator-style function for easy board editing.
    
    Args:
        board_str: Multi-line string with board representation
                   Format: 'B' or 'X' = Black, 'W' or 'O' = White, '.' or ' ' = empty
    
    Returns:
        List of (x, y, color) tuples
    
    Example:
        board_moves = visual_board('''
            . . . . . . . . . . . . . . . . . . .
            . . . . . . . . . . . . . . . . . . .
            . . . B . . . . . . . . . . . . . . .
            . . . . . . . . . . . . . . . W . . .
            ...
        ''')
    """
    return parse_visual_board(board_str)


def interactive_board_editor(initial_moves=None, title="Go Board Editor"):
    """
    Interactive graphical board editor using tkinter.
    
    Click on intersections to place stones. Right-click or use buttons to remove.
    
    Args:
        initial_moves: Optional list of (x, y, color) tuples to start with
        title: Window title
    
    Returns:
        List of (x, y, color) tuples representing the board position
        Returns None if user cancels or tkinter is not available
    """
    if not TKINTER_AVAILABLE:
        print("Error: tkinter is not available. Please install tkinter or use manual board editing.")
        return None
    
    BOARD_SIZE = 19
    CELL_SIZE = 25
    MARGIN = 30
    BOARD_WIDTH = BOARD_SIZE * CELL_SIZE + 2 * MARGIN
    BOARD_HEIGHT = BOARD_SIZE * CELL_SIZE + 2 * MARGIN + 100  # Extra space for buttons
    
    # Initialize board state: 0 = empty, 1 = Black, 2 = White
    board_state = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    current_color = 'B'  # Current player to place
    
    # Load initial moves if provided
    if initial_moves:
        for x, y, color in initial_moves:
            if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
                board_state[y][x] = 1 if color == 'B' else 2
    
    root = tk.Tk()
    root.title(title)
    root.resizable(False, False)
    
    # Canvas for board
    canvas = tk.Canvas(root, width=BOARD_WIDTH, height=BOARD_HEIGHT, bg='#DCB35C')
    canvas.pack()
    
    # Draw grid
    def draw_grid():
        # Draw lines
        for i in range(BOARD_SIZE):
            y = MARGIN + i * CELL_SIZE
            canvas.create_line(MARGIN, y, MARGIN + (BOARD_SIZE - 1) * CELL_SIZE, y, width=1)
        
        for i in range(BOARD_SIZE):
            x = MARGIN + i * CELL_SIZE
            canvas.create_line(x, MARGIN, x, MARGIN + (BOARD_SIZE - 1) * CELL_SIZE, width=1)
        
        # Draw star points
        star_points = [(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)]
        for x, y in star_points:
            px = MARGIN + x * CELL_SIZE
            py = MARGIN + y * CELL_SIZE
            canvas.create_oval(px - 3, py - 3, px + 3, py + 3, fill='black')
        
        # Draw coordinates
        coord_chars = "ABCDEFGHJKLMNOPQRST"
        for i in range(BOARD_SIZE):
            # Top and bottom labels
            x = MARGIN + i * CELL_SIZE
            canvas.create_text(x, MARGIN - 15, text=coord_chars[i], font=('Arial', 8))
            canvas.create_text(x, MARGIN + (BOARD_SIZE - 1) * CELL_SIZE + 15, text=coord_chars[i], font=('Arial', 8))
            # Left and right labels
            y = MARGIN + i * CELL_SIZE
            canvas.create_text(MARGIN - 15, y, text=str(BOARD_SIZE - i), font=('Arial', 8))
            canvas.create_text(MARGIN + (BOARD_SIZE - 1) * CELL_SIZE + 15, y, text=str(BOARD_SIZE - i), font=('Arial', 8))
    
    # Draw stones
    def draw_stones():
        # Clear existing stones
        canvas.delete("stone")
        
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if board_state[y][x] != 0:
                    px = MARGIN + x * CELL_SIZE
                    py = MARGIN + y * CELL_SIZE
                    color = 'black' if board_state[y][x] == 1 else 'white'
                    # Draw stone with border
                    canvas.create_oval(px - 10, py - 10, px + 10, py + 10, 
                                     fill=color, outline='black', width=1, tags="stone")
    
    # Convert pixel coordinates to board coordinates
    def pixel_to_board(px, py):
        x = round((px - MARGIN) / CELL_SIZE)
        y = round((py - MARGIN) / CELL_SIZE)
        if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
            return x, y
        return None, None
    
    # Handle click on board
    def on_click(event):
        x, y = pixel_to_board(event.x, event.y)
        if x is not None and y is not None:
            if event.num == 1:  # Left click - place stone
                if board_state[y][x] == 0:
                    board_state[y][x] = 1 if current_color == 'B' else 2
                    draw_stones()
            elif event.num == 3:  # Right click - remove stone
                board_state[y][x] = 0
                draw_stones()
    
    canvas.bind("<Button-1>", on_click)
    canvas.bind("<Button-3>", on_click)
    
    # Button callbacks
    def toggle_color():
        nonlocal current_color
        current_color = 'W' if current_color == 'B' else 'B'
        color_label.config(text=f"Current: {'Black' if current_color == 'B' else 'White'}")
    
    def clear_board():
        nonlocal board_state
        board_state = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        draw_stones()
    
    def get_moves():
        moves = []
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if board_state[y][x] != 0:
                    color = 'B' if board_state[y][x] == 1 else 'W'
                    moves.append((x, y, color))
        return moves
    
    def export_and_close():
        moves = get_moves()
        # Copy to clipboard
        moves_str = "board_moves = [\n"
        for x, y, color in moves:
            moves_str += f"    ({x}, {y}, '{color}'),  # {chr(65+x)}{19-y}\n"
        moves_str += "]"
        
        root.clipboard_clear()
        root.clipboard_append(moves_str)
        messagebox.showinfo("Exported", 
                          f"Board position exported!\n\n{moves_str}\n\n"
                          "Copied to clipboard. Paste it into your code.")
        root.result = moves
        root.destroy()
    
    def cancel():
        root.result = None
        root.destroy()
    
    # Draw initial board
    draw_grid()
    draw_stones()
    
    # Control panel
    button_frame = tk.Frame(root)
    button_frame.pack(side=tk.BOTTOM, pady=5)
    
    color_label = tk.Label(button_frame, text=f"Current: {'Black' if current_color == 'B' else 'White'}", 
                          font=('Arial', 10, 'bold'))
    color_label.pack(side=tk.LEFT, padx=5)
    
    tk.Button(button_frame, text="Switch Color", command=toggle_color).pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text="Clear Board", command=clear_board).pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text="Export & Close", command=export_and_close, 
             bg='#4CAF50', fg='white', font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text="Cancel", command=cancel).pack(side=tk.LEFT, padx=5)
    
    # Instructions
    instructions = tk.Label(root, 
                          text="Left-click: Place stone | Right-click: Remove stone | Switch color to alternate",
                          font=('Arial', 8), fg='gray')
    instructions.pack(side=tk.BOTTOM)
    
    root.result = None
    root.mainloop()
    
    return root.result


def example_basic_usage(use_editor=False):
    """Example: Extract features from a simple board position."""
    print("=" * 80)
    print("Example 1: Basic usage with empty board and synthetic ownership")
    print("=" * 80)
    
    # ============================================================================
    # EDIT SECTION 1: Board position
    # ============================================================================
    # Option 1: Use interactive editor (set use_editor=True or call with --editor flag)
    # Option 2: Use visual board representation (RECOMMENDED - easiest to edit!)
    # Option 3: Manually edit the board_moves list
    if use_editor:
        board_moves = interactive_board_editor(title="Example 1: Basic Usage")
        if board_moves is None:
            print("Editor cancelled, using empty board.")
            board_moves = []
    else:
        # VISUAL BOARD REPRESENTATION (easiest to edit!)
        # Edit the board below by changing 'B' (Black), 'W' (White), or '.' (empty)
        # Coordinates: A-S (left to right), 19-1 (top to bottom)
        board_moves = visual_board('''
            . . . . . . . . . . . . . . . . . . .
            . . . . . . . . . . . . . . . . . . .
            . . . . . . . . . . . . . . . . . . .
            . . . . . . . . . . . . . . . . . . .
            . . . . . . . . . . . . . . . . . . .
            . . . . . . . . . . . . . . . . . . .
            . . . . . . . . . . . . . . . . . . .
            . . . . . . . . . . . . . . . . . . .
            . . . . . . . . . . . . . . . . . . .
            . . . . . . . . . . . . . . . . . . .
            . . . . . . . . . . . . . . . . . . .
            . . . . . . . . . . . . . . . . . . .
            . . . . . . . . . . . . . . . . . . .
            . . . . . . . . . . . . . . . . . . .
            . . . . . . . . . . . . . . . . . . .
            . . . . . . . . . . . . . . . . . . .
            . . . . . . . . . . . . . . . . . . .
            . . . . . . . . . . . . . . . . . . .
            . . . . . . . . . . . . . . . . . . .
        ''')
    
    # ============================================================================
    # EDIT SECTION 2: Previous board position (optional - set to None if not needed)
    # ============================================================================
    previous_board_moves = None
    
    # ============================================================================
    # EDIT SECTION 3: Ownership map (19x19) from current player's perspective
    # ============================================================================
    # Positive values = good for current player, Negative = good for opponent
    ownership = np.zeros((19, 19))
    ownership[0:5, 0:5] = 0.8   # Strong territory in top-left corner
    ownership[14:19, 14:19] = -0.6  # Opponent territory in bottom-right
    
    # ============================================================================
    # EDIT SECTION 4: Previous ownership map (optional - set to None if not needed)
    # ============================================================================
    previous_ownership = None
    
    # ============================================================================
    # EDIT SECTION 5: Policy distribution (361 elements, should sum to ~1.0)
    # ============================================================================
    # Index = y * 19 + x for position (x, y)
    policy = np.ones(361) / 361  # Uniform policy
    
    # ============================================================================
    # Code below runs automatically - no editing needed
    # ============================================================================
    current_player = Board.BLACK
    current_move = None
    
    print("\nBoard position:")
    print(board_to_visual(board_moves))
    print()
    
    board = Board(19)
    for x, y, color in board_moves:
        player = Board.BLACK if color == 'B' else Board.WHITE
        board.play(player, board.loc(x, y))
    
    before_board = None
    if previous_board_moves is not None:
        before_board = create_board_from_moves(previous_board_moves)
    
    move_loc = None
    if current_move is not None:
        move_loc = board.loc(current_move[0], current_move[1])
    
    features = extract_features(
        board=board,
        ownership=ownership,
        policy=policy,
        player=current_player,
        move_loc=move_loc,
        before_ownership=previous_ownership,
        before_board=before_board
    )
    
    # Print some key features
    print(f"Potential territory: {features['potential_territory']}")
    print(f"Solid territory: {features['solid_territory']}")
    print(f"Urgency by region: {features['urgency']}")


def example_with_move(use_editor=False):
    """Example: Extract features with a move played."""
    print("\n" + "=" * 80)
    print("Example 2: Features with a move")
    print("=" * 80)
    
    # ============================================================================
    # EDIT SECTION 1: Board position
    # ============================================================================
    # Option 1: Use interactive editor (set use_editor=True)
    # Option 2: Use visual board representation (RECOMMENDED - easiest to edit!)
    # Option 3: Manually edit the board_moves list
    if use_editor:
        board_moves = interactive_board_editor(
            initial_moves=[(3, 3, 'B'), (15, 15, 'W'), (4, 3, 'B')],
            title="Example 2: Features with a move"
        )
        if board_moves is None:
            print("Editor cancelled, using default board.")
            board_moves = [(3, 3, 'B'), (15, 15, 'W'), (4, 3, 'B')]
    else:
        # VISUAL BOARD REPRESENTATION (easiest to edit!)
        # Edit by changing 'B' (Black), 'W' (White), or '.' (empty)
        board_moves = visual_board('''
            . . . . . . . . . . . . . . . . . . .
            . . . . . . . . . . . . . . . . . . .
            . . . . . B . . . . . . . . . . . . .
            . . . B . . . . . . . . . . . . . . .
            . . . . . . . . . . . . . . . . . . .
            . . . . . . . . . . . . . . . . . . .
            . . . . . . . . . . . . . . . . . . .
            . . . . . . . . . . . . . . . . . . .
            . . . . . . . . . . . . . . . . . . .
            . . . . . . . . . . . . . . . . . . .
            . . . . . . . . . . . . . . . . . . .
            . . . . . . . . . . . . . . . . . . .
            . . . . . . . . . . . . . . . . . . .
            . . . . . . . . . . . . . . . . . . .
            . . . . . . . . . . . . . . . W . . .
            . . . . . . . . . . . . . . . . . . .
            . . . . . . . . . . . . . . . . . . .
            . . . . . . . . . . . . . . . . . . .
            . . . . . . . . . . . . . . . . . . .
        ''')
    
    # ============================================================================
    # EDIT SECTION 2: Previous board position (optional - set to None if not needed)
    # ============================================================================
    previous_board_moves = None
    
    # ============================================================================
    # EDIT SECTION 3: Ownership map (19x19) from current player's perspective
    # ============================================================================
    # Positive values = good for current player, Negative = good for opponent
    ownership = np.zeros((19, 19))
    ownership[0:6, 0:6] = 0.7   # Black territory
    ownership[13:19, 13:19] = -0.7  # White territory
    
    # ============================================================================
    # EDIT SECTION 4: Previous ownership map (optional - set to None if not needed)
    # ============================================================================
    previous_ownership = None
    
    # ============================================================================
    # EDIT SECTION 5: Policy distribution (361 elements, should sum to ~1.0)
    # ============================================================================
    # Index = y * 19 + x for position (x, y)
    policy = np.zeros(361)
    policy[0] = 0.3    # Top-left corner (0, 0)
    policy[18] = 0.3   # Top-right corner (18, 0)
    policy[342] = 0.2  # Bottom-left corner (0, 18)
    policy[360] = 0.2  # Bottom-right corner (18, 18)
    
    # ============================================================================
    # Code below runs automatically - no editing needed
    # ============================================================================
    current_player = Board.BLACK
    current_move = (5, 3)  # Extension move at F16
    
    print("\nBoard position:")
    print(board_to_visual(board_moves, highlight_move=current_move))
    print(f"Current move: {current_move} (marked with *)")
    print()
    
    board = create_board_from_moves(board_moves)
    before_board = None
    if previous_board_moves is not None:
        before_board = create_board_from_moves(previous_board_moves)
    
    move_loc = None
    if current_move is not None:
        move_loc = board.loc(current_move[0], current_move[1])
    
    features = extract_features(
        board=board,
        ownership=ownership,
        policy=policy,
        player=current_player,
        move_loc=move_loc,
        before_ownership=previous_ownership,
        before_board=before_board
    )
    
    print(f"Connection: {features['connection']}")
    print(f"Extension: {features['extension']}")
    print(f"Building territory: {features['building_count']}")
    print(f"Urgency in corner_tl: {features['urgency']['corner_tl']:.3f}")


def example_before_after_comparison():
    """Example: Compare before and after a move."""
    print("\n" + "=" * 80)
    print("Example 3: Before/after comparison")
    print("=" * 80)
    
    # ============================================================================
    # EDIT SECTION 1: Board position
    # ============================================================================
    # VISUAL BOARD REPRESENTATION (easiest to edit!)
    # This is the board AFTER the move (includes all moves including the new one)
    board_moves = visual_board('''
        . . . . . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . . . . . .
        . . . . . B . . . . . . . . . . . . .
        . . . B . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . . W . . .
        . . . . . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . . . . . .
    ''')
    
    # ============================================================================
    # EDIT SECTION 2: Previous board position (before the move)
    # ============================================================================
    # This should include all moves from before, but NOT the new move
    previous_board_moves = visual_board('''
        . . . . . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . . . . . .
        . . . B . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . . W . . .
        . . . . . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . . . . . .
    ''')
    
    # ============================================================================
    # EDIT SECTION 3: Ownership map (19x19) from current player's perspective
    # ============================================================================
    # Positive values = good for current player, Negative = good for opponent
    # This is the ownership AFTER the move
    ownership = np.zeros((19, 19))
    ownership[0:6, 0:6] = 0.8   # Expanded black territory
    ownership[14:19, 14:19] = -0.4  # Reduced white territory
    
    # ============================================================================
    # EDIT SECTION 4: Previous ownership map (before the move)
    # ============================================================================
    # Ownership BEFORE the move (from current player's perspective)
    previous_ownership = np.zeros((19, 19))
    previous_ownership[0:5, 0:5] = 0.6   # Black territory
    previous_ownership[14:19, 14:19] = -0.4  # White territory
    
    # ============================================================================
    # EDIT SECTION 5: Policy distribution (361 elements, should sum to ~1.0)
    # ============================================================================
    # Index = y * 19 + x for position (x, y)
    policy = np.ones(361) / 361  # Uniform policy
    
    # ============================================================================
    # Code below runs automatically - no editing needed
    # ============================================================================
    current_player = Board.BLACK
    current_move = (4, 3)  # Extension move at E16 (the new move)
    
    print("\nBoard position BEFORE move:")
    print(board_to_visual(previous_board_moves))
    print("\nBoard position AFTER move:")
    print(board_to_visual(board_moves, highlight_move=current_move))
    print(f"Current move: {current_move} (marked with *)")
    print()
    
    board = create_board_from_moves(board_moves)
    before_board = create_board_from_moves(previous_board_moves)
    
    move_loc = None
    if current_move is not None:
        move_loc = board.loc(current_move[0], current_move[1])
    
    features = extract_features(
        board=board,
        ownership=ownership,
        policy=policy,
        player=current_player,
        move_loc=move_loc,
        before_ownership=previous_ownership,
        before_board=before_board
    )
    
    print(f"Building territory count: {features['building_count']}")
    print(f"Building territory intensity: {features['building_intensity']:.3f}")
    print(f"Reduction count: {features['reduction_count']}")
    print(f"Reduction intensity: {features['reduction_intensity']:.3f}")
    print(f"Group strength delta: {features['group_strength_delta']:.3f}")


def example_load_from_file():
    """Example: Load from game files."""
    print("\n" + "=" * 80)
    print("Example 4: Load from game files")
    print("=" * 80)
    
    # Check if game directory exists
    game_dir = Path(__file__).parent.parent / "games"
    if not game_dir.exists():
        print(f"Game directory not found: {game_dir}")
        print("Skipping this example.")
        return
    
    # Find first game directory
    game_dirs = [d for d in game_dir.iterdir() if d.is_dir()]
    if not game_dirs:
        print("No game directories found.")
        print("Skipping this example.")
        return
    
    game_path = game_dirs[0] / "moves.jsonl"
    if not game_path.exists():
        print(f"moves.jsonl not found in {game_dirs[0]}")
        print("Skipping this example.")
        return
    
    # Load first move
    with open(game_path, 'r') as f:
        move_data = json.loads(f.readline())
    
    # Extract data
    ownership = np.array(move_data['ownership']).reshape(19, 19)
    policy = np.array(move_data.get('policy0', [0] * 361)[:361])
    
    # Create board
    board = Board(19)
    move_loc = move_data.get('move_loc', None)
    if move_loc and move_loc != 0:
        player_str = move_data.get('player', 'b')
        player = Board.BLACK if player_str == 'b' else Board.WHITE
        try:
            board.play(player, move_loc)
        except:
            pass  # Board might already have the move
    
    # Extract features
    features = extract_features(
        board=board,
        ownership=ownership,
        policy=policy,
        player=Board.BLACK if move_data.get('player') == 'b' else Board.WHITE,
        move_loc=move_loc if move_loc else None
    )
    
    print(f"Move: {move_data.get('move_number', 'unknown')}")
    print(f"Player: {move_data.get('player', 'unknown')}")
    print(f"Potential territory: {features['potential_territory']}")
    print(f"Solid territory: {features['solid_territory']}")
    print(f"Total urgency: {sum(features['urgency'].values()):.3f}")


def example_interactive_editor():
    """Example: Use interactive board editor to create a board position."""
    print("=" * 80)
    print("Interactive Board Editor")
    print("=" * 80)
    print("\nOpening graphical board editor...")
    print("Instructions:")
    print("  - Left-click: Place stone (current color)")
    print("  - Right-click: Remove stone")
    print("  - Switch Color: Toggle between Black and White")
    print("  - Clear Board: Remove all stones")
    print("  - Export & Close: Copy board position to clipboard and return")
    print("  - Cancel: Close without exporting")
    print()
    
    # Optionally start with existing moves
    initial_moves = [
        (3, 3, 'B'),   # Example starting position
        (15, 15, 'W'),
    ]
    
    # Open editor (comment out initial_moves to start with empty board)
    board_moves = interactive_board_editor(initial_moves=None, title="Go Board Editor - Example")
    
    if board_moves is None:
        print("Editor was cancelled or tkinter is not available.")
        return
    
    print(f"\nExported board position with {len(board_moves)} stones:")
    for x, y, color in board_moves:
        coord = f"{chr(65+x)}{19-y}"
        print(f"  {color} at ({x}, {y}) = {coord}")
    
    # Now you can use board_moves in your code
    print("\nYou can now use this board_moves list in your example functions!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Example script for extract_features')
    parser.add_argument('--editor', action='store_true', 
                       help='Open interactive board editor instead of running examples')
    args = parser.parse_args()
    
    if args.editor:
        example_interactive_editor()
    else:
        # Run examples
        example_basic_usage()
        example_with_move()
        example_before_after_comparison()
        example_load_from_file()
        
        print("\n" + "=" * 80)
        print("Examples completed!")
        print("=" * 80)
        print("\nTo use interactive board editor:")
        print("  python example_extract_features.py --editor")
        print("\nTo use from command line:")
        print("  python extract_features.py --board empty --ownership ownership.npy")
        print("  python extract_features.py --game-dir games/XXX --move-number 10")

