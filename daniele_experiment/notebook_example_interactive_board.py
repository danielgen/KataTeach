"""
Example notebook cell code for using the interactive Go board.

Copy this code into a Jupyter notebook cell to play against the model.
"""

# Cell 1: Import and setup
import sys
from pathlib import Path

# Add paths
# Add the current directory (daniele_experiment) to find interactive_play_with_ownership
sys.path.insert(0, str(Path.cwd()))
# Add the parent's python directory for KataGo modules
sys.path.append(str(Path.cwd().parent / "python"))

from interactive_play_with_ownership import InteractiveGoBoard, create_interactive_board

# Cell 2: Create and show the board
# Adjust model_path to point to your model.ckpt file
board = create_interactive_board(
    model_path="model.ckpt",  # Update this path if needed
    board_size=19,
    device="auto",  # Will auto-detect mps/cuda/cpu
    prob_threshold=0.01  # Probability threshold for model moves
)

# Display the interactive board
# Click on intersections to play moves
# The model will automatically respond
board.show(figsize=(12, 12))

# Cell 3 (optional): Get ownership values programmatically
# After playing some moves, you can access ownership:
ownership = board.get_ownership()
if ownership is not None:
    print(f"Ownership shape: {ownership.shape}")
    print(f"Ownership range: [{ownership.min():.3f}, {ownership.max():.3f}]")
    
    # You can also visualize ownership separately
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.imshow(ownership, cmap='RdBu_r', vmin=-1, vmax=1, origin='upper')
    plt.colorbar(label='Ownership (positive=Black, negative=White)')
    plt.title('Current Ownership Map')
    plt.show()

# Cell 4 (optional): Get move history
moves = board.get_move_history()
print(f"Total moves played: {len(moves)}")
for i, (player, loc) in enumerate(moves):
    player_str = "Black" if player == 1 else "White"
    if loc == 0:  # Pass
        loc_str = "pass"
    else:
        # Convert to human-readable coordinates
        col_labels = 'ABCDEFGHJKLMNOPQRST'
        x = board.gs.board.loc_x(loc)
        y = board.gs.board.loc_y(loc)
        loc_str = f"{col_labels[x]}{19 - y}"
    print(f"Move {i+1}: {player_str} plays {loc_str}")

