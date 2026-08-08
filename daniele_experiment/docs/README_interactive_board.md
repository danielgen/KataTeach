# Interactive Go Board with Ownership Visualization

This script provides an interactive Go board for playing against a KataGo model in a Jupyter notebook, with real-time ownership visualization.

## Features

- **Interactive play**: Click on intersections to play moves
- **Automatic model response**: The model responds automatically after your move
- **Ownership visualization**: See ownership values as a color-coded heatmap overlay
  - Red areas = Current player's territory (positive ownership)
  - Blue areas = Opponent's territory (negative ownership)
- **Move controls**: Reset button and Pass button for game control

## Usage in Jupyter Notebook

### Basic Usage

```python
from daniele_experiment.interactive_play_with_ownership import create_interactive_board

board = create_interactive_board(
    model_path="daniele_experiment/model.ckpt",
    board_size=19,
    device="auto",
    prob_threshold=0.01,
)

board.show(figsize=(12, 12))
```

### Advanced Usage

```python
# Get ownership values programmatically
ownership = board.get_ownership()
print(f"Ownership shape: {ownership.shape}")
print(f"Ownership range: [{ownership.min():.3f}, {ownership.max():.3f}]")

# Get move history
moves = board.get_move_history()
for i, (player, loc) in enumerate(moves):
    player_str = "Black" if player == 1 else "White"
    print(f"Move {i+1}: {player_str} plays at location {loc}")
```

## How It Works

1. **Click on an intersection** to play your move
2. The model automatically evaluates the position and responds
3. **Ownership values** are displayed as a semi-transparent overlay:
   - Positive values (red) = territory controlled by the current player
   - Negative values (blue) = territory controlled by the opponent
4. Use the **Reset** button to start a new game
5. Use the **Pass** button to pass your turn

## Ownership Convention

Ownership values are normalized to the **current player's perspective**:
- Positive values = Current player's territory
- Negative values = Opponent's territory

This means the colors will flip when the turn changes, always showing the current player's perspective.

## Requirements

- Jupyter notebook or JupyterLab
- matplotlib
- numpy
- KataGo model checkpoint file
- All KataGo Python dependencies (from the `python/` directory)

Run the notebook from the repository root. The checkpoint is not included in
Git; its expected path and SHA-256 are listed in
[`../README.md`](../README.md#data-and-checkpoint-availability).

## Notes

- The board uses a 19x19 grid by default (can be changed to 9 or 13)
- Model moves are selected using probability sampling with the specified threshold
- The game ends automatically after two consecutive passes
- Ownership values update after each move
