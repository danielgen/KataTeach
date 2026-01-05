#!/usr/bin/env python3
"""
Interactive Go board for playing against KataGo model with ownership visualization.

This script provides an interactive board that can be used in a Jupyter notebook.
Click on intersections to play moves, and the model will respond automatically.
Ownership values are displayed as a heatmap overlay.

Usage in notebook:
    from interactive_play_with_ownership import InteractiveGoBoard
    
    board = InteractiveGoBoard(model_path="model.ckpt")
    board.show()  # Display the board
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import matplotlib

# Add python directory to path for KataGo modules
sys.path.append(str(Path(__file__).parent.parent / "python"))

from gamestate import GameState, Board
from load_model import load_model
from common_utils import get_device, select_move_with_sampling


class InteractiveGoBoard:
    """Interactive Go board for playing against KataGo model."""
    
    def __init__(
        self,
        model_path: str | Path = "model.ckpt",
        board_size: int = 19,
        device: str = "auto",
        prob_threshold: float = 0.01,
    ):
        """
        Initialize the interactive board.
        
        Args:
            model_path: Path to KataGo model checkpoint
            board_size: Board size (default 19)
            device: PyTorch device (default "auto")
            prob_threshold: Probability threshold for model move selection
        """
        self.board_size = board_size
        self.prob_threshold = prob_threshold
        
        # Load model
        print(f"Loading model from {model_path}...")
        dev = get_device(device)
        print(f"Using device: {dev}")
        model, swa_model, _ = load_model(str(model_path), use_swa=False, device=dev, pos_len=board_size, verbose=False)
        if swa_model is not None:
            model = swa_model
        model.eval()
        self.model = model
        
        # Initialize game state
        self.gs = GameState(board_size, GameState.RULES_TT)
        self.move_history = []
        
        # Matplotlib setup
        self.fig = None
        self.ax = None
        self.ownership_im = None
        self.click_handler = None
        
        # Current ownership data
        self.current_ownership = None
        
    def _loc_to_xy(self, loc: int) -> Tuple[int, int]:
        """Convert KataGo location to (x, y) coordinates."""
        if loc == Board.PASS_LOC:
            return None, None
        x = self.gs.board.loc_x(loc)
        y = self.gs.board.loc_y(loc)
        return x, y
    
    def _xy_to_loc(self, x: int, y: int) -> int:
        """Convert (x, y) coordinates to KataGo location."""
        return self.gs.board.loc(x, y)
    
    def _get_ownership(self) -> np.ndarray:
        """Get ownership values from model evaluation."""
        outputs = self.gs.get_model_outputs(self.model)
        ownership = outputs.get("ownership", None)
        
        if ownership is None:
            return np.zeros((self.board_size, self.board_size), dtype=np.float32)
        
        # Handle different ownership shapes
        if ownership.ndim == 3:
            ownership = ownership[0]  # Take first batch element
        
        # Reshape to board size if needed
        if ownership.shape != (self.board_size, self.board_size):
            ownership = ownership.reshape(self.board_size, self.board_size)
        
        # Normalize ownership to current player's perspective
        # KataGo outputs ownership from White's perspective (positive = White territory)
        # Flip only for Black to convert to current player's perspective
        # After normalization: positive = current player's territory, negative = opponent's territory
        if self.gs.board.pla == Board.BLACK:
            ownership = -ownership
        
        return ownership
    
    def _model_move(self) -> Optional[int]:
        """Get model's move response."""
        outputs = self.gs.get_model_outputs(self.model)
        moves_and_probs = outputs.get("moves_and_probs0", [])
        
        if not moves_and_probs:
            return None
        
        # Select move using sampling
        move_loc, move_prob, _ = select_move_with_sampling(moves_and_probs, self.prob_threshold)
        return move_loc
    
    def _draw_board(self):
        """Draw the Go board with stones and ownership overlay."""
        if self.ax is None:
            return
        
        self.ax.clear()
        
        # Draw board background
        self.ax.set_xlim(-0.5, self.board_size - 0.5)
        self.ax.set_ylim(-0.5, self.board_size - 0.5)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        # Board background color
        self.ax.add_patch(patches.Rectangle(
            (-0.5, -0.5), self.board_size, self.board_size,
            facecolor='#DCB35C', edgecolor='black', linewidth=2
        ))
        
        # Draw grid lines
        for i in range(self.board_size):
            # Horizontal lines
            self.ax.plot([-0.5, self.board_size - 0.5], [i, i], 'k-', linewidth=0.5)
            # Vertical lines
            self.ax.plot([i, i], [-0.5, self.board_size - 0.5], 'k-', linewidth=0.5)
        
        # Draw star points
        star_points = [3, 9, 15] if self.board_size == 19 else [2, 6] if self.board_size == 9 else [3, 9]
        for x in star_points:
            for y in star_points:
                if x < self.board_size and y < self.board_size:
                    self.ax.plot(x, y, 'ko', markersize=4)
        
        # Draw ownership heatmap
        ownership = self._get_ownership()
        self.current_ownership = ownership
        
        # Normalize ownership for display (-1 to 1)
        vmax = max(abs(ownership.max()), abs(ownership.min()), 0.01)
        
        # Create colormap: red for positive (current player), blue for negative (opponent)
        ownership_display = ownership.copy()
        
        # Draw ownership as semi-transparent overlay
        # Ownership array: ownership[y, x] where y=0 is top row (A19), y=18 is bottom row (A1)
        im = self.ax.imshow(
            ownership_display,
            extent=[-0.5, self.board_size - 0.5, self.board_size - 0.5, -0.5],
            cmap='RdBu_r',
            vmin=-vmax,
            vmax=vmax,
            alpha=0.5,
            interpolation='nearest',
            origin='upper'
        )
        self.ownership_im = im
        
        # Draw stones first (so ownership labels appear on top)
        stones_drawn = []
        for y in range(self.board_size):
            for x in range(self.board_size):
                loc = self._xy_to_loc(x, y)
                stone = self.gs.board.board[loc]
                
                if stone == Board.BLACK:
                    circle = patches.Circle((x, y), 0.4, facecolor='black', edgecolor='black', linewidth=0.5)
                    self.ax.add_patch(circle)
                    stones_drawn.append((x, y))
                elif stone == Board.WHITE:
                    circle = patches.Circle((x, y), 0.4, facecolor='white', edgecolor='black', linewidth=0.5)
                    self.ax.add_patch(circle)
                    stones_drawn.append((x, y))
        
        # Add ownership value text labels
        # Show values for intersections with significant ownership (threshold can be adjusted)
        ownership_threshold = 0.15  # Only show values above this threshold
        for y in range(self.board_size):
            for x in range(self.board_size):
                own_val = ownership[y, x]
                if abs(own_val) > ownership_threshold:
                    # Check if there's a stone at this position
                    has_stone = (x, y) in stones_drawn
                    
                    # Format the value (show 1 decimal place, or integer if close to integer)
                    if abs(own_val) < 0.05:
                        val_str = f'{own_val:.2f}'
                    else:
                        val_str = f'{own_val:.1f}'
                    
                    # Choose text color based on ownership value and whether there's a stone
                    if has_stone:
                        # On stones, use contrasting color
                        text_color = 'white' if own_val > 0 else 'black'
                        # Use a semi-transparent background for better readability
                        # Convert rgba to matplotlib tuple format (r, g, b, alpha) with values 0-1
                        bbox_props = dict(boxstyle='round,pad=0.15', 
                                        facecolor=(0.5, 0.5, 0.5, 0.6),  # gray with alpha
                                        edgecolor='none')
                    else:
                        # On empty intersections, use color based on ownership
                        text_color = 'black' if own_val > 0 else 'white'
                        # Use colored background matching ownership
                        # Convert rgba to matplotlib tuple format (r, g, b, alpha) with values 0-1
                        if own_val > 0:
                            bbox_color = (1.0, 0.78, 0.78, 0.7)  # light red for positive
                        else:
                            bbox_color = (0.78, 0.78, 1.0, 0.7)  # light blue for negative
                        bbox_props = dict(boxstyle='round,pad=0.15', 
                                        facecolor=bbox_color,
                                        edgecolor='none')
                    
                    self.ax.text(x, y, val_str, 
                               ha='center', va='center', 
                               fontsize=7, color=text_color,
                               weight='bold',
                               bbox=bbox_props,
                               zorder=20)  # High z-order to appear on top
        
        # Draw coordinate labels
        col_labels = 'ABCDEFGHJKLMNOPQRST'[:self.board_size]
        for i, label in enumerate(col_labels):
            self.ax.text(i, -0.8, label, ha='center', va='top', fontsize=8)
            self.ax.text(i, self.board_size - 0.2, label, ha='center', va='bottom', fontsize=8)
        
        for i in range(self.board_size):
            num_label = str(self.board_size - i)
            self.ax.text(-0.8, i, num_label, ha='right', va='center', fontsize=8)
            self.ax.text(self.board_size - 0.2, i, num_label, ha='left', va='center', fontsize=8)
        
        # Title with current player and move count
        current_player = "Black" if self.gs.board.pla == Board.BLACK else "White"
        title = f"Move {len(self.move_history)}: {current_player} to play"
        self.ax.set_title(title, fontsize=12, fontweight='bold')
        
        self.fig.canvas.draw()
    
    def _on_click(self, event):
        """Handle mouse clicks on the board."""
        if event.inaxes != self.ax:
            return
        
        # Convert click coordinates to board coordinates
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        
        if not (0 <= x < self.board_size and 0 <= y < self.board_size):
            return
        
        loc = self._xy_to_loc(x, y)
        current_player = self.gs.board.pla
        
        # Check if move is legal
        if not self.gs.board.would_be_legal(current_player, loc):
            print(f"Illegal move at ({x}, {y})")
            return
        
        # Play human move
        self.gs.play(current_player, loc)
        self.move_history.append((current_player, loc))
        print(f"Human ({'Black' if current_player == Board.BLACK else 'White'}) plays: {self._loc_to_coord_string(loc)}")
        
        # Update display
        self._draw_board()
        
        # Check for game end (two passes)
        if len(self.move_history) >= 2:
            last_two = self.move_history[-2:]
            if last_two[0][1] == Board.PASS_LOC and last_two[1][1] == Board.PASS_LOC:
                print("Game ended: Two consecutive passes")
                return
        
        # Get model response
        model_loc = self._model_move()
        if model_loc is None:
            print("No legal moves for model")
            return
        
        model_player = self.gs.board.pla
        if model_loc == Board.PASS_LOC:
            print(f"Model ({'Black' if model_player == Board.BLACK else 'White'}) passes")
        else:
            print(f"Model ({'Black' if model_player == Board.BLACK else 'White'}) plays: {self._loc_to_coord_string(model_loc)}")
        
        # Play model move
        self.gs.play(model_player, model_loc)
        self.move_history.append((model_player, model_loc))
        
        # Update display
        self._draw_board()
    
    def _loc_to_coord_string(self, loc: int) -> str:
        """Convert location to human-readable coordinate string."""
        if loc == Board.PASS_LOC:
            return "pass"
        x, y = self._loc_to_xy(loc)
        col_labels = 'ABCDEFGHJKLMNOPQRST'
        return f"{col_labels[x]}{self.board_size - y}"
    
    def _on_reset(self, event):
        """Reset the game."""
        self.gs = GameState(self.board_size, GameState.RULES_TT)
        self.move_history = []
        self._draw_board()
        print("Game reset")
    
    def _on_pass(self, event):
        """Play a pass move."""
        current_player = self.gs.board.pla
        self.gs.play(current_player, Board.PASS_LOC)
        self.move_history.append((current_player, Board.PASS_LOC))
        print(f"Human ({'Black' if current_player == Board.BLACK else 'White'}) passes")
        
        # Check for game end
        if len(self.move_history) >= 2:
            last_two = self.move_history[-2:]
            if last_two[0][1] == Board.PASS_LOC and last_two[1][1] == Board.PASS_LOC:
                print("Game ended: Two consecutive passes")
                self._draw_board()
                return
        
        # Get model response
        model_loc = self._model_move()
        if model_loc is None:
            print("No legal moves for model")
            self._draw_board()
            return
        
        model_player = self.gs.board.pla
        if model_loc == Board.PASS_LOC:
            print(f"Model ({'Black' if model_player == Board.BLACK else 'White'}) passes")
        else:
            print(f"Model ({'Black' if model_player == Board.BLACK else 'White'}) plays: {self._loc_to_coord_string(model_loc)}")
        
        self.gs.play(model_player, model_loc)
        self.move_history.append((model_player, model_loc))
        self._draw_board()
    
    def show(self, figsize: Tuple[int, int] = (10, 10)):
        """
        Display the interactive board.
        
        Args:
            figsize: Figure size (width, height) in inches
        """
        # Use non-blocking backend for notebooks
        if matplotlib.get_backend() not in ['nbAgg', 'TkAgg', 'Qt5Agg']:
            try:
                matplotlib.use('TkAgg')
            except:
                pass
        
        self.fig, self.ax = plt.subplots(figsize=figsize)
        
        # Draw initial board
        self._draw_board()
        
        # Connect click handler
        self.click_handler = self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        
        # Add control buttons
        ax_reset = plt.axes([0.02, 0.02, 0.1, 0.04])
        btn_reset = Button(ax_reset, 'Reset')
        btn_reset.on_clicked(self._on_reset)
        
        ax_pass = plt.axes([0.14, 0.02, 0.1, 0.04])
        btn_pass = Button(ax_pass, 'Pass')
        btn_pass.on_clicked(self._on_pass)
        
        plt.tight_layout()
        plt.show()
    
    def get_ownership(self) -> Optional[np.ndarray]:
        """Get current ownership values."""
        return self.current_ownership
    
    def get_move_history(self) -> list:
        """Get the move history."""
        return self.move_history.copy()


def create_interactive_board(
    model_path: str | Path = "model.ckpt",
    board_size: int = 19,
    device: str = "auto",
    prob_threshold: float = 0.01,
) -> InteractiveGoBoard:
    """
    Convenience function to create an interactive board.
    
    Usage:
        board = create_interactive_board("model.ckpt")
        board.show()
    """
    return InteractiveGoBoard(
        model_path=model_path,
        board_size=board_size,
        device=device,
        prob_threshold=prob_threshold,
    )


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive Go board vs KataGo model")
    parser.add_argument("--model", type=Path, default="model.ckpt", help="Path to model checkpoint")
    parser.add_argument("--board-size", type=int, default=19, choices=[9, 13, 19], help="Board size")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/mps/cpu)")
    parser.add_argument("--prob-threshold", type=float, default=0.01, help="Probability threshold for moves")
    
    args = parser.parse_args()
    
    board = InteractiveGoBoard(
        model_path=args.model,
        board_size=args.board_size,
        device=args.device,
        prob_threshold=args.prob_threshold,
    )
    board.show()

