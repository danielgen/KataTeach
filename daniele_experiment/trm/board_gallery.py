#!/usr/bin/env python3
"""
Create a gallery of actual Go board positions that strongly activate each concept.

Usage:
    python -m daniele_experiment.trm.board_gallery --model-path daniele_experiment/games/trm_model.pt --slates-jsonl daniele_experiment/games/slates.jsonl --sgf-dir daniele_experiment/games/ --output-dir board_gallery/ --concepts-per-gallery 6 --boards-per-concept 36
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add python directory to path for KataGo modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "python"))

from board import Board
from sgfmill import sgf, sgf_moves, common as sgf_common

from .dataset import TRMDataset
from .model import GoExplainTRM


def load_model(model_path: Path, device: str = "mps") -> GoExplainTRM:
    """Load trained TRM model."""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint["config"]
    
    model = GoExplainTRM(
        c_in=config["c_in"],
        k_concepts=config["k_concepts"],
        z_ch=config["z_ch"],
        T=config["T"]
    ).to(device)
    
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def reconstruct_board_from_sgf(sgf_path: Path, move_number: int) -> Board:
    """Reconstruct board state from SGF up to a specific move number."""
    with open(sgf_path, 'rb') as f:
        sgf_bytes = f.read()
        game = sgf.Sgf_game.from_bytes(sgf_bytes)
    
    board_size = game.get_size()
    board = Board(size=board_size)
    
    # Get the main sequence of moves
    sequence = game.get_main_sequence()
    moves_played = 0
    
    for node in sequence[1:]:  # Skip root node
        if moves_played >= move_number:
            break
            
        if node.has_property("B"):
            try:
                color, move = node.get_move()
                if color == "b" and move is not None:
                    row, col = move
                    loc = board.loc(col, row)
                    board.play(Board.BLACK, loc)
                    moves_played += 1
            except ValueError:
                pass
        elif node.has_property("W"):
            try:
                color, move = node.get_move()
                if color == "w" and move is not None:
                    row, col = move
                    loc = board.loc(col, row)
                    board.play(Board.WHITE, loc)
                    moves_played += 1
            except ValueError:
                pass
    
    return board


def draw_go_board(board: Board, ax, title: str = "", show_coordinates: bool = True):
    """Draw a Go board with stones."""
    size = board.size
    
    # Set beige background
    ax.set_facecolor('#F5E6D3')
    
    # Draw board border
    border = plt.Rectangle((-0.5, -0.5), size, size, 
                          fill=False, edgecolor='black', linewidth=2.0, zorder=1)
    ax.add_patch(border)
    
    # Draw board grid lines (only inside the board)
    for i in range(size):
        # Horizontal lines
        ax.plot([0, size-1], [i, i], color='black', linewidth=1.0, zorder=2)
        # Vertical lines  
        ax.plot([i, i], [0, size-1], color='black', linewidth=1.0, zorder=2)
    
    # Draw stones with larger radius
    for y in range(size):
        for x in range(size):
            loc = board.loc(x, y)
            if board.board[loc] == Board.BLACK:
                circle = plt.Circle((x, y), 0.35, color='black', zorder=3)
                ax.add_patch(circle)
            elif board.board[loc] == Board.WHITE:
                circle = plt.Circle((x, y), 0.35, facecolor='white', edgecolor='black', linewidth=1.0, zorder=3)
                ax.add_patch(circle)
    
    # Draw coordinates with larger font
    if show_coordinates:
        coord_labels = 'ABCDEFGHJKLMNOPQRST'
        for i in range(size):
            ax.text(i, -0.5, coord_labels[i], ha='center', va='center', fontsize=10)
            ax.text(-0.5, i, str(size - i), ha='center', va='center', fontsize=10)
    
    ax.set_xlim(-1, size)
    ax.set_ylim(-1, size)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12)
    ax.axis('off')


def find_concept_activations(
    model: GoExplainTRM,
    dataset: TRMDataset,
    device: str = "mps",
    batch_size: int = 8
) -> Tuple[np.ndarray, List[Dict]]:
    """Find concept activations for all samples."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    all_concept_activities = []
    all_sample_info = []
    
    print("Computing concept activations for all samples...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx % 50 == 0:
                print(f"Processing batch {batch_idx}/{len(loader)}")
            
            x_chw = batch["x_chw"].to(device)
            outputs = model(x_chw)
            
            YT_prob = F.softmax(outputs["YT"], dim=1)  # [B, K, H, W]
            concept_activity = torch.mean(YT_prob, dim=(2, 3))  # [B, K] - average over spatial dimensions
            
            all_concept_activities.append(concept_activity.cpu())
            
            # Store sample info - handle batched structure
            batch_size = len(batch["x_chw"])
            for i in range(batch_size):
                # When DataLoader batches samples, it creates lists for each key
                if isinstance(batch["meta"], list):
                    meta = batch["meta"][i]
                else:
                    meta = batch["meta"]
                
                # Extract slate_id properly
                slate_id = meta.get("slate_id", f"batch_{batch_idx}_sample_{i}")
                if isinstance(slate_id, list):
                    slate_id = slate_id[0]  # Take first element if it's a list
                
                sample_info = {
                    "batch_idx": batch_idx,
                    "sample_idx": i,
                    "slate_id": slate_id,
                    "trunkfinal_path": meta.get("trunkfinal_path"),
                }
                all_sample_info.append(sample_info)
    
    # Combine all batches
    all_concept_activities = torch.cat(all_concept_activities, dim=0)  # [N, K]
    return all_concept_activities.numpy(), all_sample_info


def create_board_gallery(
    model: GoExplainTRM,
    dataset: TRMDataset,
    concept_activities: np.ndarray,
    sample_info: List[Dict],
    sgf_dir: Path,
    output_dir: Path,
    concepts_per_gallery: int = 6,
    boards_per_concept: int = 36,
    device: str = "mps"
) -> None:
    """Create gallery of actual Go boards that strongly activate each concept."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_concepts = concept_activities.shape[1]
    num_galleries = (num_concepts + concepts_per_gallery - 1) // concepts_per_gallery
    
    print(f"Creating {num_galleries} board galleries with {concepts_per_gallery} concepts each...")
    
    for gallery_idx in range(num_galleries):
        start_concept = gallery_idx * concepts_per_gallery
        end_concept = min(start_concept + concepts_per_gallery, num_concepts)
        concepts_in_gallery = end_concept - start_concept
        
        # Create figure with subplots - make boards larger
        fig, axes = plt.subplots(
            concepts_in_gallery, 
            boards_per_concept, 
            figsize=(boards_per_concept * 2.5, concepts_in_gallery * 2.5)
        )
        
        if concepts_in_gallery == 1:
            axes = axes.reshape(1, -1)
        if boards_per_concept == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle(f"Board Gallery {gallery_idx + 1} - Concepts {start_concept} to {end_concept-1}", fontsize=16)
        
        for concept_idx in range(start_concept, end_concept):
            concept_row = concept_idx - start_concept
            
            # Find samples with highest activation for this concept
            concept_scores = concept_activities[:, concept_idx]
            
            # Get top samples, but ensure diversity by avoiding duplicates
            sorted_indices = np.argsort(concept_scores)[::-1]  # Sort in descending order
            top_indices = []
            seen_slate_ids = set()
            
            for idx in sorted_indices:
                if len(top_indices) >= boards_per_concept:
                    break
                
                slate_id = sample_info[idx]["slate_id"]
                if isinstance(slate_id, list):
                    slate_id = slate_id[0]
                
                # Only add if we haven't seen this slate_id before
                if slate_id not in seen_slate_ids:
                    top_indices.append(idx)
                    seen_slate_ids.add(slate_id)
            
            # If we don't have enough unique samples, fill with remaining top samples
            if len(top_indices) < boards_per_concept:
                for idx in sorted_indices:
                    if len(top_indices) >= boards_per_concept:
                        break
                    if idx not in top_indices:
                        top_indices.append(idx)
            
            print(f"Concept {concept_idx}: top activation = {concept_scores[top_indices[0]]:.4f}, unique samples = {len(set(sample_info[i]['slate_id'] if not isinstance(sample_info[i]['slate_id'], list) else sample_info[i]['slate_id'][0] for i in top_indices))}")
            
            for board_idx, sample_idx in enumerate(top_indices):
                ax = axes[concept_row, board_idx]
                
                try:
                    # Parse slate_id to get game_uuid and move number
                    slate_id = sample_info[sample_idx]["slate_id"]
                    
                    # Handle case where slate_id might be a list (due to batching)
                    if isinstance(slate_id, list):
                        slate_id = slate_id[0]  # Take first element
                    
                    if ":" in slate_id:
                        game_uuid, move_str = slate_id.split(":")
                        move_number = int(move_str)
                    else:
                        # Fallback parsing
                        game_uuid = slate_id
                        move_number = 0
                    
                    # Find corresponding SGF file
                    sgf_path = sgf_dir / f"{game_uuid}.sgf"
                    
                    if sgf_path.exists():
                        # Reconstruct board state
                        board = reconstruct_board_from_sgf(sgf_path, move_number)
                        
                        # Draw the board
                        title = f"#{board_idx+1}\n{concept_scores[sample_idx]:.3f}\nMove {move_number}"
                        draw_go_board(board, ax, title, show_coordinates=False)
                        
                    else:
                        # SGF not found, show empty board
                        ax.text(0.5, 0.5, f"SGF not found\n{game_uuid}", 
                               ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f"#{board_idx+1}\n{concept_scores[sample_idx]:.3f}")
                        ax.axis('off')
                        
                except Exception as e:
                    # Error reconstructing board, show error message
                    ax.text(0.5, 0.5, f"Error:\n{str(e)[:30]}...", 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"#{board_idx+1}\n{concept_scores[sample_idx]:.3f}")
                    ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"board_gallery_{gallery_idx + 1}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved board gallery {gallery_idx + 1}")


def main():
    parser = argparse.ArgumentParser(description="Create board position gallery for concept activations")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to trained TRM model")
    parser.add_argument("--slates-jsonl", type=Path, required=True, help="Path to slates.jsonl")
    parser.add_argument("--sgf-dir", type=Path, required=True, help="Directory containing SGF files")
    parser.add_argument("--output-dir", type=Path, default=Path("board_gallery"), help="Output directory")
    parser.add_argument("--device", type=str, default="mps", help="Device to use")
    parser.add_argument("--concepts-per-gallery", type=int, default=6, help="Number of concepts per gallery")
    parser.add_argument("--boards-per-concept", type=int, default=36, help="Number of boards per concept")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Load model and dataset
    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, args.device)
    
    print(f"Loading dataset from {args.slates_jsonl}")
    dataset = TRMDataset(slates_jsonl=args.slates_jsonl)
    
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Input shape: {dataset.chw_shape}")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find concept activations
    concept_activities, sample_info = find_concept_activations(
        model, dataset, args.device, args.batch_size
    )
    
    # Create board gallery
    print("Creating board galleries...")
    create_board_gallery(
        model, dataset, concept_activities, sample_info, args.sgf_dir, args.output_dir,
        concepts_per_gallery=args.concepts_per_gallery,
        boards_per_concept=args.boards_per_concept,
        device=args.device
    )
    
    print(f"Board gallery complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
