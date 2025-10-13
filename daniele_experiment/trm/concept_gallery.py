#!/usr/bin/env python3
"""
Create a gallery of board positions that strongly activate each concept.

Usage:
    python -m daniele_experiment.trm.concept_gallery --model-path daniele_experiment/games/trm_model.pt --slates-jsonl daniele_experiment/games/slates.jsonl --output-dir concept_gallery/ --concepts-per-gallery 6 --boards-per-concept 36
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
                
                sample_info = {
                    "batch_idx": batch_idx,
                    "sample_idx": i,
                    "slate_id": meta["slate_id"] if "slate_id" in meta else f"batch_{batch_idx}_sample_{i}",
                    "trunkfinal_path": meta["trunkfinal_path"] if "trunkfinal_path" in meta else None,
                }
                all_sample_info.append(sample_info)
    
    # Combine all batches
    all_concept_activities = torch.cat(all_concept_activities, dim=0)  # [N, K]
    return all_concept_activities.numpy(), all_sample_info


def create_concept_gallery(
    model: GoExplainTRM,
    dataset: TRMDataset,
    concept_activities: np.ndarray,
    sample_info: List[Dict],
    output_dir: Path,
    concepts_per_gallery: int = 6,
    boards_per_concept: int = 36,
    device: str = "mps"
) -> None:
    """Create gallery of boards that strongly activate each concept."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_concepts = concept_activities.shape[1]
    num_galleries = (num_concepts + concepts_per_gallery - 1) // concepts_per_gallery
    
    print(f"Creating {num_galleries} galleries with {concepts_per_gallery} concepts each...")
    
    for gallery_idx in range(num_galleries):
        start_concept = gallery_idx * concepts_per_gallery
        end_concept = min(start_concept + concepts_per_gallery, num_concepts)
        concepts_in_gallery = end_concept - start_concept
        
        # Create figure with subplots
        fig, axes = plt.subplots(
            concepts_in_gallery, 
            boards_per_concept, 
            figsize=(boards_per_concept * 1.5, concepts_in_gallery * 1.5)
        )
        
        if concepts_in_gallery == 1:
            axes = axes.reshape(1, -1)
        if boards_per_concept == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle(f"Concept Gallery {gallery_idx + 1} - Concepts {start_concept} to {end_concept-1}", fontsize=16)
        
        for concept_idx in range(start_concept, end_concept):
            concept_row = concept_idx - start_concept
            
            # Find samples with highest activation for this concept
            concept_scores = concept_activities[:, concept_idx]
            top_indices = np.argsort(concept_scores)[-boards_per_concept:][::-1]
            
            print(f"Concept {concept_idx}: top activation = {concept_scores[top_indices[0]]:.4f}")
            
            for board_idx, sample_idx in enumerate(top_indices):
                # Load the sample
                sample = dataset[sample_idx]
                x_chw = sample["x_chw"].unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(x_chw)
                    YT_prob = F.softmax(outputs["YT"], dim=1)  # [1, K, H, W]
                    concept_map = YT_prob[0, concept_idx].cpu().numpy()  # [H, W]
                
                # Plot the concept map
                ax = axes[concept_row, board_idx]
                im = ax.imshow(concept_map, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
                ax.set_title(f"#{board_idx+1}\n{concept_scores[sample_idx]:.3f}", fontsize=8)
                ax.axis('off')
                
                # Add colorbar for first board of each concept
                if board_idx == 0:
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        for concept_row in range(concepts_in_gallery):
            for board_idx in range(boards_per_concept):
                if concept_row * boards_per_concept + board_idx >= concepts_in_gallery * boards_per_concept:
                    axes[concept_row, board_idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"concept_gallery_{gallery_idx + 1}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved gallery {gallery_idx + 1}")


def create_concept_summary(
    concept_activities: np.ndarray,
    sample_info: List[Dict],
    output_dir: Path,
    top_k: int = 10
) -> None:
    """Create a summary of concept statistics and top activations."""
    num_concepts = concept_activities.shape[1]
    
    # Compute statistics
    concept_means = np.mean(concept_activities, axis=0)
    concept_stds = np.std(concept_activities, axis=0)
    concept_maxs = np.max(concept_activities, axis=0)
    
    # Find top activations for each concept
    top_activations = {}
    for concept_idx in range(num_concepts):
        concept_scores = concept_activities[:, concept_idx]
        top_indices = np.argsort(concept_scores)[-top_k:][::-1]
        top_activations[concept_idx] = [
            {
                "sample_idx": idx,
                "score": concept_scores[idx],
                "slate_id": sample_info[idx]["slate_id"]
            }
            for idx in top_indices
        ]
    
    # Save summary
    summary_path = output_dir / "concept_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("TRM Concept Gallery Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total samples: {len(concept_activities)}\n")
        f.write(f"Total concepts: {num_concepts}\n\n")
        
        f.write("Concept Statistics:\n")
        f.write("-" * 30 + "\n")
        for concept_idx in range(num_concepts):
            f.write(f"Concept {concept_idx:2d}: mean={concept_means[concept_idx]:.4f} "
                   f"std={concept_stds[concept_idx]:.4f} max={concept_maxs[concept_idx]:.4f}\n")
        
        f.write(f"\nTop {top_k} Activations per Concept:\n")
        f.write("-" * 40 + "\n")
        for concept_idx in range(num_concepts):
            f.write(f"\nConcept {concept_idx}:\n")
            for i, activation in enumerate(top_activations[concept_idx]):
                f.write(f"  {i+1:2d}. Score: {activation['score']:.4f} "
                       f"Slate: {activation['slate_id']}\n")
    
    print(f"Saved concept summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Create concept activation gallery")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to trained TRM model")
    parser.add_argument("--slates-jsonl", type=Path, required=True, help="Path to slates.jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path("concept_gallery"), help="Output directory")
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
    
    # Create concept gallery
    print("Creating concept galleries...")
    create_concept_gallery(
        model, dataset, concept_activities, sample_info, args.output_dir,
        concepts_per_gallery=args.concepts_per_gallery,
        boards_per_concept=args.boards_per_concept,
        device=args.device
    )
    
    # Create summary
    print("Creating concept summary...")
    create_concept_summary(concept_activities, sample_info, args.output_dir)
    
    print(f"Concept gallery complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
