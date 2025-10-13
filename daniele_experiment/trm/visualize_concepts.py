#!/usr/bin/env python3
"""
Visualize TRM concept maps and activations.

Usage:
    python -m daniele_experiment.trm.visualize_concepts --model-path daniele_experiment/games/trm_model.pt --slates-jsonl daniele_experiment/games/slates.jsonl --output-dir concept_viz/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

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


def visualize_concept_maps(
    model: GoExplainTRM,
    dataset: TRMDataset,
    output_dir: Path,
    num_samples: int = 5,
    top_k_concepts: int = 8,
    device: str = "mps"
) -> None:
    """Visualize concept maps for sample positions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample random positions
    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        x_chw = sample["x_chw"].unsqueeze(0).to(device)  # [1, C, H, W]
        
        with torch.no_grad():
            outputs = model(x_chw)
            Y0 = outputs["Y0"]  # [1, K, H, W] - initial concept logits
            YT = outputs["YT"]  # [1, K, H, W] - final concept logits
            Ev = outputs["Ev"]  # [1, D, H, W] - evidence features
        
        # Convert to probabilities
        Y0_prob = F.softmax(Y0, dim=1).squeeze(0).cpu().numpy()  # [K, H, W]
        YT_prob = F.softmax(YT, dim=1).squeeze(0).cpu().numpy()  # [K, H, W]
        
        # Find most active concepts
        concept_activity = np.mean(YT_prob, axis=(1, 2))  # [K]
        top_concept_indices = np.argsort(concept_activity)[-top_k_concepts:][::-1]
        
        # Create visualization
        fig, axes = plt.subplots(2, top_k_concepts + 1, figsize=(20, 8))
        fig.suptitle(f"TRM Concept Maps - Sample {i+1} (Slate: {sample['meta']['slate_id']})", fontsize=16)
        
        # Show evidence (first few channels)
        evidence_channels = min(3, Ev.shape[1])
        for c in range(evidence_channels):
            if c < 3:  # Only show first 3 evidence channels
                ev_data = Ev[0, c].cpu().numpy()
                im = axes[0, c].imshow(ev_data, cmap='viridis', interpolation='nearest')
                axes[0, c].set_title(f"Evidence {c}")
                axes[0, c].axis('off')
                # Add colorbar for evidence
                plt.colorbar(im, ax=axes[0, c], fraction=0.046, pad=0.04)
        
        # Show concept maps (initial)
        for j, concept_idx in enumerate(top_concept_indices):
            col_idx = j + evidence_channels
            if col_idx < axes.shape[1]:
                concept_data = Y0_prob[concept_idx]
                im = axes[0, col_idx].imshow(concept_data, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
                axes[0, col_idx].set_title(f"Concept {concept_idx} (init)\nActivity: {concept_activity[concept_idx]:.3f}")
                axes[0, col_idx].axis('off')
                plt.colorbar(im, ax=axes[0, col_idx], fraction=0.046, pad=0.04)
        
        # Show concept maps (final)
        for j, concept_idx in enumerate(top_concept_indices):
            col_idx = j + evidence_channels
            if col_idx < axes.shape[1]:
                concept_data = YT_prob[concept_idx]
                im = axes[1, col_idx].imshow(concept_data, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
                axes[1, col_idx].set_title(f"Concept {concept_idx} (final)\nActivity: {concept_activity[concept_idx]:.3f}")
                axes[1, col_idx].axis('off')
                plt.colorbar(im, ax=axes[1, col_idx], fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        for row in range(2):
            for col in range(evidence_channels + top_k_concepts, axes.shape[1]):
                axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"concept_maps_sample_{i+1}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved concept maps for sample {i+1}")


def analyze_concept_statistics(
    model: GoExplainTRM,
    dataset: TRMDataset,
    output_dir: Path,
    device: str = "mps",
    batch_size: int = 8
) -> None:
    """Analyze concept usage statistics across the dataset."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    all_concept_activities = []
    all_policy_logits = []
    
    print("Computing concept statistics...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx % 50 == 0:
                print(f"Processing batch {batch_idx}/{len(loader)}")
            
            x_chw = batch["x_chw"].to(device)
            outputs = model(x_chw)
            
            YT_prob = F.softmax(outputs["YT"], dim=1)  # [B, K, H, W]
            concept_activity = torch.mean(YT_prob, dim=(2, 3))  # [B, K]
            all_concept_activities.append(concept_activity.cpu())
            
            if "policy" in batch:
                policy_logits = outputs["policy_logits"]  # [B, 361+1]
                all_policy_logits.append(policy_logits.cpu())
    
    # Combine all batches
    all_concept_activities = torch.cat(all_concept_activities, dim=0)  # [N, K]
    concept_means = torch.mean(all_concept_activities, dim=0)  # [K]
    concept_stds = torch.std(all_concept_activities, dim=0)  # [K]
    
    # Create concept statistics plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Concept activity distribution
    concept_indices = range(len(concept_means))
    ax1.bar(concept_indices, concept_means.numpy(), yerr=concept_stds.numpy(), alpha=0.7)
    ax1.set_xlabel("Concept Index")
    ax1.set_ylabel("Mean Activity")
    ax1.set_title("Concept Activity Statistics")
    ax1.grid(True, alpha=0.3)
    
    # Top concepts
    top_concepts = torch.argsort(concept_means, descending=True)[:10]
    ax2.bar(range(len(top_concepts)), concept_means[top_concepts].numpy())
    ax2.set_xlabel("Rank")
    ax2.set_ylabel("Mean Activity")
    ax2.set_title("Top 10 Most Active Concepts")
    ax2.set_xticks(range(len(top_concepts)))
    ax2.set_xticklabels([f"C{idx.item()}" for idx in top_concepts])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "concept_statistics.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save statistics to file
    stats_path = output_dir / "concept_stats.txt"
    with open(stats_path, 'w') as f:
        f.write("TRM Concept Statistics\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total samples: {len(all_concept_activities)}\n")
        f.write(f"Total concepts: {len(concept_means)}\n\n")
        
        f.write("Top 10 Most Active Concepts:\n")
        for i, concept_idx in enumerate(top_concepts):
            f.write(f"{i+1:2d}. Concept {concept_idx.item():2d}: {concept_means[concept_idx]:.4f} ± {concept_stds[concept_idx]:.4f}\n")
        
        f.write(f"\nConcept Activity Summary:\n")
        f.write(f"Mean activity: {torch.mean(concept_means):.4f}\n")
        f.write(f"Std activity: {torch.std(concept_means):.4f}\n")
        f.write(f"Max activity: {torch.max(concept_means):.4f}\n")
        f.write(f"Min activity: {torch.min(concept_means):.4f}\n")
    
    print(f"Saved concept statistics to {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize TRM concept maps")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to trained TRM model")
    parser.add_argument("--slates-jsonl", type=Path, required=True, help="Path to slates.jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path("concept_viz"), help="Output directory")
    parser.add_argument("--device", type=str, default="mps", help="Device to use")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--top-k-concepts", type=int, default=8, help="Number of top concepts to show")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for statistics")
    
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
    
    # Visualize concept maps
    print("Generating concept map visualizations...")
    visualize_concept_maps(
        model, dataset, args.output_dir,
        num_samples=args.num_samples,
        top_k_concepts=args.top_k_concepts,
        device=args.device
    )
    
    # Analyze concept statistics
    print("Computing concept statistics...")
    analyze_concept_statistics(
        model, dataset, args.output_dir,
        device=args.device,
        batch_size=args.batch_size
    )
    
    print(f"Visualization complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
