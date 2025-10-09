from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add the parent directory to Python path to find cbm module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cbm.move_candidate_dataset import MoveCandidateDataset, slate_group_collate
from cbm.move_conditioned_model import MoveConditionedConceptBottleneckModel


def kl_over_candidates(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between predicted and target distributions over candidates."""
    logp = F.log_softmax(logits, dim=0)
    return torch.sum(target_probs * (torch.log(target_probs + 1e-8) - logp))


def train_one_epoch(model, loader, optimizer, device, concept_names: List[str] = None):
    """Train one epoch with concept learning losses."""
    model.train()
    total_loss = 0.0
    total_kl = 0.0
    total_concept = 0.0
    total_reg = 0.0
    num_batches = 0
    
    for batch in loader:
        optimizer.zero_grad()
        
        batch_loss = 0.0
        batch_kl = 0.0
        batch_concept = 0.0
        batch_reg = 0.0
        num_slates = 0
        
        # Process each slate in the batch
        for slate in batch:
            x = slate["x"].to(device)  # [D]
            moves = slate["moves"].to(device)  # [K]
            slate_probs = slate["slate_probs"].to(device)  # [K]
            concept_labels = slate.get("concept_labels")
            concept_mask = slate.get("concept_mask")
            
            if concept_labels is not None:
                concept_labels = concept_labels.to(device)  # [K, num_labeled_concepts]
            if concept_mask is not None:
                concept_mask = concept_mask.to(device)  # [K, num_concepts]
            
            # Expand x for all candidates
            x_expanded = x.unsqueeze(0).expand(moves.size(0), -1)  # [K, D]
            
            # Forward pass
            policy_logits, concept_loss, reg_loss, concept_logits = model.forward_move(
                x_expanded, moves,
                concept_labels=concept_labels,
                concept_mask=concept_mask,
                return_concepts=True
            )
            
            # Policy loss (KL divergence over candidates)
            kl_loss = kl_over_candidates(policy_logits, slate_probs)
            
            # Accumulate losses
            slate_loss = kl_loss
            if concept_loss is not None:
                slate_loss = slate_loss + concept_loss
                batch_concept += concept_loss.item()
            if reg_loss is not None:
                slate_loss = slate_loss + reg_loss
                batch_reg += reg_loss.item()
            
            batch_loss += slate_loss
            batch_kl += kl_loss.item()
            num_slates += 1
        
        # Average over slates in batch
        if num_slates > 0:
            batch_loss = batch_loss / num_slates
            batch_kl = batch_kl / num_slates
            batch_concept = batch_concept / num_slates
            batch_reg = batch_reg / num_slates
            
            batch_loss.backward()
            optimizer.step()
            
            total_loss += batch_loss.item()
            total_kl += batch_kl
            total_concept += batch_concept
            total_reg += batch_reg
            num_batches += 1
    
    # Return average losses
    if num_batches > 0:
        return {
            "loss": total_loss / num_batches,
            "kl": total_kl / num_batches,
            "concept": total_concept / num_batches,
            "reg": total_reg / num_batches,
        }
    else:
        return {"loss": 0.0, "kl": 0.0, "concept": 0.0, "reg": 0.0}


def analyze_learned_concepts(model, loader, device, concept_names: List[str] = None, top_k: int = 5):
    """Analyze what concepts the model has learned."""
    model.eval()
    
    concept_activations = []
    move_concept_pairs = []
    
    with torch.no_grad():
        for batch in loader:
            for slate in batch:
                x = slate["x"].to(device)
                moves = slate["moves"].to(device)
                
                # Get concepts for each move
                for move in moves:
                    move_concepts = model.concepts_for_move(x, move.item(), return_probs=True)
                    concept_activations.append(move_concepts.cpu())
                    move_concept_pairs.append((move.item(), move_concepts.cpu()))
    
    if not concept_activations:
        print("No concept activations found.")
        return
    
    # Stack all activations
    all_concepts = torch.cat(concept_activations, dim=0)  # [N, num_concepts]
    
    print(f"\n=== Concept Analysis ===")
    print(f"Total samples: {all_concepts.size(0)}")
    print(f"Total concepts: {all_concepts.size(1)}")
    print(f"Labeled concepts: {model.num_labeled_concepts}")
    print(f"Latent concepts: {model.num_latent_concepts}")
    
    # Analyze concept usage
    concept_means = torch.mean(all_concepts, dim=0)
    concept_stds = torch.std(all_concepts, dim=0)
    concept_maxs = torch.max(all_concepts, dim=0)[0]
    
    print(f"\n=== Concept Statistics ===")
    for i in range(all_concepts.size(1)):
        concept_type = "labeled" if i < model.num_labeled_concepts else "latent"
        concept_name = concept_names[i] if concept_names and i < len(concept_names) else f"concept_{i}"
        print(f"{concept_type:7} {i:2d}: {concept_name:20} "
              f"mean={concept_means[i]:.3f} std={concept_stds[i]:.3f} max={concept_maxs[i]:.3f}")
    
    # Find most active latent concepts
    if model.num_latent_concepts > 0:
        latent_start = model.num_labeled_concepts
        latent_means = concept_means[latent_start:]
        latent_indices = torch.argsort(latent_means, descending=True)
        
        print(f"\n=== Top {top_k} Most Active Latent Concepts ===")
        for i in range(min(top_k, model.num_latent_concepts)):
            idx = latent_indices[i].item()
            global_idx = latent_start + idx
            print(f"Latent {idx:2d} (global {global_idx:2d}): mean={latent_means[idx]:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Train move-conditioned CBM with concept learning")
    parser.add_argument("--slates-path", type=Path, required=True, help="Path to slates.jsonl")
    parser.add_argument("--labels-path", type=Path, help="Path to labels.jsonl (optional)")
    parser.add_argument("--output-dir", type=Path, default=Path("cbm_output"), help="Output directory")
    parser.add_argument("--num-labeled-concepts", type=int, default=0, help="Number of labeled concepts")
    parser.add_argument("--num-latent-concepts", type=int, default=10, help="Number of latent concepts to learn")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (number of slates)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--lambda-sparsity", type=float, default=0.01, help="Sparsity regularization weight")
    parser.add_argument("--lambda-orthogonality", type=float, default=0.01, help="Orthogonality regularization weight")
    parser.add_argument("--lambda-diversity", type=float, default=0.01, help="Diversity regularization weight")
    parser.add_argument("--analyze-concepts", action="store_true", help="Analyze learned concepts after training")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    dataset = MoveCandidateDataset(
        slates_path=args.slates_path,
        labels_path=args.labels_path,
        num_labeled_concepts=args.num_labeled_concepts,
        num_latent_concepts=args.num_latent_concepts,
    )
    
    print(f"Dataset loaded: {len(dataset)} slates")
    print(f"Input dim: {dataset.input_dim}")
    print(f"Labeled concepts: {dataset.num_labeled_concepts}")
    print(f"Latent concepts: {dataset.num_latent_concepts}")
    print(f"Total concepts: {dataset.num_concepts}")
    
    # Create data loader
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0, 
        collate_fn=slate_group_collate
    )
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = MoveConditionedConceptBottleneckModel(
        input_dim=dataset.input_dim,
        num_labeled_concepts=dataset.num_labeled_concepts,
        num_latent_concepts=dataset.num_latent_concepts,
        lambda_sparsity=args.lambda_sparsity,
        lambda_orthogonality=args.lambda_orthogonality,
        lambda_diversity=args.lambda_diversity,
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Load concept names if available
    concept_names = None
    if args.labels_path and args.labels_path.exists():
        # Try to load concept names from ontology or labels file
        # This is a placeholder - you might want to load from your ontology.yaml
        concept_names = [f"concept_{i}" for i in range(dataset.num_concepts)]
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        stats = train_one_epoch(model, loader, optimizer, device, concept_names)
        print(f"Epoch {epoch+1:2d}: loss={stats['loss']:.4f} "
              f"kl={stats['kl']:.4f} concept={stats['concept']:.4f} reg={stats['reg']:.4f}")
    
    # Save model
    model_path = args.output_dir / "concept_learning_cbm.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': dataset.input_dim,
            'num_labeled_concepts': dataset.num_labeled_concepts,
            'num_latent_concepts': dataset.num_latent_concepts,
            'lambda_sparsity': args.lambda_sparsity,
            'lambda_orthogonality': args.lambda_orthogonality,
            'lambda_diversity': args.lambda_diversity,
        }
    }, model_path)
    print(f"Model saved to {model_path}")
    
    # Analyze learned concepts
    if args.analyze_concepts:
        print("\nAnalyzing learned concepts...")
        analyze_learned_concepts(model, loader, device, concept_names)


if __name__ == "__main__":
    main()
# python cbm/examples/train_concept_learning_cbm.py --slates-path "games/slates.jsonl" --labels-path "games/labels.jsonl" --num-labeled-concepts 88 --num-latent-concepts 15 --lambda-sparsity 0.01 --lambda-orthogonality 0.01 --lambda-diversity 0.01 --epochs 20 --batch-size 4 --lr 3e-4 --analyze-concepts