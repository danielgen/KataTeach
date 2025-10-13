from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add the parent directory to Python path to find cbm module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cbm.move_candidate_dataset import MoveCandidateDataset, slate_group_collate
from cbm.concept_utils import (
    load_model_with_concept_learning,
    extract_concepts_for_topk_moves,
    analyze_concept_usage,
    print_concept_analysis
)
from daniele_experiment import get_device


def evaluate_move_ranking(model, dataset, device, num_samples: int = 10):
    """Evaluate how well the model ranks moves compared to KataGo policy."""
    model.eval()
    
    ranking_metrics = {
        'spearman_correlations': [],
        'top1_accuracy': [],
        'top3_accuracy': [],
        'top5_accuracy': [],
        'kl_divergences': []
    }
    
    sample_count = 0
    
    with torch.no_grad():
        for i, slate in enumerate(dataset):
            if sample_count >= num_samples:
                break
                
            x = slate["x"].to(device)
            moves = slate["moves"].to(device)
            slate_probs = slate["slate_probs"].to(device)
            
            # Get model predictions
            model_scores = model.score_candidates(x, moves)
            model_probs = F.softmax(model_scores, dim=0)
            
            # Compute ranking correlation
            spearman_corr = torch.corrcoef(torch.stack([
                model_scores.cpu(), 
                torch.log(slate_probs + 1e-8).cpu()
            ]))[0, 1].item()
            
            # Top-K accuracy (handle variable slate sizes)
            num_candidates = len(slate_probs)
            k3 = min(3, num_candidates)
            k5 = min(5, num_candidates)
            
            true_top1 = torch.argmax(slate_probs).item()
            true_top3 = torch.topk(slate_probs, k3).indices
            true_top5 = torch.topk(slate_probs, k5).indices
            
            pred_top1 = torch.argmax(model_scores).item()
            pred_top3 = torch.topk(model_scores, k3).indices
            pred_top5 = torch.topk(model_scores, k5).indices
            
            top1_acc = 1.0 if pred_top1 == true_top1 else 0.0
            top3_acc = len(set(true_top3.tolist()) & set(pred_top3.tolist())) / k3
            top5_acc = len(set(true_top5.tolist()) & set(pred_top5.tolist())) / k5
            
            # KL divergence
            kl_div = F.kl_div(
                F.log_softmax(model_scores, dim=0),
                slate_probs,
                reduction='sum'
            ).item()
            
            ranking_metrics['spearman_correlations'].append(spearman_corr)
            ranking_metrics['top1_accuracy'].append(top1_acc)
            ranking_metrics['top3_accuracy'].append(top3_acc)
            ranking_metrics['top5_accuracy'].append(top5_acc)
            ranking_metrics['kl_divergences'].append(kl_div)
            
            sample_count += 1
            
            if sample_count <= 3:  # Print details for first 3 samples
                print(f"\n=== Sample {sample_count} ===")
                print(f"Spearman correlation: {spearman_corr:.3f}")
                print(f"Top-1 accuracy: {top1_acc:.3f}")
                print(f"Top-3 accuracy: {top3_acc:.3f}")
                print(f"Top-5 accuracy: {top5_acc:.3f}")
                print(f"KL divergence: {kl_div:.3f}")
                
                # Show top moves
                print(f"\nTop {k5} KataGo moves:")
                for j, (move, prob) in enumerate(zip(moves[true_top5], slate_probs[true_top5])):
                    print(f"  {j+1}. Move {move.item():3d}: {prob.item():.3f}")
                
                print(f"\nTop {k5} Model moves:")
                for j, (move, score) in enumerate(zip(moves[pred_top5], model_scores[pred_top5])):
                    print(f"  {j+1}. Move {move.item():3d}: {score.item():.3f}")
    
    # Compute averages
    avg_metrics = {}
    for key, values in ranking_metrics.items():
        avg_metrics[key] = np.mean(values) if values else 0.0
    
    return avg_metrics


def analyze_concept_patterns(model, dataset, device, num_samples: int = 20):
    """Analyze what patterns the learned concepts capture."""
    model.eval()
    
    concept_activations = []
    move_types = []
    
    with torch.no_grad():
        sample_count = 0
        for i, slate in enumerate(dataset):
            if sample_count >= num_samples:
                break
                
            x = slate["x"].to(device)
            moves = slate["moves"].to(device)
            slate_probs = slate["slate_probs"].to(device)
            
            # Get concepts for each move
            for j, move in enumerate(moves):
                move_concepts = model.concepts_for_move(x, move.item(), return_probs=True)
                concept_activations.append(move_concepts.cpu().numpy())
                
                # Categorize move by policy strength
                move_prob = slate_probs[j].item()
                if move_prob > 0.3:
                    move_types.append("strong")
                elif move_prob > 0.1:
                    move_types.append("medium")
                else:
                    move_types.append("weak")
            
            sample_count += 1
    
    concept_activations = np.array(concept_activations)
    
    print(f"\n=== Concept Pattern Analysis ===")
    print(f"Analyzed {len(concept_activations)} moves across {num_samples} positions")
    
    # Analyze concept usage by move strength
    strong_moves = [i for i, t in enumerate(move_types) if t == "strong"]
    medium_moves = [i for i, t in enumerate(move_types) if t == "medium"]
    weak_moves = [i for i, t in enumerate(move_types) if t == "weak"]
    
    print(f"\nMove distribution: {len(strong_moves)} strong, {len(medium_moves)} medium, {len(weak_moves)} weak")
    
    for concept_idx in range(concept_activations.shape[1]):
        strong_avg = np.mean(concept_activations[strong_moves, concept_idx]) if strong_moves else 0
        medium_avg = np.mean(concept_activations[medium_moves, concept_idx]) if medium_moves else 0
        weak_avg = np.mean(concept_activations[weak_moves, concept_idx]) if weak_moves else 0
        
        print(f"Concept {concept_idx:2d}: Strong={strong_avg:.3f}, Medium={medium_avg:.3f}, Weak={weak_avg:.3f}")
        
        # Identify concepts that distinguish move quality
        if strong_avg > medium_avg + 0.05 and strong_avg > weak_avg + 0.1:
            print(f"  -> Concept {concept_idx} strongly associated with good moves!")
        elif weak_avg > strong_avg + 0.05 and weak_avg > medium_avg + 0.1:
            print(f"  -> Concept {concept_idx} associated with weak moves")


def test_concept_explanations(model, dataset, device, num_samples: int = 5):
    """Test concept explanations for top moves."""
    model.eval()
    
    print(f"\n=== Concept Explanations for Top Moves ===")
    
    with torch.no_grad():
        sample_count = 0
        for i, slate in enumerate(dataset):
            if sample_count >= num_samples:
                break
                
            x = slate["x"].to(device)
            moves = slate["moves"].to(device)
            slate_probs = slate["slate_probs"].to(device)
            
            print(f"\n--- Position {sample_count + 1} ---")
            
            # Get top 3 moves according to KataGo
            top3_indices = torch.topk(slate_probs, 3).indices
            top3_moves = moves[top3_indices]
            top3_probs = slate_probs[top3_indices]
            
            for j, (move, prob) in enumerate(zip(top3_moves, top3_probs)):
                print(f"\nMove {j+1}: {move.item():3d} (KataGo prob: {prob.item():.3f})")
                
                # Get concepts for this move
                move_concepts = model.concepts_for_move(x, move.item(), return_probs=True)
                
                # Show active concepts
                active_concepts = []
                for concept_idx, activation in enumerate(move_concepts[0]):
                    if activation > 0.3:  # Threshold for "active"
                        active_concepts.append((concept_idx, activation.item()))
                
                if active_concepts:
                    active_concepts.sort(key=lambda x: x[1], reverse=True)
                    print(f"  Active concepts: {', '.join([f'C{c[0]}({c[1]:.2f})' for c in active_concepts])}")
                else:
                    print(f"  No strongly active concepts (max: {move_concepts[0].max().item():.3f})")
            
            sample_count += 1


def main():
    parser = argparse.ArgumentParser(description="Test trained concept learning CBM")
    parser.add_argument("--model-path", type=Path, default=Path("cbm_output/concept_learning_cbm.pt"), 
                       help="Path to trained model")
    parser.add_argument("--slates-path", type=Path, default=Path("games/slates.jsonl"), 
                       help="Path to slates.jsonl")
    parser.add_argument("--num-samples", type=int, default=10, 
                       help="Number of samples to evaluate")
    parser.add_argument("--concept-threshold", type=float, default=0.3, 
                       help="Threshold for concept activation")
    
    args = parser.parse_args()
    
    # Load model
    print("Loading trained model...")
    device = torch.device(get_device())
    model = load_model_with_concept_learning(args.model_path, device)
    print(f"Model loaded on {device}")
    print(f"Concepts: {model.num_labeled_concepts} labeled + {model.num_latent_concepts} latent")
    
    # Load dataset
    print("Loading dataset...")
    dataset = MoveCandidateDataset(
        slates_path=args.slates_path,
        num_labeled_concepts=0,  # No labels for testing
        num_latent_concepts=model.num_latent_concepts,
    )
    print(f"Dataset loaded: {len(dataset)} slates")
    
    # Evaluate move ranking
    print(f"\n{'='*50}")
    print("EVALUATING MOVE RANKING")
    print(f"{'='*50}")
    
    ranking_metrics = evaluate_move_ranking(model, dataset, device, args.num_samples)
    
    print(f"\n=== Overall Ranking Performance ===")
    print(f"Average Spearman correlation: {ranking_metrics['spearman_correlations']:.3f}")
    print(f"Top-1 accuracy: {ranking_metrics['top1_accuracy']:.3f}")
    print(f"Top-3 accuracy: {ranking_metrics['top3_accuracy']:.3f}")
    print(f"Top-5 accuracy: {ranking_metrics['top5_accuracy']:.3f}")
    print(f"Average KL divergence: {ranking_metrics['kl_divergences']:.3f}")
    
    # Analyze concept patterns
    print(f"\n{'='*50}")
    print("ANALYZING CONCEPT PATTERNS")
    print(f"{'='*50}")
    
    analyze_concept_patterns(model, dataset, device, args.num_samples)
    
    # Test concept explanations
    print(f"\n{'='*50}")
    print("TESTING CONCEPT EXPLANATIONS")
    print(f"{'='*50}")
    
    test_concept_explanations(model, dataset, device, min(5, args.num_samples))
    
    # Full concept analysis
    print(f"\n{'='*50}")
    print("FULL CONCEPT ANALYSIS")
    print(f"{'='*50}")
    
    analysis = analyze_concept_usage(model, dataset, device=device)
    print_concept_analysis(analysis)


if __name__ == "__main__":
    main()
