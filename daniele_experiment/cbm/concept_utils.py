from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .move_conditioned_model import MoveConditionedConceptBottleneckModel


def extract_concepts_for_topk_moves(
    model: MoveConditionedConceptBottleneckModel,
    x_single: torch.Tensor,
    candidate_moves_361: torch.Tensor,
    concept_names: Optional[List[str]] = None,
    k: int = 5,
    threshold: float = 0.5,
    return_scores: bool = False,
) -> Tuple[List[List[str]], torch.Tensor, torch.Tensor]:
    """
    Extract concept tags for top-K moves from a position.
    
    Args:
        model: Trained move-conditioned CBM
        x_single: Position representation [input_dim]
        candidate_moves_361: Candidate move indices [K_all]
        concept_names: Names for concepts (optional)
        k: Number of top moves to analyze
        threshold: Threshold for concept activation
        return_scores: Whether to return move scores
        
    Returns:
        tags_per_move: List of concept tags for each top-K move
        topk_moves: Top-K move indices
        topk_scores: Top-K move scores (if return_scores=True)
    """
    model.eval()
    
    with torch.no_grad():
        # Get scores for all candidates
        scores = model.score_candidates(x_single, candidate_moves_361)  # [K_all]
        topk_vals, topk_idx = torch.topk(scores, k=min(k, scores.numel()))
        topk_moves = candidate_moves_361[topk_idx]
        
        # Get concepts for top-K moves
        x_rep = x_single.view(1, -1).expand(topk_moves.size(0), -1)
        _, _, _, concept_logits = model.forward_move(x_rep, topk_moves, return_concepts=True)
        concept_probs = torch.sigmoid(concept_logits)  # [K, num_concepts]
        
        # Convert to tag names
        tags_per_move = []
        for i, probs in enumerate(concept_probs):
            active_indices = (probs > threshold).nonzero(as_tuple=True)[0].tolist()
            if concept_names:
                tags = [concept_names[idx] for idx in active_indices if idx < len(concept_names)]
            else:
                tags = [f"concept_{idx}" for idx in active_indices]
            tags_per_move.append(tags)
    
    if return_scores:
        return tags_per_move, topk_moves, topk_vals
    else:
        return tags_per_move, topk_moves


def extract_concepts_for_plan(
    model: MoveConditionedConceptBottleneckModel,
    position_sequence: List[torch.Tensor],
    move_sequence: List[int],
    concept_names: Optional[List[str]] = None,
    threshold: float = 0.5,
) -> List[List[str]]:
    """
    Extract concept tags for each step in a move sequence (plan).
    
    Args:
        model: Trained move-conditioned CBM
        position_sequence: List of position representations after each move
        move_sequence: List of move indices for the plan
        concept_names: Names for concepts (optional)
        threshold: Threshold for concept activation
        
    Returns:
        tags_per_step: List of concept tags for each step in the plan
    """
    model.eval()
    
    tags_per_step = []
    
    with torch.no_grad():
        for pos, move in zip(position_sequence, move_sequence):
            move_tensor = torch.tensor([move], device=pos.device, dtype=torch.long)
            concept_probs = model.concepts_for_move(pos, move, return_probs=True)
            
            # Convert to tags
            active_indices = (concept_probs[0] > threshold).nonzero(as_tuple=True)[0].tolist()
            if concept_names:
                tags = [concept_names[idx] for idx in active_indices if idx < len(concept_names)]
            else:
                tags = [f"concept_{idx}" for idx in active_indices]
            tags_per_step.append(tags)
    
    return tags_per_step


def analyze_concept_usage(
    model: MoveConditionedConceptBottleneckModel,
    dataset,
    concept_names: Optional[List[str]] = None,
    device: torch.device = None,
) -> Dict:
    """
    Analyze how concepts are used across the dataset.
    
    Args:
        model: Trained move-conditioned CBM
        dataset: MoveCandidateDataset
        concept_names: Names for concepts (optional)
        device: Device to run analysis on
        
    Returns:
        Dictionary with concept usage statistics
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    concept_activations = []
    move_concept_pairs = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            slate = dataset[i]
            x = slate["x"].to(device)
            moves = slate["moves"].to(device)
            
            # Get concepts for each move
            for move in moves:
                move_concepts = model.concepts_for_move(x, move.item(), return_probs=True)
                concept_activations.append(move_concepts.cpu())
                move_concept_pairs.append((move.item(), move_concepts.cpu()))
    
    if not concept_activations:
        return {"error": "No concept activations found"}
    
    # Stack all activations
    all_concepts = torch.cat(concept_activations, dim=0)  # [N, num_concepts]
    
    # Compute statistics
    concept_means = torch.mean(all_concepts, dim=0)
    concept_stds = torch.std(all_concepts, dim=0)
    concept_maxs = torch.max(all_concepts, dim=0)[0]
    concept_usage = torch.sum(all_concepts > 0.5, dim=0).float() / all_concepts.size(0)
    
    # Find most/least active concepts
    labeled_means = concept_means[:model.num_labeled_concepts]
    latent_means = concept_means[model.num_labeled_concepts:]
    
    results = {
        "total_samples": all_concepts.size(0),
        "total_concepts": all_concepts.size(1),
        "labeled_concepts": model.num_labeled_concepts,
        "latent_concepts": model.num_latent_concepts,
        "concept_stats": [],
        "most_active_labeled": [],
        "most_active_latent": [],
        "least_active_latent": [],
    }
    
    # Concept statistics
    for i in range(all_concepts.size(1)):
        concept_type = "labeled" if i < model.num_labeled_concepts else "latent"
        concept_name = concept_names[i] if concept_names and i < len(concept_names) else f"concept_{i}"
        
        results["concept_stats"].append({
            "index": i,
            "name": concept_name,
            "type": concept_type,
            "mean": concept_means[i].item(),
            "std": concept_stds[i].item(),
            "max": concept_maxs[i].item(),
            "usage_rate": concept_usage[i].item(),
        })
    
    # Most active labeled concepts
    if model.num_labeled_concepts > 0:
        labeled_indices = torch.argsort(labeled_means, descending=True)
        for i in range(min(5, model.num_labeled_concepts)):
            idx = labeled_indices[i].item()
            concept_name = concept_names[idx] if concept_names and idx < len(concept_names) else f"concept_{idx}"
            results["most_active_labeled"].append({
                "index": idx,
                "name": concept_name,
                "mean": labeled_means[idx].item(),
            })
    
    # Most/least active latent concepts
    if model.num_latent_concepts > 0:
        latent_indices_desc = torch.argsort(latent_means, descending=True)
        latent_indices_asc = torch.argsort(latent_means, descending=False)
        
        for i in range(min(5, model.num_latent_concepts)):
            # Most active
            idx = latent_indices_desc[i].item()
            global_idx = model.num_labeled_concepts + idx
            concept_name = concept_names[global_idx] if concept_names and global_idx < len(concept_names) else f"latent_{idx}"
            results["most_active_latent"].append({
                "index": idx,
                "global_index": global_idx,
                "name": concept_name,
                "mean": latent_means[idx].item(),
            })
            
            # Least active
            idx = latent_indices_asc[i].item()
            global_idx = model.num_labeled_concepts + idx
            concept_name = concept_names[global_idx] if concept_names and global_idx < len(concept_names) else f"latent_{idx}"
            results["least_active_latent"].append({
                "index": idx,
                "global_index": global_idx,
                "name": concept_name,
                "mean": latent_means[idx].item(),
            })
    
    return results


def print_concept_analysis(analysis: Dict):
    """Print a formatted analysis of concept usage."""
    print(f"\n=== Concept Usage Analysis ===")
    print(f"Total samples: {analysis['total_samples']}")
    print(f"Total concepts: {analysis['total_concepts']} (labeled: {analysis['labeled_concepts']}, latent: {analysis['latent_concepts']})")
    
    print(f"\n=== Most Active Labeled Concepts ===")
    for item in analysis["most_active_labeled"]:
        print(f"  {item['index']:2d}: {item['name']:20} mean={item['mean']:.3f}")
    
    print(f"\n=== Most Active Latent Concepts ===")
    for item in analysis["most_active_latent"]:
        print(f"  {item['index']:2d}: {item['name']:20} mean={item['mean']:.3f}")
    
    print(f"\n=== Least Active Latent Concepts ===")
    for item in analysis["least_active_latent"]:
        print(f"  {item['index']:2d}: {item['name']:20} mean={item['mean']:.3f}")
    
    print(f"\n=== All Concept Statistics ===")
    print(f"{'Idx':>3} {'Type':>7} {'Name':>20} {'Mean':>6} {'Std':>6} {'Max':>6} {'Usage':>6}")
    print("-" * 60)
    for stat in analysis["concept_stats"]:
        print(f"{stat['index']:3d} {stat['type']:7} {stat['name']:20} "
              f"{stat['mean']:6.3f} {stat['std']:6.3f} {stat['max']:6.3f} {stat['usage_rate']:6.3f}")


def load_model_with_concept_learning(
    model_path: str,
    device: torch.device = None,
) -> MoveConditionedConceptBottleneckModel:
    """
    Load a trained concept learning model.
    
    Args:
        model_path: Path to saved model
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['model_config']
    
    model = MoveConditionedConceptBottleneckModel(
        input_dim=config['input_dim'],
        num_labeled_concepts=config['num_labeled_concepts'],
        num_latent_concepts=config['num_latent_concepts'],
        lambda_sparsity=config['lambda_sparsity'],
        lambda_orthogonality=config['lambda_orthogonality'],
        lambda_diversity=config['lambda_diversity'],
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model
