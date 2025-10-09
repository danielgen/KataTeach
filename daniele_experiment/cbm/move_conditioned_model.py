from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoveConditionedConceptBottleneckModel(nn.Module):
    """Move-conditioned CBM for explainable policy scoring with concept learning.

    For a position representation x (trunkfinal) and a candidate move m, the model predicts
    per-move concepts and a scalar policy logit. Softmax over all candidates yields p(m|x).
    
    Supports both labeled concepts (supervised) and latent concepts (learned) with
    sparsity, orthogonality, and diversity regularizers.

    Inputs
    ------
    - x: [B, input_dim]
    - move_idx361: [B] integer indices in [0, total_moves)
    - concept_labels (optional): [B, num_labeled_concepts] {0,1}
    - concept_mask (optional): [B, num_concepts] {0,1} mask for which concepts have labels

    Outputs
    -------
    - policy_logit: [B] scalar logit for each (x, m)
    - concept_loss: optional BCEWithLogits loss if labels provided
    - regularization_loss: sparsity + orthogonality + diversity losses
    """

    def __init__(
        self,
        *,
        input_dim: int,
        num_labeled_concepts: int,
        num_latent_concepts: int = 0,
        total_moves: int = 19 * 19 + 1,
        hidden_dim: int = 512,
        move_emb_dim: int = 64,
        concept_hidden_dim: int = 256,
        dropout: float = 0.1,
        concat_move_to_concepts: bool = True,
        # Regularization weights
        lambda_sparsity: float = 0.01,
        lambda_orthogonality: float = 0.01,
        lambda_diversity: float = 0.01,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_labeled_concepts = num_labeled_concepts
        self.num_latent_concepts = num_latent_concepts
        self.num_concepts = num_labeled_concepts + num_latent_concepts
        self.total_moves = total_moves
        self.concat_move_to_concepts = concat_move_to_concepts
        
        # Regularization weights
        self.lambda_sparsity = lambda_sparsity
        self.lambda_orthogonality = lambda_orthogonality
        self.lambda_diversity = lambda_diversity

        # Encode trunkfinal
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Move embedding
        self.move_emb = nn.Embedding(total_moves, move_emb_dim)

        # Concept head, conditioned on both x and move embedding
        concept_in = hidden_dim + move_emb_dim
        self.concept_head = nn.Sequential(
            nn.Linear(concept_in, concept_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(concept_hidden_dim, self.num_concepts),
        )

        # Policy head consumes concepts (logits) and optionally x-encoding again
        policy_in = self.num_concepts + (hidden_dim if self.concat_move_to_concepts else 0)
        self.policy_head = nn.Sequential(
            nn.Linear(policy_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),  # scalar logit
        )

        self._bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward_move(
        self,
        x: torch.Tensor,                # [B, input_dim]
        move_idx361: torch.Tensor,      # [B]
        *,
        concept_labels: Optional[torch.Tensor] = None,  # [B, num_labeled_concepts]
        concept_mask: Optional[torch.Tensor] = None,    # [B, num_concepts] {0,1}
        return_concepts: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if x.dim() != 2 or x.size(1) != self.input_dim:
            x = x.view(x.size(0), -1)
            if x.size(1) != self.input_dim:
                raise ValueError(f"Expected input_dim={self.input_dim}, got {x.size(1)}")

        h = self.encoder(x)  # [B, H]
        m = self.move_emb(move_idx361)  # [B, E]
        hm = torch.cat([h, m], dim=1)   # [B, H+E]

        concept_logits = self.concept_head(hm)  # [B, C]

        # Policy head
        if self.concat_move_to_concepts:
            policy_in = torch.cat([concept_logits, h], dim=1)  # [B, C+H]
        else:
            policy_in = concept_logits  # [B, C]
        policy_logit = self.policy_head(policy_in).squeeze(-1)  # [B]

        # Compute concept loss (only for labeled concepts)
        concept_loss = None
        if concept_labels is not None:
            # Only supervise the labeled concept dimensions
            labeled_logits = concept_logits[:, :self.num_labeled_concepts]
            concept_loss = self._bce(labeled_logits, concept_labels)
            
            # Apply mask if provided (for missing labels)
            if concept_mask is not None:
                labeled_mask = concept_mask[:, :self.num_labeled_concepts]
                if labeled_mask.sum() > 0:
                    # Only compute loss where mask is 1
                    masked_logits = labeled_logits * labeled_mask
                    masked_labels = concept_labels * labeled_mask
                    concept_loss = self._bce(masked_logits, masked_labels)

        # Compute regularization losses
        reg_loss = self._compute_regularization_loss(concept_logits)

        if return_concepts:
            return policy_logit, concept_loss, reg_loss, concept_logits
        return policy_logit, concept_loss, reg_loss, None

    def _compute_regularization_loss(self, concept_logits: torch.Tensor) -> torch.Tensor:
        """Compute sparsity, orthogonality, and diversity regularization losses."""
        reg_loss = torch.tensor(0.0, device=concept_logits.device)
        
        # Sparsity: L1 penalty on concept activations
        if self.lambda_sparsity > 0:
            concept_probs = torch.sigmoid(concept_logits)
            sparsity_loss = torch.mean(torch.sum(concept_probs, dim=1))  # Encourage sparse activations
            reg_loss = reg_loss + self.lambda_sparsity * sparsity_loss
        
        # Orthogonality: encourage concept weight vectors to be orthogonal
        if self.lambda_orthogonality > 0:
            # Get the final linear layer weights
            final_linear = self.concept_head[-1]  # nn.Linear layer
            W = final_linear.weight  # [num_concepts, concept_hidden_dim]
            # Compute W @ W^T - I (should be close to identity for orthogonality)
            gram_matrix = torch.mm(W, W.t())
            identity = torch.eye(self.num_concepts, device=W.device)
            orthogonality_loss = torch.mean((gram_matrix - identity) ** 2)
            reg_loss = reg_loss + self.lambda_orthogonality * orthogonality_loss
        
        # Diversity: encourage different concepts to activate on different samples
        if self.lambda_diversity > 0:
            concept_probs = torch.sigmoid(concept_logits)
            # Compute pairwise correlations between concepts across batch
            concept_probs_centered = concept_probs - torch.mean(concept_probs, dim=0, keepdim=True)
            corr_matrix = torch.mm(concept_probs_centered.t(), concept_probs_centered) / (concept_probs.size(0) - 1)
            # Penalize off-diagonal correlations (encourage independence)
            mask = 1 - torch.eye(self.num_concepts, device=concept_probs.device)
            diversity_loss = torch.mean((corr_matrix * mask) ** 2)
            reg_loss = reg_loss + self.lambda_diversity * diversity_loss
        
        return reg_loss

    @torch.no_grad()
    def score_candidates(
        self,
        x_single: torch.Tensor,         # [input_dim]
        move_idx361_batch: torch.Tensor # [K]
    ) -> torch.Tensor:
        x_rep = x_single.view(1, -1).expand(move_idx361_batch.size(0), -1)
        logits, _, _, _ = self.forward_move(x_rep, move_idx361_batch)
        return logits  # [K]

    @torch.no_grad()
    def concepts_for_move(
        self,
        x_single: torch.Tensor,
        move_idx361: int,
        *,
        return_probs: bool = True,
    ) -> torch.Tensor:
        x_rep = x_single.view(1, -1)
        move = torch.tensor([move_idx361], device=x_single.device, dtype=torch.long)
        _, _, _, concept_logits = self.forward_move(x_rep, move, return_concepts=True)
        return torch.sigmoid(concept_logits) if return_probs else concept_logits

    @torch.no_grad()
    def get_labeled_concepts(self, x: torch.Tensor, move_idx361: torch.Tensor) -> torch.Tensor:
        """Get only the labeled concept probabilities."""
        _, _, _, concept_logits = self.forward_move(x, move_idx361, return_concepts=True)
        labeled_logits = concept_logits[:, :self.num_labeled_concepts]
        return torch.sigmoid(labeled_logits)

    @torch.no_grad()
    def get_latent_concepts(self, x: torch.Tensor, move_idx361: torch.Tensor) -> torch.Tensor:
        """Get only the latent (learned) concept probabilities."""
        _, _, _, concept_logits = self.forward_move(x, move_idx361, return_concepts=True)
        if self.num_latent_concepts > 0:
            latent_logits = concept_logits[:, self.num_labeled_concepts:]
            return torch.sigmoid(latent_logits)
        else:
            return torch.empty(x.size(0), 0, device=x.device)



