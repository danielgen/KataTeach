from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptBottleneckModel(nn.Module):
    """Concept Bottleneck Model for KataGo trunkfinal activations.

    This module maps trunkfinal activations -> concept logits -> policy logits.

    - The bottleneck is the concept layer; trainable with supervised concept labels.
    - The final head predicts a 362-way policy over moves (19x19 + pass) by default.
      You can set `total_moves` to a different value if using a different board size.

    Expected inputs
    ---------------
    - x: Float tensor of shape [batch, input_dim] or [batch, C, ...] that can be flattened.
    - concept_labels (optional): Float tensor [batch, num_concepts] with {0,1} concept targets.
    - candidate_mask (optional): Float tensor [batch, total_moves] with 1 for allowed moves.

    Losses
    ------
    - concept_loss: BCEWithLogits over concepts if `concept_labels` is provided.
    - policy_loss: CrossEntropy over moves if `target_idx` is provided (with optional mask).
    """

    def __init__(
        self,
        input_dim: int,
        num_concepts: int,
        total_moves: int = 19 * 19 + 1,
        hidden_dim: int = 512,
        concept_hidden_dim: int = 512,
        dropout: float = 0.1,
        use_concept_probs_for_policy: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_concepts = num_concepts
        self.total_moves = total_moves
        self.use_concept_probs_for_policy = use_concept_probs_for_policy

        # Trunkfinal encoder to a shared representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Bottleneck: concept prediction head (logits)
        self.concept_head = nn.Sequential(
            nn.Linear(hidden_dim, concept_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(concept_hidden_dim, num_concepts),
        )

        # Policy head consumes either concept probabilities or concept logits concatenated
        policy_in_dim = hidden_dim + (num_concepts if use_concept_probs_for_policy else num_concepts)
        self.policy_head = nn.Sequential(
            nn.Linear(policy_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, self.total_moves),
        )

        self._bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(
        self,
        x: torch.Tensor,
        *,
        concept_labels: Optional[torch.Tensor] = None,
        target_idx: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
        soft_policy: Optional[torch.Tensor] = None,
        return_concepts: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Flatten if needed
        batch = x.shape[0]
        x_flat = x.view(batch, -1)
        if x_flat.shape[1] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x_flat.shape[1]}")

        h = self.encoder(x_flat)
        concept_logits = self.concept_head(h)
        concept_probits = torch.sigmoid(concept_logits)

        # Choose concept signal for policy head
        concept_signal = concept_probits if self.use_concept_probs_for_policy else concept_logits
        policy_in = torch.cat([h, concept_signal], dim=1)
        policy_logits = self.policy_head(policy_in)

        # Losses if labels are provided
        concept_loss = None
        policy_loss = None

        if concept_labels is not None:
            concept_loss = self._bce(concept_logits, concept_labels)

        if target_idx is not None or soft_policy is not None:
            # Optionally mask logits to candidate moves only by subtracting large negative to disallowed
            if candidate_mask is not None:
                # candidate_mask: 1 for allowed, 0 for disallowed
                # Add log(0) ~ -inf to disallowed moves
                large_neg = -1e9
                policy_logits = policy_logits + (candidate_mask - 1.0) * large_neg
            if soft_policy is not None:
                # KL-divergence between soft targets and predicted distribution
                logp = F.log_softmax(policy_logits, dim=1)
                # avoid log 0 issues; assume soft already normalized
                policy_loss = F.kl_div(logp, soft_policy, reduction="batchmean")
            elif target_idx is not None:
                policy_loss = F.cross_entropy(policy_logits, target_idx, reduction="mean")

        if return_concepts:
            return policy_logits, concept_loss, policy_loss, concept_logits
        return policy_logits, concept_loss, policy_loss

    @torch.no_grad()
    def get_concepts(self, x: torch.Tensor, *, return_probs: bool = True) -> torch.Tensor:
        batch = x.shape[0]
        x_flat = x.view(batch, -1)
        if x_flat.shape[1] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x_flat.shape[1]}")
        h = self.encoder(x_flat)
        logits = self.concept_head(h)
        return torch.sigmoid(logits) if return_probs else logits

    @torch.no_grad()
    def policy_from_concepts(self, x: torch.Tensor, concepts: torch.Tensor, *, use_probs: bool = True) -> torch.Tensor:
        batch = x.shape[0]
        x_flat = x.view(batch, -1)
        if x_flat.shape[1] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x_flat.shape[1]}")
        if concepts.shape[1] != self.num_concepts:
            raise ValueError(f"Expected num_concepts={self.num_concepts}, got {concepts.shape[1]}")
        h = self.encoder(x_flat)
        concept_signal = concepts if use_probs else concepts  # both shapes are [B, num_concepts]
        policy_in = torch.cat([h, concept_signal], dim=1)
        return self.policy_head(policy_in)


