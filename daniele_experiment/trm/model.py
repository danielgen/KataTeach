from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptProjector(nn.Module):
    def __init__(self, in_ch: int, k_concepts: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, k_concepts, kernel_size=1)
        # Initialize with very small weights to break symmetry
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> logits: [B, K, H, W]
        logits = self.proj(x)
        # Add per-concept bias to encourage diversity
        if self.training:
            concept_bias = torch.randn(x.size(0), logits.size(1), 1, 1, device=x.device) * 0.1
            logits = logits + concept_bias
        return logits


class ReasonStep(nn.Module):
    def __init__(self, in_ch: int, z_ch: int, k_concepts: int) -> None:
        super().__init__()
        self.update_z = nn.Sequential(
            nn.Conv2d(in_ch + k_concepts + z_ch, z_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(z_ch, z_ch, 3, padding=1),
        )
        self.update_y = nn.Sequential(
            nn.Conv2d(z_ch + k_concepts, k_concepts, 1),
        )

    def forward(self, evidence: torch.Tensor, y_logprob: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # evidence: [B,C,H,W], y_logprob: [B,K,H,W], z: [B,Z,H,W]
        y_prob = y_logprob.exp()
        z_new = self.update_z(torch.cat([evidence, y_prob, z], dim=1))
        dY = self.update_y(torch.cat([z_new, y_prob], dim=1))
        y_new = torch.log_softmax(y_logprob + dY, dim=1)
        return y_new, z_new


class TinyRecursor(nn.Module):
    def __init__(self, in_ch: int, z_ch: int, k_concepts: int, T: int = 3) -> None:
        super().__init__()
        self.T = T
        self.step = ReasonStep(in_ch, z_ch, k_concepts)
        self.halt_head = nn.Conv2d(k_concepts, 1, 1)

    def forward(self, evidence: torch.Tensor, y0_logprob: torch.Tensor, z0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y, z = y0_logprob, z0
        for _ in range(self.T):
            y, z = self.step(evidence, y, z)
        halt_logit = self.halt_head(y)
        return y, z, halt_logit


class GoExplainTRM(nn.Module):
    def __init__(self, c_in: int, k_concepts: int = 64, z_ch: int = 32, T: int = 3, d_feat: int = 64) -> None:
        super().__init__()
        # Light evidence encoder
        self.enc = nn.Sequential(
            nn.Conv2d(c_in, d_feat, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(d_feat, d_feat, 3, padding=1),
        )
        self.concept_head = ConceptProjector(d_feat, k_concepts)
        self.recursor = TinyRecursor(in_ch=d_feat, z_ch=z_ch, k_concepts=k_concepts, T=T)

        # Readouts: map concepts (+ evidence) to policy/value
        # Policy as distribution over 361+1 via global pooling
        self.policy_readout = nn.Sequential(
            nn.Conv2d(k_concepts + d_feat, 64, 1),
            nn.SiLU(),
            nn.Conv2d(64, 1, 1),
        )
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(k_concepts + d_feat, 64, 1),
            nn.SiLU(),
            nn.Conv2d(64, 3, 1),  # win/loss/noresult logits
        )

    def forward(self, x_chw: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x_chw: [B,C,H,W]
        ev = self.enc(x_chw)
        logits0 = self.concept_head(ev)
        
        # Add small random noise to break symmetry and encourage diversity
        if self.training:
            noise = torch.randn_like(logits0) * 0.5  # Increased noise
            logits0 = logits0 + noise
        
        # Use temperature scaling to make softmax less sharp
        temperature = 2.0  # Higher temperature = more uniform distribution
        y0 = F.log_softmax(logits0 / temperature, dim=1)
        z0 = torch.zeros(x_chw.size(0), self.recursor.step.update_z[0].out_channels, x_chw.size(2), x_chw.size(3), device=x_chw.device)
        yT, zT, halt_logit = self.recursor(ev, y0, z0)

        # Readouts
        yz = torch.cat([yT.exp(), ev], dim=1)
        # Policy: produce a single-channel spatial logit map and flatten to positions
        pol_map = self.policy_readout(yz)  # [B,1,H,W]
        pol_logits = pol_map.view(pol_map.size(0), -1)  # [B, H*W]
        # Append pass move logit as global pooled scalar
        pass_logit = torch.mean(pol_map, dim=(2, 3)).squeeze(1)  # [B]
        policy_logits = torch.cat([pol_logits, pass_logit.unsqueeze(1)], dim=1)  # [B, 361+1]

        v_logits = self.value_head(yz).squeeze(-1).squeeze(-1)  # [B,3]

        return {
            "Y0": y0,
            "YT": yT,
            "Z": zT,
            "halt_logit": halt_logit,
            "policy_logits": policy_logits,
            "value_logits": v_logits,
            "Ev": ev,
        }


