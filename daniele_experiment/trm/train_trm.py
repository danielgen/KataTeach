from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import TRMDataset
from .model import GoExplainTRM


def kl_divergence(p_logits: torch.Tensor, q_probs: torch.Tensor) -> torch.Tensor:
    # KL(q || p) where p is logits, q is probs
    log_p = F.log_softmax(p_logits, dim=-1)
    return torch.sum(q_probs * (torch.log(q_probs + 1e-8) - log_p), dim=-1).mean()


def train_epoch(model: GoExplainTRM, loader: DataLoader, opt: torch.optim.Optimizer, device: torch.device, w_cons: float, w_ent: float, w_val: float) -> Dict[str, float]:
    model.train()
    total = {"loss": 0.0, "cons": 0.0, "ent": 0.0, "val": 0.0}
    n = 0
    for batch in loader:
        opt.zero_grad()
        x = batch["x_chw"].to(device)  # [B,C,H,W]
        out = model(x)
        loss = torch.tensor(0.0, device=device)

        # Consistency with policy if available
        if "policy" in batch:
            pol = batch["policy"].to(device)
            L_cons = kl_divergence(out["policy_logits"], pol)
        else:
            L_cons = torch.tensor(0.0, device=device)

        # Value supervision if available
        if "value" in batch:
            val = batch["value"].to(device)
            L_val = F.cross_entropy(out["value_logits"], torch.argmax(val, dim=-1))
        else:
            L_val = torch.tensor(0.0, device=device)

        # Entropy regularization on concept maps (encourage diverse concept usage)
        y_prob = out["YT"].exp()
        # Calculate spatial average of concept probabilities
        y_spatial_avg = torch.mean(y_prob, dim=(2, 3))  # [B, K]
        
        # Multiple diversity losses
        # 1. Entropy loss (encourage uniform distribution)
        ent = -torch.sum(y_spatial_avg * torch.log(y_spatial_avg + 1e-8), dim=1).mean()
        
        # 2. KL divergence from uniform (encourage all concepts to be used equally)
        uniform = torch.ones_like(y_spatial_avg) / y_spatial_avg.size(1)
        kl_uniform = F.kl_div(y_spatial_avg.log(), uniform, reduction='batchmean')
        
        # 3. Penalty for max concept being too dominant
        max_concept = torch.max(y_spatial_avg, dim=1)[0]
        max_penalty = torch.mean(max_concept)
        
        # 4. Concept dropout loss - randomly mask concepts during training
        if model.training:
            # Randomly select which concepts to "force" to be active
            concept_mask = torch.rand_like(y_spatial_avg) > 0.5  # 50% chance each concept is "forced"
            forced_concepts = torch.where(concept_mask, uniform, y_spatial_avg)
            dropout_loss = F.mse_loss(y_spatial_avg, forced_concepts)
        else:
            dropout_loss = torch.tensor(0.0, device=y_spatial_avg.device)
        
        # 5. Hard constraint: force all concepts to be within a narrow range
        target_min, target_max = 0.05, 0.15  # Each concept should be 5-15% active
        below_min = torch.clamp(target_min - y_spatial_avg, min=0.0)
        above_max = torch.clamp(y_spatial_avg - target_max, min=0.0)
        range_penalty = torch.mean(below_min + above_max)
        
        # Combine diversity losses
        diversity_loss = ent + 0.5 * kl_uniform + 0.5 * max_penalty + 0.3 * dropout_loss + 2.0 * range_penalty

        loss = w_cons * L_cons + w_ent * diversity_loss + w_val * L_val
        loss.backward()
        opt.step()

        total["loss"] += float(loss.item())
        total["cons"] += float(L_cons.item())
        total["ent"] += float(diversity_loss.item())
        total["val"] += float(L_val.item())
        n += 1

    for k in total:
        total[k] = total[k] / max(1, n)
    return total


def main():
    parser = argparse.ArgumentParser(description="Train TRM on trunkfinal CHW with policy/value consistency")
    parser.add_argument("--slates-jsonl", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--k-concepts", type=int, default=64)
    parser.add_argument("--z-ch", type=int, default=32)
    parser.add_argument("--T", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--w-cons", type=float, default=1.0)
    parser.add_argument("--w-ent", type=float, default=0.01)
    parser.add_argument("--w-val", type=float, default=0.2)
    args = parser.parse_args()

    ds = TRMDataset(slates_jsonl=args.slates_jsonl)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    c, h, w = ds.chw_shape
    model = GoExplainTRM(c_in=c, k_concepts=args.k_concepts, z_ch=args.z_ch, T=args.T).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(args.epochs):
        stats = train_epoch(model, loader, opt, torch.device(args.device), args.w_cons, args.w_ent, args.w_val)
        print(f"epoch {epoch+1}: loss={stats['loss']:.4f} cons={stats['cons']:.4f} ent={stats['ent']:.4f} val={stats['val']:.4f}")

    # Save
    out_path = args.slates_jsonl.parent / "trm_model.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "c_in": c,
            "k_concepts": args.k_concepts,
            "z_ch": args.z_ch,
            "T": args.T,
        }
    }, out_path)
    print(f"Saved TRM to {out_path}")


if __name__ == "__main__":
    main()


