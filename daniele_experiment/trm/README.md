TRM (Tiny Recursive Model) for CHW trunkfinal

Overview

- Evidence: trunkfinal tensors saved by play_and_analyze are loaded as [C,H,W].
- Model: small conv encoder, soft concept projector (K channels), recursive refinement over Y (concept map) and Z (latent state).
- Readouts: map concepts+evidence to policy logits over 361+1 and value logits over 3.

Usage

1) Ensure your slates JSONL includes trunkfinal_path and either candidates with policy_slate or a full policy0, optionally value.

2) Train:

```bash
python -m daniele_experiment.trm.train_trm --slates-jsonl games/slates.jsonl --epochs 5 --batch-size 4
```

3) Inspect:
- Visualize concept maps by reading outputs["YT"] frames from a forward pass.

Shapes

- Input x_chw: [B,C,H,W]
- Concept map YT: [B,K,H,W] (log-probs)
- Policy logits: [B, H*W+1]
- Value logits: [B, 3]

Notes

- This TRM is independent of the CBM pipeline and preserves spatial structure.
- Add augmentations and contrastive pairs as needed for stronger disentanglement.


