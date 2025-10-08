from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from daniele_experiment.cbm.dataset import SlateMovesDataset, slate_collate_fn
from daniele_experiment.cbm.model import ConceptBottleneckModel


def main() -> None:
    # Paths
    output_dir = Path("games")  # must match play_and_analyze.py --output-dir
    slates_path = output_dir / "slates.jsonl"
    labels_path = output_dir / "labels.jsonl"  # optional; one JSONL per slate with concept_labels

    dataset = SlateMovesDataset(
        slates_path=slates_path,
        labels_path=labels_path if labels_path.exists() else None,
        total_moves=19 * 19 + 1,
        target_from="played",
        require_trunkfinal=True,
        num_concepts=None,
    )

    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, collate_fn=slate_collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConceptBottleneckModel(
        input_dim=dataset.input_dim,
        num_concepts=dataset.num_concepts,
        total_moves=19 * 19 + 1,
        hidden_dim=512,
        concept_hidden_dim=256,
        dropout=0.1,
        use_concept_probs_for_policy=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    model.train()
    for step, batch in enumerate(loader):
        if batch is None:
            continue
        x = batch["x"].to(device)
        target_idx = batch["target_idx"].to(device)
        candidate_mask = batch["candidate_mask"].to(device)
        concept_labels = batch["concept_labels"].to(device) if batch["concept_labels"] is not None else None

        optimizer.zero_grad(set_to_none=True)
        policy_logits, concept_loss, policy_loss = model(
            x,
            concept_labels=concept_labels,
            target_idx=target_idx,
            candidate_mask=candidate_mask,
            soft_policy=batch["soft_policy"].to(device),
        )

        loss = 0.0
        if concept_loss is not None:
            loss = loss + concept_loss
        if policy_loss is not None:
            loss = loss + policy_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 50 == 0:
            print(f"step={step} loss={float(loss):.4f} policy_loss={float(policy_loss or 0.0):.4f} concept_loss={float(concept_loss or 0.0):.4f}")


if __name__ == "__main__":
    main()


