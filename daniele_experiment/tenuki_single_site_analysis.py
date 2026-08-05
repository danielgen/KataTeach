#!/usr/bin/env python3
"""Single-site tenuki intervention analysis (F5) with broadcast decomposition.

This is follow-up experiment F5 from ``TENUKI_INTERVENTION_ANALYSIS.md``, an
**exploratory diagnostic** that cannot alter the registered validity-v5
verdict. It asks the question the broadcast intervention could not:

    Does pushing the learned tenuki direction at a *single* candidate
    location — the kind of location the local probe was actually trained
    on — make the network more likely to play that move?

Two first-order quantities are computed per held-out causal-test position by
autograd through the frozen policy head at dose zero:

1. **Single-site self-effects.** For candidate locations (the actually
   selected move, the top far candidates, and the top near candidates by
   baseline policy), the derivative of that location's legal-board policy
   probability with respect to a dose applied only at that location. A
   single-point mask has RMS one, so one dose unit equals one raw
   probe-score unit at the intervened site. Under the aligned-mediation
   hypothesis, self-effects at far candidates should be positive.

2. **Broadcast decomposition.** The confirmatory far-mass derivative is a
   sum over board sites of ``mask[site] * (grad_far_mass[:, site] @ delta)``.
   Splitting that sum into far (positive-mask) and near (negative-mask)
   sites shows which region produces the registered negative slope, and the
   unmasked per-site couplings show whether pushing "tenuki-ness" at far
   sites helps or hurts far-region policy mass.

Random-direction comparisons reuse the confirmatory run's 100 seeded control
directions, evaluated exactly through stored gradient columns.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
PYTHON_DIR = REPO_DIR / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.append(str(PYTHON_DIR))

try:  # Package import.
    from .causal_controls import random_direction_control_ids, sha256_file
    from .tenuki_gradient_analysis import (
        direction_projected_gradient,
        flat_tenuki_masks,
        stratified_mean,
    )
    from .validated_causal_eval import (
        DEFAULT_CAUSAL_TEST_POSITIONS,
        DEFAULT_RANDOM_DIRECTIONS,
        DEFAULT_SEED,
        InterventionDirection,
        PolicyHeadOnlyBackend,
        load_validated_run,
        prepare_replay_positions,
        select_positions,
        _load_model,
    )
except ImportError:  # pragma: no cover - direct CLI execution.
    from causal_controls import random_direction_control_ids, sha256_file
    from tenuki_gradient_analysis import (
        direction_projected_gradient,
        flat_tenuki_masks,
        stratified_mean,
    )
    from validated_causal_eval import (
        DEFAULT_CAUSAL_TEST_POSITIONS,
        DEFAULT_RANDOM_DIRECTIONS,
        DEFAULT_SEED,
        InterventionDirection,
        PolicyHeadOnlyBackend,
        load_validated_run,
        prepare_replay_positions,
        select_positions,
        _load_model,
    )


ANALYSIS_NAME = "tenuki_single_site_analysis_f5"
SCHEMA_VERSION = 1
DEFAULT_CANDIDATES_PER_SET = 5
CONCEPT = "tenuki"
REPRESENTATION = "local"


def ratio_readout_and_gradient(
    policy_head: Any,
    trunkfinal: np.ndarray,
    board_mask: np.ndarray,
    numerator_flat: np.ndarray,
    legal_flat: np.ndarray,
    *,
    device: Any = "cpu",
    dtype: Any = None,
) -> Tuple[float, np.ndarray]:
    """Readout ``sum(p[numerator]) / sum(p[legal])`` and its trunk gradient.

    Generalises the far-mass readout: with ``numerator = far`` it is the
    confirmatory far-mass; with a single-site numerator it is that move's
    legal-board-conditional policy probability.
    """

    import torch

    if dtype is None:
        dtype = torch.float32
    numerator_flat = np.asarray(numerator_flat, dtype=bool)
    legal_flat = np.asarray(legal_flat, dtype=bool)
    if numerator_flat.shape != legal_flat.shape or legal_flat.ndim != 1:
        raise ValueError("numerator and legal masks must be equal-length flat boolean arrays")
    if np.any(numerator_flat & ~legal_flat):
        raise ValueError("numerator mask must be a subset of the legal mask")
    if not legal_flat.any() or not numerator_flat.any():
        raise ValueError("legal and numerator action sets must be non-empty")

    trunk = np.asarray(trunkfinal)
    height, width = trunk.shape[-2:]
    if legal_flat.size != height * width:
        raise ValueError("Flat masks do not match the trunk spatial size")

    h = torch.tensor(trunk, dtype=dtype, device=device, requires_grad=True)
    board = np.asarray(board_mask, dtype=np.float32)
    torch_mask = torch.tensor(board, dtype=dtype, device=device).view(1, 1, height, width)
    mask_sum_hw = torch.sum(torch_mask, dim=(2, 3), keepdim=True)
    mask_sum = torch.sum(torch_mask)

    logits = policy_head(
        h.unsqueeze(0),
        mask=torch_mask,
        mask_sum_hw=mask_sum_hw,
        mask_sum=mask_sum,
        extra_outputs=None,
    )
    probabilities = torch.softmax(logits[0, 0, :], dim=0)
    board_probabilities = probabilities[: height * width]
    numerator_index = torch.as_tensor(np.flatnonzero(numerator_flat), device=device)
    legal_index = torch.as_tensor(np.flatnonzero(legal_flat), device=device)
    readout = (
        board_probabilities[numerator_index].sum()
        / board_probabilities[legal_index].sum()
    )
    (gradient,) = torch.autograd.grad(readout, h)
    return float(readout.detach().cpu()), gradient.detach().cpu().numpy().astype(np.float64)


def select_candidate_sites(
    baseline_board_probabilities: np.ndarray,
    legal_flat: np.ndarray,
    far_flat: np.ndarray,
    *,
    per_set: int,
) -> Dict[str, List[int]]:
    """Deterministic top-probability candidate sites in each contract region."""

    probabilities = np.asarray(baseline_board_probabilities, dtype=np.float64)
    legal_flat = np.asarray(legal_flat, dtype=bool)
    far_flat = np.asarray(far_flat, dtype=bool)
    if probabilities.shape != legal_flat.shape:
        raise ValueError("Baseline probabilities and masks have different sizes")

    def top_sites(candidate_mask: np.ndarray) -> List[int]:
        indices = np.flatnonzero(candidate_mask)
        if indices.size == 0:
            return []
        ranked = indices[np.argsort(-probabilities[indices], kind="stable")]
        return [int(index) for index in ranked[:per_set]]

    return {
        "far": top_sites(legal_flat & far_flat),
        "near": top_sites(legal_flat & ~far_flat),
    }


def single_site_flat_mask(size: int, site_index: int) -> np.ndarray:
    """Flat single-point boolean mask (RMS of the active support is one)."""

    if not 0 <= site_index < size:
        raise ValueError(f"site index {site_index} outside flat board of size {size}")
    mask = np.zeros(size, dtype=bool)
    mask[site_index] = True
    return mask


def _aggregate(values: Sequence[float]) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "fraction_positive": float(np.mean(values > 0.0)),
        "n": int(values.size),
    }


def run_analysis(args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")

    run = load_validated_run(run_dir, CONCEPT, REPRESENTATION)
    model = _load_model(Path(args.model), args.device, run.board_size)
    model.eval()
    backend = PolicyHeadOnlyBackend(model, channels=run.channels, board_size=run.board_size)
    board_points = run.board_size * run.board_size

    selected = select_positions(run, args.split, args.max_positions, seed=args.seed)
    positions = prepare_replay_positions(model, run, selected, policy_backend=backend)

    learned = InterventionDirection.load(run)
    learned_delta = np.asarray(learned.local_delta, dtype=np.float64)
    random_ids = random_direction_control_ids(args.random_directions)
    random_deltas = np.stack(
        [
            np.asarray(learned.random_control(control_id, seed=args.seed).local_delta, np.float64)
            for control_id in random_ids
        ]
    )

    labels: List[int] = []
    rows: List[Dict[str, Any]] = []
    far_contributions: List[float] = []
    near_contributions: List[float] = []
    total_broadcast: List[float] = []
    far_site_couplings: List[float] = []
    near_site_couplings: List[float] = []
    selected_self_effects: List[float] = []
    selected_is_far: List[bool] = []
    far_candidate_means: List[float] = []
    near_candidate_means: List[float] = []
    far_candidate_pool: List[float] = []
    near_candidate_pool: List[float] = []
    random_rows: List[np.ndarray] = []
    max_identity_error = 0.0

    for position in positions:
        legal_flat, far_flat = flat_tenuki_masks(position.board, position.previous_move)
        aligned_mask_flat = np.asarray(position.spatial_mask, dtype=np.float64).reshape(-1)

        # Broadcast decomposition from the far-mass gradient.
        _, far_mass_gradient = ratio_readout_and_gradient(
            model.policy_head,
            position.trunkfinal,
            position.model_board_mask,
            far_flat,
            legal_flat,
            device=backend.device,
        )
        coupling_map = direction_projected_gradient(far_mass_gradient, learned_delta).reshape(-1)
        contribution_map = coupling_map * aligned_mask_flat
        far_sites = far_flat
        near_sites = legal_flat & ~far_flat
        far_contribution = float(contribution_map[far_sites].sum())
        near_contribution = float(contribution_map[near_sites].sum())
        total = float(contribution_map.sum())
        max_identity_error = max(
            max_identity_error, abs(total - (far_contribution + near_contribution))
        )
        far_contributions.append(far_contribution)
        near_contributions.append(near_contribution)
        total_broadcast.append(total)
        far_site_couplings.append(float(coupling_map[far_sites].mean()))
        near_site_couplings.append(float(coupling_map[near_sites].mean()))

        # Single-site self-effects at deterministic candidates.
        baseline_probabilities = np.asarray(
            position.baseline["policy0"], dtype=np.float64
        )[:board_points]
        candidates = select_candidate_sites(
            baseline_probabilities, legal_flat, far_flat, per_set=args.candidates
        )

        def self_effect(site_index: int) -> Tuple[float, np.ndarray]:
            _, gradient = ratio_readout_and_gradient(
                model.policy_head,
                position.trunkfinal,
                position.model_board_mask,
                single_site_flat_mask(board_points, site_index),
                legal_flat,
                device=backend.device,
            )
            column = gradient.reshape(gradient.shape[0], -1)[:, site_index]
            return float(column @ learned_delta), column

        far_effects = []
        far_random = np.zeros(len(random_ids))
        for site_index in candidates["far"]:
            effect, column = self_effect(site_index)
            far_effects.append(effect)
            far_random += random_deltas @ column
        near_effects = [self_effect(site_index)[0] for site_index in candidates["near"]]

        selected_effect: Optional[float] = None
        if position.idx361 is not None:
            selected_effect = self_effect(int(position.idx361))[0]
            selected_self_effects.append(selected_effect)
            selected_is_far.append(bool(far_flat[int(position.idx361)]))

        far_mean = float(np.mean(far_effects))
        near_mean = float(np.mean(near_effects))
        far_candidate_means.append(far_mean)
        near_candidate_means.append(near_mean)
        far_candidate_pool.extend(far_effects)
        near_candidate_pool.extend(near_effects)
        random_rows.append(far_random / max(len(candidates["far"]), 1))

        labels.append(int(position.label))
        rows.append(
            {
                "position_id": position.position_id,
                "label": int(position.label),
                "broadcast_far_sites_contribution": far_contribution,
                "broadcast_near_sites_contribution": near_contribution,
                "far_candidates_mean_self_effect": far_mean,
                "near_candidates_mean_self_effect": near_mean,
                "selected_move_self_effect": selected_effect,
                "selected_move_is_far": bool(far_flat[int(position.idx361)])
                if position.idx361 is not None
                else None,
            }
        )

    labels_array = np.asarray(labels, dtype=int)
    random_far_candidate_means = np.stack(random_rows)
    trained_far_mean = stratified_mean(far_candidate_means, labels_array)
    random_means = np.asarray(
        [
            stratified_mean(random_far_candidate_means[:, index], labels_array)
            for index in range(random_far_candidate_means.shape[1])
        ]
    )

    result: Dict[str, Any] = {
        "analysis": ANALYSIS_NAME,
        "schema_version": SCHEMA_VERSION,
        "evidential_status": (
            "exploratory_diagnostic_only; cannot alter the registered validity_v5 verdict"
        ),
        "provenance": {
            "probe_run": str(run.run_dir),
            "checkpoint": str(Path(args.model).resolve()),
            "checkpoint_sha256": sha256_file(Path(args.model)),
            "concept": CONCEPT,
            "representation": REPRESENTATION,
            "split_role": args.split,
            "seed": int(args.seed),
            "positions": len(positions),
            "candidates_per_region": int(args.candidates),
            "random_directions": len(random_ids),
            "dose_unit": (
                "one raw probe-score unit at the intervened site "
                "(single-point mask has unit RMS)"
            ),
        },
        "broadcast_decomposition": {
            "description": (
                "The confirmatory far-mass dose derivative decomposed over board "
                "sites: contribution(site) = aligned_mask[site] * "
                "(grad_far_mass[:, site] @ learned_delta). Far sites carry positive "
                "mask weight, near sites negative."
            ),
            "stratified_mean_total": stratified_mean(total_broadcast, labels_array),
            "far_sites_contribution": {
                "stratified_mean": stratified_mean(far_contributions, labels_array),
                **_aggregate(far_contributions),
            },
            "near_sites_contribution": {
                "stratified_mean": stratified_mean(near_contributions, labels_array),
                **_aggregate(near_contributions),
            },
            "unmasked_mean_site_coupling": {
                "description": (
                    "Mean over sites of grad_far_mass[:, site] @ learned_delta: the "
                    "effect on far-region mass of one probe-score unit at one site, "
                    "before mask weighting."
                ),
                "far_sites": _aggregate(far_site_couplings),
                "near_sites": _aggregate(near_site_couplings),
            },
            "decomposition_identity_max_error": max_identity_error,
        },
        "single_site_self_effects": {
            "description": (
                "Derivative of a location's legal-board policy probability with "
                "respect to a dose applied only at that location, along the learned "
                "direction. Aligned mediation predicts positive values at far "
                "candidates."
            ),
            "far_candidates": {
                "stratified_mean_of_position_means": trained_far_mean,
                "position_means": _aggregate(far_candidate_means),
                "pooled_candidates": _aggregate(far_candidate_pool),
            },
            "near_candidates": {
                "stratified_mean_of_position_means": stratified_mean(
                    near_candidate_means, labels_array
                ),
                "position_means": _aggregate(near_candidate_means),
                "pooled_candidates": _aggregate(near_candidate_pool),
            },
            "selected_move": {
                "all": _aggregate(selected_self_effects),
                "far_selected": _aggregate(
                    [
                        value
                        for value, is_far in zip(selected_self_effects, selected_is_far)
                        if is_far
                    ]
                ),
                "near_selected": _aggregate(
                    [
                        value
                        for value, is_far in zip(selected_self_effects, selected_is_far)
                        if not is_far
                    ]
                ),
            },
            "random_direction_comparison_far_candidates": {
                "trained_stratified_mean": trained_far_mean,
                "random_stratified_means": [float(value) for value in random_means],
                "random_mean": float(random_means.mean()),
                "random_sd": float(random_means.std(ddof=1)),
                "randoms_more_positive_than_trained": int(
                    np.sum(random_means >= trained_far_mean)
                ),
                "randoms_with_larger_absolute_value": int(
                    np.sum(np.abs(random_means) >= abs(trained_far_mean))
                ),
            },
        },
        "per_position": rows,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result


def _print_summary(result: Mapping[str, Any]) -> None:
    decomposition = result["broadcast_decomposition"]
    single = result["single_site_self_effects"]
    print(
        "Broadcast total (should match F1): "
        f"{decomposition['stratified_mean_total']:+.7f}"
    )
    print(
        "  far-site contribution:  "
        f"{decomposition['far_sites_contribution']['stratified_mean']:+.7f}"
    )
    print(
        "  near-site contribution: "
        f"{decomposition['near_sites_contribution']['stratified_mean']:+.7f}"
    )
    unmasked = decomposition["unmasked_mean_site_coupling"]
    print(
        "Unmasked mean site coupling to far mass: far sites "
        f"{unmasked['far_sites']['mean']:+.3e} "
        f"({unmasked['far_sites']['fraction_positive']:.0%} positive), near sites "
        f"{unmasked['near_sites']['mean']:+.3e} "
        f"({unmasked['near_sites']['fraction_positive']:.0%} positive)"
    )
    far = single["far_candidates"]
    near = single["near_candidates"]
    chosen = single["selected_move"]
    print(
        "Single-site self-effect, far candidates: mean "
        f"{far['position_means']['mean']:+.3e} "
        f"({far['pooled_candidates']['fraction_positive']:.0%} of candidates positive)"
    )
    print(
        "Single-site self-effect, near candidates: mean "
        f"{near['position_means']['mean']:+.3e} "
        f"({near['pooled_candidates']['fraction_positive']:.0%} positive)"
    )
    print(
        "Single-site self-effect, actually selected move: mean "
        f"{chosen['all']['mean']:+.3e} ({chosen['all']['fraction_positive']:.0%} positive; "
        f"far-selected mean {chosen['far_selected']['mean']:+.3e}, "
        f"near-selected mean {chosen['near_selected']['mean']:+.3e})"
    )
    comparison = single["random_direction_comparison_far_candidates"]
    print(
        "Far-candidate self-effect vs random directions: trained "
        f"{comparison['trained_stratified_mean']:+.3e}, random mean "
        f"{comparison['random_mean']:+.3e} (sd {comparison['random_sd']:.3e}); "
        f"{comparison['randoms_more_positive_than_trained']} / "
        f"{len(comparison['random_stratified_means'])} randoms more positive, "
        f"{comparison['randoms_with_larger_absolute_value']} with larger |value|"
    )
    print(
        "Decomposition identity max error: "
        f"{decomposition['decomposition_identity_max_error']:.3g}"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default=str(SCRIPT_DIR / "artifacts" / "runs" / "validity_v5_canonical"),
    )
    parser.add_argument("--model", default=str(SCRIPT_DIR / "model.ckpt"))
    parser.add_argument(
        "--output",
        default=str(
            SCRIPT_DIR / "artifacts" / "exploratory" / "tenuki_single_site_analysis.json"
        ),
    )
    parser.add_argument(
        "--split", default="causal_test", choices=["causal_test", "control_calibration"]
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-positions", type=int, default=DEFAULT_CAUSAL_TEST_POSITIONS)
    parser.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES_PER_SET)
    parser.add_argument("--random-directions", type=int, default=DEFAULT_RANDOM_DIRECTIONS)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    result = run_analysis(args)
    _print_summary(result)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
