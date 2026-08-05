#!/usr/bin/env python3
"""Analytic first-order gradient analysis of the tenuki intervention (F1).

This is the follow-up experiment F1 from ``TENUKI_INTERVENTION_ANALYSIS.md``.
It is an **exploratory diagnostic** and cannot alter the registered
validity-v5 confirmatory verdict.

For each held-out causal position it computes the gradient of the frozen
far-region readout (policy mass on legal board moves at Manhattan distance
at least six from the previous non-pass move, renormalised over legal board
moves) with respect to the saved ``trunkfinal`` tensor, by autograd through
the unchanged policy head at dose zero.

Because the validity-v5 intervention is linear in dose,

    h'[c,y,x] = h[c,y,x] + d * m[y,x] * delta[c],

the first-order dose response of the readout for *any* direction/mask pair
is a single dot product with that per-position gradient:

    d(readout)/d(dose) |_{d=0} = sum_{c,y,x} grad[c,y,x] * m[y,x] * delta[c].

One backward pass per position therefore yields the exact first-order slope
of the trained direction, of every random-direction control, and of every
shuffled-mask control, with no further forward passes and no disruption
calibration. A finite-difference check against the validated policy-head
backend verifies the autograd path on a deterministic subset.

The tool reuses the validated loading, position-selection, direction, and
control machinery from ``validated_causal_eval`` so that positions, masks,
random directions, and shuffles are byte-identical to the confirmatory run
when invoked with the same seed (20260730).
"""

from __future__ import annotations

import argparse
import json
import math
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
    from .causal_controls import (
        random_direction_control_ids,
        sha256_file,
        shuffle_control_ids,
        shuffled_position_mask,
    )
    from .operational_definitions import legal_board_mask, tenuki_target_mask
    from .validated_causal_eval import (
        DEFAULT_CAUSAL_TEST_POSITIONS,
        DEFAULT_RANDOM_DIRECTIONS,
        DEFAULT_SEED,
        DEFAULT_SHUFFLES,
        InterventionDirection,
        PolicyHeadOnlyBackend,
        ReplayPosition,
        concept_policy_readouts,
        load_validated_run,
        prepare_replay_positions,
        select_positions,
        _load_model,
    )
except ImportError:  # pragma: no cover - direct CLI execution.
    from causal_controls import (
        random_direction_control_ids,
        sha256_file,
        shuffle_control_ids,
        shuffled_position_mask,
    )
    from operational_definitions import legal_board_mask, tenuki_target_mask
    from validated_causal_eval import (
        DEFAULT_CAUSAL_TEST_POSITIONS,
        DEFAULT_RANDOM_DIRECTIONS,
        DEFAULT_SEED,
        DEFAULT_SHUFFLES,
        InterventionDirection,
        PolicyHeadOnlyBackend,
        ReplayPosition,
        concept_policy_readouts,
        load_validated_run,
        prepare_replay_positions,
        select_positions,
        _load_model,
    )


ANALYSIS_NAME = "tenuki_gradient_analysis_f1"
SCHEMA_VERSION = 1
DEFAULT_FD_POSITIONS = 8
DEFAULT_FD_DOSE = 0.25
CONCEPT = "tenuki"
REPRESENTATION = "local"


def flat_tenuki_masks(board: Any, previous_move: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Boolean legal and far (distance >= 6) masks in row-major tensor order.

    Row-major flattening of the ``[y, x]`` board masks matches the policy
    tensor index convention ``x = idx % size, y = idx // size`` used by the
    policy head output and by ``idx361``.
    """

    legal = legal_board_mask(board)
    far = tenuki_target_mask(board, previous_move)
    return legal.reshape(-1), far.reshape(-1)


def far_mass_and_gradient(
    policy_head: Any,
    trunkfinal: np.ndarray,
    board_mask: np.ndarray,
    legal_flat: np.ndarray,
    far_flat: np.ndarray,
    *,
    device: Any = "cpu",
    dtype: Any = None,
) -> Tuple[float, np.ndarray]:
    """Far-region readout and its gradient with respect to the trunk tensor.

    ``policy_head`` must accept ``(activations, mask=, mask_sum_hw=,
    mask_sum=, extra_outputs=)`` and return logits of shape ``(N, C, H*W+1)``
    whose channel 0 is the move policy, exactly as KataGo's policy head does.
    """

    import torch

    if dtype is None:
        dtype = torch.float32
    legal_flat = np.asarray(legal_flat, dtype=bool)
    far_flat = np.asarray(far_flat, dtype=bool)
    if legal_flat.shape != far_flat.shape or legal_flat.ndim != 1:
        raise ValueError("legal and far masks must be equal-length flat boolean arrays")
    if np.any(far_flat & ~legal_flat):
        raise ValueError("far mask must be a subset of the legal mask")
    if not legal_flat.any() or not far_flat.any():
        raise ValueError("legal and far action sets must be non-empty")

    trunk = np.asarray(trunkfinal)
    if trunk.ndim != 3:
        raise ValueError(f"Expected trunkfinal (C,H,W), got shape {trunk.shape}")
    height, width = trunk.shape[-2:]
    if legal_flat.size != height * width:
        raise ValueError("Flat masks do not match the trunk spatial size")

    h = torch.tensor(trunk, dtype=dtype, device=device, requires_grad=True)
    board = np.asarray(board_mask, dtype=np.float32)
    if board.shape != (height, width):
        raise ValueError("board_mask does not match the trunk spatial size")
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
    if logits.ndim != 3 or int(logits.shape[1]) < 1:
        raise ValueError(f"policy head returned invalid shape {tuple(logits.shape)}")
    expected = height * width + 1
    if int(logits.shape[2]) != expected:
        raise ValueError(
            f"policy head returned {int(logits.shape[2])} actions; expected {expected}"
        )
    probabilities = torch.softmax(logits[0, 0, :], dim=0)
    board_probabilities = probabilities[: height * width]
    legal_index = torch.as_tensor(np.flatnonzero(legal_flat), device=device)
    far_index = torch.as_tensor(np.flatnonzero(far_flat), device=device)
    far_mass = board_probabilities[far_index].sum() / board_probabilities[legal_index].sum()
    (gradient,) = torch.autograd.grad(far_mass, h)
    return float(far_mass.detach().cpu()), gradient.detach().cpu().numpy().astype(np.float64)


def mask_projected_gradient(gradient: np.ndarray, spatial_mask: np.ndarray) -> np.ndarray:
    """Project the (C,H,W) readout gradient through a fixed spatial mask -> (C,)."""

    return np.einsum(
        "chw,hw->c",
        np.asarray(gradient, dtype=np.float64),
        np.asarray(spatial_mask, dtype=np.float64),
    )


def direction_projected_gradient(gradient: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Project the (C,H,W) readout gradient through a fixed channel direction -> (H,W)."""

    return np.einsum(
        "chw,c->hw",
        np.asarray(gradient, dtype=np.float64),
        np.asarray(delta, dtype=np.float64),
    )


def stratified_mean(values: Sequence[float], labels: Sequence[int]) -> float:
    """Equal-label-strata mean, matching the confirmatory estimand weighting."""

    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    if values.shape != labels.shape or values.ndim != 1 or values.size == 0:
        raise ValueError("values and labels must be equal-length non-empty 1-D arrays")
    strata = sorted(set(labels.tolist()))
    return float(np.mean([values[labels == stratum].mean() for stratum in strata]))


def one_sided_positive_p(trained: float, controls: Sequence[float]) -> float:
    """Empirical p for 'controls at least as extreme in the positive direction'."""

    controls = np.asarray(controls, dtype=np.float64)
    return float((1 + int(np.sum(controls >= trained))) / (1 + controls.size))


def _summary_statistics(values: Sequence[float]) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _verify_against_causal_selection(
    positions: Sequence[ReplayPosition], causal_dir: Path, split_role: str
) -> Optional[str]:
    """Cross-check position identity against the confirmatory run, if present."""

    selected_path = causal_dir / "selected_positions.parquet"
    if not selected_path.is_file():
        return None
    import pandas as pd

    frame = pd.read_parquet(selected_path)
    expected = set(
        frame.loc[frame["split_role"].astype(str).eq(split_role), "position_id"].astype(str)
    )
    observed = {position.position_id for position in positions}
    if expected != observed:
        raise RuntimeError(
            "Selected positions differ from the confirmatory causal run: "
            f"{len(expected - observed)} missing, {len(observed - expected)} extra. "
            "Use the confirmatory seed and position count to reproduce them."
        )
    return sha256_file(selected_path)


def run_analysis(args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")

    run = load_validated_run(run_dir, CONCEPT, REPRESENTATION)
    model = _load_model(Path(args.model), args.device, run.board_size)
    model.eval()
    backend = PolicyHeadOnlyBackend(
        model,
        channels=run.channels,
        board_size=run.board_size,
        policy_atol=args.policy_atol,
        activation_atol=args.activation_atol,
    )

    selected = select_positions(run, args.split, args.max_positions, seed=args.seed)
    positions = prepare_replay_positions(model, run, selected, policy_backend=backend)
    equivalence = backend.validate_equivalence(positions, seed=args.seed)
    selection_cross_check = _verify_against_causal_selection(
        positions, Path(args.causal_dir).resolve(), args.split
    )

    learned = InterventionDirection.load(run)
    learned_delta = np.asarray(learned.local_delta, dtype=np.float64)
    random_ids = random_direction_control_ids(args.random_directions)
    random_deltas = [
        np.asarray(learned.random_control(control_id, seed=args.seed).local_delta, np.float64)
        for control_id in random_ids
    ]
    shuffle_ids = shuffle_control_ids(args.shuffles)

    labels: List[int] = []
    trained_derivatives: List[float] = []
    random_derivatives = np.zeros((len(positions), len(random_deltas)), dtype=np.float64)
    shuffle_derivatives = np.zeros((len(positions), len(shuffle_ids)), dtype=np.float64)
    rows: List[Dict[str, Any]] = []
    max_readout_discrepancy = 0.0

    for position_index, position in enumerate(positions):
        legal_flat, far_flat = flat_tenuki_masks(position.board, position.previous_move)
        far_mass, gradient = far_mass_and_gradient(
            model.policy_head,
            position.trunkfinal,
            position.model_board_mask,
            legal_flat,
            far_flat,
            device=backend.device,
        )
        baseline_readout = concept_policy_readouts(
            CONCEPT,
            position.baseline,
            position.board,
            previous_move=position.previous_move,
        )["tenuki_distance6_policy_mass"]
        max_readout_discrepancy = max(
            max_readout_discrepancy, abs(far_mass - float(baseline_readout))
        )

        aligned_mask = np.asarray(position.spatial_mask, dtype=np.float64)
        gradient_by_channel = mask_projected_gradient(gradient, aligned_mask)
        trained_derivative = float(gradient_by_channel @ learned_delta)
        for control_index, delta in enumerate(random_deltas):
            random_derivatives[position_index, control_index] = float(
                gradient_by_channel @ delta
            )
        gradient_by_point = direction_projected_gradient(gradient, learned_delta)
        for control_index, control_id in enumerate(shuffle_ids):
            shuffled = shuffled_position_mask(
                position.spatial_mask,
                base_seed=args.seed,
                repeat_id=control_id,
                position_id=position.position_id,
            )
            shuffle_derivatives[position_index, control_index] = float(
                np.sum(gradient_by_point * np.asarray(shuffled, dtype=np.float64))
            )

        labels.append(int(position.label))
        trained_derivatives.append(trained_derivative)
        rows.append(
            {
                "position_id": position.position_id,
                "label": int(position.label),
                "baseline_far_mass": float(baseline_readout),
                "analytic_far_mass": far_mass,
                "trained_directional_derivative": trained_derivative,
            }
        )

    labels_array = np.asarray(labels, dtype=int)
    trained_array = np.asarray(trained_derivatives, dtype=np.float64)
    trained_mean = stratified_mean(trained_array, labels_array)
    random_means = [
        stratified_mean(random_derivatives[:, index], labels_array)
        for index in range(random_derivatives.shape[1])
    ]
    shuffle_means = [
        stratified_mean(shuffle_derivatives[:, index], labels_array)
        for index in range(shuffle_derivatives.shape[1])
    ]
    absolute_random = np.abs(np.asarray(random_means))

    finite_difference = _finite_difference_check(
        backend,
        positions,
        learned,
        trained_array,
        fd_positions=args.fd_positions,
        fd_dose=args.fd_dose,
    )

    empirical = _load_empirical_slope(Path(args.causal_dir).resolve())

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
            "random_directions": len(random_deltas),
            "spatial_shuffles": len(shuffle_ids),
            "selection_matches_confirmatory_run": selection_cross_check is not None,
            "confirmatory_selected_positions_sha256": selection_cross_check,
            "policy_head_equivalence_max_policy_error": float(
                equivalence["max_policy_abs_error"]
            ),
            "policy_head_equivalence_max_activation_error": float(
                equivalence["max_activation_abs_error"]
            ),
            "policy_equivalence_atol": float(args.policy_atol),
            "activation_replay_atol": float(args.activation_atol),
            "activation_replay_note": (
                "Saved trunk tensors are hash-bound to the confirmatory run; the "
                "activation replay tolerance only gates present-environment replay "
                "drift and does not affect which tensors are analysed."
            ),
        },
        "readout_consistency": {
            "description": (
                "Maximum absolute difference between the differentiable far-mass "
                "readout at dose 0 and the validated backend baseline readout."
            ),
            "max_abs_discrepancy": max_readout_discrepancy,
        },
        "trained_direction": {
            "stratified_mean_derivative": trained_mean,
            "mean_by_label": {
                str(stratum): float(trained_array[labels_array == stratum].mean())
                for stratum in sorted(set(labels_array.tolist()))
            },
            "positions_with_negative_derivative": int(np.sum(trained_array < 0.0)),
            "positions_total": int(trained_array.size),
            "per_position": rows,
        },
        "random_direction_controls": {
            "stratified_means": [float(value) for value in random_means],
            "statistics": _summary_statistics(random_means),
            "one_sided_positive_p": one_sided_positive_p(trained_mean, random_means),
            "fraction_with_larger_absolute_derivative": float(
                np.mean(absolute_random >= abs(trained_mean))
            ),
        },
        "shuffled_mask_controls": {
            "stratified_means": [float(value) for value in shuffle_means],
            "statistics": _summary_statistics(shuffle_means),
            "one_sided_positive_p": one_sided_positive_p(trained_mean, shuffle_means),
        },
        "finite_difference_check": finite_difference,
        "empirical_comparison": empirical
        and {
            "confirmatory_slope": empirical,
            "analytic_first_order_slope": trained_mean,
            "absolute_difference": abs(empirical - trained_mean),
            "note": (
                "The confirmatory OLS slope over doses -2..2 should match the "
                "analytic dose-0 derivative if the response is linear in dose."
            ),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result


def _finite_difference_check(
    backend: PolicyHeadOnlyBackend,
    positions: Sequence[ReplayPosition],
    learned: InterventionDirection,
    trained_derivatives: np.ndarray,
    *,
    fd_positions: int,
    fd_dose: float,
) -> Dict[str, Any]:
    """Central-difference verification of the autograd derivatives."""

    if fd_positions <= 0 or fd_dose <= 0:
        raise ValueError("fd_positions and fd_dose must be positive")
    ordered = sorted(range(len(positions)), key=lambda index: positions[index].position_id)
    chosen = ordered[: min(fd_positions, len(positions))]
    subset = [positions[index] for index in chosen]
    masks = [position.spatial_mask for position in subset]

    def far_masses(dose: float) -> List[float]:
        outputs = backend.evaluate(
            subset, direction=learned, dose=dose, spatial_masks=masks
        )
        return [
            float(
                concept_policy_readouts(
                    CONCEPT,
                    output,
                    position.board,
                    previous_move=position.previous_move,
                )["tenuki_distance6_policy_mass"]
            )
            for position, output in zip(subset, outputs)
        ]

    upper = far_masses(fd_dose)
    lower = far_masses(-fd_dose)
    rows = []
    worst_absolute = 0.0
    for local_index, position_index in enumerate(chosen):
        central = (upper[local_index] - lower[local_index]) / (2.0 * fd_dose)
        analytic = float(trained_derivatives[position_index])
        difference = abs(central - analytic)
        worst_absolute = max(worst_absolute, difference)
        rows.append(
            {
                "position_id": positions[position_index].position_id,
                "central_difference": central,
                "analytic_derivative": analytic,
                "absolute_difference": difference,
            }
        )
    return {
        "dose_step": float(fd_dose),
        "positions_checked": len(rows),
        "max_absolute_difference": worst_absolute,
        "rows": rows,
    }


def _load_empirical_slope(causal_dir: Path) -> Optional[float]:
    report_path = causal_dir / "validated_causal_results_report.json"
    if not report_path.is_file():
        return None
    with report_path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    try:
        return float(report["sole_primary_confirmatory_test"]["statistic"]["slope"])
    except (KeyError, TypeError, ValueError):
        return None


def _print_summary(result: Mapping[str, Any]) -> None:
    trained = result["trained_direction"]
    random_controls = result["random_direction_controls"]
    shuffles = result["shuffled_mask_controls"]
    fd = result["finite_difference_check"]
    print(f"Positions analysed: {trained['positions_total']}")
    print(
        "Trained-direction stratified mean derivative: "
        f"{trained['stratified_mean_derivative']:+.7f} per unit dose"
    )
    print(
        f"Negative at {trained['positions_with_negative_derivative']} / "
        f"{trained['positions_total']} positions"
    )
    stats = random_controls["statistics"]
    print(
        "Random directions (aligned mask): mean "
        f"{stats['mean']:+.7f}, sd {stats['sd']:.7f}, "
        f"range [{stats['min']:+.7f}, {stats['max']:+.7f}]"
    )
    print(
        "One-sided positive p (random): "
        f"{random_controls['one_sided_positive_p']:.3f}; "
        "fraction with |derivative| >= |trained|: "
        f"{random_controls['fraction_with_larger_absolute_derivative']:.2f}"
    )
    stats = shuffles["statistics"]
    print(
        "Shuffled masks (learned direction): mean "
        f"{stats['mean']:+.7f}, sd {stats['sd']:.7f}"
    )
    print(
        f"Finite-difference check (dose step {fd['dose_step']}): "
        f"max |analytic - central| = {fd['max_absolute_difference']:.3g} "
        f"over {fd['positions_checked']} positions"
    )
    comparison = result.get("empirical_comparison")
    if comparison:
        print(
            "Confirmatory slope "
            f"{comparison['confirmatory_slope']:+.7f} vs analytic "
            f"{comparison['analytic_first_order_slope']:+.7f} "
            f"(|difference| {comparison['absolute_difference']:.3g})"
        )
    print(
        "Max |analytic - backend| baseline readout discrepancy: "
        f"{result['readout_consistency']['max_abs_discrepancy']:.3g}"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default=str(SCRIPT_DIR / "artifacts" / "runs" / "validity_v5_canonical"),
        help="Validated probe run directory",
    )
    parser.add_argument(
        "--causal-dir",
        default=str(
            SCRIPT_DIR
            / "artifacts"
            / "runs"
            / "validity_v5_canonical"
            / "causal"
            / "tenuki_local"
        ),
        help="Confirmatory causal output directory (for cross-checks)",
    )
    parser.add_argument(
        "--model", default=str(SCRIPT_DIR / "model.ckpt"), help="Model checkpoint"
    )
    parser.add_argument(
        "--output",
        default=str(
            SCRIPT_DIR / "artifacts" / "exploratory" / "tenuki_gradient_analysis.json"
        ),
        help="Output JSON path (must not exist)",
    )
    parser.add_argument("--split", default="causal_test", choices=["causal_test", "control_calibration"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--max-positions", type=int, default=DEFAULT_CAUSAL_TEST_POSITIONS
    )
    parser.add_argument(
        "--random-directions", type=int, default=DEFAULT_RANDOM_DIRECTIONS
    )
    parser.add_argument("--shuffles", type=int, default=DEFAULT_SHUFFLES)
    parser.add_argument("--fd-positions", type=int, default=DEFAULT_FD_POSITIONS)
    parser.add_argument("--fd-dose", type=float, default=DEFAULT_FD_DOSE)
    parser.add_argument(
        "--policy-atol",
        type=float,
        default=1e-6,
        help="Policy-head equivalence tolerance (frozen confirmatory value: 1e-6)",
    )
    parser.add_argument(
        "--activation-atol",
        type=float,
        default=1e-5,
        help=(
            "Full-network trunk replay tolerance. The frozen confirmatory value is "
            "1e-5; replay in a changed torch environment can drift slightly above "
            "it without affecting the hash-bound saved tensors being analysed."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    result = run_analysis(args)
    _print_summary(result)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
