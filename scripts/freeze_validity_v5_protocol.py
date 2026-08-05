#!/usr/bin/env python3
"""Create the append-only protocol frozen before fresh holdout generation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch


COHORT = "validity_v5_postfreeze_holdout"
FIRST_GAME_SEED = 202607300000
FRESH_GAMES = 150
SPLIT_SEED = 20260730


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games-dir", type=Path, default=Path("games"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("daniele_experiment/model.ckpt")
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    games_dir = args.games_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen protocol: {output}")
    if not games_dir.is_dir() or not checkpoint.is_file():
        raise FileNotFoundError("Games directory and checkpoint must already exist")

    # The cohort identifier must be unused when the protocol is frozen. This
    # prevents pilot/refill games from being silently admitted later.
    existing = []
    for meta_path in games_dir.glob("*/meta.json"):
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if metadata.get("cohort") == COHORT:
            existing.append(str(meta_path.parent.name))
    if existing:
        raise RuntimeError(
            f"Cohort {COHORT!r} already has {len(existing)} games; freeze must precede generation"
        )

    development_ids = sorted(
        path.name
        for path in games_dir.iterdir()
        if path.is_dir()
        and (path / "moves.jsonl").is_file()
        and (path / "trunkfinal").is_dir()
    )
    if len(development_ids) != 500:
        raise RuntimeError(
            f"Expected exactly 500 raw-complete development games, found {len(development_ids)}"
        )

    source_relatives = (
        "daniele_experiment/common_utils.py",
        "daniele_experiment/generate_games_dataset.py",
        "daniele_experiment/concepts_validated_v5.yaml",
        "daniele_experiment/operational_definitions.py",
        "daniele_experiment/build_validated_labels.py",
        "daniele_experiment/validated_probe_pipeline.py",
        "daniele_experiment/checkpoint_activation_fidelity.py",
        "daniele_experiment/causal_controls.py",
        "daniele_experiment/validated_causal_eval.py",
        "daniele_experiment/validated_results_report.py",
        "daniele_experiment/validated_causal_results_report.py",
        "python/board.py",
        "python/features.py",
        "python/gamestate.py",
        "python/load_model.py",
        "python/model_pytorch.py",
        "python/modelconfigs.py",
        "python/sgfmetadata.py",
    )
    missing = [relative for relative in source_relatives if not (repo / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Protocol source files are missing: {missing}")

    seeds = list(range(FIRST_GAME_SEED, FIRST_GAME_SEED + FRESH_GAMES))
    record = {
        "schema_version": 1,
        "protocol_id": "validity_v5",
        "status": "frozen_before_fresh_data_generation",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "historical_data_scope": {
            "development_games": 500,
            "development_game_ids_sha256": hashlib.sha256(
                ",".join(development_ids).encode("utf-8")
            ).hexdigest(),
            "statement": (
                "All pre-existing games are developmental because exploratory analyses "
                "previously inspected them. They may support nested-CV probe estimates "
                "but never the final causal-test claim."
            ),
        },
        "fresh_holdout": {
            "cohort": COHORT,
            "games": FRESH_GAMES,
            "control_calibration_games": 50,
            "causal_test_games": 100,
            "game_seed_first": seeds[0],
            "game_seed_last": seeds[-1],
            "game_seed_count": len(seeds),
            "game_seed_set_sha256": hashlib.sha256(
                ",".join(map(str, seeds)).encode("ascii")
            ).hexdigest(),
            "game_identity": "UUIDv5(NAMESPACE_URL, 'katateach:<cohort>:<game_seed>')",
            "assignment": (
                "sort fresh UUIDs, permute once with NumPy default_rng(split_seed), "
                "first 50 calibration and remaining 100 causal_test"
            ),
            "split_seed": SPLIT_SEED,
            "no_refill_rule": (
                "Any failed or pilot generation shard is quarantined. The seed set is not "
                "silently extended or replaced after any label or model output is inspected."
            ),
            "write_once": True,
        },
        "game_generation": {
            "board_size": 19,
            "rules": "GameState.RULES_TT",
            "model_evaluations": "direct neural policy without MCTS",
            "device": "cpu",
            "torch_threads": 4,
            "initial_temperature": 1.2,
            "final_temperature": 0.8,
            "transition_moves": 60,
            "minimum_raw_policy_probability": 0.01,
            "top_k": 10,
            "resign_threshold": 0.10,
            "resign_consecutive_moves": 3,
            "maximum_moves": 400,
            "move_sampling_rng": "NumPy default_rng/PCG64, one unique seed per game",
            "save_html": 0,
            "run_legacy_snorkel": False,
        },
        "labels": {
            "fresh_recomputed_contracts": [
                "tenuki_distance6@2",
                "reply_peak95@2",
                "regional_policy_peak@2",
            ],
            "fresh_legacy_migrated_fields": [],
            "all_selected_alignment_gate": True,
        },
        "probes": {
            "concepts_config": "daniele_experiment/concepts_validated_v5.yaml",
            "concepts": ["tenuki", "forcing", "urgency_peak"],
            "development_games_only": 500,
            "representations": ["global", "local", "combined"],
            "all_enabled_concepts_required": True,
            "outer_group_folds": 5,
            "inner_group_folds": 4,
            "group_unit": "game_id",
            "C_values": [0.001, 0.01, 0.1, 1.0, 10.0],
            "selection_metric": "mean inner-fold average precision",
            "f1_threshold": "inner out-of-fold maximum F1",
            "probability_calibration": False,
            "quality_gate": None,
            "max_iter": 2000,
        },
        "causal": {
            "primary_hypothesis": {
                "concept": "tenuki",
                "representation": "local",
                "readout": "tenuki_distance6_policy_mass",
                "prediction": "positive dose increases readout; negative dose decreases it",
                "statistic": {
                    "name": "label_balanced_ols_slope_across_all_frozen_doses",
                    "predictor": "nominal_dose",
                    "outcome": "paired readout delta",
                    "intercept": True,
                    "dose_set": "all values in causal.doses",
                    "label_weights": {"0": 0.5, "1": 0.5},
                    "resampling": (
                        "resample games with replacement within each frozen label "
                        "stratum; reuse each draw across every dose"
                    ),
                },
                "decision_rule": {
                    "alpha": 0.05,
                    "expected_trained_slope": "positive",
                    "random_direction_test": (
                        "one-sided finite-control empirical p <= alpha"
                    ),
                    "spatial_shuffle_test": (
                        "one-sided finite-control empirical p <= alpha"
                    ),
                    "headline_support_requires": (
                        "trained slope > 0 AND random-direction p <= alpha AND "
                        "spatial-shuffle p <= alpha"
                    ),
                    "no_substitution": (
                        "No individual dose, secondary readout, or minimum p-value "
                        "may replace this aggregate conjunction."
                    ),
                },
            },
            "secondary_exploratory_concepts": ["forcing", "urgency_peak"],
            "secondary_representation": "local",
            "secondary_evaluation_rules": {
                "forcing": {
                    "scope": "best_effort_exploratory",
                    "mask_eligibility": (
                        "A selected position must contain both reply_peak95 target and "
                        "comparison candidate actions."
                    ),
                    "infeasibility_rule": (
                        "If any prospectively selected game is mask-ineligible, omit the "
                        "forcing causal analysis as infeasible; do not replace/refill games."
                    ),
                    "confirmatory_gate": False,
                },
                "urgency_peak": {
                    "scope": "exploratory",
                    "confirmatory_gate": False,
                },
            },
            "doses": [-2.0, -1.0, 0.0, 1.0, 2.0],
            "one_position_per_game": True,
            "label_stratified_position_sampling": True,
            "maximum_calibration_positions": 50,
            "maximum_test_positions": 100,
            "spatial_shuffle_controls": 50,
            "random_direction_controls": 100,
            "control_matching": (
                "control_calibration games: mean legal-plus-pass policy "
                "Jensen-Shannon divergence"
            ),
            "policy_head_batch_size": 64,
            "full_vs_head_equivalence_sample_size": 6,
            "policy_equivalence_absolute_tolerance": 1e-6,
            "activation_equivalence_absolute_tolerance": 1e-5,
            "causal_seed": SPLIT_SEED,
        },
        "inference": {
            "unit": "game",
            "bootstrap_replicates": 2000,
            "confidence_level": 0.95,
            "control_empirical_p": "(1 + controls_at_least_as_extreme) / (1 + n_controls)",
            "balanced_position_sample": (
                "label-stratified estimand; do not present as prevalence-weighted corpus effect"
            ),
        },
        "checkpoint": {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": _sha256(checkpoint),
            "use_swa": False,
            "selected_weights": "raw_model",
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "development_activation_fidelity_gate": {
            "required_before_training": True,
            "split_role": "development",
            "sampling": "one deterministic saved pre-move position per development game",
            "expected_games": 500,
            "sample_count": 500,
            "seed": SPLIT_SEED,
            "device": "cpu",
            "absolute_max_error_tolerance": 0.0001,
            "claim_scope": (
                "Empirical compatibility of the supplied checkpoint with sampled saved "
                "activations, not proof that it originally generated the historical corpus."
            ),
        },
        "source_sha256": {
            relative: _sha256(repo / relative) for relative in source_relatives
        },
        "freeze_script_sha256": _sha256(Path(__file__).resolve()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(_canonical(record))
    output.chmod(0o444)
    print(json.dumps({"protocol": str(output), "sha256": _sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
