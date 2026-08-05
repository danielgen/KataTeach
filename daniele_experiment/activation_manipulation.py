#!/usr/bin/env python3
"""Causal interventions on KataGo concept activations during 1-visit play.

The linear probes in :mod:`linear_probe_pipeline` are observational.  This
module turns a (mean-pooled) probe into a forward hook on ``act_trunkfinal``
and measures whether changing that activation changes the policy and game.

Only ``feature_mode: pre`` probes are supported. The hook runs on the forward
pass that chooses the next move, so post-move / delta probes are the wrong
temporal target for this experiment (see ``ACTIVATION_MANIPULATION.md``).

An intervention dose is expressed in probe decision-score units.  Thus a
dose of +1 asks for ``w @ x`` to increase by one (one log-odds unit for the
saved logistic probes), rather than using an architecture-dependent tensor
scale.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import joblib
import numpy as np
import yaml

sys.path.append(str(Path(__file__).parent.parent / "python"))
SCRIPT_DIR = Path(__file__).resolve().parent


def _load_concepts_config(concepts_yaml: Path) -> Dict:
    with concepts_yaml.open() as f:
        return yaml.safe_load(f) or {}


def _concept_feature_mode(config: Dict, concept: str) -> str:
    specs = config.get("concepts") or {}
    if concept not in specs:
        raise KeyError(
            f"Concept {concept!r} not found in concepts.yaml. "
            f"Known concepts: {', '.join(sorted(specs))}"
        )
    return str(specs[concept].get("feature_mode", "pre")).lower()


def list_intervenable_concepts(probes_dir: Path, concepts_yaml: Path) -> List[str]:
    """Return enabled pre-mode concepts that have probe+scaler artifacts."""
    config = _load_concepts_config(concepts_yaml)
    specs = config.get("concepts") or {}
    candidates = [probes_dir]
    if not probes_dir.is_absolute():
        candidates.append(SCRIPT_DIR / probes_dir)

    names: List[str] = []
    for name, spec in specs.items():
        if not spec.get("enabled", True):
            continue
        if str(spec.get("feature_mode", "pre")).lower() != "pre":
            continue
        if any(
            (c / f"probe_{name}.joblib").exists() and (c / f"scaler_{name}.joblib").exists()
            for c in candidates
        ):
            names.append(name)
    return sorted(names)


def _resolve_probes_dir(path: Path, concept: str) -> Path:
    """Resolve probe artifacts from either the cwd or this experiment folder."""
    candidates = [path]
    if not path.is_absolute():
        candidates.append(SCRIPT_DIR / path)
    for candidate in candidates:
        if (candidate / f"probe_{concept}.joblib").exists() and (
            candidate / f"scaler_{concept}.joblib"
        ).exists():
            return candidate

    searched = ", ".join(str(candidate.resolve()) for candidate in candidates)
    available = sorted({
        probe.stem.removeprefix("probe_")
        for candidate in candidates
        if candidate.is_dir()
        for probe in candidate.glob("probe_*.joblib")
        if (candidate / f"scaler_{probe.stem.removeprefix('probe_')}.joblib").exists()
    })
    hint = f" Available concepts: {', '.join(available)}." if available else ""
    raise FileNotFoundError(
        f"Could not find probe_{concept}.joblib and scaler_{concept}.joblib. "
        f"Searched: {searched}.{hint}"
    )


@dataclass(frozen=True)
class ConceptIntervention:
    """A channel-space direction derived from a saved linear probe."""

    concept: str
    channel_delta_per_unit: np.ndarray
    probe_raw_direction: np.ndarray
    local_channel_delta_per_unit: Optional[np.ndarray] = None
    local_probe_raw_direction: Optional[np.ndarray] = None

    @classmethod
    def load(
        cls, concept: str, probes_dir: Path, concepts_yaml: Path
    ) -> "ConceptIntervention":
        probe_path = probes_dir / f"probe_{concept}.joblib"
        scaler_path = probes_dir / f"scaler_{concept}.joblib"
        if not probe_path.exists() or not scaler_path.exists():
            raise FileNotFoundError(
                f"Need both {probe_path.name} and {scaler_path.name} in {probes_dir}"
            )

        config = _load_concepts_config(concepts_yaml)
        mode = _concept_feature_mode(config, concept)
        if mode != "pre":
            eligible = list_intervenable_concepts(probes_dir, concepts_yaml)
            hint = (
                f" Eligible pre concepts with probes: {', '.join(eligible)}."
                if eligible
                else " No pre-mode probes found; train feature_mode: pre concepts first."
            )
            raise ValueError(
                f"Concept {concept!r} has feature_mode={mode!r}. "
                "This script only intervenes on feature_mode: pre probes, because the "
                "hook runs on the forward pass that chooses the next move. "
                "Use post/delta probes for observational readout, not this causal loop."
                f"{hint}"
            )

        feature_config = config.get("feature_extraction", {})
        aggregation = feature_config.get("aggregation", "global_pool")
        pool_type = feature_config.get("pool_type", "mean")
        if aggregation not in ("global_pool", "both") or pool_type not in ("mean", "both"):
            raise ValueError(
                "Activation manipulation requires a mean-pooled global feature block; "
                f"got aggregation={aggregation!r}, pool_type={pool_type!r}"
            )

        probe = joblib.load(probe_path)
        scaler = joblib.load(scaler_path)
        coef = np.asarray(probe.coef_[0], dtype=np.float64)
        scale = np.asarray(scaler.scale_, dtype=np.float64)
        if coef.shape != scale.shape:
            raise ValueError("Probe and scaler feature dimensions do not match")

        # The first block produced by aggregate_features is mean pooling.
        # Infer C from the model's configured trunk width where possible. For
        # this pipeline every subsequent block has the same width, so the
        # scaler mean block length is available in scaler.n_features_in_ only
        # up to an integer number of blocks. concepts.yaml defaults to one.
        # The dataset builder and the ablation in train_probes currently use a
        # 512-channel trunk explicitly. Keeping that contract here prevents an
        # ablated 512-feature probe from being mistaken for two 256-wide blocks.
        channels = int(feature_config.get("trunk_channels", 512))
        if len(coef) < channels:
            raise ValueError(
                f"Probe has {len(coef)} features, fewer than {channels} trunk channels"
            )

        raw_direction = coef[:channels] / np.maximum(scale[:channels], 1e-12)
        norm_sq = float(raw_direction @ raw_direction)
        if not np.isfinite(norm_sq) or norm_sq <= 0:
            raise ValueError(f"Concept {concept!r} has a zero/invalid global direction")
        # Broadcasting this over H,W changes the mean-pooled probe score by 1.
        delta = raw_direction / norm_sq

        # aggregate_features orders blocks as mean, optional max, then the
        # activation at the played move. The latter can be treated as a 1x1
        # spatial direction and evaluated prospectively at every board point.
        local_offset = channels * (2 if pool_type == "both" else 1)
        local_raw = None
        local_delta = None
        if feature_config.get("include_move_location", True) and len(coef) >= local_offset + channels:
            candidate = coef[local_offset:local_offset + channels] / np.maximum(
                scale[local_offset:local_offset + channels], 1e-12
            )
            local_norm_sq = float(candidate @ candidate)
            if np.isfinite(local_norm_sq) and local_norm_sq > 0:
                local_raw = candidate.astype(np.float32)
                local_delta = (candidate / local_norm_sq).astype(np.float32)
        return cls(
            concept, delta.astype(np.float32), raw_direction.astype(np.float32),
            local_delta, local_raw,
        )

    @contextmanager
    def apply(
        self,
        model,
        dose: float,
        *,
        component: str = "global",
        spatial_mask: Optional[np.ndarray] = None,
        direction_override: Optional[np.ndarray] = None,
    ) -> Iterator[None]:
        """Apply this intervention for model forwards within the context."""
        if dose == 0:
            yield
            return

        if component not in {"global", "local"}:
            raise ValueError(f"Unknown intervention component: {component}")
        direction = self.channel_delta_per_unit
        if component == "local":
            if self.local_channel_delta_per_unit is None:
                raise ValueError(
                    f"Concept {self.concept!r} has no trained move-location feature block"
                )
            if spatial_mask is None:
                raise ValueError("A spatial_mask is required for a local intervention")
            direction = self.local_channel_delta_per_unit
        if direction_override is not None:
            override = np.asarray(direction_override, dtype=np.float32)
            if override.shape != direction.shape:
                raise ValueError(
                    f"Direction override has shape {override.shape}; expected {direction.shape}"
                )
            direction = override

        def hook(_module, _inputs, output):
            import torch

            delta = torch.as_tensor(
                direction,
                dtype=output.dtype,
                device=output.device,
            ).view(1, -1, 1, 1)
            if output.ndim != 4 or output.shape[1] != delta.shape[1]:
                raise ValueError(
                    f"Expected trunk tensor N,C,H,W with C={delta.shape[1]}, "
                    f"got {tuple(output.shape)}"
                )
            if component == "local":
                mask = torch.as_tensor(spatial_mask, dtype=output.dtype, device=output.device)
                if mask.ndim == 2:
                    mask = mask.view(1, 1, *mask.shape)
                if tuple(mask.shape[-2:]) != tuple(output.shape[-2:]):
                    raise ValueError(
                        f"Spatial mask {tuple(mask.shape[-2:])} does not match activation "
                        f"shape {tuple(output.shape[-2:])}"
                    )
                delta = delta * mask
            return output + float(dose) * delta

        handle = model.act_trunkfinal.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()


def _policy_vector(outputs: Dict, board) -> np.ndarray:
    """Dense policy over board points plus pass, with illegal moves set to zero."""
    size = board.size * board.size + 1
    result = np.zeros(size, dtype=np.float64)
    for loc, prob in outputs["moves_and_probs0"]:
        if loc == board.PASS_LOC:
            idx = size - 1
        else:
            idx = board.loc_y(loc) * board.size + board.loc_x(loc)
        result[idx] = float(prob)
    total = result.sum()
    return result / total if total > 0 else result


def policy_effect(baseline: Dict, intervened: Dict, board) -> Dict[str, float]:
    """Compare two evaluations of exactly the same board position."""
    p = _policy_vector(baseline, board)
    q = _policy_vector(intervened, board)
    eps = 1e-12
    m = 0.5 * (p + q)
    js = 0.5 * np.sum(p * np.log((p + eps) / (m + eps)))
    js += 0.5 * np.sum(q * np.log((q + eps) / (m + eps)))
    return {
        "policy_js": float(js),
        "policy_l1": float(np.abs(p - q).sum()),
        "top_move_changed": float(int(np.argmax(p) != np.argmax(q))),
        "winrate_delta": float(intervened["value"][0] - baseline["value"][0]),
        "scoremean_delta": float(intervened["scoremean"] - baseline["scoremean"]),
    }


def _choose_move(outputs: Dict, rng: np.random.Generator, temperature: float) -> int:
    moves = outputs["moves_and_probs0"]
    if not moves:
        raise RuntimeError("Model returned no legal moves")
    if temperature <= 0:
        return max(moves, key=lambda item: item[1])[0]
    probs = np.asarray([max(float(item[1]), 1e-30) for item in moves])
    probs = probs ** (1.0 / temperature)
    probs /= probs.sum()
    return moves[int(rng.choice(len(moves), p=probs))][0]


def _area_score(board, komi: float = 7.5) -> float:
    """Approximate Tromp-Taylor final score (black minus white)."""
    from board import Board

    area = [Board.EMPTY] * len(board.board)
    board.calculateArea(area, True, True, True, True)
    black = sum(x == Board.BLACK for x in area)
    white = sum(x == Board.WHITE for x in area)
    return float(black - white - komi)


def play_intervention_game(
    model,
    intervention: ConceptIntervention,
    dose: float,
    board_size: int = 19,
    max_moves: int = 400,
    intervene_player: str = "both",
    seed: int = 0,
    temperature: float = 0.0,
) -> Dict:
    """Play one deterministic game and collect same-position causal effects."""
    from board import Board
    from gamestate import GameState

    gs = GameState(board_size, GameState.RULES_TT)
    effects: List[Dict[str, float]] = []
    moves: List[Tuple[int, int]] = []
    consecutive_passes = 0
    rng = np.random.default_rng(seed)

    for _ in range(max_moves):
        player = gs.board.pla
        applies = intervene_player == "both" or (
            intervene_player == "black" and player == Board.BLACK
        ) or (intervene_player == "white" and player == Board.WHITE)
        baseline = gs.get_model_outputs(model)
        if applies:
            with intervention.apply(model, dose):
                chosen_outputs = gs.get_model_outputs(model)
            effects.append(policy_effect(baseline, chosen_outputs, gs.board))
        else:
            chosen_outputs = baseline

        move = _choose_move(chosen_outputs, rng, temperature)
        gs.play(player, move)
        moves.append((player, move))
        consecutive_passes = consecutive_passes + 1 if move == Board.PASS_LOC else 0
        if consecutive_passes >= 2:
            break

    means = {
        key: float(np.mean([effect[key] for effect in effects])) if effects else 0.0
        for key in ("policy_js", "policy_l1", "top_move_changed", "winrate_delta", "scoremean_delta")
    }
    score = _area_score(gs.board)
    return {
        "dose": float(dose),
        "seed": int(seed),
        "moves": len(moves),
        "area_score_black_minus_white": score,
        "winner": "black" if score > 0 else "white" if score < 0 else "draw",
        "intervened_positions": len(effects),
        **means,
    }


def dose_response(results: Iterable[Dict]) -> Dict[str, Optional[float]]:
    """Summarize monotonic effects across ordered doses."""
    ordered = sorted(results, key=lambda row: row["dose"])
    doses = np.asarray([r["dose"] for r in ordered], dtype=float)
    summary: Dict[str, Optional[float]] = {}
    for metric in ("area_score_black_minus_white", "winrate_delta", "scoremean_delta"):
        values = np.asarray([r[metric] for r in ordered], dtype=float)
        summary[f"{metric}_slope"] = (
            float(np.polyfit(doses, values, 1)[0]) if len(np.unique(doses)) > 1 else None
        )
    summary["causal_readout"] = (
        "A non-zero policy effect plus a consistent signed dose-response supports "
        "a causal role; probe accuracy alone does not. Repeat across openings/seeds."
    )
    return summary


def main() -> None:
    from common_utils import get_device
    from load_model import load_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, nargs="?", help="KataGo PyTorch checkpoint")
    parser.add_argument("concept", nargs="?", help="Pre-mode concept name from concepts.yaml")
    parser.add_argument("--probes-dir", type=Path, default=SCRIPT_DIR / "linear_probes")
    parser.add_argument("--concepts-yaml", type=Path, default=Path(__file__).with_name("concepts.yaml"))
    parser.add_argument(
        "--list-concepts",
        action="store_true",
        help="List feature_mode: pre concepts that have probe artifacts, then exit",
    )
    parser.add_argument("--doses", type=float, nargs="+", default=[-2.0, 0.0, 2.0])
    parser.add_argument("--games-per-dose", type=int, default=1)
    parser.add_argument("--board-size", type=int, default=19)
    parser.add_argument("--max-moves", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Policy sampling temperature; use 0 for argmax play")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--intervene-player", choices=("both", "black", "white"), default="both")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.list_concepts:
        names = list_intervenable_concepts(args.probes_dir, args.concepts_yaml)
        print("\n".join(names) if names else "(none)")
        return
    if args.model is None or args.concept is None:
        parser.error("model and concept are required unless --list-concepts is set")

    device = get_device(args.device)
    model, _, _ = load_model(
        args.model, use_swa=False, device=device, pos_len=args.board_size, verbose=False
    )
    probes_dir = _resolve_probes_dir(args.probes_dir, args.concept)
    intervention = ConceptIntervention.load(args.concept, probes_dir, args.concepts_yaml)
    # Reuse each seed at every dose. This is common-random-number pairing:
    # differences are less likely to be mere sampling noise.
    games = []
    for dose in args.doses:
        for game_idx in range(args.games_per_dose):
            games.append(play_intervention_game(
                model, intervention, dose, args.board_size, args.max_moves,
                args.intervene_player, args.seed + game_idx, args.temperature,
            ))
    report = {
        "concept": args.concept,
        "intervene_player": args.intervene_player,
        "games": games,
        "dose_response": dose_response(games),
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
