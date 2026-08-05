#!/usr/bin/env python3
"""Build a scored position bank and test concept steering on matched positions.

This is the primary causal evaluation.  Each saved position is evaluated twice
without changing the board: normally and with a trunkfinal concept hook.  The
report is stratified by observed concept label and baseline probe-score band.
Full-game outcomes in activation_manipulation.py are only supplementary.
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
sys.path.append(str(REPO_DIR / "python"))

try:  # Script execution and package imports have different module roots.
    from .activation_manipulation import ConceptIntervention, _resolve_probes_dir, policy_effect
except ImportError:  # pragma: no cover - exercised by CLI invocation
    from activation_manipulation import ConceptIntervention, _resolve_probes_dir, policy_effect


def _resolve(path: Path, base: Path = SCRIPT_DIR) -> Path:
    if path.is_absolute() or path.exists():
        return path
    candidate = base / path
    return candidate if candidate.exists() else path


def build_position_bank(
    dataset_path: Path,
    scores_path: Path,
    games_dir: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Create a compact bank with replay keys, scores, labels, and raw values."""
    identity = ["game_id", "move_number", "player", "move_loc", "game_phase"]
    dataset_schema = pd.read_parquet(dataset_path, columns=[]).columns
    # Some parquet engines don't expose schema through columns=[]; read column
    # names only from metadata through a cheap full schema read fallback.
    if len(dataset_schema) == 0:
        import pyarrow.parquet as pq
        dataset_columns = pq.ParquetFile(dataset_path).schema.names
        score_columns = pq.ParquetFile(scores_path).schema.names
    else:
        dataset_columns = list(dataset_schema)
        score_columns = list(pd.read_parquet(scores_path, columns=[]).columns)

    label_cols = [c for c in dataset_columns if c.startswith("label_")]
    raw_cols = [c for c in dataset_columns if c.startswith("rawval_")]
    score_cols = [
        c for c in score_columns
        if c.endswith("_score") or c.endswith("_score_pct") or c.endswith("_prob")
    ]
    left_cols = [c for c in identity + label_cols + raw_cols if c in dataset_columns]
    right_cols = ["game_id", "move_number"] + score_cols
    dataset = pd.read_parquet(dataset_path, columns=left_cols)
    scores = pd.read_parquet(scores_path, columns=right_cols)
    bank = dataset.merge(scores, on=["game_id", "move_number"], how="left", validate="one_to_one")
    bank["game_dir"] = bank["game_id"].map(lambda game: str((games_dir / game).resolve()))
    bank["last_move_loc"] = bank.groupby("game_id", sort=False)["move_loc"].shift(1)
    bank["position_id"] = bank["game_id"] + ":" + bank["move_number"].astype(str)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bank.to_parquet(output_path, index=False)
    return bank


def _load_moves(game_dir: Path) -> List[Dict]:
    path = game_dir / "moves.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Cannot replay position: missing {path}")
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _prob_by_loc(outputs: Dict) -> Dict[int, float]:
    return {int(loc): float(prob) for loc, prob in outputs["moves_and_probs0"]}


def _top_loc(outputs: Dict) -> int:
    return int(max(outputs["moves_and_probs0"], key=lambda pair: pair[1])[0])


def _distance(board, a: Optional[int], b: int) -> Optional[float]:
    if a is None or np.isnan(a) or a == board.PASS_LOC or b == board.PASS_LOC:
        return None
    # Match snorkel_board_positions.is_tenuki exactly: L1/Manhattan distance.
    return float(abs(board.loc_x(int(a)) - board.loc_x(b)) + abs(board.loc_y(int(a)) - board.loc_y(b)))


def _region_for_loc(board, loc: int) -> Optional[str]:
    if loc == board.PASS_LOC:
        return None
    x, y, n = board.loc_x(loc), board.loc_y(loc), board.size
    edge = 6 if n == 19 else max(3, round(n * 6 / 19))
    if x < edge and y < edge: return "corner_tl"
    if x >= n - edge and y < edge: return "corner_tr"
    if x < edge and y >= n - edge: return "corner_bl"
    if x >= n - edge and y >= n - edge: return "corner_br"
    if x < edge: return "side_left"
    if x >= n - edge: return "side_right"
    if y < edge: return "side_top"
    if y >= n - edge: return "side_bottom"
    return "center"


CORNER_REGIONS = ("corner_tl", "corner_tr", "corner_bl", "corner_br")


@lru_cache(maxsize=16)
def _corner_coordinates(size: int, region: str) -> Tuple[Tuple[int, int], ...]:
    coordinates = []
    # Use a tiny coordinate-only adapter through the same exact boundaries.
    edge = 6 if size == 19 else max(3, round(size * 6 / 19))
    for y in range(size):
        for x in range(size):
            if x < edge and y < edge: current = "corner_tl"
            elif x >= size - edge and y < edge: current = "corner_tr"
            elif x < edge and y >= size - edge: current = "corner_bl"
            elif x >= size - edge and y >= size - edge: current = "corner_br"
            else: continue
            if current == region:
                coordinates.append((x, y))
    return tuple(coordinates)


def _corner_counts(board, region: str) -> Tuple[int, int]:
    from board import Board

    black = white = 0
    for x, y in _corner_coordinates(board.size, region):
        stone = board.board[board.loc(x, y)]
        black += int(stone == Board.BLACK)
        white += int(stone == Board.WHITE)
    return black, white


def _eligible_corner_regions(board, concept: str) -> List[str]:
    """Corner regions in which the named move could currently instantiate."""
    from board import Board

    regions = []
    opponent = Board.get_opp(board.pla)
    for region in CORNER_REGIONS:
        black, white = _corner_counts(board, region)
        if concept == "occupy_corner" and black + white == 0:
            regions.append(region)
        elif concept == "approaching_corner" and black + white == 1:
            if (opponent == Board.BLACK and black == 1) or (opponent == Board.WHITE and white == 1):
                regions.append(region)
    return regions


def add_corner_eligibility(bank: pd.DataFrame) -> pd.DataFrame:
    """Replay games cheaply and mark whether each corner concept is possible."""
    from board import Board

    result = bank.copy()
    occupy = pd.Series(False, index=result.index)
    approach = pd.Series(False, index=result.index)
    for _game_id, positions in result.groupby("game_id", sort=False):
        board = Board(19)
        for row in positions.sort_values("move_number").itertuples():
            counts = {region: _corner_counts(board, region) for region in CORNER_REGIONS}
            opponent = Board.get_opp(board.pla)
            occupy.at[row.Index] = any(black + white == 0 for black, white in counts.values())
            approach.at[row.Index] = any(
                black + white == 1 and (
                    (opponent == Board.BLACK and black == 1)
                    or (opponent == Board.WHITE and white == 1)
                )
                for black, white in counts.values()
            )
            pla = Board.BLACK if row.player == "b" else Board.WHITE
            board.play(pla, int(row.move_loc))
    result["occupy_corner_eligible"] = occupy
    result["approaching_corner_eligible"] = approach
    return result


def _policy_region_masses(outputs: Dict, board) -> Dict[str, float]:
    masses = {region: 0.0 for region in (
        "corner_tl", "corner_tr", "corner_bl", "corner_br",
        "side_left", "side_right", "side_top", "side_bottom", "center",
    )}
    for loc, probability in outputs["moves_and_probs0"]:
        region = _region_for_loc(board, int(loc))
        if region is not None:
            masses[region] += float(probability)
    return masses


def _contrast_mask(selected: np.ndarray) -> np.ndarray:
    """Positive on selected points, negative elsewhere, zero mean and unit RMS."""
    selected = selected.astype(bool)
    if not selected.any() or selected.all():
        raise ValueError("Contrast mask needs non-empty selected and comparison regions")
    mask = selected.astype(np.float32)
    mask[~selected] = -float(selected.sum()) / float((~selected).sum())
    return mask / float(np.sqrt(np.mean(mask ** 2)))


def concept_local_mask(
    concept: str,
    baseline: Dict,
    board,
    row: Dict,
    local_radius: int = 4,
    far_distance: int = 6,
) -> np.ndarray:
    """Concept-specific spatial mask based only on the baseline position."""
    if concept == "tenuki":
        return spatial_intervention_mask(
            board, row.get("last_move_loc"), "local-contrast", local_radius, far_distance
        )
    selected = np.zeros((board.size, board.size), dtype=bool)
    if concept == "forcing":
        top = _top_loc(baseline)
        if top == board.PASS_LOC:
            raise ValueError("Cannot spatially anchor forcing when the baseline top move is pass")
        selected[board.loc_y(top), board.loc_x(top)] = True
    elif concept == "urgency_peak":
        masses = _policy_region_masses(baseline, board)
        top_region = max(masses, key=masses.get)
        for y in range(board.size):
            for x in range(board.size):
                selected[y, x] = _region_for_loc(board, board.loc(x, y)) == top_region
    else:
        raise ValueError(f"No concept-local mask is registered for {concept!r}")
    return _contrast_mask(selected)


def spatial_intervention_mask(
    board,
    last_move_loc: int,
    mode: str,
    local_radius: int,
    far_distance: int,
    control: str = "aligned",
    seed: int = 0,
) -> np.ndarray:
    """Construct an RMS-normalized spatial mask around the previous move."""
    if last_move_loc is None or pd.isna(last_move_loc) or int(last_move_loc) == board.PASS_LOC:
        raise ValueError("Spatial tenuki intervention requires a non-pass previous move")
    last_move_loc = int(last_move_loc)
    lx, ly = board.loc_x(last_move_loc), board.loc_y(last_move_loc)
    yy, xx = np.mgrid[:board.size, :board.size]
    # Match snorkel_board_positions.is_tenuki: Manhattan/L1 distance.
    distance = np.abs(xx - lx) + np.abs(yy - ly)
    far = distance >= far_distance
    near = distance <= local_radius
    if mode == "local-far":
        mask = far.astype(np.float32)
    elif mode == "local-contrast":
        mask = far.astype(np.float32)
        # Equal total positive and negative mass makes this spatially
        # zero-mean even when the two regions have different areas.
        if near.any() and far.any():
            mask[near] = -float(far.sum()) / float(near.sum())
    else:
        raise ValueError(f"No spatial mask for intervention mode {mode!r}")

    if control == "inverted":
        mask = -mask
    elif control == "shuffled":
        rng = np.random.default_rng(seed)
        mask = rng.permutation(mask.ravel()).reshape(mask.shape)
    elif control != "aligned":
        raise ValueError(f"Unknown mask control: {control}")
    rms = float(np.sqrt(np.mean(mask ** 2)))
    if rms <= 0:
        raise ValueError("Spatial intervention mask is empty")
    return mask / rms


def concept_metrics(concept: str, outputs: Dict, board, row: Dict) -> Dict[str, float]:
    """Behavioral readouts. Always includes probability of the recorded move."""
    probs = _prob_by_loc(outputs)
    actual = int(row["move_loc"])
    metrics = {"recorded_move_probability": probs.get(actual, 0.0)}

    if concept == "tenuki":
        last = row.get("last_move_loc")
        weighted = [(p, _distance(board, last, loc)) for loc, p in probs.items()]
        valid = [(p, d) for p, d in weighted if d is not None]
        metrics["tenuki_expected_distance"] = float(sum(p * d for p, d in valid))
        metrics["tenuki_far_policy_mass"] = float(sum(p for p, d in valid if d >= 6))
    elif concept == "forcing":
        ps = np.sort(np.asarray(list(probs.values()), dtype=float))[::-1]
        metrics["forcing_top_policy_mass"] = float(ps.max())
        metrics["forcing_policy_entropy"] = float(-np.sum(ps * np.log(ps + 1e-12)))
        metrics["forcing_top_margin"] = float(ps[0] - ps[1]) if len(ps) > 1 else float(ps[0])
        metrics["forcing_threshold_crossed"] = float(ps[0] > 0.95)
    elif concept in {"occupy_corner", "approaching_corner"}:
        eligible = set(_eligible_corner_regions(board, concept))
        metrics["eligible_corner_policy_mass"] = float(sum(
            p for loc, p in probs.items() if _region_for_loc(board, loc) in eligible
        ))
        metrics["top_move_in_eligible_corner"] = float(_region_for_loc(board, _top_loc(outputs)) in eligible)
    elif concept == "urgency_peak":
        masses = np.sort(np.asarray(list(_policy_region_masses(outputs, board).values())))[::-1]
        normalized = masses / max(float(masses.sum()), 1e-12)
        metrics["urgency_peak_region_mass"] = float(masses[0])
        metrics["urgency_region_margin"] = float(masses[0] - masses[1])
        metrics["urgency_region_entropy"] = float(-np.sum(normalized * np.log(normalized + 1e-12)))
    return metrics


def _position_stratum(row: pd.Series, concept: str) -> str:
    label_col = f"label_{concept}"
    if label_col in row and pd.notna(row[label_col]):
        return "label_positive" if int(row[label_col]) == 1 else "label_negative"
    pct_col = f"{concept}_score_pct"
    if pct_col in row and pd.notna(row[pct_col]):
        if row[pct_col] >= 0.8: return "score_high"
        if row[pct_col] <= 0.2: return "score_low"
    return "score_middle"


def select_positions(bank: pd.DataFrame, concept: str, max_positions: int, seed: int) -> pd.DataFrame:
    if f"{concept}_score" not in bank:
        raise ValueError(f"Position bank has no score for concept {concept!r}")
    selected = bank.copy()
    eligibility_col = f"{concept}_eligible"
    if eligibility_col in selected:
        selected = selected[selected[eligibility_col]].copy()
        if selected.empty:
            raise ValueError(f"No eligible positions found for {concept!r}")
    selected["stratum"] = selected.apply(_position_stratum, axis=1, concept=concept)
    # Equal allocation makes rare positives visible rather than drowned out.
    strata = sorted(selected["stratum"].unique())
    each = max(1, max_positions // len(strata))
    pieces = [
        group.sample(n=min(each, len(group)), random_state=seed)
        for _, group in selected.groupby("stratum", sort=True)
    ]
    result = pd.concat(pieces).sort_values(["game_id", "move_number"])
    if len(result) < max_positions:
        remaining = selected.drop(result.index)
        if len(remaining):
            result = pd.concat([result, remaining.sample(
                n=min(max_positions - len(result), len(remaining)), random_state=seed + 1
            )])
    return result.sort_values(["game_id", "move_number"])


def evaluate_bank(
    model,
    intervention: ConceptIntervention,
    bank: pd.DataFrame,
    concept: str,
    doses: Iterable[float],
    intervention_mode: str = "global",
    local_radius: int = 4,
    far_distance: int = 6,
    mask_control: str = "aligned",
    seed: int = 0,
    direction_controls: Optional[List[Tuple[str, Optional[np.ndarray]]]] = None,
) -> Tuple[List[Dict], Dict]:
    from board import Board
    from gamestate import GameState

    rows: List[Dict] = []
    for game_id, positions in bank.groupby("game_id", sort=False):
        moves = _load_moves(Path(positions.iloc[0]["game_dir"]))
        by_number = {int(row.move_number): row for row in positions.itertuples()}
        gs = GameState(model.pos_len, GameState.RULES_TT)
        for move_data in moves:
            move_number = int(move_data["move_number"])
            if move_number in by_number:
                row = by_number[move_number]._asdict()
                baseline = gs.get_model_outputs(model)
                base_metrics = concept_metrics(concept, baseline, gs.board, row)
                baseline_top = _top_loc(baseline)
                spatial_mask = None
                component = "global"
                position_evaluable = True
                if intervention_mode != "global":
                    component = "local"
                    # Stable per-position randomization for shuffled controls.
                    position_seed = seed + sum(ord(ch) for ch in row["position_id"])
                    if intervention_mode == "concept-local":
                        try:
                            spatial_mask = concept_local_mask(
                                concept, baseline, gs.board, row, local_radius, far_distance
                            )
                        except ValueError:
                            # For example, a forcing position whose top policy
                            # action is pass has no board point for a local mask.
                            position_evaluable = False
                        if mask_control == "inverted":
                            spatial_mask = -spatial_mask
                        elif mask_control == "shuffled":
                            rng = np.random.default_rng(position_seed)
                            spatial_mask = rng.permutation(spatial_mask.ravel()).reshape(spatial_mask.shape)
                    else:
                        spatial_mask = spatial_intervention_mask(
                            gs.board, row.get("last_move_loc"), intervention_mode,
                            local_radius, far_distance, mask_control, position_seed,
                        )
                controls = direction_controls or [("trained", None)]
                for direction_id, direction_override in controls if position_evaluable else []:
                    for dose in doses:
                        with intervention.apply(
                            model, float(dose), component=component, spatial_mask=spatial_mask,
                            direction_override=direction_override,
                        ):
                            steered = gs.get_model_outputs(model)
                        steered_metrics = concept_metrics(concept, steered, gs.board, row)
                        effect = policy_effect(baseline, steered, gs.board)
                        record = {
                            "position_id": row["position_id"], "game_id": game_id,
                            "move_number": move_number, "stratum": row["stratum"],
                            "label": row.get(f"label_{concept}"),
                            "probe_score": row.get(f"{concept}_score"),
                            "direction_id": direction_id,
                            "dose": float(dose),
                            "baseline_top_move": baseline_top,
                            "steered_top_move": _top_loc(steered),
                            "top_move_flip": int(baseline_top != _top_loc(steered)),
                            **effect,
                        }
                        for name, value in base_metrics.items():
                            record[f"baseline_{name}"] = value
                            record[f"steered_{name}"] = steered_metrics[name]
                            record[f"delta_{name}"] = steered_metrics[name] - value
                        rows.append(record)
            pla = Board.BLACK if move_data["player"] == "b" else Board.WHITE
            gs.play(pla, int(move_data["move_loc"]))
            if move_number >= max(by_number):
                break

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No evaluable positions remained after concept-specific masking")
    delta_metrics = [c for c in frame if c.startswith("delta_")]
    summary_rows = []
    for (direction_id, dose, stratum), group in frame.groupby(
        ["direction_id", "dose", "stratum"], dropna=False
    ):
        item = {
            "direction_id": direction_id, "dose": float(dose),
            "stratum": stratum, "n": len(group),
            "top_move_flip_rate": float(group["top_move_flip"].mean()),
            "mean_policy_js": float(group["policy_js"].mean()),
        }
        item.update({f"mean_{metric}": float(group[metric].mean()) for metric in delta_metrics})
        summary_rows.append(item)
    primary = {
        "tenuki": ["tenuki_expected_distance", "tenuki_far_policy_mass"],
        "forcing": ["forcing_top_policy_mass", "forcing_top_margin", "forcing_policy_entropy"],
        "occupy_corner": ["eligible_corner_policy_mass", "top_move_in_eligible_corner"],
        "approaching_corner": ["eligible_corner_policy_mass", "top_move_in_eligible_corner"],
        "urgency_peak": ["urgency_peak_region_mass", "urgency_region_margin", "urgency_region_entropy"],
    }.get(concept, ["recorded_move_probability"])
    summary = {
        "concept": concept,
        "positions": int(frame["position_id"].nunique()),
        "positions_requested": int(bank.shape[0]),
        "intervention_mode": intervention_mode,
        "mask_control": mask_control,
        "direction_controls": [name for name, _ in (direction_controls or [("trained", None)])],
        "local_radius": local_radius if (
            intervention_mode in {"local-far", "local-contrast"}
            or (intervention_mode == "concept-local" and concept == "tenuki")
        ) else None,
        "far_distance": far_distance if (
            intervention_mode in {"local-far", "local-contrast"}
            or (intervention_mode == "concept-local" and concept == "tenuki")
        ) else None,
        "primary_behavior_metrics": primary,
        "fallback_metric_note": (
            None if concept in {"tenuki", "forcing", "occupy_corner", "approaching_corner", "urgency_peak"}
            else "No bespoke metric is registered; recorded_move_probability measures mass on the observed concept-instantiating move."
        ),
        "by_dose_and_stratum": summary_rows,
        "interpretation": (
            "Steering is supported when signed doses change the named primary behavior metric "
            "and outperform spatial controls. Compare observed-positive and observed-negative "
            "strata as reinforcement and induction opportunities; policy change alone is not "
            "concept-specific evidence."
        ),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Build compact position bank")
    build.add_argument("--dataset", type=Path, default=SCRIPT_DIR / "linear_probes/dataset.parquet")
    build.add_argument("--scores", type=Path, default=SCRIPT_DIR / "linear_probes/move_concepts.parquet")
    build.add_argument("--games-dir", type=Path, default=REPO_DIR / "games")
    build.add_argument("--output", type=Path, default=SCRIPT_DIR / "linear_probes/position_bank.parquet")

    evaluate = sub.add_parser("evaluate", help="Run baseline versus hooked evaluations")
    evaluate.add_argument("model", type=Path)
    evaluate.add_argument("concept")
    evaluate.add_argument("--bank", type=Path, default=SCRIPT_DIR / "linear_probes/position_bank.parquet")
    evaluate.add_argument("--probes-dir", type=Path, default=SCRIPT_DIR / "linear_probes")
    evaluate.add_argument("--concepts-yaml", type=Path, default=SCRIPT_DIR / "concepts.yaml")
    evaluate.add_argument("--doses", type=float, nargs="+", default=[-2, -1, 0, 1, 2])
    evaluate.add_argument("--max-positions", type=int, default=200)
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument(
        "--intervention-mode", choices=("global", "local-far", "local-contrast", "concept-local"),
        default="global",
    )
    evaluate.add_argument("--local-radius", type=int, default=4)
    evaluate.add_argument("--far-distance", type=int, default=6)
    evaluate.add_argument(
        "--mask-control", choices=("aligned", "shuffled", "inverted"), default="aligned",
        help="Spatial control; ignored for global interventions",
    )
    evaluate.add_argument(
        "--direction-control", choices=("trained", "random", "other-concept"),
        default="trained",
        help="Channel direction while holding the spatial mask and activation norm fixed",
    )
    evaluate.add_argument(
        "--random-directions", type=int, default=10,
        help="Number of seeded norm-matched directions for --direction-control random",
    )
    evaluate.add_argument(
        "--control-concept",
        help="Concept whose local direction to use for --direction-control other-concept",
    )
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.command == "build":
        bank = build_position_bank(_resolve(args.dataset), _resolve(args.scores),
                                   _resolve(args.games_dir, REPO_DIR), args.output)
        print(f"Wrote {len(bank):,} replayable scored positions to {args.output}")
        return

    from common_utils import get_device
    from load_model import load_model
    bank_path = _resolve(args.bank)
    bank = pd.read_parquet(bank_path)
    if args.concept in {"occupy_corner", "approaching_corner"}:
        bank = add_corner_eligibility(bank)
    if args.intervention_mode in {"local-far", "local-contrast"} or (
        args.intervention_mode == "concept-local" and args.concept == "tenuki"
    ):
        bank = bank[bank["last_move_loc"].notna()].copy()
        # Passing provides no spatial anchor for a previous-move mask.
        bank = bank[bank["last_move_loc"] != 0].copy()  # Board.PASS_LOC
    if args.intervention_mode == "concept-local" and args.concept in {
        "occupy_corner", "approaching_corner"
    }:
        raise ValueError(
            f"{args.concept} was trained without move-location features; use --intervention-mode global"
        )
    selected = select_positions(bank, args.concept, args.max_positions, args.seed)
    device = get_device(args.device)
    model, _, _ = load_model(args.model, use_swa=False, device=device, pos_len=19, verbose=False)
    intervention = ConceptIntervention.load(
        args.concept, _resolve_probes_dir(args.probes_dir, args.concept), args.concepts_yaml
    )
    direction_controls: List[Tuple[str, Optional[np.ndarray]]] = [("trained", None)]
    if args.direction_control != "trained":
        if args.intervention_mode == "global":
            target_direction = intervention.channel_delta_per_unit
        else:
            target_direction = intervention.local_channel_delta_per_unit
        if target_direction is None:
            raise ValueError(f"Concept {args.concept!r} has no local direction to control")
        target_norm = float(np.linalg.norm(target_direction))
        if args.direction_control == "random":
            if args.random_directions <= 0:
                raise ValueError("--random-directions must be positive")
            direction_controls = []
            for idx in range(args.random_directions):
                rng = np.random.default_rng(args.seed + idx)
                random_direction = rng.normal(size=target_direction.shape).astype(np.float32)
                random_direction *= target_norm / float(np.linalg.norm(random_direction))
                direction_controls.append((f"random_{idx:02d}", random_direction))
        else:
            if not args.control_concept:
                raise ValueError("--control-concept is required for other-concept control")
            control = ConceptIntervention.load(
                args.control_concept,
                _resolve_probes_dir(args.probes_dir, args.control_concept),
                args.concepts_yaml,
            )
            source = (
                control.channel_delta_per_unit if args.intervention_mode == "global"
                else control.local_channel_delta_per_unit
            )
            if source is None:
                raise ValueError(f"Control concept {args.control_concept!r} has no local direction")
            matched = source * (target_norm / float(np.linalg.norm(source)))
            direction_controls = [(f"concept_{args.control_concept}", matched)]
    rows, summary = evaluate_bank(
        model, intervention, selected, args.concept, args.doses,
        args.intervention_mode, args.local_radius, args.far_distance,
        args.mask_control, args.seed, direction_controls,
    )
    output = args.output or Path(f"causal_positions_{args.concept}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "positions": rows}, indent=2, default=_json_default) + "\n")
    print(json.dumps(summary, indent=2, default=_json_default))
    print(f"Detailed per-position results: {output}")


def _json_default(value):
    if isinstance(value, np.generic): return value.item()
    if pd.isna(value): return None
    raise TypeError(type(value).__name__)


if __name__ == "__main__":
    main()
