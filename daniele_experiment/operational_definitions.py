"""Canonical operational definitions for policy-derived Go concepts.

This module deliberately has no model dependency.  Callers provide board states,
recorded actions, and model-output policies as internal KataGo ``Board``
locations.  Replay labeling and causal evaluation should both call these same
functions so that the label predicate and behavioral readout cannot drift apart.

The public contract IDs are versioned.  Persist ``contract.definition_id`` and
``contract.contract_hash`` in every derived artifact; aliases are only a
migration convenience and must not be written as provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from numbers import Integral
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np


REGIONS: Tuple[str, ...] = (
    "corner_tl",
    "corner_tr",
    "corner_bl",
    "corner_br",
    "side_left",
    "side_right",
    "side_top",
    "side_bottom",
    "center",
)


class PolicyTime(str, Enum):
    """State at which a policy was evaluated."""

    PRE_MOVE = "pre_move"
    POST_MOVE_REPLY = "post_move_reply"


class PolicySupport(str, Enum):
    """Action support over which probabilities are normalized."""

    LEGAL_PLUS_PASS = "legal_plus_pass"
    LEGAL_BOARD_CONDITIONAL = "legal_board_conditional"


class LabelScope(str, Enum):
    """Whether a label describes the observed action or the whole position."""

    SELECTED_ACTION = "selected_action"
    POSITION = "position"


class OperationalDefinitionError(ValueError):
    """Base error for an invalid operational-definition input."""


class IneligiblePositionError(OperationalDefinitionError):
    """Raised when a requested readout has no well-defined anchor."""


class MissingCandidateScoresError(OperationalDefinitionError):
    """Raised when exact candidate-level causal evaluation is impossible."""


@dataclass(frozen=True)
class PolicyView:
    """A normalized policy keyed by KataGo internal ``Board`` locations.

    ``probabilities`` must already have the support stated by ``support`` and
    sum to one.  Use :func:`legal_policy_view` rather than constructing this
    class directly at data-ingestion boundaries.
    """

    probabilities: Mapping[int, float]
    player_to_move: int
    time: PolicyTime
    support: PolicySupport
    pass_loc: int

    def __post_init__(self) -> None:
        cleaned: Dict[int, float] = {}
        for raw_loc, raw_probability in self.probabilities.items():
            if not isinstance(raw_loc, Integral):
                raise TypeError("Policy locations must be internal integer Board locations")
            loc = int(raw_loc)
            probability = float(raw_probability)
            if not math.isfinite(probability) or probability < 0.0:
                raise OperationalDefinitionError(
                    f"Invalid policy probability at location {loc}: {raw_probability!r}"
                )
            cleaned[loc] = probability

        total = float(sum(cleaned.values()))
        if not cleaned or not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise OperationalDefinitionError(
                f"PolicyView must be normalized to one; observed total={total:.12g}"
            )
        if self.support == PolicySupport.LEGAL_BOARD_CONDITIONAL and self.pass_loc in cleaned:
            raise OperationalDefinitionError(
                "LEGAL_BOARD_CONDITIONAL policy must not contain the pass action"
            )

        object.__setattr__(self, "probabilities", MappingProxyType(cleaned))
        object.__setattr__(self, "time", PolicyTime(self.time))
        object.__setattr__(self, "support", PolicySupport(self.support))

    def probability(self, loc: int) -> float:
        return float(self.probabilities.get(int(loc), 0.0))

    @property
    def pass_probability(self) -> float:
        return self.probability(self.pass_loc)


@dataclass(frozen=True)
class TransitionContext:
    """One recorded transition with explicit pre- and post-action semantics.

    ``previous_move`` is the most recent non-pass board action.  Passes do not
    replace this spatial anchor for ``tenuki_distance6``.
    """

    board_before: Any
    player: int
    previous_move: Optional[int]
    selected_move: int
    # The first recorded move has no persisted pre-move policy in the legacy
    # game format.  Keeping this optional lets action-effect labels such as
    # reply_peak95 remain usable without fabricating an initial policy.
    pre_policy: Optional[PolicyView]
    board_after: Any = None
    reply_policy: Optional[PolicyView] = None


@dataclass(frozen=True)
class OperationalContract:
    """Versioned metadata and canonical callables for an operational variable."""

    name: str
    version: int
    human_concept: str
    representation_time: PolicyTime
    label_scope: LabelScope
    label_policy_time: Optional[PolicyTime]
    label_policy_support: Optional[PolicySupport]
    readout_policy_time: PolicyTime
    readout_policy_support: PolicySupport
    parameters: Mapping[str, Any]
    raw_value: Callable[..., Optional[float]] = field(repr=False, compare=False)
    observed_label: Callable[..., Optional[int]] = field(repr=False, compare=False)
    policy_readouts: Callable[..., Mapping[str, Any]] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "representation_time", PolicyTime(self.representation_time))
        object.__setattr__(self, "label_scope", LabelScope(self.label_scope))
        object.__setattr__(
            self,
            "label_policy_time",
            None
            if self.label_policy_time is None
            else PolicyTime(self.label_policy_time),
        )
        object.__setattr__(
            self,
            "label_policy_support",
            None
            if self.label_policy_support is None
            else PolicySupport(self.label_policy_support),
        )
        object.__setattr__(self, "readout_policy_time", PolicyTime(self.readout_policy_time))
        object.__setattr__(
            self, "readout_policy_support", PolicySupport(self.readout_policy_support)
        )
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))

    @property
    def definition_id(self) -> str:
        return f"{self.name}@{self.version}"

    def metadata(self) -> Dict[str, Any]:
        return {
            "definition_id": self.definition_id,
            "human_concept": self.human_concept,
            "representation_time": self.representation_time.value,
            "label_scope": self.label_scope.value,
            "label_policy_time": (
                None if self.label_policy_time is None else self.label_policy_time.value
            ),
            "label_policy_support": (
                None
                if self.label_policy_support is None
                else self.label_policy_support.value
            ),
            "readout_policy_time": self.readout_policy_time.value,
            "readout_policy_support": self.readout_policy_support.value,
            "parameters": dict(self.parameters),
        }

    @property
    def contract_hash(self) -> str:
        encoded = json.dumps(
            self.metadata(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _location_probability_pairs(source: Any) -> Iterable[Tuple[int, float]]:
    """Extract internal-location/probability pairs without guessing coordinates."""

    if isinstance(source, Mapping):
        if "moves_and_probs0" in source:
            return source["moves_and_probs0"]
        if "policy0" in source:
            raise OperationalDefinitionError(
                "Dense policy0 uses tensor indices, not internal Board locations. "
                "Pass moves_and_probs0 or explicitly convert indices before this boundary."
            )
        if all(isinstance(loc, Integral) for loc in source):
            return source.items()
        raise TypeError(
            "Policy mapping must be keyed by internal Board locations or contain moves_and_probs0"
        )

    if isinstance(source, np.ndarray) and source.ndim == 1:
        raise OperationalDefinitionError(
            "A dense policy vector is coordinate-ambiguous; provide internal Board location pairs"
        )
    return source


def legal_policy_view(
    source: Any,
    board: Any,
    *,
    time: PolicyTime,
    support: PolicySupport,
    player_to_move: Optional[int] = None,
    coordinate_system: str,
) -> PolicyView:
    """Filter and normalize a policy over legal actions.

    Parameters
    ----------
    source:
        An iterable of ``(internal_board_loc, probability)`` pairs, a mapping
        keyed by internal locations, or a model-output mapping containing
        ``moves_and_probs0``.  Bare dense ``policy0`` arrays are rejected.
    coordinate_system:
        Must be the literal ``"board_loc"``.  Requiring this declaration keeps
        flat 0..360 tensor indices from silently entering the padded-board API.
    support:
        ``LEGAL_PLUS_PASS`` retains pass. ``LEGAL_BOARD_CONDITIONAL`` removes
        pass and renormalizes conditional on selecting a board point.
    """

    if coordinate_system != "board_loc":
        raise OperationalDefinitionError(
            "Policies must use KataGo internal Board locations; expected coordinate_system='board_loc'"
        )
    time = PolicyTime(time)
    support = PolicySupport(support)
    player = int(board.pla if player_to_move is None else player_to_move)
    pass_loc = int(board.PASS_LOC)

    filtered: Dict[int, float] = {}
    try:
        pairs = _location_probability_pairs(source)
        for entry in pairs:
            try:
                raw_loc, raw_probability = entry
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "Policy source must contain (internal_board_loc, probability) pairs"
                ) from exc
            if not isinstance(raw_loc, Integral):
                raise TypeError("Policy locations must be internal integer Board locations")
            loc = int(raw_loc)
            if loc in filtered:
                raise OperationalDefinitionError(f"Duplicate policy location: {loc}")
            probability = float(raw_probability)
            if not math.isfinite(probability) or probability < 0.0:
                raise OperationalDefinitionError(
                    f"Invalid policy probability at location {loc}: {raw_probability!r}"
                )

            if loc == pass_loc:
                if support == PolicySupport.LEGAL_PLUS_PASS:
                    filtered[loc] = probability
                continue
            if board.would_be_legal(player, loc):
                filtered[loc] = probability
    except OperationalDefinitionError:
        raise
    except TypeError:
        raise

    total = float(sum(filtered.values()))
    if total <= 0.0:
        raise OperationalDefinitionError(
            f"No positive legal policy mass remains for support={support.value}"
        )
    normalized = {loc: probability / total for loc, probability in filtered.items()}
    return PolicyView(normalized, player, time, support, pass_loc)


def _require_policy(
    policy: PolicyView,
    *,
    time: Optional[PolicyTime] = None,
    support: Optional[PolicySupport] = None,
) -> None:
    if time is not None and policy.time != PolicyTime(time):
        raise OperationalDefinitionError(
            f"Expected {PolicyTime(time).value} policy, got {policy.time.value}"
        )
    if support is not None and policy.support != PolicySupport(support):
        raise OperationalDefinitionError(
            f"Expected {PolicySupport(support).value} support, got {policy.support.value}"
        )


def _region_edge(size: int) -> int:
    if size <= 1:
        raise OperationalDefinitionError(f"Board size must exceed one, got {size}")
    if size == 19:
        return 6
    return max(1, min((size - 1) // 2, round(size * 6 / 19)))


def classify_region_xy(x: int, y: int, size: int = 19) -> str:
    """Classify an on-board coordinate into the canonical nine-region partition."""

    x, y, size = int(x), int(y), int(size)
    if not (0 <= x < size and 0 <= y < size):
        raise OperationalDefinitionError(f"Coordinate ({x}, {y}) is outside a {size}x{size} board")
    edge = _region_edge(size)
    high = size - edge
    if x < edge and y < edge:
        return "corner_tl"
    if x >= high and y < edge:
        return "corner_tr"
    if x < edge and y >= high:
        return "corner_bl"
    if x >= high and y >= high:
        return "corner_br"
    if x < edge:
        return "side_left"
    if x >= high:
        return "side_right"
    if y < edge:
        return "side_top"
    if y >= high:
        return "side_bottom"
    return "center"


def region_for_loc(board: Any, loc: int) -> Optional[str]:
    """Return the canonical region for an internal Board location; pass maps to ``None``."""

    loc = int(loc)
    if loc == int(board.PASS_LOC):
        return None
    if not board.is_on_board(loc):
        raise OperationalDefinitionError(f"Location {loc} is not an on-board internal location")
    return classify_region_xy(board.loc_x(loc), board.loc_y(loc), board.size)


def region_masses(policy: PolicyView, board: Any) -> Dict[str, float]:
    """Aggregate a board-conditional policy using the canonical region partition."""

    _require_policy(policy, support=PolicySupport.LEGAL_BOARD_CONDITIONAL)
    masses = {region: 0.0 for region in REGIONS}
    for loc, probability in policy.probabilities.items():
        region = region_for_loc(board, loc)
        if region is None:  # Guarded by support, retained as a defensive check.
            raise OperationalDefinitionError("Board-conditional policy unexpectedly contains pass")
        masses[region] += float(probability)
    return masses


def legal_board_mask(board: Any, player: Optional[int] = None) -> np.ndarray:
    """Boolean ``[y, x]`` mask of legal non-pass actions."""

    player = int(board.pla if player is None else player)
    mask = np.zeros((board.size, board.size), dtype=bool)
    for y in range(board.size):
        for x in range(board.size):
            loc = board.loc(x, y)
            mask[y, x] = bool(board.would_be_legal(player, loc))
    return mask


def rms_contrast_mask(target: np.ndarray, comparison: np.ndarray) -> np.ndarray:
    """Return a zero-mean, unit-RMS contrast on its active legal support.

    The active support is exactly ``target | comparison``.  Its target entries
    are positive, its comparison entries are negative, and values on that
    support have mean zero and root-mean-square one.  Points in neither set
    remain exactly zero; in particular, occupied and otherwise illegal points
    stay outside a semantic action-set intervention and do not dilute the RMS.
    """

    target = np.asarray(target, dtype=bool)
    comparison = np.asarray(comparison, dtype=bool)
    if target.shape != comparison.shape or target.ndim != 2:
        raise OperationalDefinitionError("Target and comparison must be same-shaped 2D masks")
    if np.any(target & comparison):
        raise OperationalDefinitionError("Target and comparison masks must be disjoint")
    target_count = int(target.sum())
    comparison_count = int(comparison.sum())
    if target_count == 0 or comparison_count == 0:
        raise IneligiblePositionError("Contrast requires non-empty target and comparison action sets")

    mask = np.zeros(target.shape, dtype=np.float32)
    mask[target] = 1.0
    mask[comparison] = -float(target_count) / float(comparison_count)
    active = target | comparison
    rms = float(np.sqrt(np.mean(np.square(mask[active], dtype=np.float64))))
    if rms <= 0.0:
        raise IneligiblePositionError("Contrast mask has zero RMS")
    return mask / rms


def _missing_loc(loc: Optional[int]) -> bool:
    if loc is None:
        return True
    try:
        return bool(math.isnan(float(loc)))
    except (TypeError, ValueError):
        return False


def manhattan_distance(board: Any, first: Optional[int], second: Optional[int]) -> Optional[int]:
    """Manhattan distance between non-pass internal locations, else ``None``."""

    if _missing_loc(first) or _missing_loc(second):
        return None
    first, second = int(first), int(second)
    if first == int(board.PASS_LOC) or second == int(board.PASS_LOC):
        return None
    if not board.is_on_board(first) or not board.is_on_board(second):
        raise OperationalDefinitionError("Manhattan distance requires on-board internal locations")
    return abs(board.loc_x(first) - board.loc_x(second)) + abs(
        board.loc_y(first) - board.loc_y(second)
    )


def tenuki_action_predicate(
    board: Any,
    previous_move: Optional[int],
    candidate_move: Optional[int],
    *,
    min_distance: int = 6,
) -> bool:
    """The ``tenuki_distance6`` action predicate: Manhattan distance at least six."""

    distance = manhattan_distance(board, previous_move, candidate_move)
    return distance is not None and distance >= int(min_distance)


def tenuki_observed_value(
    context: TransitionContext, *, min_distance: int = 6
) -> Optional[float]:
    """Observed selected-move distance, or ``None`` when the proxy is ineligible."""

    del min_distance  # Part of the uniform callable signature; raw value is unthresholded.
    distance = manhattan_distance(
        context.board_before, context.previous_move, context.selected_move
    )
    return None if distance is None else float(distance)


def tenuki_observed_label(
    context: TransitionContext, *, min_distance: int = 6
) -> Optional[int]:
    value = tenuki_observed_value(context)
    return None if value is None else int(value >= int(min_distance))


def tenuki_target_mask(
    board: Any,
    previous_move: Optional[int],
    *,
    player: Optional[int] = None,
    min_distance: int = 6,
) -> np.ndarray:
    """Legal spatial actions satisfying the exact tenuki label predicate."""

    if _missing_loc(previous_move) or int(previous_move) == int(board.PASS_LOC):
        raise IneligiblePositionError("Tenuki requires a previous non-pass move")
    legal = legal_board_mask(board, player)
    target = np.zeros_like(legal)
    for y in range(board.size):
        for x in range(board.size):
            if legal[y, x]:
                target[y, x] = tenuki_action_predicate(
                    board, previous_move, board.loc(x, y), min_distance=min_distance
                )
    return target


def tenuki_contrast_mask(
    board: Any,
    previous_move: Optional[int],
    *,
    player: Optional[int] = None,
    min_distance: int = 6,
) -> np.ndarray:
    """Canonical aligned mask: target tenuki actions versus all other legal board actions."""

    legal = legal_board_mask(board, player)
    target = tenuki_target_mask(
        board, previous_move, player=player, min_distance=min_distance
    )
    return rms_contrast_mask(target, legal & ~target)


def tenuki_policy_readouts(
    policy: PolicyView,
    board: Any,
    previous_move: Optional[int],
    *,
    min_distance: int = 6,
) -> Mapping[str, float]:
    """Policy mass and expected distance using the exact tenuki action predicate."""

    _require_policy(
        policy,
        time=PolicyTime.PRE_MOVE,
        support=PolicySupport.LEGAL_BOARD_CONDITIONAL,
    )
    if _missing_loc(previous_move) or int(previous_move) == int(board.PASS_LOC):
        raise IneligiblePositionError("Tenuki readouts require a previous non-pass move")

    target_mass = 0.0
    expected_distance = 0.0
    for loc, probability in policy.probabilities.items():
        distance = manhattan_distance(board, previous_move, loc)
        if distance is None:
            raise OperationalDefinitionError("Board policy unexpectedly contains pass")
        expected_distance += float(probability) * distance
        if distance >= int(min_distance):
            target_mass += float(probability)
    return {
        "tenuki_distance6_policy_mass": float(target_mass),
        "tenuki_distance6_complement_mass": float(1.0 - target_mass),
        "tenuki_expected_manhattan_distance": float(expected_distance),
    }


def reply_peak(reply_policy: PolicyView) -> float:
    """Maximum probability in the opponent's normalized legal reply policy."""

    _require_policy(
        reply_policy,
        time=PolicyTime.POST_MOVE_REPLY,
        support=PolicySupport.LEGAL_PLUS_PASS,
    )
    return float(max(reply_policy.probabilities.values()))


def reply_peak95_observed_value(
    context: TransitionContext, *, threshold: float = 0.95
) -> Optional[float]:
    del threshold
    if int(context.selected_move) == int(context.board_before.PASS_LOC):
        return None
    if context.reply_policy is None:
        raise OperationalDefinitionError("reply_peak95 requires a post-move opponent reply policy")
    return reply_peak(context.reply_policy)


def reply_peak95_observed_label(
    context: TransitionContext, *, threshold: float = 0.95
) -> Optional[int]:
    value = reply_peak95_observed_value(context)
    return None if value is None else int(value > float(threshold))


def _validated_candidate_scores(
    locations: Iterable[int], candidate_reply_peaks: Mapping[int, float]
) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    missing = []
    for raw_loc in locations:
        loc = int(raw_loc)
        if loc not in candidate_reply_peaks:
            missing.append(loc)
            continue
        score = float(candidate_reply_peaks[loc])
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise OperationalDefinitionError(
                f"Candidate reply peak at {loc} must lie in [0,1], got {score!r}"
            )
        scores[loc] = score
    if missing:
        preview = ", ".join(str(loc) for loc in missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise MissingCandidateScoresError(
            f"Missing reply-peak scores for {len(missing)} policy actions: {preview}{suffix}"
        )
    return scores


def reply_peak95_candidate_readouts(
    current_policy: PolicyView,
    candidate_reply_peaks: Mapping[int, float],
    *,
    threshold: float = 0.95,
) -> Mapping[str, float]:
    """Exact forcing-proxy mass under a current policy and frozen candidate map."""

    _require_policy(
        current_policy,
        time=PolicyTime.PRE_MOVE,
        support=PolicySupport.LEGAL_BOARD_CONDITIONAL,
    )
    scores = _validated_candidate_scores(
        current_policy.probabilities.keys(), candidate_reply_peaks
    )
    target_mass = 0.0
    expected_peak = 0.0
    for loc, probability in current_policy.probabilities.items():
        score = scores[loc]
        expected_peak += float(probability) * score
        if score > float(threshold):
            target_mass += float(probability)
    return {
        "reply_peak95_action_mass": float(target_mass),
        "expected_reply_peak": float(expected_peak),
    }


def reply_peak95_target_mask(
    board: Any,
    candidate_reply_peaks: Mapping[int, float],
    *,
    player: Optional[int] = None,
    threshold: float = 0.95,
) -> np.ndarray:
    """Legal candidate actions whose frozen opponent-reply peak exceeds the threshold."""

    player = int(board.pla if player is None else player)
    legal_locations = [
        board.loc(x, y)
        for y in range(board.size)
        for x in range(board.size)
        if board.would_be_legal(player, board.loc(x, y))
    ]
    scores = _validated_candidate_scores(legal_locations, candidate_reply_peaks)
    target = np.zeros((board.size, board.size), dtype=bool)
    for loc, score in scores.items():
        if score > float(threshold):
            target[board.loc_y(loc), board.loc_x(loc)] = True
    return target


def reply_peak95_contrast_mask(
    board: Any,
    candidate_reply_peaks: Mapping[int, float],
    *,
    player: Optional[int] = None,
    threshold: float = 0.95,
) -> np.ndarray:
    legal = legal_board_mask(board, player)
    target = reply_peak95_target_mask(
        board, candidate_reply_peaks, player=player, threshold=threshold
    )
    return rms_contrast_mask(target, legal & ~target)


def regional_policy_readouts(
    policy: PolicyView,
    board: Any,
    *,
    anchor_region: Optional[str] = None,
) -> Mapping[str, Any]:
    """Canonical nine-region concentration readouts for a pre-move board policy."""

    _require_policy(
        policy,
        time=PolicyTime.PRE_MOVE,
        support=PolicySupport.LEGAL_BOARD_CONDITIONAL,
    )
    masses = region_masses(policy, board)
    peak_region = max(REGIONS, key=lambda region: masses[region])
    sorted_masses = sorted(masses.values(), reverse=True)
    vector = np.asarray([masses[region] for region in REGIONS], dtype=float)
    readouts: Dict[str, Any] = {
        "regional_policy_peak": float(sorted_masses[0]),
        "regional_policy_margin": float(sorted_masses[0] - sorted_masses[1]),
        "regional_policy_entropy": float(-np.sum(vector * np.log(vector + 1e-12))),
        "regional_policy_peak_region": peak_region,
        "regional_policy_masses": masses,
    }
    if anchor_region is not None:
        if anchor_region not in REGIONS:
            raise OperationalDefinitionError(f"Unknown anchor region: {anchor_region!r}")
        readouts["regional_policy_anchor_mass"] = float(masses[anchor_region])
    return readouts


def regional_policy_peak_observed_value(
    context: TransitionContext, *, positive_quantile: float = 0.85
) -> Optional[float]:
    del positive_quantile
    if context.pre_policy is None:
        return None
    return float(regional_policy_readouts(context.pre_policy, context.board_before)["regional_policy_peak"])


def regional_policy_peak_observed_label(
    context: TransitionContext,
    *,
    high_threshold: float,
    positive_quantile: float = 0.85,
) -> Optional[int]:
    del positive_quantile
    threshold = float(high_threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise OperationalDefinitionError(
            f"Regional-policy threshold must lie in [0,1], got {high_threshold!r}"
        )
    value = regional_policy_peak_observed_value(context)
    return None if value is None else int(value >= threshold)


def regional_peak_target_mask(
    board: Any,
    policy: PolicyView,
    *,
    player: Optional[int] = None,
    anchor_region: Optional[str] = None,
) -> np.ndarray:
    """Legal actions in a fixed or baseline-peak region."""

    if anchor_region is None:
        anchor_region = str(regional_policy_readouts(policy, board)["regional_policy_peak_region"])
    if anchor_region not in REGIONS:
        raise OperationalDefinitionError(f"Unknown anchor region: {anchor_region!r}")
    legal = legal_board_mask(board, player)
    target = np.zeros_like(legal)
    for y in range(board.size):
        for x in range(board.size):
            if legal[y, x]:
                target[y, x] = classify_region_xy(x, y, board.size) == anchor_region
    return target


def regional_peak_contrast_mask(
    board: Any,
    policy: PolicyView,
    *,
    player: Optional[int] = None,
    anchor_region: Optional[str] = None,
) -> np.ndarray:
    legal = legal_board_mask(board, player)
    target = regional_peak_target_mask(
        board, policy, player=player, anchor_region=anchor_region
    )
    return rms_contrast_mask(target, legal & ~target)


TENUKI_DISTANCE6 = OperationalContract(
    name="tenuki_distance6",
    version=2,
    human_concept="tenuki",
    representation_time=PolicyTime.PRE_MOVE,
    label_scope=LabelScope.SELECTED_ACTION,
    label_policy_time=None,
    label_policy_support=None,
    readout_policy_time=PolicyTime.PRE_MOVE,
    readout_policy_support=PolicySupport.LEGAL_BOARD_CONDITIONAL,
    parameters={
        "min_manhattan_distance": 6,
        "anchor": "most_recent_nonpass_move",
    },
    raw_value=tenuki_observed_value,
    observed_label=tenuki_observed_label,
    policy_readouts=tenuki_policy_readouts,
)

REPLY_PEAK95 = OperationalContract(
    name="reply_peak95",
    version=2,
    human_concept="forcing",
    representation_time=PolicyTime.PRE_MOVE,
    label_scope=LabelScope.SELECTED_ACTION,
    label_policy_time=PolicyTime.POST_MOVE_REPLY,
    label_policy_support=PolicySupport.LEGAL_PLUS_PASS,
    readout_policy_time=PolicyTime.PRE_MOVE,
    readout_policy_support=PolicySupport.LEGAL_BOARD_CONDITIONAL,
    parameters={"reply_peak_threshold": 0.95, "strict_threshold": True},
    raw_value=reply_peak95_observed_value,
    observed_label=reply_peak95_observed_label,
    policy_readouts=reply_peak95_candidate_readouts,
)

REGIONAL_POLICY_PEAK = OperationalContract(
    name="regional_policy_peak",
    version=2,
    human_concept="urgency_proxy",
    representation_time=PolicyTime.PRE_MOVE,
    label_scope=LabelScope.POSITION,
    label_policy_time=PolicyTime.PRE_MOVE,
    label_policy_support=PolicySupport.LEGAL_BOARD_CONDITIONAL,
    readout_policy_time=PolicyTime.PRE_MOVE,
    readout_policy_support=PolicySupport.LEGAL_BOARD_CONDITIONAL,
    parameters={"positive_quantile": 0.85, "region_edge_19x19": 6},
    raw_value=regional_policy_peak_observed_value,
    observed_label=regional_policy_peak_observed_label,
    policy_readouts=regional_policy_readouts,
)

CONTRACTS: Mapping[str, OperationalContract] = MappingProxyType(
    {
        contract.definition_id: contract
        for contract in (TENUKI_DISTANCE6, REPLY_PEAK95, REGIONAL_POLICY_PEAK)
    }
)

CONTRACT_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "tenuki_distance6": TENUKI_DISTANCE6.definition_id,
        "tenuki": TENUKI_DISTANCE6.definition_id,
        "reply_peak95": REPLY_PEAK95.definition_id,
        "forcing": REPLY_PEAK95.definition_id,
        "regional_policy_peak": REGIONAL_POLICY_PEAK.definition_id,
        "urgency_peak": REGIONAL_POLICY_PEAK.definition_id,
    }
)


def get_contract(name_or_id: str) -> OperationalContract:
    """Resolve a versioned contract ID or a migration alias to its canonical contract."""

    definition_id = CONTRACT_ALIASES.get(name_or_id, name_or_id)
    try:
        return CONTRACTS[definition_id]
    except KeyError as exc:
        known = ", ".join(CONTRACTS)
        raise KeyError(f"Unknown operational definition {name_or_id!r}; known IDs: {known}") from exc


__all__ = [
    "CONTRACTS",
    "CONTRACT_ALIASES",
    "REGIONS",
    "TENUKI_DISTANCE6",
    "REPLY_PEAK95",
    "REGIONAL_POLICY_PEAK",
    "IneligiblePositionError",
    "LabelScope",
    "MissingCandidateScoresError",
    "OperationalContract",
    "OperationalDefinitionError",
    "PolicySupport",
    "PolicyTime",
    "PolicyView",
    "TransitionContext",
    "classify_region_xy",
    "get_contract",
    "legal_board_mask",
    "legal_policy_view",
    "manhattan_distance",
    "region_for_loc",
    "region_masses",
    "regional_peak_contrast_mask",
    "regional_peak_target_mask",
    "regional_policy_peak_observed_label",
    "regional_policy_peak_observed_value",
    "regional_policy_readouts",
    "reply_peak",
    "reply_peak95_candidate_readouts",
    "reply_peak95_contrast_mask",
    "reply_peak95_observed_label",
    "reply_peak95_observed_value",
    "reply_peak95_target_mask",
    "rms_contrast_mask",
    "tenuki_action_predicate",
    "tenuki_contrast_mask",
    "tenuki_observed_label",
    "tenuki_observed_value",
    "tenuki_policy_readouts",
    "tenuki_target_mask",
]
