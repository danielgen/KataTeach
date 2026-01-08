#!/usr/bin/env python3
"""
Go board position analysis with ownership-based spatial concepts.

Ownership convention:
    - Raw ownership from KataGo's get_model_outputs is from CURRENT PLAYER's perspective
      (positive = current player to move's territory)
    - This changes based on whose turn it is: after Black plays, it's from White's perspective
    - This module normalizes ownership to the analyzing player's perspective
    - All delta features compare before/after ownership from the same perspective
    - Pass ownership_frame_player to indicate whose perspective the ownership is from
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Set
from functools import lru_cache
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "python"))
from board import Board

# --- Configuration ---
TAU_POS = 0.10          # Weak ownership threshold
TAU_SOLID = 0.70        # Solid territory threshold
TAU_POS_LOW = 0.08      # Hysteresis low
TAU_POS_HIGH = 0.12     # Hysteresis high
TAU_ONLY_MOVE = 0.05    # "Only move" threshold
TAU_GROUP_IOU = 0.1     # Group matching IoU threshold
TAU_GROUP_BELONGING = 0.2  # Ownership threshold for grouping stones by influence paths
TAU_AJI_VICINITY = 5    # Aji reduction L1 radius
TAU_DELTA_MIN = 0.05     # Minimum ownership delta for solidification/reduction

REGIONS = ["corner_tl", "corner_tr", "corner_bl", "corner_br",
           "side_left", "side_right", "side_top", "side_bottom", "center"]


@dataclass
class Group:
    color: int
    head: int
    stones: List[int]
    liberties: int
    strength: float = 0.0
    connectivity: float = 0.0
    influence_area: int = 0
    influence_strength: float = 0.0


# --- Coordinate Helpers ---

def loc_to_xy(board: Board, loc: int) -> Tuple[int, int]:
    return board.loc_x(loc), board.loc_y(loc)


def xy_to_loc(board: Board, x: int, y: int) -> int:
    return board.loc(x, y)


def in_bounds(x: int, y: int, size: int = 19) -> bool:
    return 0 <= x < size and 0 <= y < size


def classify_region(x: int, y: int, size: int = 19) -> str:
    """Classify (x,y) into one of 9 regions."""
    c = 6  # corner size
    s = size - c
    if x < c and y < c: return "corner_tl"
    if x >= s and y < c: return "corner_tr"
    if x < c and y >= s: return "corner_bl"
    if x >= s and y >= s: return "corner_br"
    if c <= x < s and y < c: return "side_top"
    if c <= x < s and y >= s: return "side_bottom"
    if x < c and c <= y < s: return "side_left"
    if x >= s and c <= y < s: return "side_right"
    return "center"


@lru_cache(maxsize=4)
def region_map(size: int = 19) -> np.ndarray:
    m = np.empty((size, size), dtype=object)
    for y in range(size):
        for x in range(size):
            m[y, x] = classify_region(x, y, size)
    return m


def _empty_region_dict(default=0.0) -> Dict[str, Any]:
    return {r: default for r in REGIONS}


CORNER_REGIONS = {"corner_tl", "corner_tr", "corner_bl", "corner_br"}


def count_stones_in_corner(board: Board, corner_region: str) -> Tuple[int, int]:
    """
    Count stones in a corner region.
    
    Args:
        board: Current board state
        corner_region: One of "corner_tl", "corner_tr", "corner_bl", "corner_br"
    
    Returns:
        (black_count, white_count) tuple
    """
    black_count = 0
    white_count = 0
    size = board.size
    
    for y in range(size):
        for x in range(size):
            if classify_region(x, y, size) == corner_region:
                loc = board.loc(x, y)
                stone = board.board[loc]
                if stone == Board.BLACK:
                    black_count += 1
                elif stone == Board.WHITE:
                    white_count += 1
    
    return black_count, white_count


def is_occupy_corner(board: Board, before_board: Optional[Board], move_loc: int, player: int) -> bool:
    """
    Check if move is the first stone in a corner area.
    
    Returns True when a player places the first stone in a corner region
    (i.e., no stones existed in that corner before the move).
    """
    if before_board is None:
        return False
    
    x, y = loc_to_xy(board, move_loc)
    region = classify_region(x, y, board.size)
    
    # Must be in a corner
    if region not in CORNER_REGIONS:
        return False
    
    # Check if corner was empty before this move
    black_before, white_before = count_stones_in_corner(before_board, region)
    return black_before == 0 and white_before == 0


def is_approaching_corner(board: Board, before_board: Optional[Board], move_loc: int, player: int) -> bool:
    """
    Check if move approaches an opponent's corner stone.
    
    Returns True when the move is the second stone in a corner area
    and the only other stone in that corner is an opponent's stone.
    """
    if before_board is None:
        return False
    
    x, y = loc_to_xy(board, move_loc)
    region = classify_region(x, y, board.size)
    
    # Must be in a corner
    if region not in CORNER_REGIONS:
        return False
    
    # Check stones in corner before the move
    black_before, white_before = count_stones_in_corner(before_board, region)
    total_before = black_before + white_before
    
    # Must be exactly one stone before (opponent's)
    if total_before != 1:
        return False
    
    # The existing stone must be opponent's
    opponent = Board.get_opp(player)
    if opponent == Board.BLACK:
        return black_before == 1
    else:
        return white_before == 1


def _stone_mask(board: Board) -> np.ndarray:
    """Create mask where stones exist (True = stone present)."""
    mask = np.zeros((board.size, board.size), dtype=bool)
    for y in range(board.size):
        for x in range(board.size):
            if board.board[board.loc(x, y)] != Board.EMPTY:
                mask[y, x] = True
    return mask


def _group_to_mask(group: Group, board: Board) -> np.ndarray:
    mask = np.zeros((board.size, board.size), dtype=bool)
    for loc in group.stones:
        x, y = loc_to_xy(board, loc)
        mask[y, x] = True
    return mask


def _get_group_region(group: Group, board: Board) -> str:
    """Get dominant region for a group."""
    counts = {}
    for loc in group.stones:
        x, y = loc_to_xy(board, loc)
        r = classify_region(x, y, board.size)
        counts[r] = counts.get(r, 0) + 1
    return max(counts, key=counts.get) if counts else "none"


def _match_groups_by_iou(
    before_groups: List[Group],
    after_groups: List[Group],
    board: Board
) -> List[Tuple[Group, Group]]:
    """Match groups across plies using spatial IoU."""
    matched = []
    used = set()
    for bg in before_groups:
        bm = _group_to_mask(bg, board)
        best_iou, best_ag, best_idx = 0.0, None, None
        for i, ag in enumerate(after_groups):
            if i in used:
                continue
            am = _group_to_mask(ag, board)
            inter = np.sum(bm & am)
            union = np.sum(bm | am)
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou and iou >= TAU_GROUP_IOU:
                best_iou, best_ag, best_idx = iou, ag, i
        if best_ag is not None:
            matched.append((bg, best_ag))
            used.add(best_idx)
    return matched


# --- Ownership Normalization ---

def normalize_ownership(ownership: np.ndarray, from_player: int, to_player: int) -> np.ndarray:
    """Convert ownership from one player's perspective to another's."""
    return ownership if from_player == to_player else -ownership


# --- Group Functions ---

def enumerate_groups(
    board: Board,
    ownership: np.ndarray,
    color: int,
    tau: float = TAU_GROUP_BELONGING
) -> List[Group]:
    """
    Enumerate groups using physical adjacency first, then ownership paths.
    
    Groups are defined by:
    1. Physical adjacency: Stones of the same color that are directly adjacent 
       are ALWAYS in the same group (they share liberties and live/die together)
    2. Ownership connectivity: Groups can be extended through empty points with 
       ownership > tau to connect strategically related stones
    
    Args:
        board: Current board state
        ownership: Ownership array (19x19), normalized to color's perspective
        color: Color to group (Board.BLACK or Board.WHITE)
        tau: Ownership threshold for path connectivity through empty points
    
    Returns:
        List of Group objects
    """
    size = board.size
    
    visited_stones = set()
    groups = []
    
    # Find all stones of the given color
    all_stones = []
    for y in range(size):
        for x in range(size):
            loc = board.loc(x, y)
            if board.board[loc] == color:
                all_stones.append(loc)
    
    # BFS to find connected components
    for start_loc in all_stones:
        if start_loc in visited_stones:
            continue
        
        # BFS from this stone
        component = []
        queue = [start_loc]
        visited_stones.add(start_loc)
        visited_path = set()  # Track empty points visited in BFS
        
        while queue:
            loc = queue.pop(0)
            if board.board[loc] == color:
                component.append(loc)
            
            x, y = loc_to_xy(board, loc)
            
            # Check all 4 neighbors
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if not in_bounds(nx, ny, size):
                    continue
                
                nloc = xy_to_loc(board, nx, ny)
                
                # RULE 1: Physically adjacent stones are ALWAYS connected
                # (regardless of ownership - they share liberties)
                if board.board[nloc] == color:
                    if nloc not in visited_stones:
                        visited_stones.add(nloc)
                        queue.append(nloc)
                # RULE 2: Can traverse through empty points with high ownership
                # to connect strategically related stones
                elif board.board[nloc] == Board.EMPTY:
                    if nloc not in visited_path and ownership[ny, nx] > tau:
                        visited_path.add(nloc)
                        queue.append(nloc)
        
        if component:
            # Use first stone as head, compute liberties
            head = component[0]
            # Count liberties for the group (union of all stones' liberties)
            liberty_set = set()
            for loc in component:
                x, y = loc_to_xy(board, loc)
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = x + dx, y + dy
                    if in_bounds(nx, ny, size):
                        nloc = xy_to_loc(board, nx, ny)
                        if board.board[nloc] == Board.EMPTY:
                            liberty_set.add(nloc)
            
            groups.append(Group(
                color=color,
                head=head,
                stones=component,
                liberties=len(liberty_set)
            ))
    
    return groups


def compute_group_strengths(groups: List[Group], ownership: np.ndarray, board: Board) -> None:
    """Set group strength as mean ownership over stones."""
    for g in groups:
        vals = []
        for loc in g.stones:
            x, y = loc_to_xy(board, loc)
            vals.append(float(ownership[y, x]))
        g.strength = float(np.mean(vals)) if vals else 0.0


def compute_group_connectivity(groups: List[Group], ownership: np.ndarray, board: Board) -> None:
    """Set group connectivity as mean ownership of nearby empty points."""
    for g in groups:
        checked = set()
        vals = []
        for loc in g.stones:
            x0, y0 = loc_to_xy(board, loc)
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if abs(dx) + abs(dy) > 2:
                        continue
                    nx, ny = x0 + dx, y0 + dy
                    if not in_bounds(nx, ny, board.size):
                        continue
                    nloc = xy_to_loc(board, nx, ny)
                    if nloc in checked or board.board[nloc] != Board.EMPTY:
                        continue
                    checked.add(nloc)
                    vals.append(ownership[ny, nx])
        g.connectivity = float(np.mean(vals)) if vals else 0.0


def compute_group_influence(groups: List[Group], ownership: np.ndarray, board: Board) -> None:
    """
    Set group influence area and strength.
    
    Finds all empty points and opponent stones connected to the group via paths 
    where every point on the path has ownership >= TAU_POS. Only considers own player's ownership.
    """
    size = board.size
    for g in groups:
        # Start BFS from all stones in the group
        visited = set()
        influence_points = []  # List of (ownership_value, loc) for empty points
        
        # Initialize queue with all stones in the group
        queue = []
        for loc in g.stones:
            queue.append(loc)
            visited.add(loc)
        
        # BFS to find all reachable empty points via ownership paths
        while queue:
            loc = queue.pop(0)
            x, y = loc_to_xy(board, loc)
            
            # Check all 4 neighbors
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if not in_bounds(nx, ny, size):
                    continue
                
                nloc = xy_to_loc(board, nx, ny)
                if nloc in visited:
                    continue
                
                # Check if path is valid (ownership >= TAU_POS)
                v = ownership[ny, nx]
                if v < TAU_POS:
                    continue
                
                visited.add(nloc)
                
                # If empty point or opponent stone, add to influence points
                if (board.board[nloc] == Board.EMPTY or board.board[nloc] != g.color):
                    influence_points.append(v)
                    queue.append(nloc)
                else:
                    # Own stone - continue traversing
                    queue.append(nloc)
        
        g.influence_area = len(influence_points)
        g.influence_strength = float(np.mean(influence_points)) if influence_points else 0.0


# --- Territory Analysis ---

def count_building_territory(before: np.ndarray, after: np.ndarray, board: Board) -> Tuple[int, float]:
    """Count points that went from neutral to owned territory."""
    mask = (np.abs(before) < TAU_POS_LOW) & (after > TAU_POS_HIGH) & ~_stone_mask(board)
    count = int(np.sum(mask))
    intensity = float(np.mean(after[mask] - before[mask])) if count > 0 else 0.0
    return count, intensity


def solidify_territory_delta(before: np.ndarray, after: np.ndarray, board: Board) -> Tuple[int, float]:
    """Count points where owned territory was strengthened (delta >= 0.1)."""
    delta = after - before
    mask = (before > TAU_POS) & (after > before) & (delta >= TAU_DELTA_MIN) & ~_stone_mask(board)
    count = int(np.sum(mask))
    intensity = float(np.mean(delta[mask])) if count > 0 else 0.0
    return count, intensity


def reduce_opponent_territory(before: np.ndarray, after: np.ndarray, board: Board) -> Tuple[int, float]:
    """Count points where opponent territory was reduced (delta >= 0.1)."""
    delta = after - before
    mask = (before < -TAU_POS) & (after > -TAU_POS) & (np.abs(delta) >= TAU_DELTA_MIN) & ~_stone_mask(board)
    count = int(np.sum(mask))
    intensity = float(np.mean(np.abs(delta[mask]))) if count > 0 else 0.0
    return count, intensity


def invasion_effect(before: np.ndarray, after: np.ndarray, board: Board, move_loc: Optional[int] = None, groups: Optional[List[Group]] = None) -> Tuple[bool, float]:
    """Detect if move invades opponent territory (flips from opp to own).
    
    Requires:
    - The stone just played must have 3+ liberties
    - The area around the move must be mostly empty (>= 80% empty points within radius 3)
    - The invading stone must be the only stone in its group (if groups are provided)
    """
    empty = ~_stone_mask(board)
    if move_loc is not None and move_loc != Board.PASS_LOC:
        mx, my = loc_to_xy(board, move_loc)
        
        # Check that the stone is the only stone in its group (if groups are provided)
        if groups is not None:
            move_group = find_group_containing(move_loc, groups)
            if move_group is None or len(move_group.stones) > 1:
                return False, 0.0
        
        # Check liberties: count adjacent empty points
        liberties = 0
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = mx + dx, my + dy
            if in_bounds(nx, ny, board.size):
                nloc = xy_to_loc(board, nx, ny)
                if board.board[nloc] == Board.EMPTY:
                    liberties += 1
        
        # Require 3+ liberties
        if liberties < 3:
            return False, 0.0
        
        # Check that area around move is mostly empty
        spatial = np.zeros_like(before, dtype=bool)
        for y in range(board.size):
            for x in range(board.size):
                if np.sqrt((x - mx)**2 + (y - my)**2) <= 3.0:
                    spatial[y, x] = True
        
        # Count empty points and total points in the area (excluding the move location itself)
        empty_in_area = np.sum(empty & spatial)
        total_in_area = np.sum(spatial) - 1  # Exclude the move location itself
        empty_ratio = empty_in_area / max(1, total_in_area)
        
        # Require >= 80% empty points in the area
        if empty_ratio < 0.8:
            return False, 0.0
        
        # Check opponent territory ratio (existing logic)
        opp_ratio = np.sum(empty & spatial & (before < -TAU_POS)) / max(1, np.sum(empty & spatial))
        if opp_ratio <= 0.5:
            return False, 0.0
        
        mask = empty & spatial & (before < -TAU_POS) & (after > TAU_POS)
    else:
        mask = empty & (before < -TAU_POS) & (after > TAU_POS)
    count = int(np.sum(mask))
    intensity = float(np.mean(after[mask] - before[mask])) if count > 0 else 0.0
    return count > 0, intensity


def territory_sizes(ownership: np.ndarray, board: Board, player: Optional[int] = None) -> Tuple[int, int]:
    """
    Return (potential, solid) territory counts.
    
    Includes empty points and opponent stones under own control (ownership > TAU_POS).
    """
    empty = ~_stone_mask(board)
    
    # Include opponent stones under own control if player is provided
    if player is not None:
        opp = Board.get_opp(player)
        opp_stones = np.zeros((board.size, board.size), dtype=bool)
        for y in range(board.size):
            for x in range(board.size):
                loc = board.loc(x, y)
                if board.board[loc] == opp:
                    opp_stones[y, x] = True
        territory_mask = empty | opp_stones
    else:
        territory_mask = empty
    
    solid = int(np.sum((ownership >= TAU_SOLID) & territory_mask))
    potential = int(np.sum((ownership > TAU_POS) & territory_mask)) - solid
    return potential, solid


def territory_sizes_with_delta(
    before: np.ndarray, after: np.ndarray, board: Board, player: Optional[int] = None
) -> Dict[str, int]:
    """Return delta territory counts (change from before to after move).
    
    Includes empty points and opponent stones under own control (ownership > TAU_POS).
    
    Returns dict with:
        potential_territory: Change in weakly-owned points (excluding solid) from before to after
        solid_territory: Change in strongly-owned points from before to after
    """
    empty = ~_stone_mask(board)
    
    # Include opponent stones under own control if player is provided
    if player is not None:
        opp = Board.get_opp(player)
        opp_stones = np.zeros((board.size, board.size), dtype=bool)
        for y in range(board.size):
            for x in range(board.size):
                loc = board.loc(x, y)
                if board.board[loc] == opp:
                    opp_stones[y, x] = True
        territory_mask = empty | opp_stones
    else:
        territory_mask = empty
    
    pot_before = int(np.sum((before > TAU_POS) & territory_mask))
    pot_after = int(np.sum((after > TAU_POS) & territory_mask))
    solid_before = int(np.sum((before >= TAU_SOLID) & territory_mask))
    solid_after = int(np.sum((after >= TAU_SOLID) & territory_mask))
    
    # Return deltas (change from before to after)
    potential_before = pot_before - solid_before
    potential_after = pot_after - solid_after
    return {
        "potential_territory": potential_after - potential_before,
        "solid_territory": solid_after - solid_before,
    }


def direct_sacrifice(move_loc: int, ownership: np.ndarray, board: Board, before_ownership: Optional[np.ndarray] = None) -> Tuple[bool, float]:
    """
    Check if played stone is in opponent territory AFTER the move.
    
    A direct sacrifice occurs when the stone just played is immediately considered
    to be in opponent territory (i.e., it's a sacrifice stone that will likely die).
    
    Uses ownership (after the move) to check if the stone is in opponent territory.
    Ownership should be normalized to the player's perspective 
    (positive = good for player, negative = good for opponent).
    
    Args:
        move_loc: Location of the move
        ownership: Ownership array AFTER the move (normalized to player's perspective)
        board: Board state after the move
        before_ownership: Not used (kept for backward compatibility)
    
    Returns:
        (is_sacrifice, intensity): Whether the stone is a sacrifice and how strongly
    """
    if move_loc == Board.PASS_LOC:
        return False, 0.0
    x, y = loc_to_xy(board, move_loc)
    
    # Check if the stone is in opponent territory AFTER the move
    # (negative ownership from player's perspective means opponent's territory)
    is_sac = ownership[y, x] < -TAU_POS
    return is_sac, abs(ownership[y, x]) if is_sac else 0.0


def indirect_sacrifice(
    before: np.ndarray, after: np.ndarray, color: int, 
    board: Board, before_board: Optional[Board] = None
) -> Tuple[int, float]:
    """
    Count own stones that became opponent territory.
    
    Uses before_board to find stones that existed BEFORE the move - this is crucial
    because captured stones won't be on the current board anymore.
    Falls back to current board if before_board is not provided.
    
    Args:
        before: Ownership array before the move (normalized to player's perspective)
        after: Ownership array after the move (normalized to player's perspective)
        color: Player color to check for sacrificed stones
        board: Board state after the move
        before_board: Board state before the move (important for detecting captured stones)
    """
    # Use before_board to find stones that existed before the move
    # This is crucial for detecting captured stones that are no longer on current board
    check_board = before_board if before_board is not None else board
    
    mask = np.zeros((check_board.size, check_board.size), dtype=bool)
    for y in range(check_board.size):
        for x in range(check_board.size):
            if check_board.board[check_board.loc(x, y)] == color:
                mask[y, x] = True
    sac = mask & (before > TAU_POS) & (after < -TAU_POS)
    count = int(np.sum(sac))
    intensity = float(np.mean(before[sac] - after[sac])) if count > 0 else 0.0
    return count, intensity


# --- Regional Analysis ---

def compute_territory_delta_by_region(
    before: np.ndarray, after: np.ndarray, board: Board
) -> Tuple[Dict[str, int], Dict[str, float], Dict[str, int], Dict[str, float]]:
    """Compute building and solidification by region.
    
    Uses same thresholds as global functions:
    - Building: Uses hysteresis (TAU_POS_LOW/TAU_POS_HIGH) to match count_building_territory
    - Solidification: Uses TAU_POS to match solidify_territory_delta
    """
    build_c, build_i = _empty_region_dict(0), _empty_region_dict(0.0)
    solid_c, solid_i = _empty_region_dict(0), _empty_region_dict(0.0)
    stones = _stone_mask(board)
    for y in range(board.size):
        for x in range(board.size):
            if stones[y, x]:
                continue
            r = classify_region(x, y, board.size)
            # Building: use hysteresis thresholds to match count_building_territory
            if np.abs(before[y, x]) < TAU_POS_LOW and after[y, x] > TAU_POS_HIGH:
                build_c[r] += 1
                build_i[r] += after[y, x] - before[y, x]
            # Solidification: matches solidify_territory_delta (requires delta >= 0.1)
            delta = after[y, x] - before[y, x]
            if before[y, x] > TAU_POS and after[y, x] > before[y, x] and delta >= TAU_DELTA_MIN:
                solid_c[r] += 1
                solid_i[r] += delta
    for r in REGIONS:
        if build_c[r] > 0:
            build_i[r] /= build_c[r]
        if solid_c[r] > 0:
            solid_i[r] /= solid_c[r]
    return build_c, build_i, solid_c, solid_i


def compute_reduction_by_region(
    before: np.ndarray, after: np.ndarray, board: Board
) -> Tuple[Dict[str, int], Dict[str, float]]:
    """Compute reduction of opponent territory by region.
    
    Uses same threshold-crossing logic as reduce_opponent_territory:
    counts only points that cross from opponent territory to contested/neutral.
    """
    reduction_count, reduction_intensity = _empty_region_dict(0), _empty_region_dict(0.0)
    stones = _stone_mask(board)
    for y in range(board.size):
        for x in range(board.size):
            if stones[y, x]:
                continue
            # Match reduce_opponent_territory: count threshold crossings only (requires delta >= 0.1)
            delta = after[y, x] - before[y, x]
            if before[y, x] < -TAU_POS and after[y, x] > -TAU_POS and abs(delta) >= TAU_DELTA_MIN:
                r = classify_region(x, y, board.size)
                reduction_count[r] += 1
                reduction_intensity[r] += abs(delta)
    for r in REGIONS:
        if reduction_count[r] > 0:
            reduction_intensity[r] /= reduction_count[r]
    return reduction_count, reduction_intensity


# --- Policy Analysis ---

def urgency_by_region(policy: np.ndarray) -> Dict[str, float]:
    """Sum of policy mass by region."""
    urg = _empty_region_dict(0.0)
    for y in range(19):
        for x in range(19):
            idx = y * 19 + x
            if idx < len(policy):
                urg[classify_region(x, y, 19)] += float(policy[idx])
    return urg


def urgency_intensity_by_region(policy: np.ndarray) -> Dict[str, float]:
    """Normalized urgency (share of total policy mass)."""
    urg = urgency_by_region(policy)
    total = sum(urg.values())
    return {r: urg[r] / total if total > 0 else 0.0 for r in urg}


def is_forcing(policy: np.ndarray) -> bool:
    """Check if one move dominates (>95% probability)."""
    return float(np.max(policy)) > (1.0 - TAU_ONLY_MOVE)


def is_tenuki(selected_idx: int, last_move_loc: Optional[int], policy: np.ndarray, board: Board) -> bool:
    """Check if move is tenuki (far from last move, ignoring local follow-up)."""
    if last_move_loc is None or last_move_loc == Board.PASS_LOC:
        return False
    x_sel, y_sel = selected_idx % 19, selected_idx // 19
    x_last, y_last = loc_to_xy(board, last_move_loc)
    if abs(x_sel - x_last) + abs(y_sel - y_last) < 6:
        return False
    if classify_region(x_sel, y_sel, 19) == classify_region(x_last, y_last, 19):
        return False
    sel_prob = policy[selected_idx]
    for y in range(19):
        for x in range(19):
            if abs(x - x_last) + abs(y - y_last) <= 4:
                idx = y * 19 + x
                if idx < len(policy) and policy[idx] > sel_prob:
                    return True
    return False


# --- Tactical Analysis ---

def is_cut_move(board: Board, move_loc: int) -> bool:
    """Check if move separates 2+ opponent groups."""
    if move_loc == Board.PASS_LOC:
        return False
    # After the move, the stone at move_loc is the player who made the move
    # board.pla is the NEXT player to move (opponent of the mover)
    mover = board.board[move_loc]
    if mover == Board.EMPTY:
        return False  # Should not happen, but safety check
    opp = Board.get_opp(mover)
    x, y = loc_to_xy(board, move_loc)
    heads = set()
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nx, ny = x + dx, y + dy
        if in_bounds(nx, ny, board.size):
            nloc = xy_to_loc(board, nx, ny)
            if board.board[nloc] == opp:
                heads.add(board.group_head[nloc])
    return len(heads) >= 2


def is_connection_move(
    board: Board, move_loc: int, color: int,
    before_board: Board, before_ownership: np.ndarray, after_ownership: np.ndarray
) -> Tuple[bool, float, List[str], List[int]]:
    """
    Check if move connects 2+ previously separate groups using group enumeration logic.
    
    Args:
        board: Board state after the move
        move_loc: Location of the move
        color: Player color
        before_board: Board state before the move
        before_ownership: Ownership array before the move (normalized to color's perspective)
        after_ownership: Ownership array after the move (normalized to color's perspective)
    
    Returns:
        (is_connection, strength_gain, merged_regions, merged_head_locs):
        - is_connection: True if 2+ groups were merged
        - strength_gain: Number of groups merged minus 1
        - merged_regions: List of regions where merged groups are located
        - merged_head_locs: List of head stone locations of merged groups
    """
    if move_loc == Board.PASS_LOC:
        return False, 0.0, [], []
    
    # Enumerate groups before and after the move
    groups_before = enumerate_groups(before_board, before_ownership, color)
    groups_after = enumerate_groups(board, after_ownership, color)
    
    # Find the group containing the move after the move
    move_group_after = find_group_containing(move_loc, groups_after)
    if move_group_after is None:
        return False, 0.0, [], []
    
    # Find which groups from before were merged into this group
    # A before group is merged if any of its stones are in the move group
    merged_groups = []
    move_group_stones = set(move_group_after.stones)
    
    for before_group in groups_before:
        # Check if any stones from before group are in the move group
        before_stones = set(before_group.stones)
        if before_stones & move_group_stones:  # Non-empty intersection
            merged_groups.append(before_group)
    
    # Connection occurs if 2+ previously separate groups were merged
    # (excluding the move itself - we only count groups that existed before)
    num_merged = len(merged_groups)
    
    # If no groups matched, the move didn't connect any existing groups
    if num_merged == 0:
        return False, 0.0, [], []
    
    # Extract regions and head locations of merged groups
    merged_regions = []
    merged_head_locs = []
    for group in merged_groups:
        region = _get_group_region(group, before_board)
        merged_regions.append(region)
        merged_head_locs.append(group.head)
    
    # Connection strength gain = number of groups merged - 1
    # (if 2 groups merged, gain = 1; if 3 groups merged, gain = 2, etc.)
    is_conn = num_merged >= 2
    strength_gain = float(num_merged - 1) if is_conn else 0.0
    
    return is_conn, strength_gain, merged_regions, merged_head_locs


def is_extension_move(board: Board, move_loc: int, color: int) -> bool:
    """Check if move is adjacent to own stone."""
    if move_loc == Board.PASS_LOC:
        return False
    x, y = loc_to_xy(board, move_loc)
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nx, ny = x + dx, y + dy
        if in_bounds(nx, ny, board.size) and board.board[xy_to_loc(board, nx, ny)] == color:
            return True
    return False


def liberties_of_group(board: Board, loc: int, groups: List[Group]) -> int:
    """
    Count liberties of the group containing the given location.
    
    Looks up the group in the provided groups list and returns its liberties.
    Groups must be computed beforehand (e.g., via enumerate_groups).
    """
    if loc == Board.PASS_LOC or board.board[loc] == Board.EMPTY:
        return 0
    
    # Look up the group containing this location
    for group in groups:
        if loc in group.stones:
            return group.liberties
    
    # Should not happen if groups were computed correctly, but return 0 as fallback
    return 0


def find_group_containing(loc: int, groups: List[Group]) -> Optional[Group]:
    """Find the group that contains the given location."""
    for group in groups:
        if loc in group.stones:
            return group
    return None


def compute_current_group_delta(
    move_loc: int,
    groups_before: List[Group],
    groups_after: List[Group],
    board: Board
) -> Tuple[float, float, float, float, int, int, float, float]:
    """
    Compute strength, connectivity, and influence delta for the group containing the move.
    
    Returns:
        (current_strength, strength_delta, current_connectivity, connectivity_delta,
         current_influence_count, influence_count_delta, current_influence_strength, influence_strength_delta)
    """
    if move_loc == Board.PASS_LOC or board.board[move_loc] == Board.EMPTY:
        return 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0
    
    # Find the group containing this move after the move
    current_group = find_group_containing(move_loc, groups_after)
    if current_group is None:
        return 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0
    
    current_strength = current_group.strength
    current_connectivity = current_group.connectivity
    current_influence_count = current_group.influence_area
    current_influence_strength = current_group.influence_strength
    
    # Try to find corresponding group before the move using IoU matching
    best_match = None
    best_iou = 0.0
    current_mask = _group_to_mask(current_group, board)
    
    for before_group in groups_before:
        before_mask = _group_to_mask(before_group, board)
        inter = np.sum(current_mask & before_mask)
        union = np.sum(current_mask | before_mask)
        iou = inter / union if union > 0 else 0.0
        if iou > best_iou and iou >= TAU_GROUP_IOU:
            best_iou = iou
            best_match = before_group
    
    if best_match is not None:
        strength_delta = current_strength - best_match.strength
        connectivity_delta = current_connectivity - best_match.connectivity
        influence_count_delta = current_influence_count - best_match.influence_area
        influence_strength_delta = current_influence_strength - best_match.influence_strength
    else:
        # New group (no match before) - no meaningful comparison available
        # Set all deltas to 0 since there's no "before" state to compare against
        strength_delta = 0.0
        connectivity_delta = 0.0
        influence_count_delta = 0
        influence_strength_delta = 0.0
    
    return (current_strength, strength_delta, current_connectivity, connectivity_delta,
            current_influence_count, influence_count_delta, current_influence_strength, influence_strength_delta)

def atari_move(board: Board, move_loc: int) -> bool:
    """Check if move puts opponent group in atari.
    
    Uses Board's deterministic group tracking (stone adjacency), not ownership-based groups.
    Atari = group has exactly 1 liberty remaining (can be captured next move).
    """
    if move_loc == Board.PASS_LOC:
        return False
    # After the move, the stone at move_loc is the player who made the move
    # board.pla is the NEXT player to move (opponent of the mover)
    mover = board.board[move_loc]
    if mover == Board.EMPTY:
        return False  # Should not happen, but safety check
    opp = Board.get_opp(mover)
    x, y = loc_to_xy(board, move_loc)
    
    # Check the 4 neighbors of the move location
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nx, ny = x + dx, y + dy
        if not in_bounds(nx, ny, board.size):
            continue
        nloc = xy_to_loc(board, nx, ny)
        
        # If neighbor is opponent stone, check if its group has exactly 1 liberty (atari)
        if board.board[nloc] == opp:
            head = board.group_head[nloc]
            if board.group_liberty_count[head] == 1:
                return True
    
    return False


def creates_new_group(
    groups_before: List[Group], groups_after: List[Group], 
    move_loc: int, board: Board
) -> bool:
    """
    Check if move created a new group (didn't extend an existing group).
    
    Uses IoU matching to determine if the group containing the move has a 
    corresponding group from before the move. If no match is found, the move
    created a new group.
    
    Args:
        groups_before: List of groups before the move
        groups_after: List of groups after the move
        move_loc: Location of the stone just played
        board: Board state after the move
    """
    if move_loc == Board.PASS_LOC:
        return False
    
    # Find the group containing the played stone after the move
    current_group = find_group_containing(move_loc, groups_after)
    if current_group is None:
        return False
    
    # Try to find a matching group before the move using IoU
    current_mask = _group_to_mask(current_group, board)
    
    for before_group in groups_before:
        before_mask = _group_to_mask(before_group, board)
        inter = np.sum(current_mask & before_mask)
        union = np.sum(current_mask | before_mask)
        iou = inter / union if union > 0 else 0.0
        if iou >= TAU_GROUP_IOU:
            # Found a matching group - move extended an existing group
            return False
    
    # No matching group found - this is a new group
    return True


# --- Attack Analysis ---

def reduce_aji(before: np.ndarray, after: np.ndarray, board: Board, color: int, move_loc: Optional[int] = None) -> Tuple[bool, float]:
    """Check if move reduces aji (increases ownership over weak opponent stones)."""
    opp = Board.get_opp(color)
    deltas = []
    vicinity = None
    if move_loc is not None and move_loc != Board.PASS_LOC:
        mx, my = loc_to_xy(board, move_loc)
        vicinity = np.zeros((board.size, board.size), dtype=bool)
        for y in range(board.size):
            for x in range(board.size):
                if abs(x - mx) + abs(y - my) <= TAU_AJI_VICINITY:
                    vicinity[y, x] = True
    for y in range(board.size):
        for x in range(board.size):
            loc = xy_to_loc(board, x, y)
            if board.board[loc] == opp and before[y, x] > TAU_POS:
                if vicinity is None or vicinity[y, x]:
                    deltas.append(after[y, x] - before[y, x])
    if not deltas:
        return False, 0.0
    intensity = float(np.mean(deltas))
    return intensity >= 0.05, intensity


def attack_strength_delta(
    groups_before: List[Group], groups_after: List[Group], board: Board
) -> Tuple[bool, float, float]:
    """
    Compute attack effect by comparing opponent group strengths.
    
    Groups should already be filtered to opponent groups. Returns:
    - bool: True if at least one opponent group decreased by 0.1 or more
    - float: Average strength decrease amount
    - float: Maximum strength decrease amount
    """
    if not groups_before or not groups_after:
        return False, 0.0, 0.0
    matched = _match_groups_by_iou(groups_before, groups_after, board)
    if not matched:
        return False, 0.0, 0.0
    deltas = [a.strength - b.strength for b, a in matched]
    avg = float(np.mean(deltas))
    # Attack is True if at least one group decreased by 0.1 or more
    has_attack = any(delta <= -0.1 for delta in deltas)
    return has_attack, abs(avg), abs(min(deltas))


def get_attacked_groups_info(
    groups_before: List[Group], groups_after: List[Group], board: Board
) -> Dict[str, Any]:
    """
    Get information about groups under attack (strength decreased by >= 0.1).
    
    Groups should already be filtered to opponent groups. Returns:
    - attacked_groups_count: Number of groups under attack
    - attacked_groups_regions: List of regions where attacked groups are located
    - attacked_groups_head_locs: List of head stone locations of attacked groups
    - attacked_groups_strength_deltas: List of strength deltas for each attacked group
    """
    if not groups_before or not groups_after:
        return {
            "attacked_groups_count": 0,
            "attacked_groups_regions": [],
            "attacked_groups_head_locs": [],
            "attacked_groups_strength_deltas": [],
        }
    
    matched = _match_groups_by_iou(groups_before, groups_after, board)
    if not matched:
        return {
            "attacked_groups_count": 0,
            "attacked_groups_regions": [],
            "attacked_groups_head_locs": [],
            "attacked_groups_strength_deltas": [],
        }
    
    attacked_groups = []
    for bg, ag in matched:
        delta = ag.strength - bg.strength
        # Attack is defined as strength decrease >= 0.1
        if delta <= -0.1:
            attacked_groups.append((ag, delta))
    
    if not attacked_groups:
        return {
            "attacked_groups_count": 0,
            "attacked_groups_regions": [],
            "attacked_groups_head_locs": [],
            "attacked_groups_strength_deltas": [],
        }
    
    regions = []
    head_locs = []
    strength_deltas = []
    
    for group, delta in attacked_groups:
        region = _get_group_region(group, board)
        regions.append(region)
        head_locs.append(group.head)
        strength_deltas.append(float(delta))
    
    return {
        "attacked_groups_count": len(attacked_groups),
        "attacked_groups_regions": regions,
        "attacked_groups_head_locs": head_locs,
        "attacked_groups_strength_deltas": strength_deltas,
    }


def killing_attack(
    groups_before: List[Group], groups_after: List[Group], board: Board
) -> Tuple[bool, float]:
    """
    Check if move creates a killing attack (opp group transitions from alive to killed).
    
    Groups should already be filtered to opponent groups. Returns:
    - bool: True if any opponent group was alive before (strength > 0) and became killed after (strength <= 0)
    - float: Mean absolute strength of killed groups
    """
    if not groups_before or not groups_after:
        return False, 0.0
    matched = _match_groups_by_iou(groups_before, groups_after, board)
    killed = []
    for bg, ag in matched:
        # Check if group was alive before (strength > 0) and became killed after (strength <= 0)
        if bg.strength > 0 and ag.strength <= 0:
            killed.append(ag)
    if not killed:
        return False, 0.0
    return True, float(np.mean([abs(g.strength) for g in killed]))


# --- Main Analysis Function ---

def analyze_position_comprehensive(
    board: Board, 
    ownership: np.ndarray, 
    policy: np.ndarray,
    player: Optional[int] = None,
    move_loc: Optional[int] = None,
    last_move_loc: Optional[int] = None,
    before_ownership: Optional[np.ndarray] = None,
    before_board: Optional[Board] = None,
    ownership_frame_player: Optional[int] = None,
    pass_counterfactual_ownership: Optional[np.ndarray] = None,
    pass_counterfactual_frame_player: Optional[int] = None
) -> Dict[str, Any]:
    """
    Comprehensive position analysis returning tactical and territorial metrics.

    Args:
        board: Current board state after the move
        ownership: Ownership array (19x19) for current position
        policy: Policy array (362) for current position
        player: Player who made the move (defaults to board.pla)
        move_loc: Location of the move just played
        last_move_loc: Location of the previous move (for tenuki detection)
        before_ownership: Ownership array before the move (for delta calculations)
        before_board: Board state before the move (for group comparisons)
        ownership_frame_player: If set, ownership is already normalized to this player's perspective
        pass_counterfactual_ownership: Ownership array if player had passed instead of playing the move.
            Used for territory-related concepts to avoid "anticipatory ownership" issues where
            the pre-move ownership already reflects the model's expectation of the move.
            If provided, territory concepts (building, solidification, reduction, invasion)
            will compare pass_counterfactual vs actual ownership instead of before vs after.
        pass_counterfactual_frame_player: Player perspective for pass_counterfactual_ownership

    Returns:
        Dict with the following features:

        Territory Features (delta - change from before to after move):
            potential_territory: Change in weakly-owned empty points (ownership > TAU_POS)
            solid_territory: Change in strongly-owned empty points (ownership >= TAU_SOLID)

        Territory Change Features (delta - move effects):
            building_count: Empty points that became owned (neutral -> owned)
            building_intensity: Average ownership gain on building points
            solidification_count: Points that were owned and became stronger
            solidification_intensity: Average ownership gain on solidification points
            reduction_count: Opponent territory points that decreased in opponent ownership
            reduction_intensity: Average reduction amount
            invasion: True if move flipped opponent territory to own territory
            invasion_intensity: Average ownership swing on invaded points

        Group Strength Features (delta):
            group_strength_delta: Average change in ownership over all own groups' stones
            group_connectivity_delta: Average change in ownership of empty points near own groups
            max_group_strength_delta: Maximum single-group strength improvement
            max_group_connectivity_delta: Maximum single-group connectivity improvement

        Current Group Features (for the group containing the move):
            current_group_strength: Strength of the group containing the move after the move
            current_group_strength_delta: Change in strength of that specific group
            current_group_connectivity: Connectivity of the group containing the move after the move
            current_group_connectivity_delta: Change in connectivity of that specific group
            must_live: True if group would be dead under pass counterfactual but is alive after move
                       Only applies when move connects to existing stones (not new single-stone groups)
            counterfactual_group_strength: Strength the pre-existing stones would have if player passed

        Influence Features (delta):
            influence_count_delta: Change in count of adjacent empty points with favorable ownership
            influence_strength_delta: Change in average influence strength across all own groups

        Tactical Features:
            cut: Move separates 2+ opponent groups
            connection: Move connects 2+ own groups (using group enumeration)
            connection_strength_gain: Number of groups connected minus 1
            merged_groups_regions: List of regions where merged groups are located (when connection = True)
            merged_groups_head_locs: List of head stone locations of merged groups (when connection = True)
            extension: Move is adjacent to at least one own stone
            liberties: Liberty count of the group containing the played stone
            atari: Move puts at least one opponent group into atari
            creates_new_group: Move created a new separate group

        Attack Features:
            attack: True if average opponent group strength decreased
            avg_attack_intensity: Average decrease in opponent group strengths
            max_attack_intensity: Maximum decrease in any single opponent group strength
            attacked_groups_count: Number of opponent groups under attack (strength decreased >= 0.1)
            attacked_groups_regions: List of regions where attacked groups are located
            attacked_groups_head_locs: List of head stone locations of attacked groups
            attacked_groups_strength_deltas: List of strength deltas for each attacked group
            killing_attack: True if opponent group strength dropped to likely-dead level
            kill_intensity: Average absolute strength of killed groups
            reduce_aji: True if move reduced opponent's aji
            aji_reduction_intensity: Average aji reduction amount

        Sacrifice Features:
            direct_sacrifice: The played stone is in opponent territory AFTER the move (sacrifice stone)
            direct_sacrifice_intensity: Absolute ownership value of sacrificed stone
            indirect_sacrifice: Count of own stones that flipped to opponent territory
            indirect_sacrifice_intensity: Average ownership swing of sacrificed stones

        Policy Features:
            urgency: Dict of policy probability mass by region
            urgency_intensity: Dict of normalized urgency per region
            forcing: True if top move has >95% probability
            tenuki: Move is far from last move, ignoring local follow-up
            occupy_corner: First stone played in a corner area (corner was empty before)
            approaching_corner: Second stone in corner, responding to opponent's corner stone

        Regional Breakdowns:
            building_count_by_region, building_intensity_by_region
            solidification_count_by_region, solidification_intensity_by_region
            reduction_count_by_region, reduction_intensity_by_region

    Ownership normalization:
        - Raw ownership from get_model_outputs is from current player to move's perspective
        - ownership is from ownership_frame_player's perspective (pass the player to move when captured)
        - before_ownership is from before_board.pla's perspective (who was to move when captured)
        - The function normalizes both to the analyzing player's perspective internally
    """
    results: Dict[str, Any] = {}
    player = player if player is not None else board.pla
    is_pass = move_loc is not None and move_loc == Board.PASS_LOC
    
    # Track whether pass counterfactual is being used for territory concepts
    results["used_pass_counterfactual"] = pass_counterfactual_ownership is not None

    # Normalize ownership to current player's perspective
    frame = ownership_frame_player if ownership_frame_player is not None else Board.WHITE
    own_curr = normalize_ownership(ownership, frame, player)

    # Normalize before_ownership
    # Raw ownership from get_model_outputs is from current player to move's perspective
    # before_ownership was captured when before_board.pla was to play, so use that as the frame
    own_before = None
    if before_ownership is not None:
        if before_board is not None:
            # before_ownership is from before_board.pla's perspective
            own_before = normalize_ownership(before_ownership, before_board.pla, player)
        else:
            # Fallback: assume same as ownership_frame_player logic
            before_frame = Board.get_opp(frame) if ownership_frame_player is not None else Board.WHITE
            own_before = normalize_ownership(before_ownership, before_frame, player)

    # Normalize pass_counterfactual_ownership for territory concepts
    # This is the ownership if the player had passed instead of playing the actual move.
    # After a pass, it's the opponent's turn, so the ownership is from opponent's perspective.
    own_pass_counterfactual = None
    if pass_counterfactual_ownership is not None:
        if pass_counterfactual_frame_player is not None:
            own_pass_counterfactual = normalize_ownership(
                pass_counterfactual_ownership, pass_counterfactual_frame_player, player
            )
        else:
            # Default: after pass, it's opponent's turn (same as after actual move)
            own_pass_counterfactual = normalize_ownership(
                pass_counterfactual_ownership, Board.get_opp(player), player
            )
    
    # Determine which "before" ownership to use for all before/after comparisons
    # If pass counterfactual is available, use it to avoid anticipatory ownership issues
    # This affects both territory AND group features (group strength, attack analysis, etc.)
    own_baseline = own_pass_counterfactual if own_pass_counterfactual is not None else own_before

    # Urgency
    results["urgency"] = urgency_by_region(policy)
    results["urgency_intensity"] = urgency_intensity_by_region(policy)
    
    # Groups (using ownership-based grouping)
    groups = enumerate_groups(board, own_curr, player)
    compute_group_strengths(groups, own_curr, board)
    compute_group_connectivity(groups, own_curr, board)
    compute_group_influence(groups, own_curr, board)

    # Territory analysis
    # Use own_baseline (pass counterfactual if available, otherwise own_before)
    # This helps avoid "anticipatory ownership" where pre-move ownership already reflects
    # the model's expectation of the move being played.
    if own_baseline is not None and not is_pass:
        bc, bi = count_building_territory(own_baseline, own_curr, board)
        sc, si = solidify_territory_delta(own_baseline, own_curr, board)
        rc, ri = reduce_opponent_territory(own_baseline, own_curr, board)
        # Invasion uses own_before (not counterfactual) - we want to see actual board change
        inv, inv_i = invasion_effect(own_before if own_before is not None else own_baseline, own_curr, board, move_loc, groups)

        results.update({
            "building_count": bc, "building_intensity": bi,
            "solidification_count": sc, "solidification_intensity": si,
            "reduction_count": rc, "reduction_intensity": ri,
            "invasion": inv, "invasion_intensity": inv_i,
        })

        # Regional analysis
        bc_r, bi_r, sc_r, si_r = compute_territory_delta_by_region(own_baseline, own_curr, board)
        rc_r, ri_r = compute_reduction_by_region(own_baseline, own_curr, board)
        results.update({
            "building_count_by_region": bc_r, "building_intensity_by_region": bi_r,
            "solidification_count_by_region": sc_r, "solidification_intensity_by_region": si_r,
            "reduction_count_by_region": rc_r, "reduction_intensity_by_region": ri_r,
        })
    else:
        results.update({
            "building_count": 0, "building_intensity": 0.0,
            "solidification_count": 0, "solidification_intensity": 0.0,
            "reduction_count": 0, "reduction_intensity": 0.0,
            "invasion": False, "invasion_intensity": 0.0,
            "building_count_by_region": _empty_region_dict(0),
            "building_intensity_by_region": _empty_region_dict(0.0),
            "solidification_count_by_region": _empty_region_dict(0),
            "solidification_intensity_by_region": _empty_region_dict(0.0),
            "reduction_count_by_region": _empty_region_dict(0),
            "reduction_intensity_by_region": _empty_region_dict(0.0),
        })

    # Territory sizes (delta - change from baseline to after)
    # Use territory baseline (pass counterfactual if available) for consistent territory measurement
    if own_baseline is not None:
        territory_data = territory_sizes_with_delta(own_baseline, own_curr, board, player)
        results.update(territory_data)
    else:
        # No baseline available, so delta is 0
        results["potential_territory"] = 0
        results["solid_territory"] = 0
    
    # Sacrifices
    if move_loc is not None:
        # Use baseline ownership to check if location was in opponent territory before the move
        ds, ds_i = direct_sacrifice(move_loc, own_curr, board, before_ownership=own_baseline)
        results["direct_sacrifice"] = ds
        results["direct_sacrifice_intensity"] = ds_i
        # Keep backward compatibility
        results["sacrifice_intensity"] = ds_i
        if own_baseline is not None:
            # Pass before_board to detect captured stones that are no longer on current board
            ind, ind_i = indirect_sacrifice(own_baseline, own_curr, player, board, before_board)
            results["indirect_sacrifice"] = ind
            results["indirect_sacrifice_intensity"] = ind_i
        else:
            results["indirect_sacrifice"] = 0
            results["indirect_sacrifice_intensity"] = 0.0
    else:
        results.update({
            "direct_sacrifice": False, "direct_sacrifice_intensity": 0.0,
            "sacrifice_intensity": 0.0,  # Backward compatibility
            "indirect_sacrifice": 0, "indirect_sacrifice_intensity": 0.0,
        })

    # Tactical concepts
    if move_loc is not None and not is_pass:
        results["cut"] = is_cut_move(board, move_loc)
        # Connection check using group enumeration (requires before_board and baseline ownership)
        if before_board is not None and own_baseline is not None:
            conn, conn_gain, merged_regions, merged_head_locs = is_connection_move(board, move_loc, player, before_board, own_baseline, own_curr)
            results["merged_groups_regions"] = merged_regions
            results["merged_groups_head_locs"] = merged_head_locs
        else:
            conn, conn_gain = False, 0.0
            results["merged_groups_regions"] = []
            results["merged_groups_head_locs"] = []
        results["connection"] = conn
        results["connection_strength_gain"] = conn_gain
        results["extension"] = is_extension_move(board, move_loc, player)
        results["liberties"] = liberties_of_group(board, move_loc, groups)
        results["atari"] = atari_move(board, move_loc)
    else:
        results.update({
            "cut": False, "connection": False, "connection_strength_gain": 0.0,
            "merged_groups_regions": [], "merged_groups_head_locs": [],
            "extension": False, "liberties": 0, "atari": False,
        })

    results["forcing"] = is_forcing(policy) if not is_pass else False

    if last_move_loc is not None and move_loc is not None and not is_pass:
        move_idx = loc_to_xy(board, move_loc)[1] * 19 + loc_to_xy(board, move_loc)[0]
        results["tenuki"] = is_tenuki(move_idx, last_move_loc, policy, board)
    else:
        results["tenuki"] = False
    
    # Corner occupation features
    if move_loc is not None and not is_pass:
        results["occupy_corner"] = is_occupy_corner(board, before_board, move_loc, player)
        results["approaching_corner"] = is_approaching_corner(board, before_board, move_loc, player)
    else:
        results["occupy_corner"] = False
        results["approaching_corner"] = False
    
    # Attack analysis (uses baseline ownership for before/after comparisons)
    if own_baseline is not None:
        ra, ra_i = reduce_aji(own_baseline, own_curr, board, player, move_loc)
        results["reduce_aji"] = ra
        results["aji_reduction_intensity"] = ra_i

        if before_board is not None:
            groups_before = enumerate_groups(before_board, own_baseline, player)
            compute_group_strengths(groups_before, own_baseline, before_board)
            compute_group_connectivity(groups_before, own_baseline, before_board)
            compute_group_influence(groups_before, own_baseline, before_board)
            
            # Compute current group (the group containing the move) delta
            if move_loc is not None and not is_pass:
                (curr_str, curr_str_delta, curr_conn, curr_conn_delta,
                 curr_inf_count, curr_inf_count_delta, curr_inf_str, curr_inf_str_delta) = compute_current_group_delta(
                    move_loc, groups_before, groups, board
                )
                results["current_group_strength"] = curr_str
                results["current_group_strength_delta"] = curr_str_delta
                results["current_group_connectivity"] = curr_conn
                results["current_group_connectivity_delta"] = curr_conn_delta
                results["current_group_influence_count"] = curr_inf_count
                results["current_group_influence_count_delta"] = curr_inf_count_delta
                results["current_group_influence_strength"] = curr_inf_str
                results["current_group_influence_strength_delta"] = curr_inf_str_delta
                
                # "must_live" feature: True if group would be dead under counterfactual but is alive after move
                # This detects moves that are necessary to save a group from dying
                # Only applies to groups that existed before the move (not new single-stone groups)
                must_live = False
                counterfactual_strength = 0.0
                if own_pass_counterfactual is not None:
                    # Find the current group containing move_loc
                    current_group = None
                    for g in groups:
                        if move_loc in g.stones:
                            current_group = g
                            break
                    
                    if current_group is not None:
                        # Only consider stones that existed BEFORE the move (exclude the just-played stone)
                        # In counterfactual, the move_loc stone doesn't exist since player passed
                        pre_existing_stones = [loc for loc in current_group.stones if loc != move_loc]
                        
                        # Only compute must_live if there are pre-existing stones
                        # (i.e., the move connected to an existing group, not just a new single stone)
                        if pre_existing_stones:
                            stone_ownerships = []
                            for loc in pre_existing_stones:
                                y, x = board.loc_y(loc), board.loc_x(loc)
                                if 0 <= y < own_pass_counterfactual.shape[0] and 0 <= x < own_pass_counterfactual.shape[1]:
                                    stone_ownerships.append(own_pass_counterfactual[y, x])
                            
                            if stone_ownerships:
                                counterfactual_strength = float(np.mean(stone_ownerships))
                                # Group is "dead" under counterfactual if strength <= 0
                                # Group is "alive" after move if current strength > 0
                                must_live = (counterfactual_strength <= 0.0) and (curr_str > 0.0)
                
                results["must_live"] = must_live
                results["counterfactual_group_strength"] = counterfactual_strength
            else:
                results.update({
                    "current_group_strength": 0.0, "current_group_strength_delta": 0.0,
                    "current_group_connectivity": 0.0, "current_group_connectivity_delta": 0.0,
                    "current_group_influence_count": 0, "current_group_influence_count_delta": 0,
                    "current_group_influence_strength": 0.0, "current_group_influence_strength_delta": 0.0,
                    "must_live": False, "counterfactual_group_strength": 0.0,
                })
            
            # Also compute opponent groups for attack analysis
            opp = Board.get_opp(player)
            opp_own_curr = -own_curr  # Opponent's perspective (negated)
            opp_own_baseline = -own_baseline  # Use baseline for opponent too
            opp_groups = enumerate_groups(board, opp_own_curr, opp)
            opp_groups_before = enumerate_groups(before_board, opp_own_baseline, opp)
            compute_group_strengths(opp_groups, opp_own_curr, board)
            compute_group_connectivity(opp_groups, opp_own_curr, board)
            compute_group_influence(opp_groups, opp_own_curr, board)
            compute_group_strengths(opp_groups_before, opp_own_baseline, before_board)
            compute_group_connectivity(opp_groups_before, opp_own_baseline, before_board)
            compute_group_influence(opp_groups_before, opp_own_baseline, before_board)

            # Attack metrics (using opponent groups)
            is_atk, avg_atk, max_atk = attack_strength_delta(opp_groups_before, opp_groups, board)
            results["attack"] = is_atk
            results["avg_attack_intensity"] = avg_atk
            results["max_attack_intensity"] = max_atk

            # Get information about attacked groups (regions and head locations)
            attacked_info = get_attacked_groups_info(opp_groups_before, opp_groups, board)
            results["attacked_groups_count"] = attacked_info["attacked_groups_count"]
            results["attacked_groups_regions"] = attacked_info["attacked_groups_regions"]
            results["attacked_groups_head_locs"] = attacked_info["attacked_groups_head_locs"]
            results["attacked_groups_strength_deltas"] = attacked_info["attacked_groups_strength_deltas"]

            is_kill, kill_i = killing_attack(opp_groups_before, opp_groups, board)
            results["killing_attack"] = is_kill
            results["kill_intensity"] = kill_i
            results["creates_new_group"] = creates_new_group(groups_before, groups, move_loc, board)

            # Group strength and connectivity deltas - use individual group matching for accuracy
            avg_str_delta, max_str_delta, _ = compute_individual_group_strength_deltas(groups_before, groups, board)
            avg_conn_delta, max_conn_delta, _ = compute_individual_group_connectivity_deltas(groups_before, groups, board)
            
            # Influence delta - use accurate computation that avoids double-counting
            inf_count_delta, inf_str_delta = compute_influence_delta_accurate(
                groups_before, groups, own_baseline, own_curr, before_board, board
            )

            results["group_strength_delta"] = avg_str_delta
            results["group_connectivity_delta"] = avg_conn_delta
            results["max_group_strength_delta"] = max_str_delta
            results["max_group_connectivity_delta"] = max_conn_delta
            results["influence_count_delta"] = inf_count_delta
            results["influence_strength_delta"] = inf_str_delta
        else:
            results.update({
                "attack": False, "avg_attack_intensity": 0.0, "max_attack_intensity": 0.0,
                "attacked_groups_count": 0, "attacked_groups_regions": [], 
                "attacked_groups_head_locs": [], "attacked_groups_strength_deltas": [],
                "killing_attack": False, "kill_intensity": 0.0, "creates_new_group": False,
                "group_strength_delta": 0.0, "group_connectivity_delta": 0.0,
                "max_group_strength_delta": 0.0, "max_group_connectivity_delta": 0.0,
                "influence_count_delta": 0, "influence_strength_delta": 0.0,
                "current_group_strength": 0.0, "current_group_strength_delta": 0.0,
                "current_group_connectivity": 0.0, "current_group_connectivity_delta": 0.0,
                "current_group_influence_count": 0, "current_group_influence_count_delta": 0,
                "current_group_influence_strength": 0.0, "current_group_influence_strength_delta": 0.0,
                "must_live": False, "counterfactual_group_strength": 0.0,
            })
    else:
        results.update({
            "reduce_aji": False, "aji_reduction_intensity": 0.0,
            "attack": False, "avg_attack_intensity": 0.0, "max_attack_intensity": 0.0,
            "attacked_groups_count": 0, "attacked_groups_regions": [], 
            "attacked_groups_head_locs": [], "attacked_groups_strength_deltas": [],
            "killing_attack": False, "kill_intensity": 0.0, "creates_new_group": False,
            "group_strength_delta": 0.0, "group_connectivity_delta": 0.0,
            "max_group_strength_delta": 0.0, "max_group_connectivity_delta": 0.0,
            "influence_count_delta": 0, "influence_strength_delta": 0.0,
            "current_group_strength": 0.0, "current_group_strength_delta": 0.0,
            "current_group_connectivity": 0.0, "current_group_connectivity_delta": 0.0,
            "current_group_influence_count": 0, "current_group_influence_count_delta": 0,
            "current_group_influence_strength": 0.0, "current_group_influence_strength_delta": 0.0,
            "must_live": False, "counterfactual_group_strength": 0.0,
        })

    return results


# --- Backward Compatibility ---
TAU_CONN = TAU_POS
EPSILON_POL = 1e-12
reduce_opponent_territory_count = reduce_opponent_territory
compute_reduction_delta_by_region = compute_reduction_by_region


def weakening_territory_in_region(before: np.ndarray, after: np.ndarray, region: str, color: int) -> Tuple[int, float]:
    """Count weakening of opponent territory in specific region."""
    m = region_map(before.shape[0])
    mask = (m == region) & (before < -TAU_POS) & (after > before)
    count = int(np.sum(mask))
    intensity = float(np.mean(np.abs(before[mask] - after[mask]))) if count > 0 else 0.0
    return count, intensity


def leaving_weakness(before: np.ndarray, after: np.ndarray, color: int, board: Board) -> int:
    """Count intersections that flipped from own to opponent ownership."""
    mask = np.zeros((board.size, board.size), dtype=bool)
    for y in range(board.size):
        for x in range(board.size):
            if board.board[board.loc(x, y)] == color:
                mask[y, x] = True
    flipped = mask & (before > TAU_POS) & (after < -TAU_POS)
    return int(np.sum(flipped))


def compute_individual_group_strength_deltas(
    groups_before: List[Group], groups_after: List[Group], board: Board
) -> Tuple[float, float, float]:
    """
    Compute individual group strength deltas using IoU matching.
    
    Returns:
        (avg_delta, max_delta, min_delta) for all matched groups
    """
    matched = _match_groups_by_iou(groups_before, groups_after, board)
    if not matched:
        # No matched groups - check if there are new groups (after only)
        if groups_after and not groups_before:
            # All groups are new, their strength is the "delta"
            deltas = [g.strength for g in groups_after]
            return float(np.mean(deltas)), max(deltas), min(deltas)
        return 0.0, 0.0, 0.0
    
    deltas = [after.strength - before.strength for before, after in matched]
    
    # Also account for new groups (in after but not matched)
    matched_after_indices = {id(after) for _, after in matched}
    for g in groups_after:
        if id(g) not in matched_after_indices:
            # New group - its full strength is the delta
            deltas.append(g.strength)
    
    return float(np.mean(deltas)), max(deltas), min(deltas)


def compute_individual_group_connectivity_deltas(
    groups_before: List[Group], groups_after: List[Group], board: Board
) -> Tuple[float, float, float]:
    """
    Compute individual group connectivity deltas using IoU matching.
    
    Returns:
        (avg_delta, max_delta, min_delta) for all matched groups
    """
    matched = _match_groups_by_iou(groups_before, groups_after, board)
    if not matched:
        if groups_after and not groups_before:
            deltas = [g.connectivity for g in groups_after]
            return float(np.mean(deltas)), max(deltas), min(deltas)
        return 0.0, 0.0, 0.0
    
    deltas = [after.connectivity - before.connectivity for before, after in matched]
    
    matched_after_indices = {id(after) for _, after in matched}
    for g in groups_after:
        if id(g) not in matched_after_indices:
            deltas.append(g.connectivity)
    
    return float(np.mean(deltas)), max(deltas), min(deltas)


def compute_total_unique_influence(
    groups: List[Group], ownership: np.ndarray, board: Board
) -> Tuple[int, float, Set[int]]:
    """
    Compute total unique influence area across all groups (no double-counting).
    
    Returns:
        (total_count, avg_strength, set_of_influenced_locs)
    """
    size = board.size
    all_influenced_locs = set()
    all_ownership_values = []
    
    for g in groups:
        # BFS from all stones in the group (same logic as compute_group_influence)
        visited = set()
        queue = []
        for loc in g.stones:
            queue.append(loc)
            visited.add(loc)
        
        while queue:
            loc = queue.pop(0)
            x, y = loc_to_xy(board, loc)
            
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if not in_bounds(nx, ny, size):
                    continue
                
                nloc = xy_to_loc(board, nx, ny)
                if nloc in visited:
                    continue
                
                v = ownership[ny, nx]
                if v < TAU_POS:
                    continue
                
                visited.add(nloc)
                
                if board.board[nloc] == Board.EMPTY or board.board[nloc] != g.color:
                    # Track unique influenced points (deduplicated across groups)
                    if nloc not in all_influenced_locs:
                        all_influenced_locs.add(nloc)
                        all_ownership_values.append(v)
                    queue.append(nloc)
                else:
                    queue.append(nloc)
    
    total_count = len(all_influenced_locs)
    avg_strength = float(np.mean(all_ownership_values)) if all_ownership_values else 0.0
    return total_count, avg_strength, all_influenced_locs


def compute_influence_delta_accurate(
    groups_before: List[Group], groups_after: List[Group],
    own_before: np.ndarray, own_after: np.ndarray,
    before_board: Board, after_board: Board
) -> Tuple[int, float]:
    """
    Compute influence delta accurately by counting unique influenced points.
    
    Returns:
        (count_delta, strength_delta)
    """
    before_count, before_str, _ = compute_total_unique_influence(groups_before, own_before, before_board)
    after_count, after_str, _ = compute_total_unique_influence(groups_after, own_after, after_board)
    
    return after_count - before_count, after_str - before_str


def compute_group_strength_delta_by_region(
    groups_before: List[Group], groups_after: List[Group], board: Board
) -> Dict[str, float]:
    """Compute change in group strengths by region."""
    before_by_r = {r: [] for r in REGIONS}
    after_by_r = {r: [] for r in REGIONS}
    for g in groups_before:
        before_by_r[_get_group_region(g, board)].append(g)
    for g in groups_after:
        after_by_r[_get_group_region(g, board)].append(g)
    deltas = {}
    for r in REGIONS:
        b_str = float(np.mean([g.strength for g in before_by_r[r]])) if before_by_r[r] else 0.0
        a_str = float(np.mean([g.strength for g in after_by_r[r]])) if after_by_r[r] else 0.0
        deltas[r] = a_str - b_str
    return deltas


def compute_group_connectivity_delta_by_region(
    groups_before: List[Group], groups_after: List[Group], board: Board
) -> Dict[str, float]:
    """Compute change in group connectivity by region."""
    before_by_r = {r: [] for r in REGIONS}
    after_by_r = {r: [] for r in REGIONS}
    for g in groups_before:
        before_by_r[_get_group_region(g, board)].append(g)
    for g in groups_after:
        after_by_r[_get_group_region(g, board)].append(g)
    deltas = {}
    for r in REGIONS:
        b_conn = float(np.mean([g.connectivity for g in before_by_r[r]])) if before_by_r[r] else 0.0
        a_conn = float(np.mean([g.connectivity for g in after_by_r[r]])) if after_by_r[r] else 0.0
        deltas[r] = a_conn - b_conn
    return deltas


def compute_influence_delta_by_region(
    groups_before: List[Group], groups_after: List[Group], board: Board
) -> Tuple[Dict[str, int], Dict[str, float]]:
    """Compute change in influence by region."""
    before_by_r = {r: [] for r in REGIONS}
    after_by_r = {r: [] for r in REGIONS}
    for g in groups_before:
        before_by_r[_get_group_region(g, board)].append(g)
    for g in groups_after:
        after_by_r[_get_group_region(g, board)].append(g)
    count_d, str_d = {}, {}
    for r in REGIONS:
        b_cnt = sum(g.influence_area for g in before_by_r[r])
        a_cnt = sum(g.influence_area for g in after_by_r[r])
        count_d[r] = a_cnt - b_cnt
        b_str = float(np.mean([g.influence_strength for g in before_by_r[r]])) if before_by_r[r] else 0.0
        a_str = float(np.mean([g.influence_strength for g in after_by_r[r]])) if after_by_r[r] else 0.0
        str_d[r] = a_str - b_str
    return count_d, str_d


__all__ = [
    "Group", "REGIONS", "TAU_POS", "TAU_SOLID", "TAU_CONN", "EPSILON_POL", "TAU_GROUP_BELONGING",
    "loc_to_xy", "xy_to_loc", "in_bounds",
    "classify_region", "region_map", "normalize_ownership",
    "enumerate_groups", "compute_group_strengths", "compute_group_connectivity", 
    "compute_group_influence", "count_building_territory", "solidify_territory_delta",
    "reduce_opponent_territory", "reduce_opponent_territory_count",
    "invasion_effect", "weakening_territory_in_region", "leaving_weakness",
    "territory_sizes", "territory_sizes_with_delta", "direct_sacrifice", "indirect_sacrifice",
    "compute_territory_delta_by_region", "compute_reduction_by_region", "compute_reduction_delta_by_region",
    "compute_group_strength_delta_by_region", "compute_group_connectivity_delta_by_region",
    "compute_influence_delta_by_region",
    "urgency_by_region", "urgency_intensity_by_region", "is_forcing", "is_tenuki",
    "is_cut_move", "is_connection_move", "is_extension_move",
    "liberties_of_group", "atari_move", "creates_new_group",
    "find_group_containing", "compute_current_group_delta",
    "is_occupy_corner", "is_approaching_corner", "CORNER_REGIONS",
    "reduce_aji", "attack_strength_delta", "get_attacked_groups_info", "killing_attack",
    "analyze_position_comprehensive",
]
