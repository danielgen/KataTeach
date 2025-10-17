#!/usr/bin/env python3
"""
Comprehensive spatial concepts and metrics for Go board analysis.

This module implements the Go Explainability Ontology with enhanced features:

SPATIAL FOUNDATIONS:
- Board coordinates: aa (top-left/A19) to ss (bottom-right/S1)
- Regions: corner_tl, corner_tr, corner_bl, corner_br, side_left, side_right, side_top, side_bottom, center
- L1 distance-based connectivity (radius ≤ 2)

OWNERSHIP AND COLOR CONVENTIONS:
- Ownership map from KataGo (Black perspective)
- Thresholds: TAU_POS=0.1 (weak ownership), TAU_SOLID=0.7 (solid territory), TAU_CONN=0.1 (connectivity)

GROUP FORMATION AND ATTRIBUTES:
- Deterministic connection (adjacent stones) OR ownership-based connection (≥TAU_CONN)
- Group strength: average ownership over group stones
- Group connectivity: average ownership of empty intersections within L1 ≤ 2 radius
- Group influence area and strength

TERRITORY AND TRANSITIONS (with intensity metrics):
- Building territory: empty → own ownership ≥TAU_POS (with building_intensity)
- Solidifying territory: increase existing ownership values (with solidification_intensity)
- Reducing territory: reduce opponent's owned intersections (with reduction_intensity)
- Weakening territory: reduce opponent's average ownership in area (with weakening_intensity)
- Invasion: reduce opponent + increase own territory (with invasion_intensity)
- Leaving weakness: own → opponent ownership
- Direct/indirect sacrifice (with sacrifice_intensity)

TACTICAL RELATIONS (with intensity metrics):
- Cut: move that increases disconnected components among opponent stones
- Connection: merges groups or increases connectivity (with connection_strength_gain)
- Extension: move adjacent to existing own stone
- Liberties: number of empty orthogonal intersections adjacent to group
- Atari: opponent group with exactly one liberty
- Attack: decreases opponent group strength (with attack_intensity)
- Killing attack: results in mean own ownership ≥0.5 over opponent stones (with kill_intensity)
- Reduce aji: lowers opponent group's local connection ≥0.05 (with aji_reduction_intensity)

POLICY, URGENCY, AND INTENT:
- Only move: policy assigns nonzero probability to only one candidate
- Urgency by region: sum of policy probabilities within region (with urgency_intensity)
- Tenuki: different region + policy mass remains near previous region
- Rough intent: policy move → ownership effect simulation

COMPUTED FEATURES:
- Region-based deltas: group_strength_delta[region], group_connectivity_delta[region], etc.
- Aggregate totals: group_strength_delta, group_connectivity_delta, etc.
- Creates new group: true if number of own groups increases
- All features include both boolean/count flags and intensity metrics
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set, Iterable, Any
import json

import numpy as np

# Allow importing sibling python modules (board)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "python"))

from board import Board


# -------------------------
# Configuration thresholds
# -------------------------

TAU_POS = 0.10            # Weak ownership threshold (presence threshold)
TAU_SOLID = 0.70          # Solid territory threshold
TAU_CONN = 0.10           # Connectivity threshold
EPSILON_POL = 1e-12       # Policy probability threshold

# Legacy constants for backward compatibility
OWN_MIN = TAU_POS
TERRITORY_SOLID = TAU_SOLID


@dataclass
class Group:
    color: int                 # Board.BLACK or Board.WHITE
    head: int                  # representative loc (board.group_head)
    stones: List[int]          # list of locs belonging to the group
    liberties: int             # number of liberties
    bbox: Tuple[int,int,int,int]  # (min_x, min_y, max_x, max_y)
    strength: float            # average ownership value on group stones (signed from perspective of 'color')
    connectivity: float        # average ownership of empty intersections within bounds
    influence_area: int        # count of own ownership around group
    influence_strength: float  # average ownership around group


# -------------------------
# Coordinate and Region Functions
# -------------------------

def xy_to_loc(board: Board, x: int, y: int) -> int:
    """Convert (x,y) coordinates to board location."""
    return board.loc(x, y)


def loc_to_xy(board: Board, loc: int) -> Tuple[int, int]:
    """Convert board location to (x,y) coordinates."""
    return board.loc_x(loc), board.loc_y(loc)


def in_bounds(x: int, y: int, size: int = 19) -> bool:
    """Check if coordinates are within board bounds."""
    return 0 <= x < size and 0 <= y < size


def classify_region(x: int, y: int, size: int = 19) -> str:
    """
    Classify region into 9 distinct areas:
    - 4 Corners: top-left, top-right, bottom-left, bottom-right
    - 4 Sides: left, right, upper, lower
    - 1 Center: middle area
    
    Using bands a-f (0-5) and n-s (size-6 to size-1) for corner/side boundaries
    """
    corner_band = set(range(0, 6)) | set(range(size-6, size))  # a-f and n-s
    in_x_corner = x in corner_band
    in_y_corner = y in corner_band
    
    # Determine corner regions
    if in_x_corner and in_y_corner:
        if x < 6 and y < 6:  # top-left
            return "corner_tl"
        elif x >= size-6 and y < 6:  # top-right
            return "corner_tr"
        elif x < 6 and y >= size-6:  # bottom-left
            return "corner_bl"
        elif x >= size-6 and y >= size-6:  # bottom-right
            return "corner_br"
    
    # Determine side regions
    elif in_x_corner or in_y_corner:
        if in_x_corner and not in_y_corner:  # left or right side
            if x < 6:
                return "side_left"
            else:
                return "side_right"
        elif in_y_corner and not in_x_corner:  # upper or lower side
            if y < 6:
                return "side_top"
            else:
                return "side_bottom"
    
    # Center region
    return "center"


def region_map(size: int = 19) -> np.ndarray:
    """Create a map of regions for the board."""
    m = np.empty((size, size), dtype=object)
    for y in range(size):
        for x in range(size):
            m[y, x] = classify_region(x, y, size)
    return m


# -------------------------
# Group Analysis Functions
# -------------------------

def enumerate_groups_deterministic(board: Board) -> List[Group]:
    """Enumerate groups using deterministic stone connections."""
    seen_heads: Set[int] = set()
    groups: List[Group] = []
    size = board.size
    
    for y in range(size):
        for x in range(size):
            loc = board.loc(x, y)
            stone = board.board[loc]
            if stone != Board.BLACK and stone != Board.WHITE:
                continue
            head = board.group_head[loc]
            if head in seen_heads:
                continue
            seen_heads.add(head)
            
            # Walk the circular linked list to gather stones
            stones: List[int] = []
            cur = loc
            while True:
                if board.group_head[cur] == head:
                    stones.append(cur)
                cur = board.group_next[cur]
                if cur == loc:
                    break
            
            # Compute bbox
            xs = [board.loc_x(s) for s in stones]
            ys = [board.loc_y(s) for s in stones]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            
            groups.append(Group(
                color=stone,
                head=head,
                stones=stones,
                liberties=board.group_liberty_count[head],
                bbox=bbox,
                strength=0.0,
                connectivity=0.0,
                influence_area=0,
                influence_strength=0.0,
            ))
    
    return groups


def enumerate_groups_ownership(board: Board, ownership: np.ndarray, color: int) -> List[Group]:
    """Enumerate groups using ownership map (>=0.1 threshold) but only where actual stones exist."""
    size = board.size
    visited = np.zeros((size, size), dtype=bool)
    groups: List[Group] = []
    
    def flood_fill(x: int, y: int) -> List[Tuple[int, int]]:
        """Flood fill to find connected ownership area with actual stones."""
        stack = [(x, y)]
        group_stones = []
        
        while stack:
            cx, cy = stack.pop()
            if (not in_bounds(cx, cy, size) or visited[cy, cx] or 
                abs(ownership[cy, cx]) < TAU_POS or 
                (ownership[cy, cx] > 0) != (color == Board.BLACK)):
                continue
            
            # Only include if there's actually a stone at this location
            loc = board.loc(cx, cy)
            if board.board[loc] != color:
                continue
                
            visited[cy, cx] = True
            group_stones.append((cx, cy))
            
            # Add neighbors
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                stack.append((cx + dx, cy + dy))
        
        return group_stones
    
    for y in range(size):
        for x in range(size):
            loc = board.loc(x, y)
            # Only consider positions that have actual stones of the right color
            if (board.board[loc] == color and 
                not visited[y, x] and 
                abs(ownership[y, x]) >= TAU_POS and
                (ownership[y, x] > 0) == (color == Board.BLACK)):
                
                group_stones = flood_fill(x, y)
                if group_stones:
                    # Convert to board locations
                    stones = [board.loc(cx, cy) for cx, cy in group_stones]
                    xs, ys = zip(*group_stones)
                    bbox = (min(xs), min(ys), max(xs), max(ys))
                    
                    # Calculate liberties (simplified)
                    liberties = 0
                    for cx, cy in group_stones:
                        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                            nx, ny = cx + dx, cy + dy
                            if (in_bounds(nx, ny, size) and 
                                board.board[board.loc(nx, ny)] == Board.EMPTY):
                                liberties += 1
                    
                    groups.append(Group(
                        color=color,
                        head=stones[0],  # Use first stone as head
                        stones=stones,
                        liberties=liberties,
                        bbox=bbox,
                        strength=0.0,
                        connectivity=0.0,
                        influence_area=0,
                        influence_strength=0.0,
                    ))
    
    return groups


def compute_group_strengths(groups: List[Group], ownership: np.ndarray, player_perspective: int, board: Board) -> None:
    """Compute group strength as average ownership of group stones."""
    for g in groups:
        vals: List[float] = []
        sign = +1.0 if g.color == player_perspective else -1.0
        for loc in g.stones:
            x, y = loc_to_xy(board, loc)
            vals.append(sign * float(ownership[y, x]))
        g.strength = float(np.mean(vals)) if vals else 0.0


def compute_group_connectivity(groups: List[Group], ownership: np.ndarray, board: Board) -> None:
    """Compute group connectivity as average ownership of empty intersections within L1 ≤ 2 radius."""
    for g in groups:
        sign = 1.0 if g.color == Board.BLACK else -1.0
        vals: List[float] = []
        
        # For each stone in group, check L1 ≤ 2 radius
        checked = set()
        for loc in g.stones:
            x0, y0 = loc_to_xy(board, loc)
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if abs(dx) + abs(dy) > 2:  # L1 distance check
                        continue
                    nx, ny = x0 + dx, y0 + dy
                    if not in_bounds(nx, ny, board.size):
                        continue
                    nloc = xy_to_loc(board, nx, ny)
                    if nloc in checked:
                        continue
                    checked.add(nloc)
                    if board.board[nloc] == Board.EMPTY:
                        v = ownership[ny, nx] * sign
                        vals.append(v)
        
        g.connectivity = float(np.mean(vals)) if vals else 0.0


def compute_group_influence(groups: List[Group], ownership: np.ndarray, board: Board) -> None:
    """Compute group influence area and strength."""
    for g in groups:
        sign = 1.0 if g.color == Board.BLACK else -1.0
        influence_points = []
        
        # Find all points around the group with same-sign ownership
        for loc in g.stones:
            x, y = loc_to_xy(board, loc)
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                nx, ny = x + dx, y + dy
                if (in_bounds(nx, ny, board.size) and 
                    board.board[xy_to_loc(board, nx, ny)] == Board.EMPTY and
                    ownership[ny, nx] * sign > TAU_POS):
                    influence_points.append(ownership[ny, nx] * sign)
        
        g.influence_area = len(influence_points)
        g.influence_strength = float(np.mean(influence_points)) if influence_points else 0.0


# -------------------------
# Territory Analysis Functions
# -------------------------

def count_building_territory(before: np.ndarray, after: np.ndarray, color: int) -> Tuple[int, float]:
    """Count intersections that changed from empty (<0.1) to own ownership (>0.1) and compute intensity."""
    same_sign = (1 if color == Board.BLACK else -1)
    prev = before * same_sign
    post = after * same_sign
    
    # Find intersections that were built
    built_mask = (np.abs(prev) < TAU_POS) & (post > TAU_POS)
    count = int(np.sum(built_mask))
    
    # Compute intensity as mean ownership increase of built intersections
    if count > 0:
        intensity = float(np.mean(post[built_mask] - prev[built_mask]))
    else:
        intensity = 0.0
    
    return count, intensity


def solidify_territory_delta(before: np.ndarray, after: np.ndarray, color: int) -> Tuple[int, float]:
    """Calculate increase in ownership values of previously owned intersections and compute intensity."""
    same_sign = (1 if color == Board.BLACK else -1)
    prev = before * same_sign
    post = after * same_sign
    owned_mask = prev > TAU_POS
    
    # Find intersections that were solidified (owned before and after, with increase)
    solidified_mask = owned_mask & (post > prev)
    count = int(np.sum(solidified_mask))
    
    # Compute intensity as mean ownership gain across solidified points
    if count > 0:
        intensity = float(np.mean(post[solidified_mask] - prev[solidified_mask]))
    else:
        intensity = 0.0
    
    return count, intensity


def reduce_opponent_territory_count(before: np.ndarray, after: np.ndarray, color: int) -> Tuple[int, float]:
    """Count reduction in opponent's owned intersections and compute intensity."""
    opp_sign = (-1 if color == Board.BLACK else 1)
    prev = before * opp_sign
    post = after * opp_sign
    
    # Find intersections that were reduced (opponent owned before, not after)
    reduced_mask = (prev > TAU_POS) & (post <= TAU_POS)
    count = int(np.sum(reduced_mask))
    
    # Compute intensity as mean ownership change magnitude of affected points
    if count > 0:
        intensity = float(np.mean(np.abs(post[reduced_mask] - prev[reduced_mask])))
    else:
        intensity = 0.0
    
    return count, intensity


def invasion_effect(before: np.ndarray, after: np.ndarray, color: int) -> Tuple[bool, float]:
    """Calculate invasion effect: reduced opponent territory + built own territory."""
    built_count, built_intensity = count_building_territory(before, after, color)
    reduced_count, reduced_intensity = reduce_opponent_territory_count(before, after, color)
    
    # Invasion occurs if both building and reduction happened
    is_invasion = (built_count > 0) and (reduced_count > 0)
    
    # Invasion intensity is the combined effect
    invasion_intensity = built_intensity + reduced_intensity
    
    return is_invasion, invasion_intensity


def weakening_territory_in_region(before: np.ndarray, after: np.ndarray, region: str, color: int) -> Tuple[int, float]:
    """Calculate weakening of opponent territory in specific region."""
    m = region_map(before.shape[0])
    opp_sign = (-1 if color == Board.BLACK else 1)
    prev = before * opp_sign
    post = after * opp_sign
    mask = (m == region)
    
    if not np.any(mask):
        return 0, 0.0
    
    # Count points that were weakened (opponent owned before, weakened after)
    weakened_mask = mask & (prev > TAU_POS) & (post < prev)
    count = int(np.sum(weakened_mask))
    
    # Intensity is the average reduction magnitude
    if count > 0:
        intensity = float(np.mean(prev[weakened_mask] - post[weakened_mask]))
    else:
        intensity = 0.0
    
    return count, intensity


def leaving_weakness(before: np.ndarray, after: np.ndarray, color: int) -> int:
    """Count intersections that flipped from own to opponent ownership."""
    own_sign = (1 if color == Board.BLACK else -1)
    prev = before * own_sign
    post = after * own_sign
    return int(np.sum((prev > TAU_POS) & (post < -TAU_POS)))


def territory_sizes(ownership: np.ndarray, color: int) -> Tuple[int, int]:
    """Calculate potential and solid territory sizes."""
    sign = (1 if color == Board.BLACK else -1)
    v = ownership * sign
    potential = int(np.sum(v > TAU_POS) - np.sum(v >= TAU_SOLID))
    solid = int(np.sum(v >= TAU_SOLID))
    return potential, solid


def direct_sacrifice(move_loc: int, after: np.ndarray, color: int, board: Board) -> Tuple[bool, float]:
    """Check if the played stone becomes opponent's territory and compute sacrifice intensity."""
    if move_loc == Board.PASS_LOC:
        return False, 0.0
    x, y = loc_to_xy(board, move_loc)
    sign = (1 if color == Board.BLACK else -1)
    
    is_sacrifice = bool((after[y, x] * sign) < -TAU_POS)
    sacrifice_intensity = abs(after[y, x] * sign) if is_sacrifice else 0.0
    
    return is_sacrifice, sacrifice_intensity


def indirect_sacrifice(before: np.ndarray, after: np.ndarray, color: int, board: Board) -> Tuple[int, float]:
    """Check if any own stone becomes opponent's territory and compute intensity."""
    sign = (1 if color == Board.BLACK else -1)
    prev = before * sign
    post = after * sign
    
    # Find stones that became opponent's territory
    sacrificed_mask = (prev > TAU_POS) & (post < -TAU_POS)
    count = int(np.sum(sacrificed_mask))
    
    # Compute intensity as mean ownership loss of affected stones
    if count > 0:
        intensity = float(np.mean(prev[sacrificed_mask] - post[sacrificed_mask]))
    else:
        intensity = 0.0
    
    return count, intensity


# -------------------------
# Region-Based Delta Functions
# -------------------------

def compute_group_strength_delta_by_region(
    groups_before: List[Group], 
    groups_after: List[Group], 
    board: Board
) -> Dict[str, float]:
    """Compute change in group strengths by region."""
    regions = ["corner_tl", "corner_tr", "corner_bl", "corner_br",
               "side_left", "side_right", "side_top", "side_bottom", "center"]
    deltas = {region: 0.0 for region in regions}
    
    # Group groups by region
    before_by_region = {region: [] for region in regions}
    after_by_region = {region: [] for region in regions}
    
    for g in groups_before:
        # Find dominant region of group
        region_counts = {region: 0 for region in regions}
        for loc in g.stones:
            x, y = loc_to_xy(board, loc)
            region = classify_region(x, y, board.size)
            region_counts[region] += 1
        dominant_region = max(region_counts, key=region_counts.get)
        before_by_region[dominant_region].append(g)
    
    for g in groups_after:
        # Find dominant region of group
        region_counts = {region: 0 for region in regions}
        for loc in g.stones:
            x, y = loc_to_xy(board, loc)
            region = classify_region(x, y, board.size)
            region_counts[region] += 1
        dominant_region = max(region_counts, key=region_counts.get)
        after_by_region[dominant_region].append(g)
    
    # Compute deltas by region
    for region in regions:
        before_strength = np.mean([g.strength for g in before_by_region[region]]) if before_by_region[region] else 0.0
        after_strength = np.mean([g.strength for g in after_by_region[region]]) if after_by_region[region] else 0.0
        deltas[region] = float(after_strength - before_strength)
    
    return deltas


def compute_group_connectivity_delta_by_region(
    groups_before: List[Group], 
    groups_after: List[Group], 
    board: Board
) -> Dict[str, float]:
    """Compute change in group connectivity by region."""
    regions = ["corner_tl", "corner_tr", "corner_bl", "corner_br",
               "side_left", "side_right", "side_top", "side_bottom", "center"]
    deltas = {region: 0.0 for region in regions}
    
    # Group groups by region (same logic as above)
    before_by_region = {region: [] for region in regions}
    after_by_region = {region: [] for region in regions}
    
    for g in groups_before:
        region_counts = {region: 0 for region in regions}
        for loc in g.stones:
            x, y = loc_to_xy(board, loc)
            region = classify_region(x, y, board.size)
            region_counts[region] += 1
        dominant_region = max(region_counts, key=region_counts.get)
        before_by_region[dominant_region].append(g)
    
    for g in groups_after:
        region_counts = {region: 0 for region in regions}
        for loc in g.stones:
            x, y = loc_to_xy(board, loc)
            region = classify_region(x, y, board.size)
            region_counts[region] += 1
        dominant_region = max(region_counts, key=region_counts.get)
        after_by_region[dominant_region].append(g)
    
    # Compute deltas by region
    for region in regions:
        before_connectivity = np.mean([g.connectivity for g in before_by_region[region]]) if before_by_region[region] else 0.0
        after_connectivity = np.mean([g.connectivity for g in after_by_region[region]]) if after_by_region[region] else 0.0
        deltas[region] = float(after_connectivity - before_connectivity)
    
    return deltas


def compute_influence_delta_by_region(
    groups_before: List[Group], 
    groups_after: List[Group], 
    board: Board
) -> Tuple[Dict[str, int], Dict[str, float]]:
    """Compute change in influence count and strength by region."""
    regions = ["corner_tl", "corner_tr", "corner_bl", "corner_br",
               "side_left", "side_right", "side_top", "side_bottom", "center"]
    count_deltas = {region: 0 for region in regions}
    strength_deltas = {region: 0.0 for region in regions}
    
    # Group groups by region (same logic as above)
    before_by_region = {region: [] for region in regions}
    after_by_region = {region: [] for region in regions}
    
    for g in groups_before:
        region_counts = {region: 0 for region in regions}
        for loc in g.stones:
            x, y = loc_to_xy(board, loc)
            region = classify_region(x, y, board.size)
            region_counts[region] += 1
        dominant_region = max(region_counts, key=region_counts.get)
        before_by_region[dominant_region].append(g)
    
    for g in groups_after:
        region_counts = {region: 0 for region in regions}
        for loc in g.stones:
            x, y = loc_to_xy(board, loc)
            region = classify_region(x, y, board.size)
            region_counts[region] += 1
        dominant_region = max(region_counts, key=region_counts.get)
        after_by_region[dominant_region].append(g)
    
    # Compute deltas by region
    for region in regions:
        before_count = sum(g.influence_area for g in before_by_region[region])
        after_count = sum(g.influence_area for g in after_by_region[region])
        count_deltas[region] = after_count - before_count
        
        before_strength = np.mean([g.influence_strength for g in before_by_region[region]]) if before_by_region[region] else 0.0
        after_strength = np.mean([g.influence_strength for g in after_by_region[region]]) if after_by_region[region] else 0.0
        strength_deltas[region] = float(after_strength - before_strength)
    
    return count_deltas, strength_deltas


def compute_territory_delta_by_region(
    before: np.ndarray, 
    after: np.ndarray, 
    color: int
) -> Tuple[Dict[str, int], Dict[str, float], Dict[str, int], Dict[str, float]]:
    """Compute building and solidification deltas by region."""
    regions = ["corner_tl", "corner_tr", "corner_bl", "corner_br",
               "side_left", "side_right", "side_top", "side_bottom", "center"]
    
    building_count = {region: 0 for region in regions}
    building_intensity = {region: 0.0 for region in regions}
    solidification_count = {region: 0 for region in regions}
    solidification_intensity = {region: 0.0 for region in regions}
    
    same_sign = (1 if color == Board.BLACK else -1)
    prev = before * same_sign
    post = after * same_sign
    
    for y in range(before.shape[0]):
        for x in range(before.shape[1]):
            region = classify_region(x, y, before.shape[0])
            
            # Building territory
            if (np.abs(prev[y, x]) < TAU_POS) and (post[y, x] > TAU_POS):
                building_count[region] += 1
                building_intensity[region] += post[y, x] - prev[y, x]
            
            # Solidification
            if (prev[y, x] > TAU_POS) and (post[y, x] > prev[y, x]):
                solidification_count[region] += 1
                solidification_intensity[region] += post[y, x] - prev[y, x]
    
    # Convert intensity sums to averages
    for region in regions:
        if building_count[region] > 0:
            building_intensity[region] /= building_count[region]
        if solidification_count[region] > 0:
            solidification_intensity[region] /= solidification_count[region]
    
    return building_count, building_intensity, solidification_count, solidification_intensity


def compute_reduction_delta_by_region(
    before: np.ndarray, 
    after: np.ndarray, 
    color: int
) -> Tuple[Dict[str, int], Dict[str, float]]:
    """Compute reduction deltas by region."""
    regions = ["corner_tl", "corner_tr", "corner_bl", "corner_br",
               "side_left", "side_right", "side_top", "side_bottom", "center"]
    
    reduction_count = {region: 0 for region in regions}
    reduction_intensity = {region: 0.0 for region in regions}
    
    opp_sign = (-1 if color == Board.BLACK else 1)
    prev = before * opp_sign
    post = after * opp_sign
    
    for y in range(before.shape[0]):
        for x in range(before.shape[1]):
            region = classify_region(x, y, before.shape[0])
            
            # Reduction
            if (prev[y, x] > TAU_POS) and (post[y, x] <= TAU_POS):
                reduction_count[region] += 1
                reduction_intensity[region] += abs(post[y, x] - prev[y, x])
    
    # Convert intensity sums to averages
    for region in regions:
        if reduction_count[region] > 0:
            reduction_intensity[region] /= reduction_count[region]
    
    return reduction_count, reduction_intensity


# -------------------------
# Policy and Move Analysis Functions
# -------------------------

def urgency_by_region(policy: np.ndarray) -> Dict[str, float]:
    """Calculate urgency as sum of policy mass by region."""
    urg: Dict[str, float] = {
        "corner_tl": 0.0,
        "corner_tr": 0.0,
        "corner_bl": 0.0,
        "corner_br": 0.0,
        "side_left": 0.0,
        "side_right": 0.0,
        "side_top": 0.0,
        "side_bottom": 0.0,
        "center": 0.0
    }
    size = 19
    for y in range(size):
        for x in range(size):
            idx = y * size + x
            if idx < len(policy):
                r = classify_region(x, y, size)
                urg[r] += float(policy[idx])
    return urg


def urgency_intensity_by_region(policy: np.ndarray) -> Dict[str, float]:
    """Calculate urgency intensity as normalized share of total policy mass by region."""
    urg = urgency_by_region(policy)
    total_policy = sum(urg.values())
    
    if total_policy > 0:
        return {region: urg[region] / total_policy for region in urg}
    else:
        return {region: 0.0 for region in urg}


def is_cut_move(board: Board, move_loc: int) -> bool:
    """Check if move creates a cut (w-b/b-w configuration separating groups)."""
    if move_loc == Board.PASS_LOC:
        return False
    pla = board.pla
    opp = Board.get_opp(pla)
    x, y = loc_to_xy(board, move_loc)
    adj_opp_heads: Set[int] = set()
    
    for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
        nx, ny = x + dx, y + dy
        if not in_bounds(nx, ny, board.size):
            continue
        nloc = xy_to_loc(board, nx, ny)
        if board.board[nloc] == opp:
            adj_opp_heads.add(board.group_head[nloc])
    
    return len(adj_opp_heads) >= 2


def is_only_move(policy: np.ndarray, eps: float = EPSILON_POL) -> bool:
    """Check if policy has only one non-zero value."""
    non_zero = np.sum(policy > eps)
    return non_zero == 1


def rough_intent_effects(ownership: np.ndarray, policy: np.ndarray, color: int, threshold: float = 0.001) -> Dict[int, Dict[str, float]]:
    """Calculate rough intent effects for candidate moves only (policy > threshold)."""
    size = 19
    sign = (1 if color == Board.BLACK else -1)
    effects: Dict[int, Dict[str, float]] = {}
    
    for y in range(size):
        for x in range(size):
            idx = y * size + x
            if idx >= len(policy) or policy[idx] <= threshold:
                continue
            
            # Simulate placing stone with ownership=1.0
            after = ownership.copy()
            after[y, x] = 1.0 * sign
            
            # Simple local smoothing (cross shape)
            for dx, dy, w in [(1,0,0.5), (-1,0,0.5), (0,1,0.5), (0,-1,0.5)]:
                nx, ny = x + dx, y + dy
                if in_bounds(nx, ny, size):
                    after[ny, nx] = np.clip(after[ny, nx] + w * sign, -1.0, 1.0)
            
            # Measure territorial effects
            potential, solid = territory_sizes(after, color)
            effects[idx] = {
                "potential_territory": float(potential), 
                "solid_territory": float(solid)
            }
    
    return effects


def is_tenuki(selected_idx: int, last_move_loc: Optional[int], policy: np.ndarray, board: Board) -> bool:
    """Check if move is tenuki (different area + closer candidates exist)."""
    if last_move_loc is None or selected_idx is None:
        return False
    
    size = 19
    x_sel, y_sel = selected_idx % size, selected_idx // size
    x_last, y_last = loc_to_xy(board, last_move_loc)
    
    region_sel = classify_region(x_sel, y_sel, size)
    region_last = classify_region(x_last, y_last, size)
    
    if region_sel == region_last:
        return False
    
    # Check if there are candidates in last region with higher probability
    selected_prob = policy[selected_idx]
    for y in range(size):
        for x in range(size):
            if classify_region(x, y, size) == region_last:
                idx = y * size + x
                if idx < len(policy) and policy[idx] > selected_prob:
                    return True
    
    return False


def creates_new_group(board_before: Board, board_after: Board, player: int) -> bool:
    """Check if number of own groups increased after the move."""
    groups_before = enumerate_groups_deterministic(board_before)
    groups_after = enumerate_groups_deterministic(board_after)
    count_before = sum(1 for g in groups_before if g.color == player)
    count_after = sum(1 for g in groups_after if g.color == player)
    return count_after > count_before


def is_connection_move(board: Board, move_loc: int, color: int) -> Tuple[bool, float]:
    """Check if move connects stones or increases connectivity and compute strength gain."""
    if move_loc == Board.PASS_LOC:
        return False, 0.0
    x, y = loc_to_xy(board, move_loc)
    heads: List[int] = []
    
    for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
        nx, ny = x + dx, y + dy
        if not in_bounds(nx, ny, board.size):
            continue
        nloc = xy_to_loc(board, nx, ny)
        if board.board[nloc] == color:
            h = board.group_head[nloc]
            if h not in heads:
                heads.append(h)
    
    is_connection = len(heads) >= 2
    
    # For now, return a simple strength gain based on number of groups connected
    # This could be enhanced to compute actual connectivity increase
    connection_strength_gain = float(len(heads) - 1) if is_connection else 0.0
    
    return is_connection, connection_strength_gain


def is_extension_move(board: Board, move_loc: int, color: int) -> bool:
    """Check if move is next to existing own stone."""
    if move_loc == Board.PASS_LOC:
        return False
    x, y = loc_to_xy(board, move_loc)
    
    for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
        nx, ny = x + dx, y + dy
        if not in_bounds(nx, ny, board.size):
            continue
        nloc = xy_to_loc(board, nx, ny)
        if board.board[nloc] == color:
            return True
    
    return False


def liberties_of_group(board: Board, any_stone_loc: int) -> int:
    """Get number of liberties for a group."""
    if any_stone_loc == Board.PASS_LOC or board.board[any_stone_loc] == Board.EMPTY:
        return 0
    return int(board.num_liberties(any_stone_loc))


def atari_move(board: Board, move_loc: int) -> bool:
    """Check if move leaves opponent group in atari (1 liberty)."""
    if move_loc == Board.PASS_LOC:
        return False
    
    pla = board.pla
    opp = Board.get_opp(pla)
    size = board.size
    seen: Set[int] = set()
    
    for y in range(size):
        for x in range(size):
            loc = xy_to_loc(board, x, y)
            if board.board[loc] == opp:
                head = board.group_head[loc]
                if head in seen:
                    continue
                seen.add(head)
                if board.group_liberty_count[head] == 1:
                    return True
    
    return False


def reduce_aji(before: np.ndarray, after: np.ndarray, board: Board, color: int) -> Tuple[bool, float]:
    """Calculate aji reduction (increase in own ownership over opponent groups) and intensity."""
    opp = Board.get_opp(color)
    size = board.size
    delta: List[float] = []
    
    for y in range(size):
        for x in range(size):
            loc = xy_to_loc(board, x, y)
            if board.board[loc] == opp:
                delta.append((after[y, x] - before[y, x]) * (1 if color == Board.BLACK else -1))
    
    if not delta:
        return False, 0.0
    
    aji_reduction_intensity = float(np.mean(delta))
    # Aji reduction occurs if the mean reduction is >= 0.05
    reduces_aji = aji_reduction_intensity >= 0.05
    
    return reduces_aji, aji_reduction_intensity


def attack_strength_delta(groups_before: List[Group], groups_after: List[Group], opp_color: int) -> Tuple[bool, float]:
    """Calculate attack strength (negative change in opponent group strengths) and intensity."""
    before_map = {g.head: g for g in groups_before if g.color == opp_color}
    after_map = {g.head: g for g in groups_after if g.color == opp_color}
    common_heads = set(before_map.keys()) & set(after_map.keys())
    
    if not common_heads:
        return False, 0.0
    
    deltas = [after_map[h].strength - before_map[h].strength for h in common_heads]
    attack_intensity = float(np.mean(deltas))
    is_attack = attack_intensity < 0  # Negative change means attack
    
    return is_attack, abs(attack_intensity)  # Return positive intensity


def killing_attack(groups_after: List[Group], opp_color: int) -> Tuple[bool, float]:
    """Check if any opponent group has strength <= -0.5 (>=0.5 own ownership) and compute kill intensity."""
    killed_groups = []
    for g in groups_after:
        if g.color == opp_color and g.strength <= -0.5:
            killed_groups.append(g)
    
    if not killed_groups:
        return False, 0.0
    
    # Kill intensity is the final ownership mean over targeted stones
    kill_intensity = float(np.mean([abs(g.strength) for g in killed_groups]))
    return True, kill_intensity


# -------------------------
# Comprehensive Analysis Function
# -------------------------

def analyze_position_comprehensive(
    board: Board, 
    ownership: np.ndarray, 
    policy: np.ndarray,
    player: int,
    move_loc: Optional[int] = None,
    last_move_loc: Optional[int] = None,
    before_ownership: Optional[np.ndarray] = None,
    before_board: Optional[Board] = None
) -> Dict[str, Any]:
    """
    Perform comprehensive analysis of a position using all 28 concepts.
    
    Args:
        board: Current board state
        ownership: Current ownership map (19x19)
        policy: Policy distribution (361)
        player: Current player (Board.BLACK or Board.WHITE)
        move_loc: Location of current move (optional)
        last_move_loc: Location of last move (optional)
        before_ownership: Ownership before current move (optional)
        before_board: Board state before current move (optional)
    
    Returns:
        Dictionary containing all analysis results matching the specification.
        Note: For annotation purposes, these can be treated as tags with values
        and directions (plus/minus) as mentioned in the original specification.
    """
    results = {}
    
    # 1-2. Urgency by region (regions are static, not needed in output)
    results["urgency"] = urgency_by_region(policy)
    results["urgency_intensity"] = urgency_intensity_by_region(policy)
    
    # 3-7. Groups and influence
    groups_det = enumerate_groups_deterministic(board)
    groups_own = enumerate_groups_ownership(board, ownership, player)
    
    compute_group_strengths(groups_det, ownership, player, board)
    compute_group_strengths(groups_own, ownership, player, board)
    compute_group_connectivity(groups_det, ownership, board)
    compute_group_connectivity(groups_own, ownership, board)
    compute_group_influence(groups_det, ownership, board)
    compute_group_influence(groups_own, ownership, board)
    
    # Groups are used for computation but not returned (per note #1)
    # Only derived metrics are included in results
    
    # 8-16. Territory analysis
    if before_ownership is not None:
        # Building territory
        building_count, building_intensity = count_building_territory(before_ownership, ownership, player)
        results["building_count"] = building_count
        results["building_intensity"] = building_intensity
        
        # Solidification
        solidification_count, solidification_value = solidify_territory_delta(before_ownership, ownership, player)
        results["solidification_count"] = solidification_count
        results["solidification_value"] = solidification_value
        
        # Reduction
        reduction_count, reduction_intensity = reduce_opponent_territory_count(before_ownership, ownership, player)
        results["reduction_count"] = reduction_count
        results["reduction_intensity"] = reduction_intensity
        
        # Invasion
        is_invasion, invasion_intensity = invasion_effect(before_ownership, ownership, player)
        results["invasion"] = is_invasion
        results["invasion_intensity"] = invasion_intensity
        
        # Leaving weakness
        results["leaves_weakness"] = leaving_weakness(before_ownership, ownership, player)
        
        # Regional weakening
        weakening_count = {}
        weakening_intensity = {}
        for region in ["corner_tl", "corner_tr", "corner_bl", "corner_br",
                       "side_left", "side_right", "side_top", "side_bottom", "center"]:
            count, intensity = weakening_territory_in_region(before_ownership, ownership, region, player)
            weakening_count[region] = count
            weakening_intensity[region] = intensity
        results["weakening_count_by_region"] = weakening_count
        results["weakening_intensity_by_region"] = weakening_intensity
        
        # Regional territory deltas
        building_count_by_region, building_intensity_by_region, solidification_count_by_region, solidification_value_by_region = compute_territory_delta_by_region(before_ownership, ownership, player)
        results["building_count_by_region"] = building_count_by_region
        results["building_intensity_by_region"] = building_intensity_by_region
        results["solidification_count_by_region"] = solidification_count_by_region
        results["solidification_value_by_region"] = solidification_value_by_region
        
        reduction_count_by_region, reduction_intensity_by_region = compute_reduction_delta_by_region(before_ownership, ownership, player)
        results["reduction_count_by_region"] = reduction_count_by_region
        results["reduction_intensity_by_region"] = reduction_intensity_by_region
        
    else:
        # Set defaults when no before_ownership available
        results["building_count"] = 0
        results["building_intensity"] = 0.0
        results["solidification_count"] = 0
        results["solidification_value"] = 0.0
        results["reduction_count"] = 0
        results["reduction_intensity"] = 0.0
        results["invasion"] = False
        results["invasion_intensity"] = 0.0
        results["leaves_weakness"] = 0
        results["weakening_count_by_region"] = {region: 0 for region in ["corner_tl", "corner_tr", "corner_bl", "corner_br", "side_left", "side_right", "side_top", "side_bottom", "center"]}
        results["weakening_intensity_by_region"] = {region: 0.0 for region in ["corner_tl", "corner_tr", "corner_bl", "corner_br", "side_left", "side_right", "side_top", "side_bottom", "center"]}
        results["building_count_by_region"] = {region: 0 for region in ["corner_tl", "corner_tr", "corner_bl", "corner_br", "side_left", "side_right", "side_top", "side_bottom", "center"]}
        results["building_intensity_by_region"] = {region: 0.0 for region in ["corner_tl", "corner_tr", "corner_bl", "corner_br", "side_left", "side_right", "side_top", "side_bottom", "center"]}
        results["solidification_count_by_region"] = {region: 0 for region in ["corner_tl", "corner_tr", "corner_bl", "corner_br", "side_left", "side_right", "side_top", "side_bottom", "center"]}
        results["solidification_value_by_region"] = {region: 0.0 for region in ["corner_tl", "corner_tr", "corner_bl", "corner_br", "side_left", "side_right", "side_top", "side_bottom", "center"]}
        results["reduction_count_by_region"] = {region: 0 for region in ["corner_tl", "corner_tr", "corner_bl", "corner_br", "side_left", "side_right", "side_top", "side_bottom", "center"]}
        results["reduction_intensity_by_region"] = {region: 0.0 for region in ["corner_tl", "corner_tr", "corner_bl", "corner_br", "side_left", "side_right", "side_top", "side_bottom", "center"]}
    
    # 14-16. Territory sizes and sacrifices
    potential, solid = territory_sizes(ownership, player)
    results["potential_territory"] = potential
    results["solid_territory"] = solid
    
    if move_loc is not None:
        is_direct_sacrifice, sacrifice_intensity = direct_sacrifice(move_loc, ownership, player, board)
        results["direct_sacrifice"] = is_direct_sacrifice
        results["sacrifice_intensity"] = sacrifice_intensity
        
        if before_ownership is not None:
            indirect_count, indirect_sacrifice_intensity = indirect_sacrifice(before_ownership, ownership, player, board)
            results["indirect_sacrifice"] = indirect_count
            results["indirect_sacrifice_intensity"] = indirect_sacrifice_intensity
        else:
            results["indirect_sacrifice"] = 0
            results["indirect_sacrifice_intensity"] = 0.0
    else:
        results["direct_sacrifice"] = False
        results["sacrifice_intensity"] = 0.0
        results["indirect_sacrifice"] = 0
        results["indirect_sacrifice_intensity"] = 0.0
    
    # 18-25. Tactical concepts
    if move_loc is not None:
        results["cut"] = is_cut_move(board, move_loc)
        
        is_connection, connection_strength_gain = is_connection_move(board, move_loc, player)
        results["connection"] = is_connection
        results["connection_strength_gain"] = connection_strength_gain
        
        results["extension"] = is_extension_move(board, move_loc, player)
        results["liberties"] = liberties_of_group(board, move_loc)
        results["atari"] = atari_move(board, move_loc)
    else:
        results["cut"] = False
        results["connection"] = False
        results["connection_strength_gain"] = 0.0
        results["extension"] = False
        results["liberties"] = 0
        results["atari"] = False
    
    results["only_move"] = is_only_move(policy)
    results["rough_intent"] = rough_intent_effects(ownership, policy, player)
    
    if last_move_loc is not None and move_loc is not None:
        move_idx = loc_to_xy(board, move_loc)[1] * 19 + loc_to_xy(board, move_loc)[0]
        results["tenuki"] = is_tenuki(move_idx, last_move_loc, policy, board)
    else:
        results["tenuki"] = False
    
    # 26-28. Attack concepts
    if before_ownership is not None:
        reduces_aji, aji_reduction_intensity = reduce_aji(before_ownership, ownership, board, player)
        results["reduce_aji"] = reduces_aji
        results["aji_reduction_intensity"] = aji_reduction_intensity
        
        # Calculate attack effects
        if before_board is not None:
            groups_before = enumerate_groups_deterministic(before_board)
            is_attack, attack_intensity = attack_strength_delta(groups_before, groups_det, Board.get_opp(player))
            results["attack"] = is_attack
            results["attack_intensity"] = attack_intensity
            
            # Group deltas by region
            group_strength_delta_by_region = compute_group_strength_delta_by_region(groups_before, groups_det, board)
            results["group_strength_delta_by_region"] = group_strength_delta_by_region
            results["group_strength_delta"] = sum(group_strength_delta_by_region.values())
            
            group_connectivity_delta_by_region = compute_group_connectivity_delta_by_region(groups_before, groups_det, board)
            results["group_connectivity_delta_by_region"] = group_connectivity_delta_by_region
            results["group_connectivity_delta"] = sum(group_connectivity_delta_by_region.values())
            
            influence_count_delta_by_region, influence_strength_delta_by_region = compute_influence_delta_by_region(groups_before, groups_det, board)
            results["influence_count_delta_by_region"] = influence_count_delta_by_region
            results["influence_count_delta"] = sum(influence_count_delta_by_region.values())
            results["influence_strength_delta_by_region"] = influence_strength_delta_by_region
            results["influence_strength_delta"] = sum(influence_strength_delta_by_region.values())
            
            # Creates new group
            results["creates_new_group"] = creates_new_group(before_board, board, player)
        else:
            results["attack"] = False
            results["attack_intensity"] = 0.0
            results["group_strength_delta_by_region"] = {region: 0.0 for region in ["corner_tl", "corner_tr", "corner_bl", "corner_br", "side_left", "side_right", "side_top", "side_bottom", "center"]}
            results["group_strength_delta"] = 0.0
            results["group_connectivity_delta_by_region"] = {region: 0.0 for region in ["corner_tl", "corner_tr", "corner_bl", "corner_br", "side_left", "side_right", "side_top", "side_bottom", "center"]}
            results["group_connectivity_delta"] = 0.0
            results["influence_count_delta_by_region"] = {region: 0 for region in ["corner_tl", "corner_tr", "corner_bl", "corner_br", "side_left", "side_right", "side_top", "side_bottom", "center"]}
            results["influence_count_delta"] = 0
            results["influence_strength_delta_by_region"] = {region: 0.0 for region in ["corner_tl", "corner_tr", "corner_bl", "corner_br", "side_left", "side_right", "side_top", "side_bottom", "center"]}
            results["influence_strength_delta"] = 0.0
            results["creates_new_group"] = False
        
        is_killing_attack, kill_intensity = killing_attack(groups_det, Board.get_opp(player))
        results["killing_attack"] = is_killing_attack
        results["kill_intensity"] = kill_intensity
    else:
        results["reduce_aji"] = False
        results["aji_reduction_intensity"] = 0.0
        results["attack"] = False
        results["attack_intensity"] = 0.0
        results["group_strength_delta_by_region"] = {region: 0.0 for region in ["corner_tl", "corner_tr", "corner_bl", "corner_br", "side_left", "side_right", "side_top", "side_bottom", "center"]}
        results["group_strength_delta"] = 0.0
        results["group_connectivity_delta_by_region"] = {region: 0.0 for region in ["corner_tl", "corner_tr", "corner_bl", "corner_br", "side_left", "side_right", "side_top", "side_bottom", "center"]}
        results["group_connectivity_delta"] = 0.0
        results["influence_count_delta_by_region"] = {region: 0 for region in ["corner_tl", "corner_tr", "corner_bl", "corner_br", "side_left", "side_right", "side_top", "side_bottom", "center"]}
        results["influence_count_delta"] = 0
        results["influence_strength_delta_by_region"] = {region: 0.0 for region in ["corner_tl", "corner_tr", "corner_bl", "corner_br", "side_left", "side_right", "side_top", "side_bottom", "center"]}
        results["influence_strength_delta"] = 0.0
        results["creates_new_group"] = False
        results["killing_attack"] = False
        results["kill_intensity"] = 0.0
    
    return results


# Export all functions for use in other modules
__all__ = [
    "Group",
    "classify_region",
    "region_map",
    "enumerate_groups_deterministic",
    "enumerate_groups_ownership",
    "compute_group_strengths",
    "compute_group_connectivity",
    "compute_group_influence",
    "count_building_territory",
    "solidify_territory_delta",
    "reduce_opponent_territory_count",
    "invasion_effect",
    "weakening_territory_in_region",
    "leaving_weakness",
    "territory_sizes",
    "direct_sacrifice",
    "indirect_sacrifice",
    "urgency_by_region",
    "is_cut_move",
    "is_only_move",
    "rough_intent_effects",
    "is_tenuki",
    "is_connection_move",
    "is_extension_move",
    "liberties_of_group",
    "atari_move",
    "reduce_aji",
    "attack_strength_delta",
    "killing_attack",
    "analyze_position_comprehensive",
]