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

COMPUTED FEATURES:
- Region-based deltas: group_strength_delta[region], group_connectivity_delta[region], etc.
- Aggregate totals: group_strength_delta, group_connectivity_delta, etc.
- Creates new group: true if number of own groups increases
- All features include both boolean/count flags and intensity metrics
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set, Iterable, Any
import json
from functools import lru_cache

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
EPSILON_POL = 1e-12       # Policy probability threshold (legacy, too small)

# Improved thresholds for more reliable weak supervision
TAU_POS_LOW = 0.08        # Lower threshold for building territory (hysteresis)
TAU_POS_HIGH = 0.12       # Higher threshold for building territory (hysteresis)
TAU_ONLY_MOVE = 0.05      # Threshold for "only move" detection (policy > 0.95)
TAU_GROUP_IOU = 0.1       # Minimum IoU for group matching across plies
TAU_AJI_VICINITY = 5      # L1 distance for aji reduction vicinity mask

# Legacy constants for backward compatibility
OWN_MIN = TAU_POS
TERRITORY_SOLID = TAU_SOLID


@dataclass
class Group:
    color: int                 # Board.BLACK or Board.WHITE
    head: int                  # representative loc (board.group_head)
    stones: List[int]          # list of locs belonging to the group
    liberties: int             # number of liberties
    strength: float            # average ownership value on group stones (signed from perspective of 'color')
    connectivity: float        # average ownership of empty intersections within L1 ≤ 2 radius (legacy)
    connectivity_signed: float # signed connectivity (+ if helps group's color, - if helps opponent)
    connectivity_mag: float    # magnitude of connectivity (turn-invariant)
    influence_area: int        # count of own ownership around group
    influence_strength: float  # average ownership around group
    influence_spread: float    # measure of how spread out the influence is (0-1, higher = more spread)


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
    Classify region into 9 distinct areas using a simple grid-based approach:
    - 4 Corners: top-left, top-right, bottom-left, bottom-right (first 6x6 areas)
    - 4 Sides: left, right, upper, lower (remaining edge areas)
    - 1 Center: middle area (7x7 center)
    
    Grid layout for 19x19 board:
    - Corners: 0-5 x 0-5, 14-18 x 0-5, 0-5 x 14-18, 14-18 x 14-18
    - Sides: 6-13 x 0-5, 6-13 x 14-18, 0-5 x 6-13, 14-18 x 6-13  
    - Center: 6-13 x 6-13
    """
    # Define the grid boundaries
    corner_size = 6  # 6x6 corner areas
    side_start = corner_size  # 6
    side_end = size - corner_size  # 13
    
    # Corner regions (6x6 areas)
    if x < corner_size and y < corner_size:
        return "corner_tl"  # 0-5 x 0-5
    elif x >= side_end and y < corner_size:
        return "corner_tr"  # 14-18 x 0-5
    elif x < corner_size and y >= side_end:
        return "corner_bl"  # 0-5 x 14-18
    elif x >= side_end and y >= side_end:
        return "corner_br"  # 14-18 x 14-18
    
    # Side regions
    elif side_start <= x < side_end and y < corner_size:
        return "side_top"  # 6-13 x 0-5
    elif side_start <= x < side_end and y >= side_end:
        return "side_bottom"  # 6-13 x 14-18
    elif x < corner_size and side_start <= y < side_end:
        return "side_left"  # 0-5 x 6-13
    elif x >= side_end and side_start <= y < side_end:
        return "side_right"  # 14-18 x 6-13
    
    # Center region
    else:
        return "center"  # 6-13 x 6-13


@lru_cache(maxsize=4)  # Cache for common board sizes (9, 13, 19, 21)
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
            
            groups.append(Group(
                color=stone,
                head=head,
                stones=stones,
                liberties=board.group_liberty_count[head],
                strength=0.0,  # Will be computed later with ownership data
                connectivity=0.0,  # Legacy field
                connectivity_signed=0.0,  # Will be computed later
                connectivity_mag=0.0,  # Will be computed later
                influence_area=0,
                influence_strength=0.0,
                influence_spread=0.0,
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
                    
                    # Calculate liberties using board's union-find data to avoid double counting
                    # Find the head of the group in the board's union-find structure
                    if group_stones:
                        first_stone_loc = board.loc(group_stones[0][0], group_stones[0][1])
                        liberties = board.group_liberty_count[board.group_head[first_stone_loc]]
                    else:
                        liberties = 0
                    
                    groups.append(Group(
                        color=color,
                        head=stones[0],  # Use first stone as head
                        stones=stones,
                        liberties=liberties,
                        strength=0.0,  # Will be computed later with ownership data
                        connectivity=0.0,  # Legacy field
                        connectivity_signed=0.0,  # Will be computed later
                        connectivity_mag=0.0,  # Will be computed later
                        influence_area=0,
                        influence_strength=0.0,
                        influence_spread=0.0,
                    ))
    
    return groups


def compute_group_strengths(groups: List[Group], ownership: np.ndarray, player_perspective: int, board: Board) -> None:
    """Compute group strength as average ownership of group stones (ownership already from player perspective)."""
    for g in groups:
        vals: List[float] = []
        for loc in g.stones:
            x, y = loc_to_xy(board, loc)
            # Ownership is already from player perspective, no sign correction needed
            vals.append(float(ownership[y, x]))
        g.strength = float(np.mean(vals)) if vals else 0.0


def compute_group_connectivity(groups: List[Group], ownership: np.ndarray, board: Board) -> None:
    """Compute group connectivity as average ownership of empty intersections within L1 ≤ 2 radius."""
    for g in groups:
        vals: List[float] = []
        abs_vals: List[float] = []
        
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
                        # Ownership is from current player's perspective
                        v = ownership[ny, nx]
                        vals.append(v)
                        abs_vals.append(abs(v))
        
        # Legacy connectivity (frame-dependent)
        g.connectivity = float(np.mean(vals)) if vals else 0.0
        
        # New connectivity metrics (more robust)
        g.connectivity_signed = float(np.mean(vals)) if vals else 0.0
        g.connectivity_mag = float(np.mean(abs_vals)) if abs_vals else 0.0


def compute_group_influence(groups: List[Group], ownership: np.ndarray, board: Board) -> None:
    """Compute group influence area, strength, and spread."""
    for g in groups:
        influence_points = []
        influence_locations = []
        
        # Find all points around the group with same-sign ownership
        for loc in g.stones:
            x, y = loc_to_xy(board, loc)
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                nx, ny = x + dx, y + dy
                if (in_bounds(nx, ny, board.size) and 
                    board.board[xy_to_loc(board, nx, ny)] == Board.EMPTY):
                    # For own groups, positive ownership indicates influence
                    # For opponent groups, negative ownership indicates influence
                    if g.color == board.pla and ownership[ny, nx] > TAU_POS:
                        influence_points.append(ownership[ny, nx])
                        influence_locations.append((nx, ny))
                    elif g.color != board.pla and ownership[ny, nx] < -TAU_POS:
                        influence_points.append(-ownership[ny, nx])  # Convert to positive for consistency
                        influence_locations.append((nx, ny))
        
        g.influence_area = len(influence_points)
        g.influence_strength = float(np.mean(influence_points)) if influence_points else 0.0
        
        # Calculate influence spread (how spread out the influence is)
        if len(influence_locations) > 1:
            # Calculate the standard deviation of distances from group center
            group_center_x = np.mean([loc_to_xy(board, stone)[0] for stone in g.stones])
            group_center_y = np.mean([loc_to_xy(board, stone)[1] for stone in g.stones])
            
            distances = []
            for inf_x, inf_y in influence_locations:
                dist = np.sqrt((inf_x - group_center_x)**2 + (inf_y - group_center_y)**2)
                distances.append(dist)
            
            # Normalize spread (0-1, higher = more spread)
            max_possible_dist = np.sqrt(2) * 19  # Diagonal of board
            g.influence_spread = float(np.std(distances) / max_possible_dist) if distances else 0.0
        else:
            g.influence_spread = 0.0


# -------------------------
# Territory Analysis Functions
# -------------------------

def count_building_territory(before: np.ndarray, after: np.ndarray, color: int, board: Board) -> Tuple[int, float, float]:
    """Count intersections that changed from empty to own ownership using hysteresis and compute intensity.
    Excludes actual stone positions from territory counting.
    
    Args:
        before: Ownership map before move (from current player's perspective)
        after: Ownership map after move (from current player's perspective)
        color: Current player (Board.BLACK or Board.WHITE)
        board: Board state to check for stones
    
    Returns:
        (count, intensity, sum_delta) - count of built intersections, mean intensity, sum of deltas
    """
    # Use hysteresis thresholds to reduce flip-flop
    # Building: before < TAU_POS_LOW and after > TAU_POS_HIGH
    built_mask = (np.abs(before) < TAU_POS_LOW) & (after > TAU_POS_HIGH)
    
    # Exclude actual stone positions from territory counting
    stone_mask = np.zeros_like(built_mask, dtype=bool)
    for y in range(board.size):
        for x in range(board.size):
            loc = board.loc(x, y)
            if board.board[loc] != Board.EMPTY:
                stone_mask[y, x] = True
    
    # Only count empty intersections for territory
    built_mask = built_mask & ~stone_mask
    
    count = int(np.sum(built_mask))
    
    # Compute intensity and sum
    if count > 0:
        deltas = after[built_mask] - before[built_mask]
        intensity = float(np.mean(deltas))
        sum_delta = float(np.sum(deltas))
    else:
        intensity = 0.0
        sum_delta = 0.0
    
    return count, intensity, sum_delta


def solidify_territory_delta(before: np.ndarray, after: np.ndarray, color: int, board: Board) -> Tuple[int, float, float]:
    """Calculate increase in ownership values of previously owned intersections and compute intensity.
    Excludes actual stone positions from territory counting.
    
    Args:
        before: Ownership map before move (from current player's perspective)
        after: Ownership map after move (from current player's perspective)
        color: Current player (Board.BLACK or Board.WHITE)
        board: Board state to check for stones
    
    Returns:
        (count, intensity, sum_delta) - count of solidified intersections, mean intensity, sum of deltas
    """
    # Determine what constitutes "owned" territory for current player
    # For current player, owned territory means positive values
    owned_mask = before > TAU_POS
    
    # Find intersections that were solidified (owned before and after, with increase)
    solidified_mask = owned_mask & (after > before)  # Increase in positive values
    
    # Exclude actual stone positions from territory counting
    stone_mask = np.zeros_like(solidified_mask, dtype=bool)
    for y in range(board.size):
        for x in range(board.size):
            loc = board.loc(x, y)
            if board.board[loc] != Board.EMPTY:
                stone_mask[y, x] = True
    
    # Only count empty intersections for territory
    solidified_mask = solidified_mask & ~stone_mask
    
    count = int(np.sum(solidified_mask))
    
    # Compute intensity and sum
    if count > 0:
        deltas = after[solidified_mask] - before[solidified_mask]
        intensity = float(np.mean(deltas))
        sum_delta = float(np.sum(deltas))
    else:
        intensity = 0.0
        sum_delta = 0.0
    
    return count, intensity, sum_delta


def reduce_opponent_territory_count(before: np.ndarray, after: np.ndarray, color: int, board: Board) -> Tuple[int, float, float]:
    """Count reduction in opponent's owned intersections and compute intensity.
    Excludes actual stone positions from territory counting.
    
    Args:
        before: Ownership map before move (from current player's perspective)
        after: Ownership map after move (from current player's perspective)
        color: Current player (Board.BLACK or Board.WHITE)
        board: Board state to check for stones
    
    Returns:
        (count, intensity, sum_delta) - count of reduced intersections, mean intensity, sum of deltas
    """
    # Determine what constitutes opponent's territory
    # For current player, opponent territory means negative values
    opp_owned_mask = before < -TAU_POS
    
    # Find intersections that were reduced (opponent owned before, not after)
    # Opponent territory reduction means negative values becoming less negative (closer to 0)
    reduced_mask = opp_owned_mask & (after > before)
    
    # Exclude actual stone positions from territory counting
    stone_mask = np.zeros_like(reduced_mask, dtype=bool)
    for y in range(board.size):
        for x in range(board.size):
            loc = board.loc(x, y)
            if board.board[loc] != Board.EMPTY:
                stone_mask[y, x] = True
    
    # Only count empty intersections for territory
    reduced_mask = reduced_mask & ~stone_mask
    
    count = int(np.sum(reduced_mask))
    
    # Compute intensity and sum
    if count > 0:
        deltas = after[reduced_mask] - before[reduced_mask]
        intensity = float(np.mean(np.abs(deltas)))
        sum_delta = float(np.sum(deltas))  # Sum of actual changes (can be negative)
    else:
        intensity = 0.0
        sum_delta = 0.0
    
    return count, intensity, sum_delta


def invasion_effect(before: np.ndarray, after: np.ndarray, color: int, board: Board, move_loc: Optional[int] = None) -> Tuple[bool, float]:
    """Calculate invasion effect: same points flip from opponent ownership to own ownership.
    
    Args:
        before: Ownership map before move (from current player's perspective)
        after: Ownership map after move (from current player's perspective)
        color: Current player (Board.BLACK or Board.WHITE)
        board: Board state to check for stones
        move_loc: Location of the move (optional, for spatial constraint)
    """
    # Create mask to exclude stone positions
    stone_mask = np.zeros_like(before, dtype=bool)
    for y in range(board.size):
        for x in range(board.size):
            loc = board.loc(x, y)
            if board.board[loc] != Board.EMPTY:
                stone_mask[y, x] = True
    
    # Only count empty intersections for territory
    empty_mask = ~stone_mask
    
    # Create spatial constraint mask (L2 distance <= 3 from move)
    spatial_mask = np.ones_like(before, dtype=bool)  # Default: no spatial constraint
    if move_loc is not None and move_loc != Board.PASS_LOC:
        move_x, move_y = loc_to_xy(board, move_loc)
        spatial_mask = np.zeros_like(before, dtype=bool)
        for y in range(board.size):
            for x in range(board.size):
                # Calculate L2 distance from move location
                l2_dist = np.sqrt((x - move_x)**2 + (y - move_y)**2)
                if l2_dist <= 3.0:
                    spatial_mask[y, x] = True
        
        # Check if >0.5% of points within L2<=3 are opponent territory
        spatial_empty_mask = empty_mask & spatial_mask
        total_spatial_points = int(np.sum(spatial_empty_mask))
        if total_spatial_points > 0:
            opponent_spatial_points = int(np.sum(spatial_empty_mask & (before < -TAU_POS)))
            opponent_percentage = opponent_spatial_points / total_spatial_points
            if opponent_percentage <= 0.5:  # Less than 50% opponent territory
                # Not enough opponent territory in vicinity, no invasion possible
                return False, 0.0
    
    # Invasion: same points flip from opponent territory to own territory
    # For current player, opponent territory is negative, own territory is positive
    # AND the points must be within L2 distance <= 3 of the move
    invasion_mask = empty_mask & spatial_mask & (before < -TAU_POS) & (after > TAU_POS)
    
    invasion_count = int(np.sum(invasion_mask))
    
    # Invasion intensity is the average ownership change of invaded points
    if invasion_count > 0:
        invasion_intensity = float(np.mean(after[invasion_mask] - before[invasion_mask]))
    else:
        invasion_intensity = 0.0
    
    is_invasion = invasion_count > 0
    
    return is_invasion, invasion_intensity


def weakening_territory_in_region(before: np.ndarray, after: np.ndarray, region: str, color: int) -> Tuple[int, float]:
    """Calculate weakening of opponent territory in specific region.
    
    Args:
        before: Ownership map before move (from current player's perspective)
        after: Ownership map after move (from current player's perspective)
        region: Region name
        color: Current player (Board.BLACK or Board.WHITE)
    """
    m = region_map(before.shape[0])
    mask = (m == region)
    
    if not np.any(mask):
        return 0, 0.0
    
    # Determine what constitutes opponent territory and weakening
    # For current player, opponent territory is negative, weakening means less negative
    opp_owned_mask = before < -TAU_POS
    weakened_mask = mask & opp_owned_mask & (after > before)  # Less negative = weakened
    
    count = int(np.sum(weakened_mask))
    
    # Intensity is the average reduction magnitude
    if count > 0:
        intensity = float(np.mean(np.abs(before[weakened_mask] - after[weakened_mask])))
    else:
        intensity = 0.0
    
    return count, intensity


def leaving_weakness(before: np.ndarray, after: np.ndarray, color: int, board: Board) -> int:
    """Count intersections that flipped from own to opponent ownership.
    
    Args:
        before: Ownership map before move (from current player's perspective)
        after: Ownership map after move (from current player's perspective)
        color: Current player (Board.BLACK or Board.WHITE)
        board: Board state
    """
    # Only check positions where there are actual stones of the current player
    stone_mask = np.zeros_like(before, dtype=bool)
    for y in range(board.size):
        for x in range(board.size):
            loc = board.loc(x, y)
            if board.board[loc] == color:  # Only current player's stones
                stone_mask[y, x] = True
    
    # Determine what constitutes own vs opponent territory from current player's perspective
    # For current player, own territory is positive, opponent territory is negative
    own_territory_mask = before > TAU_POS
    opponent_territory_mask = after < -TAU_POS
    
    # Only count stone positions that flipped from own to opponent territory
    flipped_mask = stone_mask & own_territory_mask & opponent_territory_mask
    
    return int(np.sum(flipped_mask))


def territory_sizes(ownership: np.ndarray, color: int, board: Board) -> Tuple[int, int]:
    """Calculate potential and solid territory sizes.
    Excludes actual stone positions from territory counting.
    
    Args:
        ownership: Ownership map (from current player's perspective)
        color: Current player (Board.BLACK or Board.WHITE)
        board: Board state to check for stones
    """
    # Create mask to exclude stone positions
    stone_mask = np.zeros_like(ownership, dtype=bool)
    for y in range(board.size):
        for x in range(board.size):
            loc = board.loc(x, y)
            if board.board[loc] != Board.EMPTY:
                stone_mask[y, x] = True
    
    # For current player, positive values represent own territory
    # Only count empty intersections for territory
    empty_mask = ~stone_mask
    potential = int(np.sum((ownership > TAU_POS) & empty_mask) - np.sum((ownership >= TAU_SOLID) & empty_mask))
    solid = int(np.sum((ownership >= TAU_SOLID) & empty_mask))
    return potential, solid


def direct_sacrifice(move_loc: int, after: np.ndarray, color: int, board: Board) -> Tuple[bool, float]:
    """Check if the played stone becomes opponent's territory and compute sacrifice intensity.
    
    Args:
        move_loc: Location of the move
        after: Ownership map after move (from current player's perspective)
        color: Current player (Board.BLACK or Board.WHITE)
        board: Board state
    """
    if move_loc == Board.PASS_LOC:
        return False, 0.0
    x, y = loc_to_xy(board, move_loc)
    
    # For current player, sacrifice means the move location becomes opponent territory (negative)
    is_sacrifice = bool(after[y, x] < -TAU_POS)
    sacrifice_intensity = abs(after[y, x]) if is_sacrifice else 0.0
    
    return is_sacrifice, sacrifice_intensity


def indirect_sacrifice(before: np.ndarray, after: np.ndarray, color: int, board: Board) -> Tuple[int, float]:
    """Check if any own stone becomes opponent's territory and compute intensity.
    
    Args:
        before: Ownership map before move (from current player's perspective)
        after: Ownership map after move (from current player's perspective)
        color: Current player (Board.BLACK or Board.WHITE)
        board: Board state
    """
    # Only check positions where there are actual stones of the current player
    stone_mask = np.zeros_like(before, dtype=bool)
    for y in range(board.size):
        for x in range(board.size):
            loc = board.loc(x, y)
            if board.board[loc] == color:  # Only current player's stones
                stone_mask[y, x] = True
    
    # Find stones that became opponent's territory
    # For current player, own territory is positive, opponent territory is negative
    sacrificed_mask = stone_mask & (before > TAU_POS) & (after < -TAU_POS)
    
    count = int(np.sum(sacrificed_mask))
    
    # Compute intensity as mean ownership loss of affected stones
    if count > 0:
        intensity = float(np.mean(before[sacrificed_mask] - after[sacrificed_mask]))
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


def compute_max_group_strength_delta(
    groups_before: List[Group], 
    groups_after: List[Group], 
    board: Board
) -> Tuple[float, str]:
    """Compute the maximum group strength delta and which region it occurred in."""
    deltas_by_region = compute_group_strength_delta_by_region(groups_before, groups_after, board)
    
    if not deltas_by_region:
        return 0.0, "none"
    
    max_region = max(deltas_by_region, key=deltas_by_region.get)
    max_delta = deltas_by_region[max_region]
    
    return max_delta, max_region


def compute_group_metrics_avg_max(
    groups_before: List[Group], 
    groups_after: List[Group], 
    board: Board,
    focus_color: int
) -> Dict[str, Any]:
    """Compute average and max group strength/connectivity deltas for groups of focus_color.
    
    Args:
        groups_before: Groups before the move
        groups_after: Groups after the move  
        board: Board state
        focus_color: Color to focus on (Board.BLACK or Board.WHITE)
    
    Returns:
        Dictionary with avg/max metrics and group locations
    """
    # Filter groups by focus color
    before_groups = [g for g in groups_before if g.color == focus_color]
    after_groups = [g for g in groups_after if g.color == focus_color]
    
    if not before_groups or not after_groups:
        return {
            "avg_strength_delta": 0.0,
            "max_strength_delta": 0.0,
            "max_strength_group_location": "none",
            "avg_connectivity_delta": 0.0,
            "max_connectivity_delta": 0.0,
            "max_connectivity_group_location": "none"
        }
    
    # Match groups by spatial overlap (IoU)
    def group_to_mask(group: Group, size: int) -> np.ndarray:
        mask = np.zeros((size, size), dtype=bool)
        for loc in group.stones:
            x, y = loc_to_xy(board, loc)
            mask[y, x] = True
        return mask
    
    def get_group_location(group: Group, board: Board) -> str:
        """Get the most frequent region for stones in the group."""
        region_counts = {}
        for loc in group.stones:
            x, y = loc_to_xy(board, loc)
            region = classify_region(x, y, board.size)
            region_counts[region] = region_counts.get(region, 0) + 1
        
        if not region_counts:
            return "none"
        
        return max(region_counts, key=region_counts.get)
    
    matched_pairs = []
    used_after = set()
    
    for before_group in before_groups:
        before_mask = group_to_mask(before_group, board.size)
        best_iou = 0.0
        best_after_group = None
        
        for i, after_group in enumerate(after_groups):
            if i in used_after:
                continue
            
            after_mask = group_to_mask(after_group, board.size)
            
            # Calculate IoU
            intersection = np.sum(before_mask & after_mask)
            union = np.sum(before_mask | after_mask)
            iou = intersection / union if union > 0 else 0.0
            
            if iou > best_iou and iou >= TAU_GROUP_IOU:
                best_iou = iou
                best_after_group = after_group
        
        if best_after_group is not None:
            matched_pairs.append((before_group, best_after_group))
            used_after.add(after_groups.index(best_after_group))
    
    if not matched_pairs:
        return {
            "avg_strength_delta": 0.0,
            "max_strength_delta": 0.0,
            "max_strength_group_location": "none",
            "avg_connectivity_delta": 0.0,
            "max_connectivity_delta": 0.0,
            "max_connectivity_group_location": "none"
        }
    
    # Calculate deltas
    strength_deltas = [after.strength - before.strength for before, after in matched_pairs]
    connectivity_deltas = [after.connectivity - before.connectivity for before, after in matched_pairs]
    
    # Find max deltas and their group locations
    max_strength_idx = strength_deltas.index(max(strength_deltas))
    max_connectivity_idx = connectivity_deltas.index(max(connectivity_deltas))
    
    max_strength_group = matched_pairs[max_strength_idx][1]  # after group
    max_connectivity_group = matched_pairs[max_connectivity_idx][1]  # after group
    
    return {
        "avg_strength_delta": float(np.mean(strength_deltas)),
        "max_strength_delta": float(max(strength_deltas)),
        "max_strength_group_location": get_group_location(max_strength_group, board),
        "avg_connectivity_delta": float(np.mean(connectivity_deltas)),
        "max_connectivity_delta": float(max(connectivity_deltas)),
        "max_connectivity_group_location": get_group_location(max_connectivity_group, board)
    }


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


def compute_max_group_connectivity_delta(
    groups_before: List[Group], 
    groups_after: List[Group], 
    board: Board
) -> Tuple[float, str]:
    """Compute the maximum group connectivity delta and which region it occurred in."""
    deltas_by_region = compute_group_connectivity_delta_by_region(groups_before, groups_after, board)
    
    if not deltas_by_region:
        return 0.0, "none"
    
    max_region = max(deltas_by_region, key=deltas_by_region.get)
    max_delta = deltas_by_region[max_region]
    
    return max_delta, max_region


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
    color: int,
    board: Board
) -> Tuple[Dict[str, int], Dict[str, float], Dict[str, int], Dict[str, float]]:
    """Compute building and solidification deltas by region.
    Excludes actual stone positions from territory counting.
    
    Args:
        before: Ownership map before move (from current player's perspective)
        after: Ownership map after move (from current player's perspective)
        color: Current player (Board.BLACK or Board.WHITE)
        board: Board state to check for stones
    """
    regions = ["corner_tl", "corner_tr", "corner_bl", "corner_br",
               "side_left", "side_right", "side_top", "side_bottom", "center"]
    
    building_count = {region: 0 for region in regions}
    building_intensity = {region: 0.0 for region in regions}
    solidification_count = {region: 0 for region in regions}
    solidification_intensity = {region: 0.0 for region in regions}
    
    for y in range(before.shape[0]):
        for x in range(before.shape[1]):
            loc = board.loc(x, y)
            # Skip stone positions
            if board.board[loc] != Board.EMPTY:
                continue
                
            region = classify_region(x, y, before.shape[0])
            
            # Building territory (both maps are from current player's perspective)
            # For current player, building means positive values
            if (np.abs(before[y, x]) < TAU_POS) and (after[y, x] > TAU_POS):
                building_count[region] += 1
                building_intensity[region] += after[y, x] - before[y, x]
            
            # Solidification for current player
            if (before[y, x] > TAU_POS) and (after[y, x] > before[y, x]):
                solidification_count[region] += 1
                solidification_intensity[region] += after[y, x] - before[y, x]
    
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
    color: int,
    board: Board
) -> Tuple[Dict[str, int], Dict[str, float]]:
    """Compute reduction deltas by region.
    Excludes actual stone positions from territory counting.
    
    Args:
        before: Ownership map before move (from current player's perspective)
        after: Ownership map after move (from current player's perspective)
        color: Current player (Board.BLACK or Board.WHITE)
        board: Board state to check for stones
    """
    regions = ["corner_tl", "corner_tr", "corner_bl", "corner_br",
               "side_left", "side_right", "side_top", "side_bottom", "center"]
    
    reduction_count = {region: 0 for region in regions}
    reduction_intensity = {region: 0.0 for region in regions}
    
    for y in range(before.shape[0]):
        for x in range(before.shape[1]):
            loc = board.loc(x, y)
            # Skip stone positions
            if board.board[loc] != Board.EMPTY:
                continue
                
            region = classify_region(x, y, before.shape[0])
            
            # Reduction (both maps are from current player's perspective)
            # For current player, opponent territory is negative, reduction means less negative
            if (before[y, x] < -TAU_POS) and (after[y, x] > before[y, x]):
                reduction_count[region] += 1
                reduction_intensity[region] += abs(after[y, x] - before[y, x])
    
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


def is_only_move(policy: np.ndarray, eps: float = TAU_ONLY_MOVE) -> bool:
    """Check if policy has only one move with significant probability (>0.95)."""
    # Only move if the maximum probability is > 0.95
    max_prob = float(np.max(policy))
    return max_prob > (1.0 - eps)




def is_tenuki(selected_idx: int, last_move_loc: Optional[int], policy: np.ndarray, board: Board) -> bool:
    """Check if move is tenuki (different area + distance > 6 + local follow-up ignored)."""
    if last_move_loc is None or selected_idx is None or last_move_loc == Board.PASS_LOC:
        return False
    
    size = 19
    x_sel, y_sel = selected_idx % size, selected_idx // size
    x_last, y_last = loc_to_xy(board, last_move_loc)
    
    # Check distance requirement (L1 >= 6)
    l1_distance = abs(x_sel - x_last) + abs(y_sel - y_last)
    if l1_distance < 6:
        return False
    
    region_sel = classify_region(x_sel, y_sel, size)
    region_last = classify_region(x_last, y_last, size)
    
    if region_sel == region_last:
        return False
    
    # Check if there are candidates within L1 <= 4 of last move with higher probability
    selected_prob = policy[selected_idx]
    for y in range(size):
        for x in range(size):
            # Check if within L1 <= 4 of last move
            if abs(x - x_last) + abs(y - y_last) <= 4:
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
                
                # Check if this group is almost in atari (2 liberty) AND the move is one of its liberties
                if board.group_liberty_count[head] == 2:
                    # Get the liberties of this group and check if move_loc is one of them
                    group_liberties = get_group_liberties(board, head)
                    if move_loc in group_liberties:
                        return True
    
    return False


def get_group_liberties(board: Board, group_head: int) -> Set[int]:
    """Get all liberties of a deterministic group."""
    if board.board[group_head] == Board.EMPTY:
        return set()
    
    liberties = set()
    cur = group_head
    
    # Walk through all stones in the group
    while True:
        stone_x, stone_y = loc_to_xy(board, cur)
        
        # Check all 4 orthogonal directions for liberties
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            lib_x, lib_y = stone_x + dx, stone_y + dy
            if in_bounds(lib_x, lib_y, board.size):
                lib_loc = xy_to_loc(board, lib_x, lib_y)
                if board.board[lib_loc] == Board.EMPTY:
                    liberties.add(lib_loc)
        
        cur = board.group_next[cur]
        if cur == group_head:
            break
    
    return liberties


def reduce_aji(before: np.ndarray, after: np.ndarray, board: Board, color: int, move_loc: Optional[int] = None) -> Tuple[bool, float]:
    """Calculate aji reduction (increase in own ownership over opponent groups that were under own ownership).
    
    Args:
        before: Ownership map before move (from current player's perspective)
        after: Ownership map after move (from current player's perspective)
        board: Board state
        color: Current player (Board.BLACK or Board.WHITE)
        move_loc: Location of the move (optional, for vicinity mask)
    """
    opp = Board.get_opp(color)
    size = board.size
    delta: List[float] = []
    
    # If move_loc is provided, create vicinity mask
    vicinity_mask = None
    if move_loc is not None and move_loc != Board.PASS_LOC:
        move_x, move_y = loc_to_xy(board, move_loc)
        vicinity_mask = np.zeros((size, size), dtype=bool)
        for y in range(size):
            for x in range(size):
                if abs(x - move_x) + abs(y - move_y) <= TAU_AJI_VICINITY:
                    vicinity_mask[y, x] = True
    
    for y in range(size):
        for x in range(size):
            loc = xy_to_loc(board, x, y)
            if board.board[loc] == opp:
                # Only consider opponent stones that were under own ownership before
                # For current player, own ownership is positive
                if before[y, x] > TAU_POS:  # Was under own ownership
                    # Apply vicinity mask if available
                    if vicinity_mask is None or vicinity_mask[y, x]:
                        ownership_increase = after[y, x] - before[y, x]
                        delta.append(ownership_increase)
    
    if not delta:
        return False, 0.0
    
    aji_reduction_intensity = float(np.mean(delta))
    # Aji reduction occurs if the mean increase is >= 0.05
    reduces_aji = aji_reduction_intensity >= 0.05
    
    return reduces_aji, aji_reduction_intensity


def attack_strength_delta(groups_before: List[Group], groups_after: List[Group], opp_color: int, board: Board) -> Tuple[bool, float, float]:
    """Calculate attack strength using spatial overlap matching instead of head matching."""
    before_groups = [g for g in groups_before if g.color == opp_color]
    after_groups = [g for g in groups_after if g.color == opp_color]
    
    if not before_groups or not after_groups:
        return False, 0.0, 0.0
    
    # Create spatial masks for each group
    def group_to_mask(group: Group, size: int) -> np.ndarray:
        mask = np.zeros((size, size), dtype=bool)
        for loc in group.stones:
            x, y = loc_to_xy(board, loc)
            mask[y, x] = True
        return mask
    
    # Match groups by spatial overlap (IoU)
    matched_pairs = []
    used_after = set()
    
    for before_group in before_groups:
        before_mask = group_to_mask(before_group, board.size)
        best_iou = 0.0
        best_after_group = None
        
        for i, after_group in enumerate(after_groups):
            if i in used_after:
                continue
            
            after_mask = group_to_mask(after_group, board.size)
            
            # Calculate IoU
            intersection = np.sum(before_mask & after_mask)
            union = np.sum(before_mask | after_mask)
            iou = intersection / union if union > 0 else 0.0
            
            if iou > best_iou and iou >= TAU_GROUP_IOU:
                best_iou = iou
                best_after_group = after_group
        
        if best_after_group is not None:
            matched_pairs.append((before_group, best_after_group))
            used_after.add(after_groups.index(best_after_group))
    
    if not matched_pairs:
        return False, 0.0, 0.0
    
    # Calculate strength deltas for matched pairs
    deltas = [after.strength - before.strength for before, after in matched_pairs]
    avg_attack_intensity = float(np.mean(deltas))
    max_attack_intensity = float(min(deltas))  # Most negative (strongest attack)
    
    is_attack = avg_attack_intensity < 0  # Negative change means attack
    
    return is_attack, abs(avg_attack_intensity), abs(max_attack_intensity)


def compute_attack_metrics_avg_max(
    groups_before: List[Group], 
    groups_after: List[Group], 
    board: Board,
    opp_color: int
) -> Dict[str, Any]:
    """Compute average and max attack strength deltas for opponent groups.
    
    Args:
        groups_before: Groups before the move
        groups_after: Groups after the move  
        board: Board state
        opp_color: Opponent color being attacked
    
    Returns:
        Dictionary with avg/max attack metrics and group locations
    """
    # Filter groups by opponent color
    before_groups = [g for g in groups_before if g.color == opp_color]
    after_groups = [g for g in groups_after if g.color == opp_color]
    
    if not before_groups or not after_groups:
        return {
            "avg_attack_intensity": 0.0,
            "max_attack_intensity": 0.0,
            "max_attack_group_location": "none",
            "is_attack": False
        }
    
    # Match groups by spatial overlap (IoU) - same logic as above
    def group_to_mask(group: Group, size: int) -> np.ndarray:
        mask = np.zeros((size, size), dtype=bool)
        for loc in group.stones:
            x, y = loc_to_xy(board, loc)
            mask[y, x] = True
        return mask
    
    def get_group_location(group: Group, board: Board) -> str:
        """Get the most frequent region for stones in the group."""
        region_counts = {}
        for loc in group.stones:
            x, y = loc_to_xy(board, loc)
            region = classify_region(x, y, board.size)
            region_counts[region] = region_counts.get(region, 0) + 1
        
        if not region_counts:
            return "none"
        
        return max(region_counts, key=region_counts.get)
    
    matched_pairs = []
    used_after = set()
    
    for before_group in before_groups:
        before_mask = group_to_mask(before_group, board.size)
        best_iou = 0.0
        best_after_group = None
        
        for i, after_group in enumerate(after_groups):
            if i in used_after:
                continue
            
            after_mask = group_to_mask(after_group, board.size)
            
            # Calculate IoU
            intersection = np.sum(before_mask & after_mask)
            union = np.sum(before_mask | after_mask)
            iou = intersection / union if union > 0 else 0.0
            
            if iou > best_iou and iou >= TAU_GROUP_IOU:
                best_iou = iou
                best_after_group = after_group
        
        if best_after_group is not None:
            matched_pairs.append((before_group, best_after_group))
            used_after.add(after_groups.index(best_after_group))
    
    if not matched_pairs:
        return {
            "avg_attack_intensity": 0.0,
            "max_attack_intensity": 0.0,
            "max_attack_group_location": "none",
            "is_attack": False
        }
    
    # Calculate attack deltas (negative values = attack)
    attack_deltas = [after.strength - before.strength for before, after in matched_pairs]
    
    # Find max attack (most negative delta)
    max_attack_idx = attack_deltas.index(min(attack_deltas))
    max_attack_group = matched_pairs[max_attack_idx][1]  # after group
    
    avg_attack_intensity = float(np.mean([abs(d) for d in attack_deltas if d < 0]))
    max_attack_intensity = float(abs(min(attack_deltas)))
    is_attack = any(d < 0 for d in attack_deltas)
    
    return {
        "avg_attack_intensity": avg_attack_intensity,
        "max_attack_intensity": max_attack_intensity,
        "max_attack_group_location": get_group_location(max_attack_group, board),
        "is_attack": is_attack
    }


def killing_attack(groups_before: List[Group], groups_after: List[Group], opp_color: int, board: Board, move_loc: Optional[int] = None) -> Tuple[bool, float]:
    """Check if move creates a killing attack by comparing before/after group strengths.
    
    A killing attack occurs when:
    1. An opponent group's strength significantly decreases (becomes more negative)
    2. The final strength is <= -0.5 (strong own ownership over opponent stones)
    3. The move is spatially related to the attacked group (within L1 distance 3)
    """
    if move_loc is None or move_loc == Board.PASS_LOC:
        return False, 0.0
    
    # Find opponent groups that were significantly weakened
    killed_groups = []
    
    # Create spatial masks for each group to match them
    def group_to_mask(group: Group, size: int) -> np.ndarray:
        mask = np.zeros((size, size), dtype=bool)
        for loc in group.stones:
            x, y = loc_to_xy(board, loc)
            mask[y, x] = True
        return mask
    
    # Match groups by spatial overlap (IoU)
    before_groups = [g for g in groups_before if g.color == opp_color]
    after_groups = [g for g in groups_after if g.color == opp_color]
    
    matched_pairs = []
    used_after = set()
    
    for before_group in before_groups:
        before_mask = group_to_mask(before_group, board.size)
        best_iou = 0.0
        best_after_group = None
        
        for i, after_group in enumerate(after_groups):
            if i in used_after:
                continue
            
            after_mask = group_to_mask(after_group, board.size)
            
            # Calculate IoU
            intersection = np.sum(before_mask & after_mask)
            union = np.sum(before_mask | after_mask)
            iou = intersection / union if union > 0 else 0.0
            
            if iou > best_iou and iou >= TAU_GROUP_IOU:
                best_iou = iou
                best_after_group = after_group
        
        if best_after_group is not None:
            matched_pairs.append((before_group, best_after_group))
            used_after.add(after_groups.index(best_after_group))
    
    # Check for killing attacks in matched groups
    move_x, move_y = loc_to_xy(board, move_loc)
    
    for before_group, after_group in matched_pairs:
        # Check if group was significantly weakened
        strength_delta = after_group.strength - before_group.strength
        is_significantly_weakened = strength_delta < -0.2  # Significant weakening
        
        # Check if final strength indicates killing
        is_killed = after_group.strength <= -0.5

        if is_significantly_weakened and is_killed:
            killed_groups.append(after_group)
    
    if not killed_groups:
        return False, 0.0
    
    # Kill intensity is the mean of the absolute strength values of killed groups
    kill_intensity = float(np.mean([abs(g.strength) for g in killed_groups]))
    return True, kill_intensity


# -------------------------
# Scope and Local Analysis Helpers
# -------------------------

def create_local_mask(move_loc: int, board: Board, radius: int) -> np.ndarray:
    """Create a mask for local analysis around a move location."""
    if move_loc == Board.PASS_LOC:
        return np.zeros((board.size, board.size), dtype=bool)
    
    mask = np.zeros((board.size, board.size), dtype=bool)
    move_x, move_y = loc_to_xy(board, move_loc)
    
    for y in range(board.size):
        for x in range(board.size):
            if abs(x - move_x) + abs(y - move_y) <= radius:
                mask[y, x] = True
    
    return mask


def apply_local_mask_to_territory_deltas(
    before: np.ndarray, 
    after: np.ndarray, 
    local_mask: np.ndarray,
    color: int, 
    board: Board
) -> Tuple[int, float, float, int, float, float, int, float, float]:
    """Apply local mask to territory delta calculations and return both local and global metrics."""
    # Global metrics
    building_count_global, building_intensity_global, building_sum_global = count_building_territory(before, after, color, board)
    solidification_count_global, solidification_intensity_global, solidification_sum_global = solidify_territory_delta(before, after, color, board)
    reduction_count_global, reduction_intensity_global, reduction_sum_global = reduce_opponent_territory_count(before, after, color, board)
    
    # Local metrics (apply mask)
    building_count_local, building_intensity_local, building_sum_local = count_building_territory(
        before * local_mask, after * local_mask, color, board
    )
    solidification_count_local, solidification_intensity_local, solidification_sum_local = solidify_territory_delta(
        before * local_mask, after * local_mask, color, board
    )
    reduction_count_local, reduction_intensity_local, reduction_sum_local = reduce_opponent_territory_count(
        before * local_mask, after * local_mask, color, board
    )
    
    return (
        building_count_local, building_intensity_local, building_sum_local,
        solidification_count_local, solidification_intensity_local, solidification_sum_local,
        reduction_count_local, reduction_intensity_local, reduction_sum_local,
        building_count_global, building_intensity_global, building_sum_global,
        solidification_count_global, solidification_intensity_global, solidification_sum_global,
        reduction_count_global, reduction_intensity_global, reduction_sum_global
    )


# -------------------------
# Ownership Frame Normalization Helper
# -------------------------

def normalize_ownership_to_player_frame(
    ownership: np.ndarray,
    ownership_frame_player: int,
    target_player: int
) -> np.ndarray:
    """
    Return `ownership` expressed in `target_player` frame (positive = target_player).
    If the current `ownership` is from the opponent's frame, flip sign.
    """
    return ownership if ownership_frame_player == target_player else -ownership


def normalize_before_by_alignment(
    before_ownership_raw: np.ndarray,
    after_ownership_player_frame: np.ndarray,
    tau: float = 0.15
) -> np.ndarray:
    """
    Heuristically align `before_ownership_raw` to the same frame as `after_ownership_player_frame`
    by choosing the sign that maximizes agreement on stable-magnitude points.

    We select points where either |before| or |after| is reasonably strong (>= tau),
    then compute dot product. If negative, flip `before`.
    """
    b = before_ownership_raw.astype(np.float64, copy=False)
    a = after_ownership_player_frame.astype(np.float64, copy=False)

    # Focus on points with meaningful ownership to avoid noise
    mask = (np.abs(a) >= tau) | (np.abs(b) >= tau)
    if not np.any(mask):
        # If everything is tiny, just return as-is (no basis to align)
        print(f"[warn] No meaningful ownership points found for alignment (tau={tau})")
        return b.copy()

    score_same = float(np.sum(a[mask] * b[mask]))
    score_flipped = float(np.sum(a[mask] * (-b[mask])))
    
    print(f"[debug] Alignment scores: same={score_same:.3f}, flipped={score_flipped:.3f}")
    
    # If agreeing is worse than disagreeing, flip
    if score_same < score_flipped:
        print(f"[debug] Flipping before ownership map")
        return -b
    else:
        print(f"[debug] Keeping before ownership map as-is")
        return b.copy()


def verify_coordinate_mappings():
    """
    Verify that policy index mapping and region classification work correctly.
    This is a diagnostic function to ensure coordinate systems are aligned.
    """
    print("=== COORDINATE MAPPING VERIFICATION ===")
    
    # Test policy index mapping: (0,0) should be index 0, (18,18) should be index 360
    size = 19
    for test_case in [(0, 0, 0), (18, 18, 360), (9, 9, 180)]:
        x, y, expected_idx = test_case
        actual_idx = y * size + x
        print(f"({x},{y}) -> idx {actual_idx} (expected {expected_idx}): {'✓' if actual_idx == expected_idx else '✗'}")
    
    # Test region classification for star points
    star_points = [(3, 3, "corner_tl"), (15, 3, "corner_tr"), (3, 15, "corner_bl"), (15, 15, "corner_br"), (9, 9, "center")]
    for x, y, expected_region in star_points:
        actual_region = classify_region(x, y, size)
        print(f"Star point ({x},{y}) -> {actual_region} (expected {expected_region}): {'✓' if actual_region == expected_region else '✗'}")
    
    # Test specific move 16 location
    move_16_x, move_16_y = 4, 5
    move_16_idx = move_16_y * size + move_16_x
    move_16_region = classify_region(move_16_x, move_16_y, size)
    print(f"Move 16 location ({move_16_x},{move_16_y}) -> idx {move_16_idx}, region {move_16_region}")
    
    # Test the areas that were previously misclassified
    test_areas = [
        (4, 4, "e5"), (5, 5, "f6"),  # These should now be corner_tl
        (13, 13, "n14"), (14, 14, "o15")  # These should now be corner_br
    ]
    print("\nTesting previously misclassified areas:")
    for x, y, coord in test_areas:
        region = classify_region(x, y, size)
        print(f"{coord} ({x},{y}) -> {region}")
    
    print("=====================================")


# -------------------------
# Comprehensive Analysis Function
# -------------------------

def analyze_position_comprehensive(
    board: Board, 
    ownership: np.ndarray, 
    policy: np.ndarray,
    player: Optional[int] = None,
    move_loc: Optional[int] = None,
    last_move_loc: Optional[int] = None,
    before_ownership: Optional[np.ndarray] = None,
    before_board: Optional[Board] = None,
    scope_radius: int = 5
) -> Dict[str, Any]:
    """
    Perform comprehensive analysis of a position using all 28 concepts.
    
    Args:
        board: Current board state
        ownership: Current ownership map (19x19) - from current player's perspective
        policy: Policy distribution (361) - should be pre-move policy
        player: Current player (Board.BLACK or Board.WHITE)
        move_loc: Location of current move (optional)
        last_move_loc: Location of last move (optional)
        before_ownership: Ownership before current move (optional) - from current player's perspective
        before_board: Board state before current move (optional) - required for accurate frame alignment
        scope_radius: Radius for local vs global metrics (default 5)
    
    Returns:
        Dictionary containing all analysis results matching the specification.
        Note: For annotation purposes, these can be treated as tags with values
        and directions (plus/minus) as mentioned in the original specification.
        Pass moves short-circuit territory delta metrics and tenuki/only-move checks.
    """
    results = {}
    
    # Auto-sync player to board.pla (single source of truth)
    if player is None:
        player = board.pla
    elif player != board.pla:
        # Prefer being forgiving in production: auto-sync and log once
        if not hasattr(analyze_position_comprehensive, '_player_sync_warned'):
            print(f"[warn] player({player}) != board.pla({board.pla}); using board.pla")
            analyze_position_comprehensive._player_sync_warned = True
        player = board.pla
    
    # Handle pass moves - short-circuit territory deltas and some tactical checks
    is_pass_move = (move_loc is not None and move_loc == Board.PASS_LOC)
    
    # Verify coordinate mappings (run once for debugging)
    if not hasattr(analyze_position_comprehensive, '_coordinate_verified'):
        verify_coordinate_mappings()
        analyze_position_comprehensive._coordinate_verified = True
    
    # Normalize AFTER map: assumed to be in current player's perspective
    ownership_current = ownership.copy()
    
    # Normalize BEFORE map with fallback to heuristic alignment
    before_ownership_current = None
    if before_ownership is not None:
        if before_board is not None:
            # Use exact normalization when available
            before_ownership_current = normalize_ownership_to_player_frame(
                before_ownership,
                ownership_frame_player=before_board.pla,  # who the before map was positive for
                target_player=player                      # who we want positivity for now
            )
        else:
            # Fall back to alignment heuristic (no need to raise)
            before_ownership_current = normalize_before_by_alignment(
                before_ownership_raw=before_ownership,
                after_ownership_player_frame=ownership_current,
                tau=0.15
            )
    
    # Diagnostic prints for ownership frame validation (remove after testing)
    if before_ownership_current is not None:
        print("=== OWNERSHIP FRAME DIAGNOSTICS ===")
        if before_board is not None:
            print(f"player: {player}, before was framed for: {before_board.pla}")
        else:
            print(f"player: {player}, before frame: heuristic alignment")
        print(f"mean(before_ownership_current): {float(before_ownership_current.mean()):.4f}")
        print(f"mean(after_ownership_current): {float(ownership_current.mean()):.4f}")
        print(f"before>TAU_POS count: {np.sum(before_ownership_current > TAU_POS)}")
        print(f"after>TAU_POS count: {np.sum(ownership_current > TAU_POS)}")
        print(f"before<-TAU_POS count: {np.sum(before_ownership_current < -TAU_POS)}")
        print(f"after<-TAU_POS count: {np.sum(ownership_current < -TAU_POS)}")
        
        # Alignment verification
        dot = float(np.sum(before_ownership_current * ownership_current))
        print(f"[debug] alignment dot={dot:.3f}")
        
        # Additional debugging for move location
        if move_loc is not None and move_loc != Board.PASS_LOC:
            move_x, move_y = loc_to_xy(board, move_loc)
            print(f"move_loc: {move_loc} -> ({move_x}, {move_y})")
            print(f"before[move]: {float(before_ownership_current[move_y, move_x]):.4f}")
            print(f"after[move]: {float(ownership_current[move_y, move_x]):.4f}")
            print(f"move region: {classify_region(move_x, move_y, board.size)}")
        
        print("===================================")
    
    # 1-2. Urgency by region (regions are static, not needed in output)
    # Note: Urgency is computed from policy before the move, so it represents
    # the urgency of the position before the current move was played
    results["urgency"] = urgency_by_region(policy)
    results["urgency_intensity"] = urgency_intensity_by_region(policy)
    
    # 3-7. Groups and influence
    groups_det = enumerate_groups_deterministic(board)
    groups_own = enumerate_groups_ownership(board, ownership_current, player)
    
    compute_group_strengths(groups_det, ownership_current, player, board)
    compute_group_strengths(groups_own, ownership_current, player, board)
    compute_group_connectivity(groups_det, ownership_current, board)
    compute_group_connectivity(groups_own, ownership_current, board)
    compute_group_influence(groups_det, ownership_current, board)
    compute_group_influence(groups_own, ownership_current, board)
    
    # Groups are used for computation but not returned (per note #1)
    # Only derived metrics are included in results
    
    # 8-16. Territory analysis (skip for pass moves)
    if before_ownership_current is not None and not is_pass_move:
        # Create local mask for scope-based analysis
        local_mask = create_local_mask(move_loc, board, scope_radius)
        
        # Get both local and global territory metrics
        (building_count_local, building_intensity_local, building_sum_local,
         solidification_count_local, solidification_intensity_local, solidification_sum_local,
         reduction_count_local, reduction_intensity_local, reduction_sum_local,
         building_count_global, building_intensity_global, building_sum_global,
         solidification_count_global, solidification_intensity_global, solidification_sum_global,
         reduction_count_global, reduction_intensity_global, reduction_sum_global) = apply_local_mask_to_territory_deltas(
            before_ownership_current, ownership_current, local_mask, player, board
        )
        
        # Store both local and global metrics
        results["building_count_local"] = building_count_local
        results["building_intensity_local"] = building_intensity_local
        results["building_sum_local"] = building_sum_local
        results["building_count_global"] = building_count_global
        results["building_intensity_global"] = building_intensity_global
        results["building_sum_global"] = building_sum_global
        
        results["solidification_count_local"] = solidification_count_local
        results["solidification_intensity_local"] = solidification_intensity_local
        results["solidification_sum_local"] = solidification_sum_local
        results["solidification_count_global"] = solidification_count_global
        results["solidification_intensity_global"] = solidification_intensity_global
        results["solidification_sum_global"] = solidification_sum_global
        
        results["reduction_count_local"] = reduction_count_local
        results["reduction_intensity_local"] = reduction_intensity_local
        results["reduction_sum_local"] = reduction_sum_local
        results["reduction_count_global"] = reduction_count_global
        results["reduction_intensity_global"] = reduction_intensity_global
        results["reduction_sum_global"] = reduction_sum_global
        
        # Legacy fields (use global for backward compatibility)
        results["building_count"] = building_count_global
        results["building_intensity"] = building_intensity_global
        results["building_sum"] = building_sum_global
        results["solidification_count"] = solidification_count_global
        results["solidification_intensity"] = solidification_intensity_global
        results["solidification_sum"] = solidification_sum_global
        results["reduction_count"] = reduction_count_global
        results["reduction_intensity"] = reduction_intensity_global
        results["reduction_sum"] = reduction_sum_global
        
        # Invasion (global only)
        is_invasion, invasion_intensity = invasion_effect(before_ownership_current, ownership_current, player, board, move_loc)
        results["invasion"] = is_invasion
        results["invasion_intensity"] = invasion_intensity
        
        # Additional diagnostic prints for territory analysis
        print(f"building_count (local/global): {building_count_local}/{building_count_global}")
        print(f"reduction_count (local/global): {reduction_count_local}/{reduction_count_global}")
        print(f"invasion: {results['invasion']}, invasion_intensity: {results['invasion_intensity']:.4f}")
        
        # Leaving weakness
        results["leaves_weakness"] = leaving_weakness(before_ownership_current, ownership_current, player, board)
        
        # Regional weakening
        weakening_count = {}
        weakening_intensity = {}
        for region in ["corner_tl", "corner_tr", "corner_bl", "corner_br",
                       "side_left", "side_right", "side_top", "side_bottom", "center"]:
            count, intensity = weakening_territory_in_region(before_ownership_current, ownership_current, region, player)
            weakening_count[region] = count
            weakening_intensity[region] = intensity
        results["weakening_count_by_region"] = weakening_count
        results["weakening_intensity_by_region"] = weakening_intensity
        
        # Regional territory deltas
        building_count_by_region, building_intensity_by_region, solidification_count_by_region, solidification_value_by_region = compute_territory_delta_by_region(before_ownership_current, ownership_current, player, board)
        results["building_count_by_region"] = building_count_by_region
        results["building_intensity_by_region"] = building_intensity_by_region
        results["solidification_count_by_region"] = solidification_count_by_region
        results["solidification_value_by_region"] = solidification_value_by_region
        
        reduction_count_by_region, reduction_intensity_by_region = compute_reduction_delta_by_region(before_ownership_current, ownership_current, player, board)
        results["reduction_count_by_region"] = reduction_count_by_region
        results["reduction_intensity_by_region"] = reduction_intensity_by_region
        
        # Debug regional reduction
        print("=== REGIONAL REDUCTION DEBUG ===")
        for region, count in reduction_count_by_region.items():
            if count > 0:
                print(f"{region}: {count} reductions (intensity: {reduction_intensity_by_region[region]:.3f})")
        print("================================")
        
    else:
        # Set defaults when no before_ownership available or pass move
        results["building_count_local"] = 0
        results["building_intensity_local"] = 0.0
        results["building_sum_local"] = 0.0
        results["building_count_global"] = 0
        results["building_intensity_global"] = 0.0
        results["building_sum_global"] = 0.0
        results["solidification_count_local"] = 0
        results["solidification_intensity_local"] = 0.0
        results["solidification_sum_local"] = 0.0
        results["solidification_count_global"] = 0
        results["solidification_intensity_global"] = 0.0
        results["solidification_sum_global"] = 0.0
        results["reduction_count_local"] = 0
        results["reduction_intensity_local"] = 0.0
        results["reduction_sum_local"] = 0.0
        results["reduction_count_global"] = 0
        results["reduction_intensity_global"] = 0.0
        results["reduction_sum_global"] = 0.0
        results["invasion"] = False
        results["invasion_intensity"] = 0.0
        results["leaves_weakness"] = 0
        
        # Legacy fields
        results["building_count"] = 0
        results["building_intensity"] = 0.0
        results["building_sum"] = 0.0
        results["solidification_count"] = 0
        results["solidification_intensity"] = 0.0
        results["solidification_sum"] = 0.0
        results["reduction_count"] = 0
        results["reduction_intensity"] = 0.0
        results["reduction_sum"] = 0.0
        results["weakening_count_by_region"] = {region: 0 for region in ["corner_tl", "corner_tr", "corner_bl", "corner_br", "side_left", "side_right", "side_top", "side_bottom", "center"]}
        results["weakening_intensity_by_region"] = {region: 0.0 for region in ["corner_tl", "corner_tr", "corner_bl", "corner_br", "side_left", "side_right", "side_top", "side_bottom", "center"]}
        results["building_count_by_region"] = {region: 0 for region in ["corner_tl", "corner_tr", "corner_bl", "corner_br", "side_left", "side_right", "side_top", "side_bottom", "center"]}
        results["building_intensity_by_region"] = {region: 0.0 for region in ["corner_tl", "corner_tr", "corner_bl", "corner_br", "side_left", "side_right", "side_top", "side_bottom", "center"]}
        results["solidification_count_by_region"] = {region: 0 for region in ["corner_tl", "corner_tr", "corner_bl", "corner_br", "side_left", "side_right", "side_top", "side_bottom", "center"]}
        results["solidification_value_by_region"] = {region: 0.0 for region in ["corner_tl", "corner_tr", "corner_bl", "corner_br", "side_left", "side_right", "side_top", "side_bottom", "center"]}
        results["reduction_count_by_region"] = {region: 0 for region in ["corner_tl", "corner_tr", "corner_bl", "corner_br", "side_left", "side_right", "side_top", "side_bottom", "center"]}
        results["reduction_intensity_by_region"] = {region: 0.0 for region in ["corner_tl", "corner_tr", "corner_bl", "corner_br", "side_left", "side_right", "side_top", "side_bottom", "center"]}
    
    # 14-16. Territory sizes and sacrifices
    potential, solid = territory_sizes(ownership_current, player, board)
    results["potential_territory"] = potential
    results["solid_territory"] = solid
    
    if move_loc is not None:
        is_direct_sacrifice, sacrifice_intensity = direct_sacrifice(move_loc, ownership_current, player, board)
        results["direct_sacrifice"] = is_direct_sacrifice
        results["sacrifice_intensity"] = sacrifice_intensity
        
        if before_ownership_current is not None:
            indirect_count, indirect_sacrifice_intensity = indirect_sacrifice(before_ownership_current, ownership_current, player, board)
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
    
    # 18-25. Tactical concepts (skip for pass moves)
    if move_loc is not None and not is_pass_move:
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
    
    results["only_move"] = is_only_move(policy) if not is_pass_move else False
    
    if last_move_loc is not None and move_loc is not None and not is_pass_move:
        move_idx = loc_to_xy(board, move_loc)[1] * 19 + loc_to_xy(board, move_loc)[0]
        results["tenuki"] = is_tenuki(move_idx, last_move_loc, policy, board)
    else:
        results["tenuki"] = False
    
    # 26-28. Attack concepts
    if before_ownership_current is not None:
        reduces_aji, aji_reduction_intensity = reduce_aji(before_ownership_current, ownership_current, board, player, move_loc)
        results["reduce_aji"] = reduces_aji
        results["aji_reduction_intensity"] = aji_reduction_intensity
        
        # Calculate attack effects
        if before_board is not None:
            groups_before = enumerate_groups_deterministic(before_board)
            is_attack, avg_attack_intensity, max_attack_intensity = attack_strength_delta(groups_before, groups_det, Board.get_opp(player), board)
            results["attack"] = is_attack
            results["avg_attack_intensity"] = avg_attack_intensity
            results["max_attack_intensity"] = max_attack_intensity
            
            # Group deltas by region (legacy)
            group_strength_delta_by_region = compute_group_strength_delta_by_region(groups_before, groups_det, board)
            results["group_strength_delta_by_region"] = group_strength_delta_by_region
            results["group_strength_delta"] = sum(group_strength_delta_by_region.values())
            
            # Max group strength delta (which group is being helped most) (legacy)
            max_strength_delta, max_strength_region = compute_max_group_strength_delta(groups_before, groups_det, board)
            results["max_group_strength_delta"] = max_strength_delta
            results["max_group_strength_region"] = max_strength_region
            
            group_connectivity_delta_by_region = compute_group_connectivity_delta_by_region(groups_before, groups_det, board)
            results["group_connectivity_delta_by_region"] = group_connectivity_delta_by_region
            results["group_connectivity_delta"] = sum(group_connectivity_delta_by_region.values())
            
            # Max group connectivity delta (which group's connectivity is being improved most) (legacy)
            max_connectivity_delta, max_connectivity_region = compute_max_group_connectivity_delta(groups_before, groups_det, board)
            results["max_group_connectivity_delta"] = max_connectivity_delta
            results["max_group_connectivity_region"] = max_connectivity_region
            
            # NEW: Improved group metrics with avg/max and locations
            own_group_metrics = compute_group_metrics_avg_max(groups_before, groups_det, board, player)
            results.update({f"own_{k}": v for k, v in own_group_metrics.items()})
            
            # Attack metrics for opponent groups
            attack_metrics = compute_attack_metrics_avg_max(groups_before, groups_det, board, Board.get_opp(player))
            results.update(attack_metrics)
            
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
            results["max_group_strength_delta"] = 0.0
            results["max_group_strength_region"] = "none"
            results["group_connectivity_delta_by_region"] = {region: 0.0 for region in ["corner_tl", "corner_tr", "corner_bl", "corner_br", "side_left", "side_right", "side_top", "side_bottom", "center"]}
            results["group_connectivity_delta"] = 0.0
            results["max_group_connectivity_delta"] = 0.0
            results["max_group_connectivity_region"] = "none"
            
            # NEW: Default values for improved metrics
            results["own_avg_strength_delta"] = 0.0
            results["own_max_strength_delta"] = 0.0
            results["own_max_strength_group_location"] = "none"
            results["own_avg_connectivity_delta"] = 0.0
            results["own_max_connectivity_delta"] = 0.0
            results["own_max_connectivity_group_location"] = "none"
            results["avg_attack_intensity"] = 0.0
            results["max_attack_intensity"] = 0.0
            results["max_attack_group_location"] = "none"
            results["influence_count_delta_by_region"] = {region: 0 for region in ["corner_tl", "corner_tr", "corner_bl", "corner_br", "side_left", "side_right", "side_top", "side_bottom", "center"]}
            results["influence_count_delta"] = 0
            results["influence_strength_delta_by_region"] = {region: 0.0 for region in ["corner_tl", "corner_tr", "corner_bl", "corner_br", "side_left", "side_right", "side_top", "side_bottom", "center"]}
            results["influence_strength_delta"] = 0.0
            results["creates_new_group"] = False
        
        if before_board is not None:
            groups_before = enumerate_groups_deterministic(before_board)
            is_killing_attack, kill_intensity = killing_attack(groups_before, groups_det, Board.get_opp(player), board, move_loc)
        else:
            is_killing_attack, kill_intensity = False, 0.0
        results["killing_attack"] = is_killing_attack
        results["kill_intensity"] = kill_intensity
    else:
        results["reduce_aji"] = False
        results["aji_reduction_intensity"] = 0.0
        results["attack"] = False
        results["attack_intensity"] = 0.0
        results["group_strength_delta_by_region"] = {region: 0.0 for region in ["corner_tl", "corner_tr", "corner_bl", "corner_br", "side_left", "side_right", "side_top", "side_bottom", "center"]}
        results["group_strength_delta"] = 0.0
        results["max_group_strength_delta"] = 0.0
        results["max_group_strength_region"] = "none"
        results["group_connectivity_delta_by_region"] = {region: 0.0 for region in ["corner_tl", "corner_tr", "corner_bl", "corner_br", "side_left", "side_right", "side_top", "side_bottom", "center"]}
        results["group_connectivity_delta"] = 0.0
        results["max_group_connectivity_delta"] = 0.0
        results["max_group_connectivity_region"] = "none"
        
        # NEW: Default values for improved metrics
        results["own_avg_strength_delta"] = 0.0
        results["own_max_strength_delta"] = 0.0
        results["own_max_strength_group_location"] = "none"
        results["own_avg_connectivity_delta"] = 0.0
        results["own_max_connectivity_delta"] = 0.0
        results["own_max_connectivity_group_location"] = "none"
        results["avg_attack_intensity"] = 0.0
        results["max_attack_intensity"] = 0.0
        results["max_attack_group_location"] = "none"
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
    "is_tenuki",
    "is_connection_move",
    "is_extension_move",
    "liberties_of_group",
    "atari_move",
    "reduce_aji",
    "attack_strength_delta",
    "killing_attack",
    "compute_group_strength_delta_by_region",
    "compute_max_group_strength_delta",
    "compute_group_connectivity_delta_by_region",
    "compute_max_group_connectivity_delta",
    "compute_group_metrics_avg_max",
    "compute_attack_metrics_avg_max",
    "create_local_mask",
    "apply_local_mask_to_territory_deltas",
    "normalize_ownership_to_player_frame",
    "normalize_before_by_alignment",
    "verify_coordinate_mappings",
    "analyze_position_comprehensive",
]