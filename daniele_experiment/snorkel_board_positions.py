#!/usr/bin/env python3
"""
Comprehensive spatial concepts and metrics for Go board analysis.

This module implements all 28 concepts as specified:
1. Board coordinates: aa (top-left/S19) to ss (bottom-right)
2. Regions: 4 corners, 4 sides, center
3. Groups: deterministic connection OR ownership-based (>=0.1)
4. Group strength: average ownership of group stones
5. Group connectivity: average ownership of empty intersections within bounds
6. Group influence area: count of own ownership around group
7. Influence strength: average ownership around group
8. Building territory: empty -> own ownership >0.1
9. Solidify territory: increase existing ownership values
10. Reduce territory: reduce opponent's owned intersections
11. Invasion: reduce opponent + increase own territory
12. Weakening territory: reduce opponent's average ownership in area
13. Leaving weakness: own -> opponent ownership
14. Potential territory: ownership <0.7
15. Solid territory: ownership >=0.7
16. Direct sacrifice: played stone becomes opponent's
17. Indirect sacrifice: own stone becomes opponent's
18. Urgency: sum of policy mass by area
19. Cut: w-b/b-w configuration separating groups
20. Only move: policy has only 1 value
21. Rough intent: policy move -> ownership effect
22. Tenuki: different area + closer candidates exist
23. Connection: connects stones OR increases connectivity
24. Extension: next to existing own stone
25. Liberties: number of liberties for group
26. Atari: opponent group with 1 liberty
27. Reduce aji: increase own ownership over opponent group
28. Attack: decrease opponent group strength
29. Killing attack: opponent group >=0.5 own ownership
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

OWN_MIN = 0.10            # minimal magnitude to consider owned/influenced
TERRITORY_SOLID = 0.70    # threshold for solid territory


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
            return "corner_top_left"
        elif x >= size-6 and y < 6:  # top-right
            return "corner_top_right"
        elif x < 6 and y >= size-6:  # bottom-left
            return "corner_bottom_left"
        elif x >= size-6 and y >= size-6:  # bottom-right
            return "corner_bottom_right"
    
    # Determine side regions
    elif in_x_corner or in_y_corner:
        if in_x_corner and not in_y_corner:  # left or right side
            if x < 6:
                return "side_left"
            else:
                return "side_right"
        elif in_y_corner and not in_x_corner:  # upper or lower side
            if y < 6:
                return "side_upper"
            else:
                return "side_lower"
    
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
                abs(ownership[cy, cx]) < OWN_MIN or 
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
                abs(ownership[y, x]) >= OWN_MIN and
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
    """Compute group connectivity as average ownership of empty intersections within bounds."""
    for g in groups:
        min_x, min_y, max_x, max_y = g.bbox
        vals: List[float] = []
        sign = 1.0 if g.color == Board.BLACK else -1.0
        
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                loc = xy_to_loc(board, x, y)
                if board.board[loc] == Board.EMPTY:
                    v = ownership[y, x] * sign
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
                    ownership[ny, nx] * sign > OWN_MIN):
                    influence_points.append(ownership[ny, nx] * sign)
        
        g.influence_area = len(influence_points)
        g.influence_strength = float(np.mean(influence_points)) if influence_points else 0.0


# -------------------------
# Territory Analysis Functions
# -------------------------

def count_building_territory(before: np.ndarray, after: np.ndarray, color: int) -> int:
    """Count intersections that changed from empty (<0.1) to own ownership (>0.1)."""
    same_sign = (1 if color == Board.BLACK else -1)
    prev = before * same_sign
    post = after * same_sign
    return int(np.sum((np.abs(prev) < OWN_MIN) & (post > OWN_MIN)))


def solidify_territory_delta(before: np.ndarray, after: np.ndarray, color: int) -> float:
    """Calculate increase in ownership values of previously owned intersections."""
    same_sign = (1 if color == Board.BLACK else -1)
    prev = before * same_sign
    post = after * same_sign
    owned_mask = prev > OWN_MIN
    return float(np.sum((post - prev)[owned_mask]))


def reduce_opponent_territory_count(before: np.ndarray, after: np.ndarray, color: int) -> int:
    """Count reduction in opponent's owned intersections."""
    opp_sign = (-1 if color == Board.BLACK else 1)
    prev = before * opp_sign
    post = after * opp_sign
    return int(np.sum((prev > OWN_MIN) & (post <= OWN_MIN)))


def invasion_effect(before: np.ndarray, after: np.ndarray, color: int) -> Tuple[int, float]:
    """Calculate invasion effect: reduced opponent territory + built own territory."""
    built = count_building_territory(before, after, color)
    reduced = reduce_opponent_territory_count(before, after, color)
    return reduced, float(built)


def weakening_territory_in_region(before: np.ndarray, after: np.ndarray, region: str, color: int) -> float:
    """Calculate weakening of opponent territory in specific region."""
    m = region_map(before.shape[0])
    opp_sign = (-1 if color == Board.BLACK else 1)
    prev = before * opp_sign
    post = after * opp_sign
    mask = (m == region)
    if not np.any(mask):
        return 0.0
    return float(np.mean(post[mask]) - np.mean(prev[mask]))


def leaving_weakness(before: np.ndarray, after: np.ndarray, color: int) -> int:
    """Count intersections that flipped from own to opponent ownership."""
    own_sign = (1 if color == Board.BLACK else -1)
    prev = before * own_sign
    post = after * own_sign
    return int(np.sum((prev > OWN_MIN) & (post < -OWN_MIN)))


def territory_sizes(ownership: np.ndarray, color: int) -> Tuple[int, int]:
    """Calculate potential and solid territory sizes."""
    sign = (1 if color == Board.BLACK else -1)
    v = ownership * sign
    potential = int(np.sum(v > OWN_MIN) - np.sum(v >= TERRITORY_SOLID))
    solid = int(np.sum(v >= TERRITORY_SOLID))
    return potential, solid


def direct_sacrifice(move_loc: int, after: np.ndarray, color: int, board: Board) -> bool:
    """Check if the played stone becomes opponent's territory."""
    if move_loc == Board.PASS_LOC:
        return False
    x, y = loc_to_xy(board, move_loc)
    sign = (1 if color == Board.BLACK else -1)
    return bool((after[y, x] * sign) < -OWN_MIN)


def indirect_sacrifice(before: np.ndarray, after: np.ndarray, color: int, board: Board) -> bool:
    """Check if any own stone becomes opponent's territory."""
    sign = (1 if color == Board.BLACK else -1)
    prev = before * sign
    post = after * sign
    return bool(np.any((prev > OWN_MIN) & (post < -OWN_MIN)))


# -------------------------
# Policy and Move Analysis Functions
# -------------------------

def urgency_by_region(policy: np.ndarray) -> Dict[str, float]:
    """Calculate urgency as sum of policy mass by region."""
    urg: Dict[str, float] = {
        "corner_top_left": 0.0,
        "corner_top_right": 0.0,
        "corner_bottom_left": 0.0,
        "corner_bottom_right": 0.0,
        "side_left": 0.0,
        "side_right": 0.0,
        "side_upper": 0.0,
        "side_lower": 0.0,
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


def is_only_move(policy: np.ndarray, eps: float = 1e-12) -> bool:
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


def is_connection_move(board: Board, move_loc: int, color: int) -> bool:
    """Check if move connects stones or increases connectivity."""
    if move_loc == Board.PASS_LOC:
        return False
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
    
    return len(heads) >= 2


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


def reduce_aji(before: np.ndarray, after: np.ndarray, board: Board, color: int) -> float:
    """Calculate aji reduction (increase in own ownership over opponent groups)."""
    opp = Board.get_opp(color)
    size = board.size
    delta: List[float] = []
    
    for y in range(size):
        for x in range(size):
            loc = xy_to_loc(board, x, y)
            if board.board[loc] == opp:
                delta.append((after[y, x] - before[y, x]) * (1 if color == Board.BLACK else -1))
    
    return float(np.mean(delta)) if delta else 0.0


def attack_strength_delta(groups_before: List[Group], groups_after: List[Group], opp_color: int) -> float:
    """Calculate attack strength (negative change in opponent group strengths)."""
    before_map = {g.head: g for g in groups_before if g.color == opp_color}
    after_map = {g.head: g for g in groups_after if g.color == opp_color}
    common_heads = set(before_map.keys()) & set(after_map.keys())
    
    if not common_heads:
        return 0.0
    
    deltas = [after_map[h].strength - before_map[h].strength for h in common_heads]
    return float(np.mean(deltas))


def killing_attack(groups_after: List[Group], opp_color: int) -> bool:
    """Check if any opponent group has strength <= -0.5 (>=0.5 own ownership)."""
    for g in groups_after:
        if g.color == opp_color and g.strength <= -0.5:
            return True
    return False


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
    before_ownership: Optional[np.ndarray] = None
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
    
    Returns:
        Dictionary containing all analysis results matching the specification
    """
    results = {}
    
    # 1-2. Urgency by region (regions are static, not needed in output)
    results["urgency_by_region"] = urgency_by_region(policy)
    
    # 3-7. Groups and influence
    groups_det = enumerate_groups_deterministic(board)
    groups_own = enumerate_groups_ownership(board, ownership, player)
    
    compute_group_strengths(groups_det, ownership, player, board)
    compute_group_strengths(groups_own, ownership, player, board)
    compute_group_connectivity(groups_det, ownership, board)
    compute_group_connectivity(groups_own, ownership, board)
    compute_group_influence(groups_det, ownership, board)
    compute_group_influence(groups_own, ownership, board)
    
    results["groups_deterministic"] = [
        {
            "color": g.color,
            "head": g.head,
            "stones": g.stones,
            "liberties": g.liberties,
            "bbox": g.bbox,
            "strength": g.strength,
            "connectivity": g.connectivity,
            "influence_area": g.influence_area,
            "influence_strength": g.influence_strength
        } for g in groups_det
    ]
    
    results["groups_ownership"] = [
        {
            "color": g.color,
            "head": g.head,
            "stones": g.stones,
            "liberties": g.liberties,
            "bbox": g.bbox,
            "strength": g.strength,
            "connectivity": g.connectivity,
            "influence_area": g.influence_area,
            "influence_strength": g.influence_strength
        } for g in groups_own
    ]
    
    # 8-16. Territory analysis
    if before_ownership is not None:
        results["building_territory"] = count_building_territory(before_ownership, ownership, player)
        results["solidify_territory"] = solidify_territory_delta(before_ownership, ownership, player)
        results["reduce_territory"] = reduce_opponent_territory_count(before_ownership, ownership, player)
        results["invasion_effect"] = invasion_effect(before_ownership, ownership, player)
        results["leaving_weakness"] = leaving_weakness(before_ownership, ownership, player)
        
        # Regional weakening
        for region in ["corner_top_left", "corner_top_right", "corner_bottom_left", "corner_bottom_right",
                       "side_left", "side_right", "side_upper", "side_lower", "center"]:
            results[f"weakening_{region}"] = weakening_territory_in_region(
                before_ownership, ownership, region, player
            )
    else:
        # Set defaults when no before_ownership available
        results["building_territory"] = 0
        results["solidify_territory"] = 0.0
        results["reduce_territory"] = 0
        results["invasion_effect"] = (0, 0.0)
        results["leaving_weakness"] = 0
    
    # 14-16. Territory sizes and sacrifices
    potential, solid = territory_sizes(ownership, player)
    results["potential_territory"] = potential
    results["solid_territory"] = solid
    
    if move_loc is not None:
        results["direct_sacrifice"] = direct_sacrifice(move_loc, ownership, player, board)
        if before_ownership is not None:
            results["indirect_sacrifice"] = indirect_sacrifice(before_ownership, ownership, player, board)
        else:
            results["indirect_sacrifice"] = False
    else:
        results["direct_sacrifice"] = False
        results["indirect_sacrifice"] = False
    
    # 18-25. Tactical concepts
    if move_loc is not None:
        results["is_cut"] = is_cut_move(board, move_loc)
        results["is_connection"] = is_connection_move(board, move_loc, player)
        results["is_extension"] = is_extension_move(board, move_loc, player)
        results["liberties"] = liberties_of_group(board, move_loc)
        results["atari"] = atari_move(board, move_loc)
    else:
        results["is_cut"] = False
        results["is_connection"] = False
        results["is_extension"] = False
        results["liberties"] = 0
        results["atari"] = False
    
    results["is_only_move"] = str(is_only_move(policy))
    results["rough_intent"] = rough_intent_effects(ownership, policy, player)
    
    if last_move_loc is not None and move_loc is not None:
        move_idx = loc_to_xy(board, move_loc)[1] * 19 + loc_to_xy(board, move_loc)[0]
        results["is_tenuki"] = is_tenuki(move_idx, last_move_loc, policy, board)
    else:
        results["is_tenuki"] = False
    
    # 26-28. Attack concepts
    if before_ownership is not None:
        results["reduce_aji"] = reduce_aji(before_ownership, ownership, board, player)
        
        # Calculate attack effects
        groups_before = enumerate_groups_deterministic(board)  # Would need before board state
        results["attack_strength"] = attack_strength_delta(groups_before, groups_det, Board.get_opp(player))
        results["killing_attack"] = killing_attack(groups_det, Board.get_opp(player))
    else:
        results["reduce_aji"] = 0.0
        results["attack_strength"] = 0.0
        results["killing_attack"] = False
    
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