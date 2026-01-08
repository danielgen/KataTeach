# Complete Snorkel Features Documentation

This document provides a comprehensive account of all snorkel features defined in `snorkel_board_positions.py`, including the exact logic and thresholds used.

## Table of Contents
1. [Configuration Constants](#configuration-constants)
2. [Core Concepts](#core-concepts)
3. [Territory Features](#territory-features)
4. [Group Features](#group-features)
5. [Tactical Features](#tactical-features)
6. [Attack Features](#attack-features)
7. [Sacrifice Features](#sacrifice-features)
8. [Policy Features](#policy-features)
9. [Regional Features](#regional-features)

---

## Configuration Constants

All features use these threshold constants:

- **TAU_POS = 0.10**: Weak ownership threshold (territory is "owned" if ownership > 0.10)
- **TAU_SOLID = 0.70**: Solid territory threshold (territory is "solid" if ownership >= 0.70)
- **TAU_POS_LOW = 0.08**: Hysteresis low threshold (for building territory detection)
- **TAU_POS_HIGH = 0.12**: Hysteresis high threshold (for building territory detection)
- **TAU_ONLY_MOVE = 0.05**: "Only move" threshold (move has >95% probability)
- **TAU_GROUP_IOU = 0.1**: Group matching IoU threshold (groups match if IoU >= 0.1)
- **TAU_GROUP_BELONGING = 0.2**: Ownership threshold for grouping stones by influence paths
- **TAU_AJI_VICINITY = 5**: Aji reduction L1 radius (Manhattan distance)
- **TAU_DELTA_MIN = 0.1**: Minimum ownership delta for solidification/reduction (change must be >= 0.1)

**Regions**: The board is divided into 9 regions:
- `corner_tl`, `corner_tr`, `corner_bl`, `corner_br` (corners, 6x6 each)
- `side_left`, `side_right`, `side_top`, `side_bottom` (sides)
- `center` (remaining area)

---

## Core Concepts

### Ownership Normalization

**Key Convention**: Raw ownership from KataGo is from the **current player to move's perspective**. This module normalizes all ownership to the **analyzing player's perspective** for consistent comparisons.

```python
normalize_ownership(ownership, from_player, to_player)
```
- If `from_player == to_player`: returns ownership unchanged
- Otherwise: returns `-ownership` (flips perspective)

### Group Enumeration

Groups are defined using a two-rule system:

1. **Physical Adjacency Rule**: Stones of the same color that are directly adjacent (4-neighbors) are **ALWAYS** in the same group (they share liberties and live/die together).

2. **Ownership Connectivity Rule**: Groups can be extended through empty points with `ownership > TAU_GROUP_BELONGING (0.2)` to connect strategically related stones.

**Algorithm**: BFS from each unvisited stone:
- Traverse to adjacent same-color stones (always connected)
- Traverse through empty points with ownership > 0.2
- Count liberties as union of all stones' adjacent empty points

### Group Properties

Each group has these computed properties:

- **strength**: Mean ownership over all stones in the group
- **connectivity**: Mean ownership of nearby empty points (within L1 distance ≤ 2, excluding stones)
- **influence_area**: Count of empty points and opponent stones reachable via ownership paths (all points on path must have ownership >= TAU_POS)
- **influence_strength**: Mean ownership value of influenced points

---

## Territory Features

### 1. `potential_territory` (delta)
**Definition**: Change in weakly-owned territory from before to after move.

**Logic**:
- Counts empty points (and opponent stones under control if `player` provided) where `ownership > TAU_POS (0.10)` but `ownership < TAU_SOLID (0.70)`
- Delta = (count_after - count_before)
- Includes opponent stones under own control (ownership > TAU_POS)

**Code**: `territory_sizes_with_delta()` lines 413-450

### 2. `solid_territory` (delta)
**Definition**: Change in strongly-owned territory from before to after move.

**Logic**:
- Counts empty points (and opponent stones under control) where `ownership >= TAU_SOLID (0.70)`
- Delta = (count_after - count_before)

**Code**: `territory_sizes_with_delta()` lines 413-450

### 3. `building_count`
**Definition**: Number of empty points that transitioned from neutral to owned territory.

**Logic**:
- Mask: `(abs(before) < TAU_POS_LOW (0.08)) & (after > TAU_POS_HIGH (0.12)) & ~stone_mask`
- Uses hysteresis to avoid counting noise: must be neutral before (< 0.08) and owned after (> 0.12)
- Counts only empty points (excludes stones)

**Code**: `count_building_territory()` lines 342-347

### 4. `building_intensity`
**Definition**: Average ownership gain on points that became owned territory.

**Logic**:
- Same mask as `building_count`
- Intensity = `mean(after[mask] - before[mask])`
- Returns 0.0 if count = 0

**Code**: `count_building_territory()` lines 342-347

### 5. `solidification_count`
**Definition**: Number of points where owned territory was strengthened (delta >= 0.1).

**Logic**:
- Mask: `(before > TAU_POS (0.10)) & (after > before) & (delta >= TAU_DELTA_MIN (0.1)) & ~stone_mask`
- Counts points that were already owned and became more strongly owned by at least 0.1
- Excludes stones
- Requires ownership increase of at least 0.1 to be counted

**Code**: `solidify_territory_delta()` lines 350-355

### 6. `solidification_intensity`
**Definition**: Average ownership gain on points that were solidified.

**Logic**:
- Same mask as `solidification_count`
- Intensity = `mean(after[mask] - before[mask])`
- Returns 0.0 if count = 0

**Code**: `solidify_territory_delta()` lines 350-355

### 7. `reduction_count`
**Definition**: Number of points where opponent territory was reduced (crossed threshold, delta >= 0.1).

**Logic**:
- Mask: `(before < -TAU_POS (-0.10)) & (after > -TAU_POS) & (abs(delta) >= TAU_DELTA_MIN (0.1)) & ~stone_mask`
- Counts points that crossed from opponent territory to contested/neutral
- Excludes stones
- Requires ownership change of at least 0.1 (absolute value) to be counted

**Code**: `reduce_opponent_territory()` lines 358-363

### 8. `reduction_intensity`
**Definition**: Average reduction amount on points where opponent territory decreased.

**Logic**:
- Same mask as `reduction_count`
- Intensity = `mean(abs(after[mask] - before[mask]))`
- Returns 0.0 if count = 0

**Code**: `reduce_opponent_territory()` lines 358-363

### 9. `invasion` (boolean)
**Definition**: True if move flipped opponent territory to own territory (with strict requirements).

**Logic**:
- If `move_loc` provided and not pass:
  - **Requirement 1**: Stone must have 3+ liberties (count adjacent empty points)
  - **Requirement 2**: Area around move (radius 3) must be ≥80% empty points
  - **Requirement 3**: If groups provided, the invading stone must be the only stone in its group
  - Creates spatial mask: points within Euclidean distance ≤ 3.0 from move
  - Checks if ≥50% of empty points in vicinity were opponent territory before
  - If all requirements met: mask = `empty & spatial & (before < -TAU_POS) & (after > TAU_POS)`
- Otherwise: mask = `empty & (before < -TAU_POS) & (after > TAU_POS)`
- Returns `True` if any points match and all requirements are satisfied

**Code**: `invasion_effect()` lines 366-416

### 10. `invasion_intensity`
**Definition**: Average ownership swing on invaded points.

**Logic**:
- Same mask as `invasion`
- Intensity = `mean(after[mask] - before[mask])`
- Returns 0.0 if no invasion

**Code**: `invasion_effect()` lines 366-384

---

## Group Features

### 11. `group_strength_delta`
**Definition**: Average change in ownership over all own groups' stones.

**Logic**:
- Matches groups before/after using IoU (threshold = TAU_GROUP_IOU = 0.1)
- For each matched group: delta = `after.strength - before.strength`
- Also includes new groups (not matched): their full strength is the delta
- Returns mean of all deltas

**Code**: `compute_individual_group_strength_deltas()` lines 1253-1280

### 12. `group_connectivity_delta`
**Definition**: Average change in ownership of empty points near own groups.

**Logic**:
- Matches groups before/after using IoU
- For each matched group: delta = `after.connectivity - before.connectivity`
- Also includes new groups: their full connectivity is the delta
- Returns mean of all deltas

**Code**: `compute_individual_group_connectivity_deltas()` lines 1283-1306

### 13. `max_group_strength_delta`
**Definition**: Maximum single-group strength improvement.

**Logic**:
- Same computation as `group_strength_delta` but returns maximum delta instead of mean

**Code**: `compute_individual_group_strength_deltas()` lines 1253-1280

### 14. `max_group_connectivity_delta`
**Definition**: Maximum single-group connectivity improvement.

**Logic**:
- Same computation as `group_connectivity_delta` but returns maximum delta instead of mean

**Code**: `compute_individual_group_connectivity_deltas()` lines 1283-1306

### 15. `current_group_strength`
**Definition**: Strength of the group containing the move after the move.

**Logic**:
- Finds group containing `move_loc` in `groups_after`
- Returns `group.strength` (mean ownership over stones in group)
- Returns 0.0 if move is pass or group not found

**Code**: `compute_current_group_delta()` lines 701-755

### 16. `current_group_strength_delta`
**Definition**: Change in strength of the group containing the move.

**Logic**:
- Finds group containing move after the move
- Matches to group before using IoU (threshold = 0.1)
- Delta = `current_strength - matched_before.strength`
- If no match (new group): delta = 0.0

**Code**: `compute_current_group_delta()` lines 701-755

### 17. `current_group_connectivity`
**Definition**: Connectivity of the group containing the move after the move.

**Logic**:
- Finds group containing `move_loc` in `groups_after`
- Returns `group.connectivity` (mean ownership of nearby empty points)
- Returns 0.0 if move is pass or group not found

**Code**: `compute_current_group_delta()` lines 701-755

### 18. `current_group_connectivity_delta`
**Definition**: Change in connectivity of the group containing the move.

**Logic**:
- Finds group containing move after the move
- Matches to group before using IoU
- Delta = `current_connectivity - matched_before.connectivity`
- If no match: delta = 0.0

**Code**: `compute_current_group_delta()` lines 701-755

### 19. `current_group_influence_count`
**Definition**: Influence area count of the group containing the move after the move.

**Logic**:
- Finds group containing `move_loc` in `groups_after`
- Returns `group.influence_area` (count of influenced points)
- Returns 0 if move is pass or group not found

**Code**: `compute_current_group_delta()` lines 701-755

### 20. `current_group_influence_count_delta`
**Definition**: Change in influence area count of the group containing the move.

**Logic**:
- Finds group containing move after the move
- Matches to group before using IoU
- Delta = `current_influence_area - matched_before.influence_area`
- If no match: delta = 0

**Code**: `compute_current_group_delta()` lines 701-755

### 21. `current_group_influence_strength`
**Definition**: Influence strength of the group containing the move after the move.

**Logic**:
- Finds group containing `move_loc` in `groups_after`
- Returns `group.influence_strength` (mean ownership of influenced points)
- Returns 0.0 if move is pass or group not found

**Code**: `compute_current_group_delta()` lines 701-755

### 22. `current_group_influence_strength_delta`
**Definition**: Change in influence strength of the group containing the move.

**Logic**:
- Finds group containing move after the move
- Matches to group before using IoU
- Delta = `current_influence_strength - matched_before.influence_strength`
- If no match: delta = 0.0

**Code**: `compute_current_group_delta()` lines 701-755

### 23. `influence_count_delta`
**Definition**: Change in total unique influence area across all own groups.

**Logic**:
- Computes total unique influenced points (no double-counting) for all groups before and after
- Uses BFS from each group's stones, following ownership paths (ownership >= TAU_POS)
- Counts unique empty points and opponent stones reachable
- Delta = `after_count - before_count`

**Code**: `compute_influence_delta_accurate()` lines 1363-1377, `compute_total_unique_influence()` lines 1309-1360

### 24. `influence_strength_delta`
**Definition**: Change in average influence strength across all own groups.

**Logic**:
- Same as `influence_count_delta` but tracks mean ownership of influenced points
- Delta = `after_mean_strength - before_mean_strength`

**Code**: `compute_influence_delta_accurate()` lines 1363-1377

### 25. `creates_new_group` (boolean)
**Definition**: True if move created a new separate group (didn't extend existing group).

**Logic**:
- Finds group containing `move_loc` in `groups_after`
- Tries to match to any group in `groups_before` using IoU (threshold = 0.1)
- Returns `True` if no match found (new group)
- Returns `False` if match found (extended existing group) or move is pass

**Code**: `creates_new_group()` lines 789-827

---

## Tactical Features

### 26. `cut` (boolean)
**Definition**: True if move separates 2+ opponent groups.

**Logic**:
- Gets mover color from `board.board[move_loc]`
- Gets opponent color
- Checks 4 neighbors of move location
- Collects unique group heads of opponent stones
- Returns `True` if `len(heads) >= 2`

**Code**: `is_cut_move()` lines 623-641

### 27. `connection` (boolean)
**Definition**: True if move connects 2+ previously separate own groups (using group enumeration).

**Logic**:
- Enumerates groups before and after the move using ownership-based grouping
- Finds the group containing the move after the move
- Checks which groups from before have stones in the move group (non-empty intersection)
- Returns `True` if 2+ previously separate groups were merged
- Only counts groups that existed before the move (excludes the move itself)

**Code**: `is_connection_move()` lines 688-741

### 28. `connection_strength_gain`
**Definition**: Number of groups merged minus 1.

**Logic**:
- Same computation as `connection` but returns `num_merged - 1`
- Measures how many groups were merged (2 groups → gain = 1, 3 groups → gain = 2, etc.)

**Code**: `is_connection_move()` lines 688-741

### 29. `merged_groups_regions`
**Definition**: List of regions where merged groups are located (when connection = True).

**Logic**:
- Same computation as `connection` but collects regions
- For each merged group: determines dominant region using `_get_group_region()`
- Returns list of region strings (e.g., `["corner_tl", "center"]`)

**Code**: `is_connection_move()` lines 688-741

### 30. `merged_groups_head_locs`
**Definition**: List of head stone locations of merged groups (when connection = True).

**Logic**:
- Same computation as `connection` but collects head locations
- For each merged group: returns `group.head` (board location of head stone)
- Returns list of location integers (e.g., `[123, 456]`)

**Code**: `is_connection_move()` lines 688-741

### 31. `extension` (boolean)
**Definition**: True if move is adjacent to at least one own stone.

**Logic**:
- Checks 4 neighbors of move location
- Returns `True` if any neighbor is own color
- Returns `False` if move is pass

**Code**: `is_extension_move()` lines 662-671

### 32. `liberties`
**Definition**: Liberty count of the group containing the played stone.

**Logic**:
- Looks up group containing `move_loc` in provided groups list
- Returns `group.liberties` (union of all stones' adjacent empty points)
- Returns 0 if move is pass, location is empty, or group not found

**Code**: `liberties_of_group()` lines 674-690

### 33. `atari` (boolean)
**Definition**: True if move puts at least one opponent group into atari.

**Logic**:
- Uses Board's deterministic group tracking (stone adjacency), not ownership-based groups
- Gets mover color from `board.board[move_loc]`
- Gets opponent color
- Checks 4 neighbors of move location
- If neighbor is opponent stone: checks if its group has exactly 1 liberty (`board.group_liberty_count[head] == 1`)
- Returns `True` if any opponent group has 1 liberty (can be captured next move)
- Returns `False` if move is pass

**Code**: `atari_move()` lines 757-786

---

## Attack Features

### 34. `attack` (boolean)
**Definition**: True if at least one opponent group strength decreased by 0.1 or more.

**Logic**:
- Matches opponent groups before/after using IoU
- Computes strength delta for each matched group: `after.strength - before.strength`
- Returns `True` if any delta <= -0.1 (strength decreased by at least 0.1)
- Returns `False` if no groups or no matches

**Code**: `attack_strength_delta()` lines 856-876

### 35. `avg_attack_intensity`
**Definition**: Average absolute decrease in opponent group strengths.

**Logic**:
- Same computation as `attack` but returns `abs(mean(deltas))`
- Measures average attack strength across all opponent groups

**Code**: `attack_strength_delta()` lines 856-876

### 36. `max_attack_intensity`
**Definition**: Maximum absolute decrease in any single opponent group strength.

**Logic**:
- Same computation as `attack` but returns `abs(min(deltas))` (most negative delta)
- Measures strongest attack on any single opponent group

**Code**: `attack_strength_delta()` lines 856-876

### 37. `attacked_groups_count`
**Definition**: Number of opponent groups under attack (strength decreased >= 0.1).

**Logic**:
- Matches opponent groups before/after using IoU
- For each matched group: delta = `after.strength - before.strength`
- Counts groups where `delta <= -0.1` (strength decreased by at least 0.1)
- Returns count of attacked groups

**Code**: `get_attacked_groups_info()` lines 923-981

### 38. `attacked_groups_regions`
**Definition**: List of regions where attacked groups are located.

**Logic**:
- Same computation as `attacked_groups_count` but collects regions
- For each attacked group: determines dominant region using `_get_group_region()`
- Returns list of region strings (e.g., `["corner_tl", "center"]`)

**Code**: `get_attacked_groups_info()` lines 923-981

### 39. `attacked_groups_head_locs`
**Definition**: List of head stone locations of attacked groups.

**Logic**:
- Same computation as `attacked_groups_count` but collects head locations
- For each attacked group: returns `group.head` (board location of head stone)
- Returns list of location integers (e.g., `[123, 456]`)

**Code**: `get_attacked_groups_info()` lines 923-981

### 40. `attacked_groups_strength_deltas`
**Definition**: List of strength deltas for each attacked group.

**Logic**:
- Same computation as `attacked_groups_count` but collects deltas
- For each attacked group: delta = `after.strength - before.strength`
- Returns list of delta values (e.g., `[-0.15, -0.12]`)

**Code**: `get_attacked_groups_info()` lines 923-981

### 41. `killing_attack` (boolean)
**Definition**: True if any opponent group transitions from alive to killed.

**Logic**:
- Matches opponent groups before/after using IoU
- For each matched group:
  - Checks if group was alive before (`before.strength > 0`)
  - Checks if group became killed after (`after.strength <= 0`)
- Returns `True` if any group transitioned from alive to killed, else `False`
- Only counts groups that were NOT already killed before the move

**Code**: `killing_attack()` lines 1036-1057

### 42. `kill_intensity`
**Definition**: Mean absolute strength of killed groups.

**Logic**:
- Same computation as `killing_attack` but returns `mean(abs(strength) for killed groups)`
- Measures how strongly the killed groups were (higher = stronger groups killed)
- Returns 0.0 if no groups killed

**Code**: `killing_attack()` lines 1036-1057

### 43. `reduce_aji` (boolean)
**Definition**: True if move reduces aji (increases ownership over weak opponent stones).

**Logic**:
- If `move_loc` provided: creates vicinity mask (L1 distance ≤ TAU_AJI_VICINITY = 5)
- Finds all opponent stones where `before > TAU_POS (0.10)` (weak opponent stones in own territory)
- If vicinity mask exists: only considers stones within vicinity
- Computes ownership delta for each: `after - before`
- Returns `True` if `mean(deltas) >= 0.05` (ownership increased by at least 0.05 on average)
- Returns `False` if no such stones or mean delta < 0.05

**Code**: `reduce_aji()` lines 832-853

### 44. `aji_reduction_intensity`
**Definition**: Average aji reduction amount.

**Logic**:
- Same computation as `reduce_aji` but returns `mean(deltas)` (can be negative)
- Measures average ownership increase over weak opponent stones
- Returns 0.0 if no such stones

**Code**: `reduce_aji()` lines 832-853

---

## Sacrifice Features

### 45. `direct_sacrifice` (boolean)
**Definition**: True if the played stone is in opponent territory AFTER the move.

**Logic**:
- Gets coordinates of `move_loc`
- Checks ownership at that location: `ownership[y, x] < -TAU_POS (-0.10)`
- Returns `True` if stone is in opponent territory (negative ownership from player's perspective)
- Returns `False` if move is pass
- **Note**: Uses ownership AFTER the move, not before

**Code**: `direct_sacrifice()` lines 453-480

### 46. `sacrifice_intensity`
**Definition**: Absolute ownership value of sacrificed stone.

**Logic**:
- Same as `direct_sacrifice` but returns `abs(ownership[y, x])` if sacrifice, else 0.0
- Measures how strongly the stone is in opponent territory

**Code**: `direct_sacrifice()` lines 453-480

### 47. `indirect_sacrifice`
**Definition**: Count of own stones that flipped from own territory to opponent territory.

**Logic**:
- Uses `before_board` to find stones that existed BEFORE the move (crucial for detecting captured stones)
- Creates mask of own stones from before_board (or current board if before_board not provided)
- Mask: `own_stones & (before > TAU_POS) & (after < -TAU_POS)`
- Counts stones that were in own territory before and opponent territory after
- Returns count

**Code**: `indirect_sacrifice()` lines 483-513

### 48. `indirect_sacrifice_intensity`
**Definition**: Average ownership swing of sacrificed stones.

**Logic**:
- Same mask as `indirect_sacrifice`
- Intensity = `mean(before[mask] - after[mask])`
- Measures average ownership loss (positive value = loss)
- Returns 0.0 if count = 0

**Code**: `indirect_sacrifice()` lines 483-513

---

## Policy Features

### 47. `urgency` (dict)
**Definition**: Sum of policy probability mass by region.

**Logic**:
- Iterates over all 19x19 board positions
- For each position: `idx = y * 19 + x`
- Adds `policy[idx]` to region's urgency sum
- Returns dict with sum for each of 9 regions

**Code**: `urgency_by_region()` lines 578-586

### 48. `urgency_intensity` (dict)
**Definition**: Normalized urgency (share of total policy mass per region).

**Logic**:
- Computes `urgency` (sum by region)
- Total = `sum(urgency.values())`
- For each region: `urgency_intensity[region] = urgency[region] / total` (if total > 0, else 0.0)
- Measures what fraction of policy mass is in each region

**Code**: `urgency_intensity_by_region()` lines 589-593

### 49. `forcing` (boolean)
**Definition**: True if one move dominates (>95% probability).

**Logic**:
- Returns `True` if `max(policy) > (1.0 - TAU_ONLY_MOVE)` = `max(policy) > 0.95`
- Returns `False` if move is pass

**Code**: `is_forcing()` lines 596-598

### 50. `tenuki` (boolean)
**Definition**: True if move is far from last move, ignoring local follow-up.

**Logic**:
- Returns `False` if `last_move_loc` is None or pass
- Computes L1 distance: `abs(x_sel - x_last) + abs(y_sel - y_last)`
- Returns `False` if distance < 6 (too close)
- Returns `False` if same region (not tenuki if in same region)
- Checks if any move within L1 distance ≤ 4 of last move has higher policy than selected move
- Returns `True` if selected move is far AND there's a better local follow-up (ignoring it = tenuki)
- Returns `False` otherwise

**Code**: `is_tenuki()` lines 601-618

---

## Regional Features

All regional features break down global features by the 9 board regions. Each region feature is a dict with keys: `corner_tl`, `corner_tr`, `corner_bl`, `corner_br`, `side_left`, `side_right`, `side_top`, `side_bottom`, `center`.

### 51. `building_count_by_region` (dict)
**Definition**: Count of points that became owned territory, by region.

**Logic**:
- Same logic as `building_count` but computed per region
- Uses hysteresis: `abs(before) < TAU_POS_LOW (0.08)` and `after > TAU_POS_HIGH (0.12)`
- Excludes stones
- Returns dict with count for each region

**Code**: `compute_territory_delta_by_region()` lines 518-548

### 52. `building_intensity_by_region` (dict)
**Definition**: Average ownership gain on building points, by region.

**Logic**:
- Same mask as `building_count_by_region`
- For each region: `intensity = sum(deltas) / count` (if count > 0, else 0.0)
- Returns dict with average intensity for each region

**Code**: `compute_territory_delta_by_region()` lines 518-548

### 53. `solidification_count_by_region` (dict)
**Definition**: Count of points where owned territory was strengthened (delta >= 0.1), by region.

**Logic**:
- Same logic as `solidification_count` but computed per region
- Mask: `(before > TAU_POS) & (after > before) & (delta >= TAU_DELTA_MIN (0.1)) & ~stones`
- Requires ownership increase of at least 0.1 to be counted
- Returns dict with count for each region

**Code**: `compute_territory_delta_by_region()` lines 518-548

### 54. `solidification_intensity_by_region` (dict)
**Definition**: Average ownership gain on solidification points, by region.

**Logic**:
- Same mask as `solidification_count_by_region`
- For each region: `intensity = sum(deltas) / count` (if count > 0, else 0.0)
- Returns dict with average intensity for each region

**Code**: `compute_territory_delta_by_region()` lines 518-548

### 55. `reduction_count_by_region` (dict)
**Definition**: Count of points where opponent territory was reduced (delta >= 0.1), by region.

**Logic**:
- Same logic as `reduction_count` but computed per region
- Mask: `(before < -TAU_POS) & (after > -TAU_POS) & (abs(delta) >= TAU_DELTA_MIN (0.1)) & ~stones`
- Requires ownership change of at least 0.1 (absolute value) to be counted
- Returns dict with count for each region

**Code**: `compute_reduction_by_region()` lines 551-573

### 56. `reduction_intensity_by_region` (dict)
**Definition**: Average reduction amount on points where opponent territory decreased, by region.

**Logic**:
- Same mask as `reduction_count_by_region`
- For each region: `intensity = sum(abs(deltas)) / count` (if count > 0, else 0.0)
- Returns dict with average intensity for each region

**Code**: `compute_reduction_by_region()` lines 551-573

---

## Summary

Total features: **58 features** organized into:
- **10 Territory Features** (including 2 deltas, 8 move effects)
- **15 Group Features** (strength, connectivity, influence deltas and current group features)
- **9 Tactical Features** (cut, connection, connection_strength_gain, merged_groups_regions, merged_groups_head_locs, extension, liberties, atari, creates_new_group)
- **11 Attack Features** (attack, killing_attack, reduce_aji, and attacked groups info)
- **4 Sacrifice Features** (direct and indirect sacrifice)
- **4 Policy Features** (urgency, forcing, tenuki)
- **6 Regional Features** (breakdowns of territory features by region)

All features are computed by `analyze_position_comprehensive()` which normalizes ownership to the analyzing player's perspective and handles edge cases (pass moves, missing before state, etc.).

