# Snorkel position features

`snorkel_board_positions.py` derives interpretable Go-position features from a
board state, a move, KataGo ownership estimates, policy values, and the
preceding position. These features support exploratory labels and
visualizations. The validated-v5 headline concepts are recomputed separately
under the versioned contracts in `operational_definitions.py`.

## Perspective and timing

Ownership is normalized to the player being analysed before before/after
comparisons are made. This matters because KataGo's raw ownership output is
relative to the current player. Features that describe a change require a
before-board or before-ownership array; without one they return a neutral value.

The module uses the move location, board legality, groups and liberties, and
ownership changes to describe what changed at that move. It does not infer a
complete strategic explanation of the position.

## Feature families

### Spatial context

Board points are assigned to four corners, four sides, or the centre. Several
features provide both a board-wide value and a regional breakdown. Corner
occupancy and approach features use the pre-move contents of the relevant
corner.

### Groups and tactics

The module tracks connected stones, liberties, strength, connectivity, and
influence. Derived tactical indicators include connections, cuts, extensions,
atari, attacks, killing attacks, new groups, and direct or indirect sacrifice.

### Ownership and territory

Ownership thresholds distinguish weakly controlled points from solid
territory. The current default thresholds are:

| Setting | Value | Purpose |
| --- | ---: | --- |
| weak ownership | `0.10` | minimum meaningful ownership |
| solid ownership | `0.70` | solid-territory boundary |
| hysteresis band | `0.08`–`0.12` | stabilizes threshold crossings |
| minimum ownership change | `0.05` | solidification and reduction |

Count building measures new weakly controlled space. Solidification measures
existing controlled points becoming more secure. Reduction measures opposing
control being weakened, and invasion identifies play inside opposing influence.
Regional summaries record where these changes occur.

### Strategic proxies

Forcing, tenuki, urgency, territory building or securing, influence change,
and aji reduction are operational proxies built from the lower-level signals.
They are useful labels for exploration, but their names should not be read as
proof that the network represents the corresponding human concept.

The validated-v5 results use stricter versioned definitions for the canonical
tenuki, forcing, and urgency variables. See
[`VALIDITY_V5_EXPERIMENTS_AND_RESULTS.md`](VALIDITY_V5_EXPERIMENTS_AND_RESULTS.md)
for their contracts and evidential status.

## Output contract

`analyze_position_comprehensive()` returns JSON-compatible scalar, Boolean,
array, and regional-dictionary values after conversion by the visualization
pipeline. The source function and tests are authoritative for exact field
names. [`FEATURE_VERIFICATION.md`](FEATURE_VERIFICATION.md) describes the
Python-to-visualizer boundary.
