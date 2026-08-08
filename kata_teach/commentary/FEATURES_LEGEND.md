# KataTeach Features Legend

The move analysis panel displays the following features and concepts.

---

## Understanding the Format

Many features show two values:
- **Raw value**: The actual measurement (e.g., `38` points of territory)
- **Percentile (pXX)**: How this compares to all moves in the dataset (e.g., `p91` means this value is higher than 91% of all moves)

---

## Territory Features

These measure how the move affects territorial control on the board.

| Feature | Short Label | Description |
|---------|------------|-------------|
| **Potential Δ** | weak territory | Change in weakly-owned territory (ownership > 10% but < 70%). Positive = gained weak claims. |
| **Solid Δ** | strong territory | Change in strongly-owned territory (ownership ≥ 70%). Positive = gained solid points. |
| **Building** | new territory | Number of empty points that went from neutral (unowned) to owned. Measures new territorial claims. |
| **Solidification** | strengthened | Number of points where already-owned territory became stronger (increased by ≥ 5%). |
| **Reduction** | opponent lost | Number of opponent territory points that crossed from opponent-controlled to contested/neutral. |
| **Invasion** | flipped to own | Yes/No - Did the move flip opponent territory to own territory? Requires 3+ liberties and mostly empty surroundings. |
| **Intensities** | B:S:R | Average intensity of Building, Solidification, and Reduction effects (how strongly each effect occurred). |

---

## This Move's Group

Properties of the specific group (connected stones) that contains the move just played.

| Feature | Short Label | Description |
|---------|------------|-------------|
| **Strength** | ownership | Mean ownership value over all stones in this group. Positive = alive/healthy, negative = likely dead. Range: -1 to +1. |
| **Strength Δ** | change | How much the group's strength changed due to this move. Positive = group became healthier. |
| **Connectivity** | nearby | Mean ownership of empty points within 2 intersections of the group. Higher = better local control. |
| **Connectivity Δ** | change | How connectivity changed due to this move. |
| **Influence** | area | Count of empty points reachable via paths of favorable ownership (≥ 10%). Measures sphere of influence. |
| **Influence Δ** | change | How the influence area changed. |
| **Influence Str** | strength | Average ownership strength in the influenced area. |
| **Influence Str Δ** | change | How influence strength changed. |
| **Liberties** | empty adj | Number of empty points adjacent to stones in this group. More liberties = harder to capture. |
| **New Group** | isolated | Yes = This move created a new separate group (not connected to existing stones). |
| **Must Live** | saves group | Yes = This move saved a group that would have died if player passed. Critical defensive move. |

---

## All Groups (Average)

Aggregate statistics across ALL of the player's groups on the board.

| Feature | Short Label | Description |
|---------|------------|-------------|
| **Strength Δ** | avg change | Average strength change across all player's groups. Positive = overall position improved. |
| **Connectivity Δ** | avg change | Average connectivity change across all groups. |
| **Influence Count Δ** | points | Change in total unique influenced points (no double-counting between groups). |
| **Influence Str Δ** | avg | Average change in influence strength. |
| **Max Str Δ** | max | Maximum strength improvement of any single group. Shows biggest local gain. |
| **Max Conn Δ** | max | Maximum connectivity improvement of any single group. |

---

## Tactics

Tactical properties of the move.

| Feature | Short Label | Description |
|---------|------------|-------------|
| **Cut** | separates | Yes = Move separates opponent groups that were previously connected. |
| **Groups Split** | count | Number of new opponent groups created by the cut. |
| **Cut Regions** | locations | Board regions where the cut groups are located (e.g., "corner br"). |
| **Cut Heads** | locations | Representative stone locations of cut groups (e.g., "Q5, P3"). |
| **Connection** | joins | Yes = Move connects 2+ previously separate own groups. |
| **Conn. Gain** | groups | Number of groups connected minus 1 (connecting 3 groups = gain of 2). |
| **Extension** | adjacent | Yes = Move is adjacent to at least one own stone (not a new isolated stone). |
| **Atari** | 1 liberty | Yes = Move puts at least one opponent group into atari (1 liberty - can be captured next). |
| **Occupy Corner** | first stone | Yes = First stone placed in a corner area (opening play). |
| **Approach** | kakari | Yes = Second stone in corner responding to opponent's corner stone. |

---

## Attack

How the move affects opponent groups.

| Feature | Short Label | Description |
|---------|------------|-------------|
| **Attack** | weakened | Yes = At least one opponent group's strength decreased by ≥ 0.1 (10%). |
| **Killing** | likely dead | Yes = An opponent group transitioned from alive (strength > 0) to dead (strength ≤ 0). |
| **Reduce Aji** | potential | Yes = Move strengthens control over weak opponent stones (makes dead stones "more dead"). |
| **Intensity** | avg/max | Average and maximum attack intensity across opponent groups. Higher = stronger attack. |
| **Groups Attacked** | count | Number of opponent groups whose strength decreased ≥ 10%. |
| **Attacked Regions** | locations | Board regions where attacked groups are located. |
| **Attacked Heads** | locations | Representative stone locations of attacked groups. |
| **Group Intensities** | deltas | List of strength decreases for each attacked group. |

---

## Sacrifice

Whether the move involves sacrificing stones.

| Feature | Short Label | Description |
|---------|------------|-------------|
| **Direct** | stone lost | Yes = The stone just played is in opponent territory AFTER the move (sacrifice stone). |
| **Direct Intensity** | ownership | How strongly the sacrifice stone is in opponent territory (absolute value). |
| **Indirect** | stones lost | Count of own stones that flipped from own territory to opponent territory due to this move. |
| **Indirect Intensity** | avg swing | Average ownership swing of stones that were sacrificed. |

---

## Policy

Analysis based on the AI's policy (move preferences) at this position.

| Feature | Short Label | Description |
|---------|------------|-------------|
| **Only Move** | >95% prob | Yes = One move has >95% probability (essentially forced). |
| **Tenuki** | distant | Yes = Move ignores a local situation where a better local follow-up exists. |
| **Urgency** | regions | Shows policy probability mass by region for the opponent's next move. Higher % = AI considers that region more urgent. |

---

## Regional Breakdown

Territory changes broken down by board region.

| Region | B | S | R |
|--------|---|---|---|
| corner tl/tr/bl/br | Building count | Solidification count | Reduction count |
| side left/right/top/bottom | (empty = 0) | (empty = 0) | (empty = 0) |
| center | | | |

Format: `B:N(X%) S:N(X%)` shows Building: N points (X% of that region's total change), Solidification: N points, Reduction: N points.

---

## Neural Network Concepts (Probe Scores)

These are learned pattern detections from neural network linear probes. Each concept has:
- **Score**: Activation strength (higher = more confident the concept applies)
- **Delta**: Change from previous position (positive = concept became more relevant)

### Tactical Concepts

| Concept | Description |
|---------|-------------|
| **cut** | Move that separates opponent groups |
| **connect** | Move that joins own groups |
| **multi_connect** | Connects multiple groups at once |
| **extend** | Adjacent to existing own stone |
| **atari** | Puts opponent in atari |
| **forcing** | Creates a forcing/sente move |

### Fighting Concepts

| Concept | Description |
|---------|-------------|
| **fight_pressure** | Strong local attack pressure |
| **fight_wide** | Attacking multiple opponent groups |
| **kill_attack** | Move that threatens to kill a group |
| **aji_reduction** | Reduces opponent's potential (aji) |

### Territory Concepts

| Concept | Description |
|---------|-------------|
| **territory_building** | Creating new territorial claims |
| **territory_securing** | Strengthening existing territory |
| **opponent_reduction** | Reducing opponent's territory |
| **invasion** | Playing into opponent's sphere of influence |
| **influence_surge** | Rapidly expanding influence area |

### Group Status Concepts

| Concept | Description |
|---------|-------------|
| **group_strength_shift** | Major change in group health/safety |
| **group_connectivity_shift** | Major change in group connections |
| **must_live** | Critical move to save a group |

### Special Moves

| Concept | Description |
|---------|-------------|
| **sacrifice_direct** | Stone immediately "lost" to opponent |
| **sacrifice_indirect** | Causes own stones to flip to opponent |
| **sacrifice_commitment** | Large sacrifice for strategic gain |
| **tenuki** | Ignoring local situation to play elsewhere |
| **urgency_peak** | Playing in the most urgent region |
| **occupy_corner** | First stone in a corner (opening) |
| **approaching_corner** | Approaching opponent's corner stone |

---

## Board Regions

The board is divided into 9 regions:
- **Corners**: `corner_tl` (top-left), `corner_tr` (top-right), `corner_bl` (bottom-left), `corner_br` (bottom-right) - 6x6 each
- **Sides**: `side_left`, `side_right`, `side_top`, `side_bottom`
- **Center**: The remaining middle area

---

## Key Thresholds

| Threshold | Value | Used For |
|-----------|-------|----------|
| TAU_POS | 0.10 | Weak ownership (territory "owned" if > 10%) |
| TAU_SOLID | 0.70 | Solid territory (strongly owned if ≥ 70%) |
| TAU_DELTA_MIN | 0.05 | Minimum change to count as solidification/reduction |
| TAU_GROUP_IOU | 0.4 | Group matching threshold (40% overlap) |
| TAU_GROUP_BELONGING | 0.2 | Ownership threshold for grouping stones (20%) |

---

## Interpreting Your Example Move

Looking at your example:
- **Potential Δ: 38 (p91)** - Large gain in weak territory claims, higher than 91% of moves
- **Reduction: 43 (p90)** - Strongly reduced opponent territory, top 10% effect
- **Cut: Yes** - This move separated opponent groups
- **Attack: Yes** - Weakened an opponent group
- **New Group: Yes** - Created an isolated stone/group
- **B:S:R = 0.10:0.11:0.44** - Strongest effect is Reduction (0.44 intensity)

This appears to be an aggressive move that cuts opponent groups while significantly impacting territorial balance in the player's favor.
