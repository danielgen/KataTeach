# KataTeach Commentary Generation

This module generates natural-language commentary for Go game moves using OpenAI models. The commentary is grounded in evidence from snorkel analysis and probe-based concept detection.

## Overview

The commentary generation pipeline:

1. **Evidence Packet Builder** - Combines snorkel analysis (tactical features) with probe concept deltas
2. **Concept Gating** - Only uses concepts supported by snorkel evidence (grounding)
3. **LLM Generation** - Calls OpenAI to produce beginner-friendly commentary
4. **Caching** - Stores results to avoid regenerating on reruns

## Installation

Requires the `openai` Python package:

```bash
pip install openai
```

## Usage

### CLI

```bash
# Set your OpenAI API key
export OPENAI_API_KEY='your-key-here'

# Optional: Use a specific model (default: gpt-4.1-mini)
export OPENAI_MODEL='gpt-4o-mini'

# Process all games
python scripts/generate_commentary.py

# Process a specific game
python scripts/generate_commentary.py --game-id 00700686-05d0-4feb-bd80-0e9978faf6b2

# Limit moves per game
python scripts/generate_commentary.py --max-moves 20

# Overwrite existing commentary
python scripts/generate_commentary.py --overwrite

# Dry run (show what would be processed)
python scripts/generate_commentary.py --dry-run

# Verbose output
python scripts/generate_commentary.py -v
```

### Python API

```python
from pathlib import Path
from kata_teach.commentary import generate_game_commentary, CommentaryCache

# Generate for a single game
results = generate_game_commentary(
    game_id="00700686-05d0-4feb-bd80-0e9978faf6b2",
    games_dir=Path("games"),
    html_data_dir=Path("daniele_experiment/linear_probes/html_data"),
    max_moves=50,
)

# Access cached commentary
cache = CommentaryCache(Path("daniele_experiment/linear_probes/html_data"))
commentary = cache.get_commentary("00700686-05d0-4feb-bd80-0e9978faf6b2", move_number=10)
print(commentary.comment)
```

## Input Files

The pipeline requires two files per game:

1. **Snorkel Analysis**: `games/<game_id>/snorkel.jsonl`
   - Per-move tactical analysis (cut, connect, atari, territory, etc.)

2. **Probe Concepts**: `linear_probes/html_data/<game_id>/concepts.json`
   - Per-move concept scores and deltas from trained probes

## Output Files

For each game, the pipeline creates:

1. **Commentary Cache**: `linear_probes/html_data/<game_id>/commentary.jsonl`
   - JSONL with one commentary entry per line
   - Used for caching to avoid regeneration

2. **Merged Concepts**: `linear_probes/html_data/<game_id>/concepts_with_commentary.json`
   - Original concepts.json with `commentary` field added to each move
   - Ready for frontend display

## Commentary Schema

Each commentary entry follows this structure:

```json
{
  "move_number": 10,
  "comment": "Connects two groups. Territory building (21 points, 0.21 intensity).",
  "concepts_used": ["connect", "territory_building"]
}
```

## Concept Gating Rules

Concepts are only used in commentary if supported by snorkel evidence:

| Concept | Snorkel Gate |
|---------|--------------|
| cut | `snorkel.cut == True` |
| connect | `snorkel.connection == True` |
| multi_connect | `snorkel.connection == True && connection_strength_gain >= 2` |
| extend | `snorkel.extension == True` |
| atari | `snorkel.atari == True` |
| forcing | `snorkel.forcing == True` |
| tenuki | `snorkel.tenuki == True` |
| invasion | `snorkel.invasion == True` |
| must_live | `snorkel.must_live == True` |
| kill_attack | `snorkel.killing_attack == True` |
| aji_reduction | `snorkel.reduce_aji == True` |
| fight_pressure | `attacked_groups_count >= 1 && max_attack_intensity >= 0.20` |
| fight_wide | `attacked_groups_count >= 2` |
| territory_building | `move >= 8 && building_intensity > 0 && building_count > 0` |
| territory_securing | `move >= 8 && solidification_intensity > 0 && solidification_count > 0` |
| opponent_reduction | `move >= 8 && reduction_intensity > 0 && reduction_count > 0` |
| influence_surge | `move >= 8 && influence_count_delta > 0` |
| urgency_peak | `urgency_max >= 0.80` |
| group_strength_shift | `abs(group_strength_delta) >= 0.08` |
| group_connectivity_shift | `abs(current_group_connectivity_delta) >= 0.08` |
| sacrifice_direct | `snorkel.direct_sacrifice == True` |
| sacrifice_indirect | `snorkel.indirect_sacrifice >= 1` |
| sacrifice_commitment | `snorkel.direct_sacrifice == True` |
| occupy_corner | `snorkel.occupy_corner == True` |
| approaching_corner | `snorkel.approaching_corner == True` |

Selection is **snorkel-first**: concepts that pass gates are admitted; probe deltas rank among them. Unknown concepts are rejected. Low-trust concepts (`invasion`, `sacrifice_commitment`) are demoted in ranking.

## Configuration

### Environment Variables

- `OPENAI_API_KEY` (required): Your OpenAI API key
- `OPENAI_MODEL` (optional): Model to use (default: `gpt-4.1-mini`)

### Tuneable Thresholds

Edit `evidence.py` to adjust:

```python
GROUP_STRENGTH_THRESHOLD = 0.08
GROUP_CONNECTIVITY_THRESHOLD = 0.08
URGENCY_PEAK_THRESHOLD = 0.80
HIGH_ATTACK_INTENSITY_THRESHOLD = 0.5
```

## Rate Limiting

The module includes exponential backoff for rate limits:
- Initial backoff: 1 second
- Maximum backoff: 60 seconds
- Maximum retries: 5

## Quality Guardrails

After generation, each commentary is validated:
- All required JSON fields must be present (`move_number`, `comment`, `concepts_used`)
- `concepts_used` must be a subset of `selected_concepts`

If validation fails, the model is prompted to correct its output once.

## Module Structure

```
kata_teach/commentary/
├── __init__.py           # Public API exports
├── schema.py             # Data classes and JSON schema
├── evidence.py           # Evidence packet builder and gating
├── prompts.py            # System and user prompts
├── cache.py              # JSONL caching
├── generate_commentary.py # Main generation logic
└── README.md             # This file

scripts/
└── generate_commentary.py # CLI entry point
```

