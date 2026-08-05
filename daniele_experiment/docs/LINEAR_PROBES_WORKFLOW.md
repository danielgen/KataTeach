# Linear probes and game commentary

> **Deprecated for inferential analysis.** This document describes the legacy
> pre-`idx361`, non-nested workflow. Its outputs have been quarantined and must
> not be cited or merged into validated results. Use the prospectively frozen
> `validated_probe_pipeline.py` workflow and its run-scoped artifacts instead.

The linear-probe workflow scores every move, writes browser-friendly concept
data, and can inject that data into each game's `viz.html`.

## Environment

Run the commands from the repository root using the `ml` Conda environment:

```bash
conda run -n ml python -m daniele_experiment.linear_probe_pipeline --help
```

## Rebuild scores and visualizations

To reuse the existing trained probes, compute per-move scores and update every
game visualizer:

```bash
conda run -n ml python -m daniele_experiment.linear_probe_pipeline \
  --skip-training \
  --integrate-html
```

To regenerate only browser data from `move_concepts.parquet` and reinject it:

```bash
conda run -n ml python -m daniele_experiment.linear_probe_pipeline \
  --html-only \
  --integrate-html
```

To update one game after generating or editing its commentary:

```bash
conda run -n ml python -m daniele_experiment.integrate_concepts_html \
  --game-id 00700686-05d0-4feb-bd80-0e9978faf6b2
```

## Per-game artifact contract

Processing a game makes its directory self-contained:

```text
games/<game-id>/
  viz.html
  concepts.json
  concepts_meta.json
  commentary.json
```

`commentary.json` is the canonical portable format:

```json
{
  "game_id": "<game-id>",
  "moves": [
    {
      "move_number": 1,
      "comment": "Commentary for the move.",
      "concepts_used": ["occupy_corner"]
    }
  ]
}
```

The legacy `commentary.jsonl` cache remains supported. When it exists, it takes
precedence because it may contain a newer generation than the combined concept
file.

Because Snorkel features and concept gates can change, cached prose is validated
against the current evidence before reuse. Audit or repair an older cache without
calling an LLM using:

```bash
conda run -n ml python scripts/repair_commentary.py --game-id <game-id>
conda run -n ml python scripts/repair_commentary.py --game-id <game-id> --apply
```

The repair command preserves grounded entries and replaces stale or unsupported
ones with concise commentary assembled directly from the current gated evidence.

## Demo

The game `00700686-05d0-4feb-bd80-0e9978faf6b2` has commentary for every move.
Open its `viz.html`, move the slider, and inspect the commentary section and the
Concept Analysis panel. Top-concept badges use calibrated probabilities when
available; the expanded table includes raw scores, deltas, probabilities, and
cross-validation AUC.
