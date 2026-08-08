# Feature data flow

`snorkel_board_positions.py` computes position features. The visualization path
converts NumPy values to ordinary Python values, serializes the analysis as
JSON, and reads it from `data.analysis` in
`visualize_katago_outputs_custom.py`.

## Feature interface

The current analysis includes these groups of values:

- group strength, connectivity, influence, liberties, attack, and capture;
- potential and solid territory sizes;
- count building, solidification, reduction, and invasion;
- connection, cut, extension, sacrifice, and aji reduction;
- forcing, tenuki, urgency, and corner-related indicators;
- regional count and intensity summaries where applicable.

The exact returned keys are defined by `analyze_position_comprehensive()` in
`snorkel_board_positions.py`. That function is the authoritative interface.
The visualizer deliberately treats display fields as optional so that older
saved games can still be opened when the feature schema changes.

`potential_territory_delta` and `solid_territory_delta` are not current output
keys. Territory change is represented through the current territory,
solidification, and reduction outputs instead.

## Serialization

`convert_numpy_to_python()` recursively converts arrays, NumPy scalar values,
dictionaries, sequences, and sets into JSON-compatible Python values. Features
that need a preceding position return neutral values when the before-state is
not available.
