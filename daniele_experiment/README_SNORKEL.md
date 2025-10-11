# Snorkel Integration for Go Board Positions

This directory contains scripts for applying Snorkel's weak supervision framework to Go board positions using KataGo analysis data.

## Overview

The Snorkel integration provides a complete pipeline for:

1. **Loading board position data** from KataGo policy analysis files
2. **Applying weak labeling functions** based on Go concepts and heuristics
3. **Training a label model** to combine weak labels into probabilistic predictions
4. **Generating high-quality labels** for downstream machine learning tasks

## Files

- `snorkel_board_positions.py` - Main Snorkel processor
- `example_snorkel_usage.py` - Example usage and demonstrations
- `requirements_snorkel.txt` - Python dependencies
- `README_SNORKEL.md` - This documentation

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements_snorkel.txt
```

2. Ensure you have the existing daniele_experiment dependencies installed (KataGo Python modules, etc.)

## Quick Start

### Process a single policy file:
```bash
python snorkel_board_positions.py --input-file games/policy/game.json --output-dir snorkel_output/
```

### Process all policy files in a directory:
```bash
python snorkel_board_positions.py --input-dir games/policy/ --output-dir snorkel_output/
```

### Run the example script:
```bash
python example_snorkel_usage.py
```

## Labeling Functions

The system includes several types of labeling functions:

### Existing Go Concept Functions
- `tenuki_ok` - Labels when tenuki (playing elsewhere) is acceptable
- `invasion_viable` - Labels when invasion is viable
- `cut_available` - Labels when cutting moves are available
- `ladder_works` - Labels when ladders are favorable
- `sente_line` - Labels when a move creates sente (initiative)

### Additional Heuristic Functions
- `high_winrate_move` - Labels moves with very high winrates (>70%)
- `close_competition` - Labels positions with multiple similar-quality moves
- `policy_concentration` - Labels when policy is concentrated on few moves
- `many_candidates` - Labels positions with many candidate moves

## Output Files

The processor generates several output files:

- `snorkel_results.csv` - Main results with probabilistic labels
- `label_model.pkl` - Trained Snorkel label model
- `lf_statistics.json` - Statistics for each labeling function
- `summary_report.txt` - Human-readable summary report

## Data Format

### Input
The processor expects JSON files with the structure:
```json
{
  "sgf": "...",
  "policy": {
    "0": {
      "suggestions": [
        {"move": "C16", "winrate": 0.34, "policy_prob": 0.07},
        ...
      ],
      "actual_move": {"move": "D16", "winrate": 0.32, "player": "b"}
    },
    ...
  }
}
```

### Output
The CSV output includes:
- Original position data
- Individual labeling function outputs
- Snorkel probabilistic predictions
- Derived features (winrate spread, number of suggestions, etc.)

## Example Usage

```python
from snorkel_board_positions import SnorkelBoardPositionProcessor
from pathlib import Path

# Create processor
processor = SnorkelBoardPositionProcessor(Path("output/"))

# Process data
df_result, label_model = processor.process(Path("games/policy/"))

# Analyze results
print(f"Processed {len(df_result)} positions")
print(f"Average positive probability: {df_result['snorkel_prob_positive'].mean():.3f}")
```

## Customization

### Adding New Labeling Functions

You can add custom labeling functions by:

1. Creating a new function with the `@labeling_function()` decorator
2. Adding it to the `labeling_functions` list in `SnorkelBoardPositionProcessor`
3. The function should return 1 (positive), 0 (negative), or -1 (abstain)

Example:
```python
@labeling_function()
def lf_custom_heuristic(x):
    """Custom labeling function."""
    try:
        # Your logic here
        if some_condition(x):
            return 1
        else:
            return 0
    except:
        return -1  # Abstain on error
```

### Modifying Features

To add new features for labeling functions:

1. Modify the `_extract_features()` method in `BoardPositionData`
2. Update the `to_dict()` method to include new features
3. Create labeling functions that use the new features

## Integration with Existing Workflow

This Snorkel integration works seamlessly with the existing daniele_experiment workflow:

1. **Generate games**: Use `play_and_analyze.py` to create games with policy analysis
2. **Apply Snorkel**: Use this script to generate weak supervision labels
3. **Train models**: Use the probabilistic labels to train downstream models
4. **Evaluate**: Compare Snorkel labels with human annotations

## Performance Considerations

- The processor loads all positions into memory at once
- For very large datasets, consider processing in batches
- Label model training time scales with the number of positions and labeling functions
- Most computation is in the labeling function application phase

## Troubleshooting

### Common Issues

1. **No positions found**: Ensure your input files have the correct JSON structure
2. **All labels abstain**: Check that your labeling functions can handle the data format
3. **Memory issues**: Process smaller batches of files or reduce the number of positions

### Debug Mode

Add debug prints to labeling functions to understand their behavior:
```python
@labeling_function()
def lf_debug_example(x):
    print(f"Debug: Q values = {x.get('Q', [])}")
    # ... rest of function
```

## Future Enhancements

Potential improvements for the system:

1. **More sophisticated features**: Extract board state, move patterns, etc.
2. **Active learning**: Use uncertainty to select positions for human annotation
3. **Multi-class labels**: Extend to multiple Go concepts simultaneously
4. **Online learning**: Update label model as new data arrives
5. **Integration with CBM**: Use Snorkel labels to train concept bottleneck models

## References

- [Snorkel Documentation](https://snorkel.readthedocs.io/)
- [Weak Supervision Paper](https://hazyresearch.stanford.edu/snorkel/)
- [KataGo Analysis Engine](https://github.com/lightvector/KataGo)
