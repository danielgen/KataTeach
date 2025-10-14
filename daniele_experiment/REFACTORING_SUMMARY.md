# Code Refactoring Summary

## Overview
This document summarizes the cleanup and refactoring of the relationship between the game generation and analysis scripts in the `daniele_experiment` package.

## Changes Made

### 1. Created Common Utilities Module (`common_utils.py`)
Created a new module containing shared functions that were duplicated across multiple scripts:

- `get_device()` - Auto-detect best available PyTorch device (MPS/CUDA/CPU)
- `select_move_with_sampling()` - Sample moves within probability threshold
- `loc_to_sgf_coords()` - Convert internal location to SGF coordinates
- `create_sgf()` - Create SGF content from move list
- `calculate_dynamic_threshold()` - Calculate dynamic probability thresholds
- `_idx361_from_loc()` - Convert KataGo loc to 361-style index
- `_xy_from_loc()` - Convert KataGo loc to [x, y] coordinates
- `_loc_to_sgf()` - Convert internal loc to SGF coordinate string
- `_loc_to_human_coord()` - Convert to human-readable coordinates
- `convert_numpy_to_python()` - Convert numpy types for JSON serialization

### 2. Refactored Scripts

#### `generate_games_dataset.py`
- Removed duplicate functions: `get_device()`, `convert_numpy_to_python()`, `_idx361_from_loc()`, `_sgf_from_moves()`, `select_move_with_sampling()`
- Updated imports to use common utilities
- Replaced `_sgf_from_moves()` with `create_sgf()` from common utilities
- Simplified dynamic threshold calculation using `calculate_dynamic_threshold()`

#### `play_and_analyze.py`
- Removed duplicate functions: `get_device()`, `_idx361_from_loc()`, `_xy_from_loc()`, `_loc_to_sgf()`, `loc_to_sgf_coords()`, `_loc_to_human_coord()`, `calculate_dynamic_threshold()`, `select_move_with_sampling()`, `create_sgf()`
- Updated imports to use common utilities
- Simplified function calls to use common utilities

#### `play_games.py`
- Removed duplicate functions: `loc_to_sgf_coords()`, `select_move_with_sampling()`, `create_sgf()`
- Updated imports to use common utilities
- Removed dependency on `daniele_experiment.get_device` in favor of common utilities

#### `visualize_katago_outputs_custom.py`
- Updated to use common utilities for `get_device()` and `convert_numpy_to_python()`
- Removed duplicate function definitions
- Maintained backward compatibility by keeping the same function signatures

#### `__init__.py`
- Updated to import `get_device` from common utilities for backward compatibility
- Removed duplicate function definition

## Benefits

1. **Reduced Code Duplication**: Eliminated ~200 lines of duplicate code across the scripts
2. **Improved Maintainability**: Changes to common functions only need to be made in one place
3. **Consistent Behavior**: All scripts now use the same implementations for shared functionality
4. **Better Organization**: Common utilities are clearly separated and documented
5. **Backward Compatibility**: Existing imports continue to work through the `__init__.py` file

## File Structure After Refactoring

```
daniele_experiment/
├── common_utils.py          # NEW: Shared utilities
├── generate_games_dataset.py # REFACTORED: Uses common utilities
├── play_and_analyze.py      # REFACTORED: Uses common utilities  
├── play_games.py            # REFACTORED: Uses common utilities
├── visualize_katago_outputs_custom.py # REFACTORED: Uses common utilities
├── snorkel_board_positions.py # UNCHANGED: No redundant code found
└── __init__.py              # UPDATED: Imports from common utilities
```

## Testing

- All files pass Python syntax compilation
- No linting errors detected
- Import structure maintained for backward compatibility
- Function signatures preserved where needed

## Usage

The refactored scripts maintain the same command-line interfaces and functionality. Users can continue using them exactly as before:

```bash
# Generate games dataset
python daniele_experiment/generate_games_dataset.py model.ckpt 10 --output-dir games

# Play and analyze games  
python daniele_experiment/play_and_analyze.py model.ckpt 5 --output-dir games

# Play simple games
python daniele_experiment/play_games.py model.ckpt 3 --output-dir games
```

The refactoring is transparent to end users while significantly improving the codebase maintainability.
