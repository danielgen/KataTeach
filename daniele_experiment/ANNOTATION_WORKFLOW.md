# Annotation Workflow Guide

This guide explains how to create, export, and process game annotations with full traceability back to the original games.

## MacBook Pro Setup

This project now supports **MPS (Metal Performance Shaders)** for Apple Silicon Macs, providing GPU acceleration for PyTorch operations. The scripts will automatically detect and use the best available device:

1. **MPS** (Apple Silicon Macs) - Fastest option
2. **CUDA** (NVIDIA GPUs) - For Linux/Windows systems  
3. **CPU** - Fallback for all systems

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For Apple Silicon Macs, ensure you have PyTorch with MPS support
# This should be included in the requirements.txt, but you can verify with:
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

## Overview

The annotation system now provides much better traceability and data management:

1. **Enhanced Export**: UI exports now include game metadata, statistics, and timestamps
2. **Enriched Game Data**: Merge annotations back with original game data  
3. **Batch Processing**: Process multiple annotated games at once
4. **Summary Reports**: Human-readable summaries of annotations

## Workflow Steps

### 1. Generate Games and Analysis

```bash
# Generate games with analysis (auto-detects MPS/CUDA/CPU)
python play_and_analyze.py model.ckpt 5 --output-dir games/

# On MacBook Pro, this will automatically use MPS for faster processing
# You can also explicitly specify device if needed:
python play_and_analyze.py model.ckpt 5 --device mps  # Force MPS
python play_and_analyze.py model.ckpt 5 --device cpu  # Force CPU

# This creates:
# games/
#   ├── uuid1.sgf
#   ├── uuid2.sgf  
#   └── policy/
#       ├── uuid1.json  (combined SGF + policy data)
#       └── uuid2.json
```

### 2. Create Annotation UI

```bash
# Create labeling page for a specific game
python web/label_page.py games/policy/uuid1.json game1_labels.html configs/ontology.yaml

# Open game1_labels.html in browser to annotate
```

### 3. Export Annotations (Enhanced)

When you click "Export Labels" in the UI, you now get a much richer file:

**Old format:**
```json
{
  "perMoveLabels": {...},
  "globalLabels": {...}
}
```

**New format:**
```json
{
  "format_version": "1.0",
  "exported_at": "2025-01-21T10:30:00.000Z",
  "game_metadata": {
    "black_player": "KataGo-1visit",
    "white_player": "KataGo-1visit", 
    "date": "2025-01-21",
    "game_name": "1-visit-game-1"
  },
  "annotation_statistics": {
    "total_positions_annotated": 3,
    "total_move_annotations": 12,
    "total_global_annotations": 2
  },
  "annotations": {
    "per_move_labels": {...},
    "global_labels": {...}
  },
  "source_info": {
    "sgf_preview": "(;FF[4]GM[1]SZ[19]KM[7.5]...",
    "policy_positions": 45
  }
}
```

**Filename:** `KataGo-1visit_vs_KataGo-1visit_annotations_2025-01-21T10-30-00.json`

### 4. Create Enriched Game Data

```bash
# Enrich single game with annotations
python enrich_labels.py games/policy/uuid1.json annotations/game1_annotations.json \
  --output enriched_games/game1_enriched.json \
  --summary

# This creates:
# enriched_games/
#   ├── game1_enriched.json      (complete enriched data)
#   └── game1_enriched.summary.txt  (human-readable summary)
```

**Enriched data structure:**
```json
{
  "format_version": "1.0",
  "created_at": "2025-01-21T10:35:00.000Z",
  "annotation_session_id": "session-uuid",
  
  "sgf": "...",                    // Original SGF
  "policy": {...},                 // Original policy analysis
  
  "game_metadata": {
    "black_player": "KataGo-1visit",
    "total_moves": 89,
    "source_file": "uuid1.json"
  },
  
  "annotations": {
    "per_move_labels": {...},
    "global_labels": {...},
    "annotation_metadata": {
      "annotated_positions": 3,
      "total_move_annotations": 12
    }
  },
  
  "positions": {
    "1": {
      "position_number": 1,
      "policy_analysis": {
        "suggestions": [...],
        "actual_move": {...}
      },
      "move_annotations": {
        "C16": {"joseki": true, "profit_viable": true}
      },
      "global_annotations": {"tenuki_ok": true},
      "annotation_summary": {
        "annotated_moves": ["C16", "D17"],
        "total_tags_applied": 8
      }
    }
  }
}
```

### 5. Batch Processing

```bash
# Process all annotations at once
python batch_enrich.py annotations/ games/policy/ --output enriched_games/

# This automatically matches:
# annotations/game1_annotations_*.json -> games/policy/uuid1.json
# annotations/game2_annotations_*.json -> games/policy/uuid2.json
```

## Benefits

### 1. **Full Traceability**
- Every annotation file includes game metadata
- Enriched data preserves original SGF, policy analysis, AND annotations
- Session IDs and timestamps for audit trails

### 2. **Better File Management**
- Descriptive filenames: `KataGo_vs_KataGo_annotations_2025-01-21T10-30-00.json`
- Automatic matching of annotation files to original games
- Summary reports for quick overview

### 3. **Research-Ready Data**
- All data in one place (SGF + policy + annotations)
- Structured format for easy analysis
- Statistics and metadata included

### 4. **Human-Readable Summaries**
```
ANNOTATED GAME SUMMARY
=====================

Game Information:
- Players: KataGo-1visit (Black) vs KataGo-1visit (White)
- Date: 2025-01-21
- Total Moves: 89

Annotation Summary:
- Annotated Positions: 3
- Total Move Annotations: 12
- Global Annotations: 2

Position Details:
Position 1:
  C16: joseki, profit_viable, enclose_own_corner
  D17: joseki, profit_viable, enclose_own_corner
  Global: tenuki_ok
```

## Example Complete Workflow

```bash
# 1. Generate games (auto-detects best device)
python play_and_analyze.py model.ckpt 3 --output-dir games/

# 2. Create annotation UIs  
for file in games/policy/*.json; do
  python web/label_page.py "$file" "${file%.json}_labels.html" configs/ontology.yaml
done

# 3. Annotate games (open HTML files in browser, export when done)

# 4. Batch enrich all annotated games
python batch_enrich.py annotations/ games/policy/ --output enriched_games/

# 5. Analyze enriched data
ls enriched_games/
# enriched_uuid1.json
# enriched_uuid1.summary.txt  
# enriched_uuid2.json
# enriched_uuid2.summary.txt
```

This workflow ensures you never lose track of which annotations belong to which games, and provides rich, research-ready datasets for further analysis.

## Performance Notes

### MacBook Pro Performance
- **MPS acceleration**: On Apple Silicon Macs, the scripts automatically use MPS for significant speedup
- **Memory usage**: MPS may use more memory than CPU, but provides much faster inference
- **Compatibility**: All scripts maintain backward compatibility - you can still force CPU with `--device cpu`

### Device Selection
The scripts now use intelligent device detection:
- **Auto (default)**: Automatically selects the best available device
- **MPS**: Apple Silicon GPU acceleration (recommended for MacBook Pro)
- **CUDA**: NVIDIA GPU acceleration (for Linux/Windows with NVIDIA GPUs)
- **CPU**: CPU-only processing (slower but most compatible)

Example device usage:
```bash
# Auto-detect (recommended)
python play_and_analyze.py model.ckpt 5

# Force specific device
python play_and_analyze.py model.ckpt 5 --device mps    # Apple Silicon
python play_and_analyze.py model.ckpt 5 --device cuda   # NVIDIA GPU
python play_and_analyze.py model.ckpt 5 --device cpu    # CPU only
```
