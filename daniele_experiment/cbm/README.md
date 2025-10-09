# Concept Bottleneck Model (CBM) for Go Move Analysis

This directory contains a Concept Bottleneck Model implementation for analyzing Go moves and learning interpretable concepts. The CBM learns to predict move quality while discovering meaningful concepts that explain the decision-making process.

## Overview

The CBM consists of:
- **Move-conditioned concept learning**: Learns concepts specific to each move candidate
- **Interpretable concepts**: Discovers both labeled (from annotations) and latent concepts
- **Policy prediction**: Predicts move quality based on learned concepts
- **Visualization**: Interactive HTML visualization of concepts and game analysis

## Prerequisites

Make sure you have the required data files:
- `games/slates.jsonl` - Move candidate data with KataGo policy scores
- `games/trunkfinal/` - Neural network activations for each position
- `games/policy/` - Policy analysis data (optional, for visualization)
- `games/*.sgf` - SGF game files (optional, for visualization)
- `configs/ontology.yaml` - Concept definitions and names

## Quick Start

### 1. Train the CBM Model

Train a concept learning CBM with latent concepts:

```bash
python cbm/examples/train_concept_learning_cbm.py \
    --slates-path games/slates.jsonl \
    --output-dir cbm_output \
    --num-labeled-concepts 0 \
    --num-latent-concepts 20 \
    --epochs 20 \
    --batch-size 8 \
    --lr 3e-4 \
    --lambda-sparsity 0.01 \
    --lambda-orthogonality 0.01 \
    --lambda-diversity 0.01 \
    --analyze-concepts
```

**Parameters:**
- `--num-labeled-concepts`: Number of pre-labeled concepts (0 for pure latent learning)
- `--num-latent-concepts`: Number of latent concepts to discover
- `--epochs`: Training epochs
- `--batch-size`: Number of slates per batch
- `--lr`: Learning rate
- `--lambda-*`: Regularization weights for sparsity, orthogonality, and diversity

### 2. Test the Trained Model

Evaluate the model's performance and analyze learned concepts:

```bash
python cbm/examples/test_concept_model.py \
    --model-path cbm_output/concept_learning_cbm.pt \
    --slates-path games/slates.jsonl \
    --output-dir cbm_output \
    --num-samples 50 \
    --concept-threshold 0.3 \
    --analyze-concepts
```

**Parameters:**
- `--model-path`: Path to trained model
- `--num-samples`: Number of test samples to evaluate
- `--concept-threshold`: Threshold for concept activation
- `--analyze-concepts`: Generate detailed concept analysis

### 3. Visualize Concepts and Games

Create an interactive HTML visualization showing:
- Actual Go games with WGo.js board
- KataGo policy labels
- CBM concept activations
- Move analysis with concept names

```bash
python cbm/examples/visualize_concepts.py \
    --model-path cbm_output/concept_learning_cbm.pt \
    --slates-path games/slates.jsonl \
    --output cbm_output/concept_visualization.html \
    --games-dir games \
    --ontology configs/ontology.yaml \
    --num-positions 10 \
    --concept-threshold 0.3
```

**Parameters:**
- `--games-dir`: Directory containing SGF and policy files
- `--ontology`: Path to ontology.yaml with concept names
- `--num-positions`: Number of positions to visualize
- `--concept-threshold`: Threshold for showing active concepts

## Advanced Usage

### Training with Labeled Concepts

If you have concept annotations, train with both labeled and latent concepts:

```bash
python cbm/examples/train_concept_learning_cbm.py \
    --slates-path games/slates.jsonl \
    --labels-path games/labels.jsonl \
    --output-dir cbm_output \
    --num-labeled-concepts 50 \
    --num-latent-concepts 10 \
    --epochs 30 \
    --batch-size 4
```

### Move-Conditioned CBM Training

Train a move-conditioned CBM that learns concepts specific to each move:

```bash
python cbm/examples/train_move_conditioned_cbm.py \
    --slates-path games/slates.jsonl \
    --output-dir cbm_output \
    --num-concepts 30 \
    --epochs 25 \
    --batch-size 6
```

### Convert Annotations to Labels

Convert human annotations to training labels:

```bash
python cbm/examples/convert_annotations_to_labels.py \
    --annotations-dir games/labels \
    --ontology configs/ontology.yaml \
    --output games/labels.jsonl
```

## Output Files

After training and testing, you'll find:

### Training Outputs
- `cbm_output/concept_learning_cbm.pt` - Trained model weights
- `cbm_output/training_log.txt` - Training progress log
- `cbm_output/concept_analysis.txt` - Learned concept analysis

### Test Outputs
- `cbm_output/test_results.json` - Evaluation metrics
- `cbm_output/concept_usage_analysis.json` - Concept usage statistics
- `cbm_output/move_ranking_analysis.txt` - Move ranking performance

### Visualization Outputs
- `cbm_output/concept_visualization.html` - Interactive concept visualization
- `cbm_output/concept_visualization.html` - Open in browser to explore

## Understanding the Results

### Concept Analysis
The model learns interpretable concepts such as:
- **Strategic concepts**: `building_move`, `influence_move`, `invasion_move`
- **Tactical concepts**: `atari`, `ladder_favourable`, `tesuji`
- **Shape concepts**: `good_shape`, `bad_shape`, `connection`
- **Latent concepts**: Automatically discovered patterns

### Performance Metrics
- **Spearman correlation**: How well model rankings match KataGo
- **Top-K accuracy**: How often the best move is in top-K predictions
- **KL divergence**: Distribution similarity between model and KataGo
- **Concept usage**: Which concepts are most/least active

### Visualization Features
- **Interactive Go board**: Navigate through actual games
- **Policy labels**: See KataGo winrates on each move
- **Concept activations**: View which concepts are active for each move
- **Move analysis**: Compare model vs KataGo predictions

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `--batch-size` or use CPU
2. **No concepts found**: Lower `--concept-threshold` or increase `--num-latent-concepts`
3. **Poor performance**: Increase `--epochs` or adjust learning rate
4. **Missing SGF files**: Ensure `--games-dir` contains `.sgf` files for visualization

### Performance Tips

- Use GPU for faster training: `CUDA_VISIBLE_DEVICES=0 python ...`
- Increase batch size if you have more GPU memory
- Use more epochs for better concept learning
- Adjust regularization weights based on sparsity needs

## File Structure

```
cbm/
├── examples/
│   ├── train_concept_learning_cbm.py    # Main training script
│   ├── test_concept_model.py            # Model evaluation
│   ├── visualize_concepts.py            # HTML visualization
│   ├── train_move_conditioned_cbm.py    # Alternative training
│   └── convert_annotations_to_labels.py # Label conversion
├── move_conditioned_model.py            # CBM model implementation
├── move_candidate_dataset.py            # Dataset handling
├── concept_utils.py                     # Utility functions
└── README.md                            # This file
```

## Example Workflow

```bash
# 1. Train the model
python cbm/examples/train_concept_learning_cbm.py \
    --slates-path games/slates.jsonl \
    --output-dir cbm_output \
    --num-latent-concepts 20 \
    --epochs 20 \
    --analyze-concepts

# 2. Test the model
python cbm/examples/test_concept_model.py \
    --model-path cbm_output/concept_learning_cbm.pt \
    --slates-path games/slates.jsonl \
    --output-dir cbm_output \
    --num-samples 50

# 3. Visualize results
python cbm/examples/visualize_concepts.py \
    --model-path cbm_output/concept_learning_cbm.pt \
    --slates-path games/slates.jsonl \
    --output cbm_output/concept_visualization.html \
    --games-dir games \
    --num-positions 10

# 4. Open visualization in browser
open cbm_output/concept_visualization.html
```

This will give you a complete pipeline from training to visualization, allowing you to understand what concepts the CBM has learned and how they relate to Go move quality.
