#!/usr/bin/env python3
"""
Linear Probe Pipeline for Go Concept Discovery

This pipeline:
1. Builds a parquet dataset from games/* aligning trunkfinal activations with move labels
2. Trains linear probes (logistic regression / LinearSVC) with grouped CV by game_id
3. Saves concept vectors (probe weights)
4. Computes per-move delta concept scores and generates output for HTML visualization
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# ML imports
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    precision_recall_curve, average_precision_score,
)
from sklearn.calibration import CalibratedClassifierCV
import joblib


@dataclass
class ConceptDefinition:
    """Definition of a concept to probe for."""
    name: str
    type: str  # binary, threshold, threshold_negative, range, quantile
    source: str
    description: str
    threshold: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    threshold_key: Optional[str] = None  # For dict sources: "max", "sum", "mean", etc.
    # Quantile-specific fields
    q: Optional[float] = None  # Quantile fraction (default 0.1)
    direction: Optional[str] = None  # "high" or "low" for quantile
    use_abs: Optional[bool] = None  # Use absolute value for quantile (default False)
    filters: Optional[List[Dict[str, Any]]] = None  # Filter conditions before quantile
    stratify_by_phase: Optional[bool] = None  # If True, compute quantiles separately per game phase
    no_drop: bool = False  # If True, middle quantile samples labeled 0 instead of dropped
    enabled: bool = True  # If False, skip training
    exclude_move_location: bool = False  # Ablate local move channels
    feature_mode: str = "pre"  # pre | post | delta


def load_concepts(yaml_path: str) -> Tuple[List[ConceptDefinition], Dict]:
    """Load concept definitions from YAML file (skips enabled: false)."""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    concepts = []
    for name, spec in config['concepts'].items():
        if not spec.get('enabled', True):
            print(f"Skipping disabled concept: {name}")
            continue
        concepts.append(ConceptDefinition(
            name=name,
            type=spec['type'],
            source=spec['source'],
            description=spec['description'],
            threshold=spec.get('threshold'),
            min_val=spec.get('min_val'),
            max_val=spec.get('max_val'),
            threshold_key=spec.get('threshold_key'),
            q=spec.get('q', 0.1 if spec.get('type') == 'quantile' else None),
            direction=spec.get('direction'),
            use_abs=spec.get('use_abs', False),
            filters=spec.get('filters'),
            stratify_by_phase=spec.get('stratify_by_phase', False),
            no_drop=bool(spec.get('no_drop', False)),
            enabled=bool(spec.get('enabled', True)),
            exclude_move_location=bool(spec.get('exclude_move_location', False)),
            feature_mode=spec.get('feature_mode', 'pre'),
        ))
    
    return concepts, config


def passes_filters(row_or_analysis: Dict[str, Any], filters: Optional[List[Dict[str, Any]]]) -> bool:
    """Return True if analysis/row satisfies all filter conditions."""
    if not filters:
        return True
    for filt in filters:
        col = filt['column']
        op = filt['operator']
        val = filt['value']
        if col not in row_or_analysis:
            return False
        cur = row_or_analysis[col]
        if cur is None or (isinstance(cur, float) and np.isnan(cur)):
            return False
        if op == '<=' and not (cur <= val):
            return False
        if op == '>=' and not (cur >= val):
            return False
        if op == '==' and not (cur == val):
            return False
        if op == '!=' and not (cur != val):
            return False
        if op == '<' and not (cur < val):
            return False
        if op == '>' and not (cur > val):
            return False
    return True


def get_game_phase(move_number: int) -> str:
    """
    Determine game phase from move number.
    
    Uses absolute move ranges to handle variable-length games:
    - Early: moves 1-60 (opening)
    - Mid: moves 61-150 (middle game)
    - End: moves 151+ (endgame)
    
    Args:
        move_number: Move number (1-based)
    
    Returns:
        'early', 'mid', or 'end'
    """
    if move_number <= 60:
        return 'early'
    elif move_number <= 150:
        return 'mid'
    else:
        return 'end'


def extract_value(analysis: Dict, concept: ConceptDefinition) -> Optional[float]:
    """Extract raw value for a concept from analysis dict."""
    if concept.source not in analysis:
        return None
    
    value = analysis[concept.source]
    
    # Handle dict-based sources (e.g., urgency, regional concepts)
    if isinstance(value, dict) and concept.threshold_key:
        if concept.threshold_key == 'max':
            value = max(value.values()) if value else 0.0
        elif concept.threshold_key == 'sum':
            value = sum(value.values()) if value else 0.0
        elif concept.threshold_key == 'mean':
            value = sum(value.values()) / len(value) if value else 0.0
        else:
            # Default to max if unknown key
            value = max(value.values()) if value else 0.0
    
    # Convert to float if possible
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def extract_label(analysis: Dict, concept: ConceptDefinition, 
                  quantile_thresholds: Optional[Dict[str, Tuple[float, float]]] = None) -> Optional[int]:
    """
    Extract binary label for a concept from analysis dict.
    
    Args:
        analysis: Analysis dictionary
        concept: Concept definition
        quantile_thresholds: Dict mapping concept name to (low_thr, high_thr) for quantile concepts
    """
    value = extract_value(analysis, concept)
    if value is None:
        return None
    
    if concept.type == 'binary':
        return int(bool(value))
    elif concept.type == 'threshold':
        return int(value >= concept.threshold)
    elif concept.type == 'threshold_negative':
        return int(value <= concept.threshold)
    elif concept.type == 'range':
        return int(concept.min_val <= value <= concept.max_val)
    elif concept.type == 'quantile':
        if quantile_thresholds is None or concept.name not in quantile_thresholds:
            return None
        low_thr, high_thr = quantile_thresholds[concept.name]
        
        # Apply use_abs if specified
        if concept.use_abs:
            value = abs(value)
        
        # Apply direction
        if concept.direction == 'high':
            # Positive if value >= high_thr, negative if value <= low_thr, else NaN
            if value >= high_thr:
                return 1
            elif value <= low_thr:
                return 0
            else:
                return 0 if concept.no_drop else None  # Middle: keep as neg or drop
        elif concept.direction == 'low':
            # Positive if value <= low_thr, negative if value >= high_thr, else NaN
            if value <= low_thr:
                return 1
            elif value >= high_thr:
                return 0
            else:
                return 0 if concept.no_drop else None
        else:
            return None
    
    return None


def check_filter_columns(df: pd.DataFrame, filters: List[Dict[str, Any]], concept_name: str, warned_concepts: set) -> bool:
    """Check if all filter columns exist in dataframe. Returns True if all present, False otherwise."""
    if not filters:
        return True
    
    missing_cols = []
    for filt in filters:
        col = filt['column']
        if col not in df.columns:
            missing_cols.append(col)
    
    if missing_cols:
        # Only warn once per concept
        if concept_name not in warned_concepts:
            import warnings
            warnings.warn(
                f"Concept '{concept_name}': Missing filter columns {missing_cols}. "
                f"Skipping labeling for this concept.",
                UserWarning
            )
            warned_concepts.add(concept_name)
        return False
    
    return True


def apply_filters(df: pd.DataFrame, filters: List[Dict[str, Any]]) -> pd.DataFrame:
    """Apply filter conditions to dataframe."""
    if not filters:
        return df
    
    mask = pd.Series(True, index=df.index)
    for filt in filters:
        col = filt['column']
        op = filt['operator']  # '<=', '>=', '==', '!=', '<', '>'
        val = filt['value']
        
        if col not in df.columns:
            continue
        
        if op == '<=':
            mask &= (df[col] <= val)
        elif op == '>=':
            mask &= (df[col] >= val)
        elif op == '==':
            mask &= (df[col] == val)
        elif op == '!=':
            mask &= (df[col] != val)
        elif op == '<':
            mask &= (df[col] < val)
        elif op == '>':
            mask &= (df[col] > val)
    
    return df[mask]


def compute_quantile_thresholds(
    df: pd.DataFrame,
    concept: ConceptDefinition,
    train_indices: np.ndarray,
) -> Optional[Tuple[float, float]]:
    """
    Compute quantile thresholds for a concept on training data only.
    Applies filters before computing thresholds.
    Optionally stratifies by game phase if concept.stratify_by_phase is True.
    
    Args:
        df: Full dataframe (must have 'game_phase' column if stratify_by_phase=True)
        concept: Concept definition
        train_indices: Indices of training samples
    
    Returns:
        (low_threshold, high_threshold) tuple or None if insufficient data
        If stratify_by_phase=True, returns dict mapping phase -> (low, high) thresholds
    """
    train_df = df.iloc[train_indices].copy()
    
    # Get raw values using per-concept column name
    value_col = f"rawval_{concept.name}"
    if value_col not in train_df.columns:
        return None
    
    # Apply filters first if specified
    if concept.filters:
        train_df = apply_filters(train_df, concept.filters)
        if len(train_df) == 0:
            return None
    
    # If stratifying by phase, compute thresholds per phase
    if concept.stratify_by_phase:
        if 'game_phase' not in train_df.columns:
            # Fall back to non-stratified if phase column missing
            pass
        else:
            phase_thresholds = {}
            for phase in ['early', 'mid', 'end']:
                phase_df = train_df[train_df['game_phase'] == phase]
                if len(phase_df) == 0:
                    continue
                
                phase_values = phase_df[value_col].values
                phase_values = phase_values[~pd.isna(phase_values)]
                
                if len(phase_values) < 10:  # Need at least 10 samples per phase
                    continue
                
                # Apply use_abs if specified
                if concept.use_abs:
                    phase_values = np.abs(phase_values)
                
                q = concept.q or 0.1
                low_thr = float(np.quantile(phase_values, q))
                high_thr = float(np.quantile(phase_values, 1 - q))
                phase_thresholds[phase] = (low_thr, high_thr)
            
            # Return dict if we got thresholds for at least one phase
            if phase_thresholds:
                return phase_thresholds
            # Fall through to non-stratified if no phases had enough data
    
    # Non-stratified: compute thresholds on all data
    values = train_df[value_col].values
    values = values[~pd.isna(values)]
    
    if len(values) == 0:
        return None
    
    # Apply use_abs if specified
    if concept.use_abs:
        values = np.abs(values)
    
    if len(values) < 10:  # Need at least 10 samples for meaningful quantiles
        return None
    
    q = concept.q or 0.1
    low_thr = float(np.quantile(values, q))
    high_thr = float(np.quantile(values, 1 - q))
    
    return (low_thr, high_thr)


def build_dataset(
    games_dir: str,
    concepts: List[ConceptDefinition],
    output_path: str,
    aggregation: str = "global_pool",
    pool_type: str = "mean",
    include_move_location: bool = True,
) -> pd.DataFrame:
    """
    Build parquet dataset from games directory.
    
    Aligns trunkfinal activations (h) with move labels from snorkel analysis.
    Also includes h_next (activations after the move) for delta computation.
    
    Args:
        games_dir: Path to games directory
        concepts: List of concept definitions
        output_path: Path to save parquet file
        aggregation: How to aggregate (512, 19, 19) -> vector
        pool_type: mean, max, or both
        include_move_location: Include features from move location
    
    Returns:
        DataFrame with columns: game_id, move_number, h_pooled, h_next_pooled, labels...
    """
    games_path = Path(games_dir)
    all_rows = []
    
    game_dirs = sorted([d for d in games_path.iterdir() if d.is_dir()])
    
    for game_dir in tqdm(game_dirs, desc="Processing games"):
        game_id = game_dir.name
        
        # Load snorkel analysis data (contains concept labels)
        snorkel_path = game_dir / "snorkel.jsonl"
        if not snorkel_path.exists():
            continue
        
        # Build analysis dict indexed by move number
        analysis_by_move = {}
        with open(snorkel_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                analysis_by_move[data['move_number']] = data
        
        # Load moves data for basic info
        moves_path = game_dir / "moves.jsonl"
        if not moves_path.exists():
            continue
        
        moves = []
        with open(moves_path, 'r') as f:
            for line in f:
                move_data = json.loads(line)
                # Merge in snorkel analysis if available
                move_num = move_data['move_number']
                if move_num in analysis_by_move:
                    move_data['analysis'] = analysis_by_move[move_num].get('analysis', {})
                moves.append(move_data)
        
        # Load trunkfinal activations
        trunk_dir = game_dir / "trunkfinal"
        if not trunk_dir.exists():
            continue
        
        for move_data in moves:
            move_num = move_data['move_number']
            analysis = move_data.get('analysis', {})
            player = move_data['player']
            move_loc = move_data['move_loc']
            # KataGo's `move_loc` uses a padded 20-wide Board array. Local
            # trunk features require the explicit row-major tensor coordinate
            # persisted during generation. Fail closed rather than silently
            # interpreting a padded location as a 0..360 index.
            if 'idx361' not in move_data:
                raise ValueError(
                    f"Missing idx361 for {game_id} move {move_num}; "
                    "migrate/regenerate move metadata before building probes"
                )
            move_idx361 = int(move_data['idx361'])
            
            # Load current activation (h)
            h_path = trunk_dir / f"move_{move_num:03d}.npy"
            if not h_path.exists():
                continue
            
            h = np.load(h_path)  # (512, 19, 19)
            
            # Load next activation (h_next) if available
            h_next_path = trunk_dir / f"move_{move_num + 1:03d}.npy"
            h_next = np.load(h_next_path) if h_next_path.exists() else None
            
            # Aggregate features
            features = aggregate_features(
                h, move_idx361, aggregation, pool_type, include_move_location
            )
            
            features_next = None
            if h_next is not None:
                # For h_next, we use same move_loc since we want to see how
                # the representation at that location changed
                features_next = aggregate_features(
                    h_next, move_idx361, aggregation, pool_type, include_move_location
                )
            
            # Labels will be added per-concept below (only for non-quantile)
            
            # Build row
            row = {
                'game_id': game_id,
                'move_number': move_num,
                'player': player,
                'move_loc': move_loc,
                'idx361': move_idx361,
                'game_phase': get_game_phase(move_num),  # Add game phase for stratification
            }
            
            # Add features (store as list for parquet compatibility)
            row['h_features'] = features.tolist()
            row['h_features_dim'] = len(features)
            
            if features_next is not None:
                row['h_next_features'] = features_next.tolist()
            else:
                row['h_next_features'] = None
            
            # Filter-compatible analysis columns (include values needed by YAML filters)
            for key in [
                'building_count', 'solidification_count', 'reduction_count',
                'potential_territory', 'solid_territory', 'liberties',
                'group_strength_delta', 'influence_count_delta',
                'current_group_connectivity_delta', 'attacked_groups_count',
                'building_intensity', 'solidification_intensity', 'reduction_intensity',
                'max_attack_intensity', 'direct_sacrifice', 'connection_strength_gain',
            ]:
                value = analysis.get(key)
                row[key] = value
                if f"raw_{key}" not in row:
                    row[f"raw_{key}"] = value

            # Add labels (only for fixed/binary concepts, not quantile)
            # Apply YAML filters for non-quantile concepts (e.g. fight_wide)
            for concept in concepts:
                if concept.type != "quantile":
                    if concept.filters and not passes_filters({**analysis, **row}, concept.filters):
                        row[f"label_{concept.name}"] = None
                    else:
                        row[f"label_{concept.name}"] = extract_label(analysis, concept)
            
            # Add raw analysis values per concept (needed for quantile thresholds)
            for concept in concepts:
                raw_value = extract_value(analysis, concept)
                row[f"rawval_{concept.name}"] = raw_value
            
            all_rows.append(row)
    
    df = pd.DataFrame(all_rows)
    
    # Save to parquet
    df.to_parquet(output_path, index=False)
    print(f"Saved dataset with {len(df)} samples to {output_path}")
    
    return df


def aggregate_features(
    h: np.ndarray,
    idx361: int,
    aggregation: str,
    pool_type: str,
    include_move_location: bool,
) -> np.ndarray:
    """
    Aggregate (512, 19, 19) activations to a feature vector.
    
    Args:
        h: Activation tensor (512, 19, 19)
        idx361: Explicit row-major tensor index (0-360; 361 means pass)
        aggregation: global_pool, move_location, or both
        pool_type: mean, max, or both
        include_move_location: Whether to include move location features
    
    Returns:
        Feature vector (always the same dimension regardless of pass)
    """
    features = []
    n_channels = h.shape[0]  # 512
    
    # Global pooling
    if aggregation in ["global_pool", "both"]:
        if pool_type in ["mean", "both"]:
            features.append(h.mean(axis=(1, 2)))  # (512,)
        if pool_type in ["max", "both"]:
            features.append(h.max(axis=(1, 2)))  # (512,)
    
    # Move-location features - always include the same dimension for consistency.
    if include_move_location:
        if 0 <= idx361 < 361:
            y, x = divmod(idx361, 19)
            features.append(h[:, y, x])  # (512,)
        elif idx361 == 361:
            features.append(np.zeros(n_channels, dtype=h.dtype))
        else:
            raise ValueError(f"idx361 must be in [0, 361], got {idx361}")
    
    return np.concatenate(features) if features else h.mean(axis=(1, 2))


def get_concept_features(df: pd.DataFrame, concept: ConceptDefinition) -> Optional[np.ndarray]:
    """
    Build feature matrix for a concept based on feature_mode.

    - pre: h_features (pre-move trunk)
    - post: h_next_features (post-move trunk); rows without h_next excluded by caller via NaN mask
    - delta: h_next - h
    """
    mode = (concept.feature_mode or "pre").lower()
    h = np.array(df["h_features"].tolist())

    if mode == "pre":
        return h

    has_next = df["h_next_features"].notna().to_numpy()
    if not has_next.any():
        return None

    # Build aligned next features; missing next -> zeros (masked out later via labels)
    h_next = np.zeros_like(h)
    next_list = df["h_next_features"].tolist()
    for i, feat in enumerate(next_list):
        if feat is not None:
            h_next[i] = np.array(feat)

    if mode == "post":
        return h_next
    if mode == "delta":
        return h_next - h
    return h


def train_probes(
    df: pd.DataFrame,
    concepts: List[ConceptDefinition],
    config: Dict,
    output_dir: str,
    labeling: str = "quantile",
) -> Dict[str, Dict]:
    """
    Train linear probes for each concept.
    
    Uses GroupKFold cross-validation grouped by game_id.
    For quantile concepts, computes thresholds per fold to avoid leakage.
    Moves scaling inside CV folds to avoid test leakage.
    
    Note on quantile labeling:
    - Final model thresholds are computed on all data (for consistency)
    - CV metrics (AUC) use per-fold thresholds (honest generalization estimates)
    - Training set metrics (train_accuracy, train_f1) are optimistic (same data used for training)
    - Use CV metrics for honest performance estimates
    
    Note on sparse sources:
    - For sparse quantile sources (e.g., building_count, reduction_count), add filters
      to ensure source > 0 before computing quantiles. Otherwise "top 10%" might mean ">= 1"
      and labels become binary noise.
    
    Args:
        df: Dataset DataFrame
        concepts: List of concepts to train probes for
        config: Config dict with training settings
        output_dir: Directory to save models and vectors
        labeling: "fixed" or "quantile" - how to label continuous concepts
    
    Returns:
        Dict of results per concept
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract feature matrix (not scaled yet - will scale per fold)
    # Per-concept feature matrices are built inside the loop via get_concept_features
    groups = df['game_id'].values
    
    results = {}
    concept_vectors = {}
    all_quantile_thresholds = {}  # Store thresholds for reporting
    warned_concepts = set()  # Track concepts that have been warned about missing filters
    
    train_config = config.get('training', {})
    n_folds = train_config.get('cv_folds', 5)
    C_values = train_config.get('C_values', [0.01, 0.1, 1.0])
    min_val_samples = train_config.get('min_val_samples', 50)  # Minimum labeled samples in val fold
    do_calibrate = train_config.get('calibrate', True)
    min_cv_auc = train_config.get('min_cv_auc', 0.75)
    min_cv_f1 = train_config.get('min_cv_f1', 0.25)
    min_cv_ap = train_config.get('min_cv_avg_precision', 0.20)
    
    for concept in tqdm(concepts, desc="Training probes"):
        # For quantile concepts with quantile labeling, we'll create labels per fold
        # For fixed labeling or binary concepts, use pre-computed labels
        if labeling == "quantile" and concept.type == "quantile":
            # Quantile labeling - labels computed per fold
            label_col = None  # Will compute per fold
        else:
            # Fixed labeling - use pre-computed labels
            label_col = f"label_{concept.name}"
            if label_col not in df.columns:
                print(f"Skipping {concept.name}: label column not found")
                continue
        
        # Train with GroupKFold (scaling inside folds)
        unique_groups = np.unique(groups)
        if len(unique_groups) < 2:
            print(f"Skipping {concept.name}: insufficient groups for CV")
            continue
        
        gkf = GroupKFold(n_splits=min(n_folds, len(unique_groups)))
        
        fold_scores = []
        fold_ap_scores = []
        fold_f1_scores = []
        fold_thresholds = []
        fold_pos_counts = []
        fold_neg_counts = []
        fold_best_Cs = []
        fold_best_thresholds = []  # Store optimal thresholds per fold
        fold_val_probs = []  # for calibration diagnostics
        fold_val_labels = []
        
        # Define helper functions for labeling (will be used as closures over thresholds)
        # Precompute arrays for performance (avoid df.iloc in loops)
        raw_col = f"rawval_{concept.name}"
        if raw_col not in df.columns:
            print(f"Skipping {concept.name}: raw value column not found")
            continue
        
        raw_values = df[raw_col].to_numpy()
        raw_values_abs = np.abs(raw_values) if concept.use_abs else raw_values
        
        # Precompute filter masks if filters exist
        # If any filter column is missing, skip the concept entirely
        filters_ok = True
        filter_masks = []
        if concept.filters:
            for filt in concept.filters:
                col = filt['column']
                op = filt['operator']
                val = filt['value']
                
                if col not in df.columns:
                    print(f"Skipping {concept.name}: filter column {col} not found")
                    filters_ok = False
                    break
                
                col_values = df[col].to_numpy()
                if op == '<=':
                    mask = col_values <= val
                elif op == '>=':
                    mask = col_values >= val
                elif op == '==':
                    mask = col_values == val
                elif op == '!=':
                    mask = col_values != val
                elif op == '<':
                    mask = col_values < val
                elif op == '>':
                    mask = col_values > val
                else:
                    mask = np.ones(len(df), dtype=bool)
                
                # Also check for NaN
                mask = mask & ~pd.isna(col_values)
                filter_masks.append(mask)
        
        if not filters_ok:
            continue
        
        # Precompute game_phase array if stratifying by phase
        game_phases = None
        if concept.stratify_by_phase and 'game_phase' in df.columns:
            game_phases = df['game_phase'].to_numpy()
        
        def make_label_sample_fn_vectorized(thresholds):
            """Create a vectorized labeling function.
            
            Args:
                thresholds: Either (low_thr, high_thr) tuple or dict mapping phase -> (low_thr, high_thr)
            """
            # Normalize to dict format for uniform handling
            if isinstance(thresholds, tuple):
                # Non-stratified: use same thresholds for all phases
                phase_thresholds = {'early': thresholds, 'mid': thresholds, 'end': thresholds}
            else:
                # Stratified: use provided phase thresholds
                phase_thresholds = thresholds
            
            def label_sample_vectorized(indices):
                """Label samples using vectorized operations.
                
                Returns:
                    (labels, valid_mask) where:
                    - labels: np.array with 0, 1, or -1 (for invalid/unlabeled)
                    - valid_mask: boolean array indicating which samples have valid labels
                """
                # Apply filters if they exist
                if filter_masks:
                    # Combine all filter masks
                    combined_mask = np.ones(len(indices), dtype=bool)
                    for mask in filter_masks:
                        combined_mask = combined_mask & mask[indices]
                else:
                    combined_mask = np.ones(len(indices), dtype=bool)
                
                # Get raw values for these indices
                vals = raw_values_abs[indices]
                
                # Check for NaN
                valid_mask = combined_mask & ~pd.isna(vals)
                
                # Initialize labels with -1 (invalid/unlabeled)
                labels = np.full(len(indices), -1, dtype=np.int8)
                
                # If stratifying by phase, label per phase
                if game_phases is not None:
                    for phase in ['early', 'mid', 'end']:
                        if phase not in phase_thresholds:
                            continue
                        low_thr, high_thr = phase_thresholds[phase]
                        phase_mask = (game_phases[indices] == phase) & valid_mask
                        
                        if concept.direction == 'high':
                            labels[phase_mask & (vals >= high_thr)] = 1
                            labels[phase_mask & (vals <= low_thr)] = 0
                            if concept.no_drop:
                                mid = phase_mask & (vals > low_thr) & (vals < high_thr)
                                labels[mid] = 0
                        elif concept.direction == 'low':
                            labels[phase_mask & (vals <= low_thr)] = 1
                            labels[phase_mask & (vals >= high_thr)] = 0
                            if concept.no_drop:
                                mid = phase_mask & (vals > low_thr) & (vals < high_thr)
                                labels[mid] = 0
                else:
                    # Non-stratified: use first available threshold (all phases same)
                    low_thr, high_thr = next(iter(phase_thresholds.values()))
                    
                    if concept.direction == 'high':
                        labels[valid_mask & (vals >= high_thr)] = 1
                        labels[valid_mask & (vals <= low_thr)] = 0
                        if concept.no_drop:
                            mid = valid_mask & (vals > low_thr) & (vals < high_thr)
                            labels[mid] = 0
                    elif concept.direction == 'low':
                        labels[valid_mask & (vals <= low_thr)] = 1
                        labels[valid_mask & (vals >= high_thr)] = 0
                        if concept.no_drop:
                            mid = valid_mask & (vals > low_thr) & (vals < high_thr)
                            labels[mid] = 0
                
                # Valid mask: samples that passed filters, aren't NaN, and have a label (0 or 1)
                valid_mask = valid_mask & (labels >= 0)
                
                return labels, valid_mask
            return label_sample_vectorized
        
        # Select feature matrix for this concept (pre / post / delta)
        X_concept = get_concept_features(df, concept)
        if X_concept is None:
            print(f"Skipping {concept.name}: features unavailable for mode={concept.feature_mode}")
            continue
        n_channels = 512
        if concept.exclude_move_location and X_concept.shape[1] >= 2 * n_channels:
            # Drop local move-location half (second 512 dims when global+local)
            X_concept = X_concept[:, :n_channels]
        
        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_concept, groups=groups)):
            X_train_fold = X_concept[train_idx]
            X_val_fold = X_concept[val_idx]
            groups_train = groups[train_idx]
            groups_val = groups[val_idx]
            
            # Scale features per fold (fit on train, transform both)
            scaler_fold = StandardScaler()
            X_train_scaled = scaler_fold.fit_transform(X_train_fold)
            X_val_scaled = scaler_fold.transform(X_val_fold)
            
            # Get labels for this fold
            if labeling == "quantile" and concept.type == "quantile":
                # Compute quantile thresholds on training fold only
                thresholds = compute_quantile_thresholds(df, concept, train_idx)
                
                if thresholds is None:
                    continue
                
                # Normalize thresholds format
                if isinstance(thresholds, dict):
                    # Phase-stratified: store dict
                    fold_thresholds.append(thresholds)
                else:
                    # Non-stratified: store tuple
                    fold_thresholds.append(thresholds)
                
                # Create vectorized labeling function for this fold's thresholds
                label_samples = make_label_sample_fn_vectorized(thresholds)
                
                # Label train samples (vectorized)
                train_labels, train_valid_mask = label_samples(train_idx)
                y_train_fold = train_labels[train_valid_mask].astype(int)
                
                # Label val samples (vectorized)
                val_labels, val_valid_mask = label_samples(val_idx)
                y_val_fold = val_labels[val_valid_mask].astype(int)
                
                if len(y_train_fold) < 20:
                    continue
                
                # Check class balance in train
                pos_rate_train = y_train_fold.mean()
                if pos_rate_train < 0.01 or pos_rate_train > 0.99:
                    continue
                
                # Robust fold scoring: check val set
                if len(y_val_fold) < min_val_samples:
                    print(f"  Fold {fold_idx} skipped: only {len(y_val_fold)} labeled val samples (min={min_val_samples})")
                    continue
                
                # Check if val set has both classes
                val_pos_count = int(y_val_fold.sum())
                val_neg_count = int((1 - y_val_fold).sum())
                if val_pos_count == 0 or val_neg_count == 0:
                    print(f"  Fold {fold_idx} skipped: single class in val set (pos={val_pos_count}, neg={val_neg_count})")
                    continue
                
                fold_pos_counts.append(int(y_train_fold.sum()))
                fold_neg_counts.append(int((1 - y_train_fold).sum()))
                
                X_train_labeled = X_train_scaled[train_valid_mask]
                X_val_labeled = X_val_scaled[val_valid_mask]
                
                # Assertions: verify lengths match
                assert len(X_train_labeled) == len(y_train_fold), \
                    f"Train X/y length mismatch: X={len(X_train_labeled)}, y={len(y_train_fold)}"
                assert len(X_val_labeled) == len(y_val_fold), \
                    f"Val X/y length mismatch: X={len(X_val_labeled)}, y={len(y_val_fold)}"
                
            else:
                # Fixed labeling - use pre-computed labels
                y_all = df[label_col].values
                y_train_fold = y_all[train_idx]
                y_val_fold = y_all[val_idx]
                
                train_valid_mask = ~pd.isna(y_train_fold)
                val_valid_mask = ~pd.isna(y_val_fold)
                
                if train_valid_mask.sum() < 20:
                    continue
                
                y_train_fold = y_train_fold[train_valid_mask].astype(int)
                y_val_fold = y_val_fold[val_valid_mask].astype(int)
                
                pos_rate_train = y_train_fold.mean()
                if pos_rate_train < 0.01 or pos_rate_train > 0.99:
                    continue
                
                X_train_labeled = X_train_scaled[train_valid_mask]
                X_val_labeled = X_val_scaled[val_valid_mask]
                
                # Assertions: verify lengths match
                assert len(X_train_labeled) == len(y_train_fold), \
                    f"Train X/y length mismatch: X={len(X_train_labeled)}, y={len(y_train_fold)}"
                assert len(X_val_labeled) == len(y_val_fold), \
                    f"Val X/y length mismatch: X={len(X_val_labeled)}, y={len(y_val_fold)}"
            
            # Try different C values and pick best for this fold
            best_fold_score = -1
            best_fold_C = None
            best_fold_model = None
            best_fold_val_prob = None
            
            for C in C_values:
                model = LogisticRegression(
                    C=C,
                    class_weight='balanced',
                    max_iter=1000,
                    solver='lbfgs',
                    random_state=42,
                )
                
                try:
                    model.fit(X_train_labeled, y_train_fold)
                    y_val_prob = model.predict_proba(X_val_labeled)[:, 1]
                    score = roc_auc_score(y_val_fold, y_val_prob)
                    
                    if score > best_fold_score:
                        best_fold_score = score
                        best_fold_C = C
                        best_fold_model = model
                        best_fold_val_prob = y_val_prob
                except Exception as e:
                    continue
            
            if best_fold_C is not None:
                fold_scores.append(best_fold_score)
                fold_best_Cs.append(best_fold_C)
                fold_val_probs.append(best_fold_val_prob)
                fold_val_labels.append(y_val_fold)
                
                try:
                    fold_ap_scores.append(average_precision_score(y_val_fold, best_fold_val_prob))
                except Exception:
                    pass
                
                # Optimize threshold on validation set to maximize F1
                try:
                    precision, recall, thresholds = precision_recall_curve(
                        y_val_fold, best_fold_val_prob
                    )
                    if len(thresholds) > 0:
                        f1_scores = 2 * (precision[1:] * recall[1:]) / (precision[1:] + recall[1:] + 1e-10)
                        best_threshold_idx = np.argmax(f1_scores)
                        best_threshold = thresholds[best_threshold_idx]
                        fold_best_thresholds.append(best_threshold)
                        fold_f1_scores.append(float(f1_scores[best_threshold_idx]))
                    else:
                        fold_best_thresholds.append(0.5)
                        y_hat = (best_fold_val_prob >= 0.5).astype(int)
                        fold_f1_scores.append(float(f1_score(y_val_fold, y_hat, zero_division=0)))
                except Exception as e:
                    fold_best_thresholds.append(0.5)
        
        if len(fold_scores) == 0:
            print(f"Skipping {concept.name}: no valid folds")
            continue
        
        # Train final model on all labeled data
        # First, create labels for all data if quantile
        if labeling == "quantile" and concept.type == "quantile":
            # Compute final thresholds on full filtered dataset (deterministic)
            # This is more consistent than averaging fold thresholds
            # Note: CV AUC uses per-fold thresholds (honest generalization),
            # while final model uses global thresholds (for consistency)
            
            all_indices = np.arange(len(df))
            final_thresholds = compute_quantile_thresholds(df, concept, all_indices)
            
            if final_thresholds is None:
                print(f"Skipping {concept.name}: could not compute final thresholds on full dataset")
                continue
            
            all_quantile_thresholds[concept.name] = final_thresholds
            
            # Create labels for all data (with filters applied) - vectorized
            all_labels, valid_mask_all = make_label_sample_fn_vectorized(final_thresholds)(all_indices)
            
            y_all = all_labels[valid_mask_all].astype(int)
        else:
            y_all = df[label_col].values
            valid_mask_all = ~pd.isna(y_all)
            y_all = y_all[valid_mask_all].astype(int)
        
        if valid_mask_all.sum() < 100:
            print(f"Skipping {concept.name}: only {valid_mask_all.sum()} valid samples")
            continue
        
        # Fit per-concept scaler on all labeled data for this concept
        concept_scaler = StandardScaler()
        X_all = X_concept[valid_mask_all]
        X_all_scaled = concept_scaler.fit_transform(X_all)
        
        # Find best C - use the C that appears most often in best folds
        best_score = np.mean(fold_scores) if len(fold_scores) > 0 else 0.0
        best_ap = np.mean(fold_ap_scores) if len(fold_ap_scores) > 0 else 0.0
        best_cv_f1 = np.mean(fold_f1_scores) if len(fold_f1_scores) > 0 else 0.0
        
        if len(fold_best_Cs) > 0:
            # Use the most common C value from folds
            c_counts = Counter(fold_best_Cs)
            best_C = c_counts.most_common(1)[0][0]
        else:
            best_C = C_values[len(C_values) // 2]  # Default to middle value
        
        # Train final model
        final_model = LogisticRegression(
            C=best_C,
            class_weight='balanced',
            max_iter=1000,
            solver='lbfgs',
            random_state=42,
        )
        final_model.fit(X_all_scaled, y_all)

        # Optional probability calibration on held-out style fit
        calibrated_model = None
        if do_calibrate and len(np.unique(y_all)) > 1 and len(y_all) >= 200:
            try:
                calibrated_model = CalibratedClassifierCV(
                    estimator=LogisticRegression(
                        C=best_C,
                        class_weight='balanced',
                        max_iter=1000,
                        solver='lbfgs',
                        random_state=42,
                    ),
                    method='isotonic',
                    cv=min(3, max(2, len(np.unique(groups[valid_mask_all])))),
                )
                calibrated_model.fit(X_all_scaled, y_all)
            except Exception as e:
                print(f"  Calibration failed for {concept.name}: {e}")
                calibrated_model = None
        
        # Get concept vector
        concept_vector = final_model.coef_[0]
        concept_vectors[concept.name] = concept_vector
        
        # Get optimal threshold from CV folds (use median for robustness)
        if len(fold_best_thresholds) > 0:
            optimal_threshold = np.median(fold_best_thresholds)
        else:
            optimal_threshold = 0.5  # Default threshold
        
        # Evaluate with optimal threshold (prefer calibrated probs when available)
        if calibrated_model is not None:
            y_prob = calibrated_model.predict_proba(X_all_scaled)[:, 1]
        else:
            y_prob = final_model.predict_proba(X_all_scaled)[:, 1]
        y_pred_default = final_model.predict(X_all_scaled)  # Default 0.5 threshold
        y_pred_optimal = (y_prob >= optimal_threshold).astype(int)  # Optimal threshold
        
        pos_rate = y_all.mean()
        pos_count = int(y_all.sum())
        neg_count = int((1 - y_all).sum())
        
        # Compute metrics with default threshold (for comparison)
        train_precision_default = precision_score(y_all, y_pred_default, zero_division=0)
        train_recall_default = recall_score(y_all, y_pred_default, zero_division=0)
        train_f1_default = f1_score(y_all, y_pred_default, zero_division=0)
        
        # Compute metrics with optimal threshold
        train_precision = precision_score(y_all, y_pred_optimal, zero_division=0)
        train_recall = recall_score(y_all, y_pred_optimal, zero_division=0)
        train_f1 = f1_score(y_all, y_pred_optimal, zero_division=0)

        export_ok = bool(
            best_score >= min_cv_auc
            and best_cv_f1 >= min_cv_f1
            and best_ap >= min_cv_ap
        )
        
        results[concept.name] = {
            'cv_auc': float(best_score),
            'cv_auc_std': float(np.std(fold_scores) if len(fold_scores) > 1 else 0.0),
            'cv_avg_precision': float(best_ap),
            'cv_avg_precision_std': float(np.std(fold_ap_scores) if len(fold_ap_scores) > 1 else 0.0),
            'cv_f1': float(best_cv_f1),
            'cv_f1_std': float(np.std(fold_f1_scores) if len(fold_f1_scores) > 1 else 0.0),
            'n_folds_scored': int(len(fold_scores)),
            'best_C': float(best_C) if not isinstance(best_C, str) else best_C,
            'optimal_threshold': float(optimal_threshold),
            'threshold_std': float(np.std(fold_best_thresholds)) if len(fold_best_thresholds) > 1 else 0.0,
            'calibrated': bool(calibrated_model is not None),
            'feature_mode': concept.feature_mode,
            'exclude_move_location': bool(concept.exclude_move_location),
            'export_ok': export_ok,
            # Training set metrics with optimal threshold (optimistic - same data used for training)
            # Use CV metrics for honest generalization estimates
            'train_accuracy': accuracy_score(y_all, y_pred_optimal),
            'train_f1': train_f1,
            'train_precision': train_precision,
            'train_recall': train_recall,
            # Also store default threshold metrics for comparison
            'train_f1_default_threshold': train_f1_default,
            'train_precision_default_threshold': train_precision_default,
            'train_recall_default_threshold': train_recall_default,
            'pos_rate': pos_rate,
            'pos_count': pos_count,
            'neg_count': neg_count,
            'n_samples': len(y_all),
            'vector_norm': float(np.linalg.norm(concept_vector)),
        }
        
        # Add quantile thresholds if applicable
        if labeling == "quantile" and concept.type == "quantile":
            if concept.name in all_quantile_thresholds:
                thresholds = all_quantile_thresholds[concept.name]
                
                # Handle both non-stratified (tuple) and phase-stratified (dict) cases
                if isinstance(thresholds, dict):
                    # Phase-stratified: store per-phase thresholds
                    results[concept.name]['quantile_thresholds_by_phase'] = thresholds
                    results[concept.name]['stratify_by_phase'] = True
                    # Also compute average thresholds across phases for summary
                    all_lows = [t[0] for t in thresholds.values()]
                    all_highs = [t[1] for t in thresholds.values()]
                    results[concept.name]['quantile_avg_low_threshold'] = np.mean(all_lows)
                    results[concept.name]['quantile_avg_high_threshold'] = np.mean(all_highs)
                else:
                    # Non-stratified: store as before
                    low_thr, high_thr = thresholds
                    results[concept.name]['quantile_low_threshold'] = low_thr
                    results[concept.name]['quantile_high_threshold'] = high_thr
                    results[concept.name]['stratify_by_phase'] = False
                
                results[concept.name]['quantile_q'] = concept.q or 0.1
                results[concept.name]['quantile_direction'] = concept.direction
                results[concept.name]['labeling_scheme'] = (
                    'extreme_quantiles_keep_middle_as_neg' if concept.no_drop
                    else 'extreme_quantiles_drop_middle'
                )
                results[concept.name]['no_drop'] = concept.no_drop
                
                # Report fold thresholds for reference (average across folds)
                if len(fold_thresholds) > 0:
                    # Handle both tuple and dict fold thresholds
                    if isinstance(fold_thresholds[0], dict):
                        # Phase-stratified folds: average per phase
                        phase_avgs = {}
                        for phase in ['early', 'mid', 'end']:
                            phase_lows = [t.get(phase, (None, None))[0] for t in fold_thresholds if phase in t]
                            phase_highs = [t.get(phase, (None, None))[1] for t in fold_thresholds if phase in t]
                            phase_lows = [x for x in phase_lows if x is not None]
                            phase_highs = [x for x in phase_highs if x is not None]
                            if phase_lows and phase_highs:
                                phase_avgs[phase] = (np.mean(phase_lows), np.mean(phase_highs))
                        if phase_avgs:
                            results[concept.name]['quantile_fold_avg_thresholds_by_phase'] = phase_avgs
                    else:
                        # Non-stratified folds: average as before
                        avg_fold_low = np.mean([t[0] for t in fold_thresholds])
                        avg_fold_high = np.mean([t[1] for t in fold_thresholds])
                        results[concept.name]['quantile_fold_avg_low_threshold'] = avg_fold_low
                        results[concept.name]['quantile_fold_avg_high_threshold'] = avg_fold_high
        elif concept.type == "binary":
            results[concept.name]['labeling_scheme'] = 'binary'
        elif concept.type == "threshold":
            results[concept.name]['labeling_scheme'] = 'threshold'
            results[concept.name]['threshold'] = concept.threshold
        
        # Save model and per-concept scaler (and calibrator if present)
        joblib.dump(final_model, output_path / f"probe_{concept.name}.joblib")
        joblib.dump(concept_scaler, output_path / f"scaler_{concept.name}.joblib")
        if calibrated_model is not None:
            joblib.dump(calibrated_model, output_path / f"calibrator_{concept.name}.joblib")
        
        export_mark = "EXPORT" if export_ok else "LOW_QUALITY"
        print(f"  {concept.name} [{export_mark}]: AUC={best_score:.3f}±{results[concept.name]['cv_auc_std']:.3f}, "
              f"AP={best_ap:.3f}, cv_F1={best_cv_f1:.3f}, "
              f"train_F1={train_f1:.3f} (thresh={optimal_threshold:.3f}), "
              f"mode={concept.feature_mode}, "
              f"pos={pos_count}, neg={neg_count}, n={len(y_all)}")
    
    # Save concept vectors
    np.savez(output_path / "concept_vectors.npz", **concept_vectors)
    
    # Save results
    with open(output_path / "probe_results.json", 'w') as f:
        json.dump(_json_safe(results), f, indent=2)
    
    return results


def _json_safe(obj: Any) -> Any:
    """Convert numpy / pandas scalars to JSON-serializable Python types."""
    if obj is None:
        return None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if np.isnan(val) else val
    if isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    # Fallback for numpy 0-d arrays / pandas scalars
    try:
        if hasattr(obj, "item"):
            return _json_safe(obj.item())
    except Exception:
        pass
    return obj


def compute_move_concepts(
    df: pd.DataFrame,
    output_dir: str,
    concepts: Optional[List[ConceptDefinition]] = None,
    results: Optional[Dict[str, Dict]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Compute per-move concept scores and delta scores.

    Uses feature_mode from concept defs / results when available.
    Prefers calibrated probabilities when calibrator_*.joblib exists.
    Only exports concepts with results[export_ok]=True when results provided.
    """
    output_path = Path(output_dir)

    concept_by_name = {c.name: c for c in (concepts or [])}
    if results is None:
        results_path = output_path / "probe_results.json"
        if results_path.exists():
            with open(results_path, "r") as f:
                results = json.load(f)
        else:
            results = {}

    # Find all available concepts (must have both probe and scaler)
    available_concepts = set()
    for probe_file in output_path.glob("probe_*.joblib"):
        concept_name = probe_file.stem.replace("probe_", "")
        scaler_path = output_path / f"scaler_{concept_name}.joblib"
        if not scaler_path.exists():
            print(f"Warning: Found probe_{concept_name}.joblib but missing scaler, skipping")
            continue
        # Prefer current YAML concept list when provided (drops disabled/stale probes)
        if concept_by_name and concept_name not in concept_by_name:
            print(f"Skipping stale/disabled concept probe: {concept_name}")
            continue
        # Prefer current training results when provided
        if results and concept_name not in results:
            print(f"Skipping probe without current results: {concept_name}")
            continue
        # Quality gate: skip low-quality (use == not `is` — numpy.bool_ breaks identity)
        meta = results.get(concept_name, {}) if results else {}
        if meta and bool(meta.get("export_ok", True)) is False:
            print(f"Skipping low-quality concept for export: {concept_name}")
            continue
        available_concepts.add(concept_name)

    if len(available_concepts) == 0:
        print("Warning: No complete concept models found (need both probe and scaler files)")
        return pd.DataFrame(), []

    concept_scalers = {}
    concept_models = {}
    concept_calibrators = {}
    for concept_name in available_concepts:
        concept_scalers[concept_name] = joblib.load(output_path / f"scaler_{concept_name}.joblib")
        concept_models[concept_name] = joblib.load(output_path / f"probe_{concept_name}.joblib")
        cal_path = output_path / f"calibrator_{concept_name}.joblib"
        if cal_path.exists():
            concept_calibrators[concept_name] = joblib.load(cal_path)

    print(f"Computing scores for {len(available_concepts)} concepts: {sorted(available_concepts)}")

    results_rows = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing concept scores"):
        h = np.array(row["h_features"])
        h_next = np.array(row["h_next_features"]) if row["h_next_features"] is not None else None

        move_result = {
            "game_id": row["game_id"],
            "move_number": row["move_number"],
            "player": row["player"],
            "move_loc": row["move_loc"],
        }

        for concept_name in available_concepts:
            scaler = concept_scalers[concept_name]
            model = concept_models[concept_name]
            calibrator = concept_calibrators.get(concept_name)

            mode = "pre"
            exclude_loc = False
            if concept_name in concept_by_name:
                mode = concept_by_name[concept_name].feature_mode or "pre"
                exclude_loc = concept_by_name[concept_name].exclude_move_location
            elif concept_name in results:
                mode = results[concept_name].get("feature_mode", "pre")
                exclude_loc = results[concept_name].get("exclude_move_location", False)

            def prep(vec: np.ndarray) -> np.ndarray:
                if exclude_loc and vec.shape[0] >= 1024:
                    vec = vec[:512]
                return scaler.transform(vec.reshape(1, -1))

            # Primary score uses feature_mode semantics
            if mode == "post" and h_next is not None:
                primary = prep(h_next)
            elif mode == "delta" and h_next is not None:
                primary = prep(h_next - h)
            else:
                primary = prep(h)

            score_logit = float(model.decision_function(primary)[0])
            move_result[f"{concept_name}_score"] = score_logit

            if calibrator is not None:
                prob = float(calibrator.predict_proba(primary)[0, 1])
            else:
                # Prefer logits for ranking when uncalibrated; still emit prob
                prob = float(model.predict_proba(primary)[0, 1])
            move_result[f"{concept_name}_prob"] = prob

            # Delta: post - pre on probe scores (always compute when h_next exists)
            if h_next is not None:
                pre_x = prep(h)
                post_x = prep(h_next)
                score_pre = float(model.decision_function(pre_x)[0])
                score_post = float(model.decision_function(post_x)[0])
                delta_logit = score_post - score_pre
                move_result[f"{concept_name}_delta_logit"] = delta_logit
                move_result[f"{concept_name}_delta"] = delta_logit

                if calibrator is not None:
                    prob_pre = float(calibrator.predict_proba(pre_x)[0, 1])
                    prob_post = float(calibrator.predict_proba(post_x)[0, 1])
                else:
                    prob_pre = float(model.predict_proba(pre_x)[0, 1])
                    prob_post = float(model.predict_proba(post_x)[0, 1])
                move_result[f"{concept_name}_delta_prob"] = prob_post - prob_pre
            else:
                move_result[f"{concept_name}_delta_logit"] = None
                move_result[f"{concept_name}_delta_prob"] = None
                move_result[f"{concept_name}_delta"] = None

        results_rows.append(move_result)

    results_df = pd.DataFrame(results_rows)

    # Within-game percentile ranks for scores (more comparable across concepts)
    for concept_name in available_concepts:
        score_col = f"{concept_name}_score"
        if score_col in results_df.columns:
            results_df[f"{concept_name}_score_pct"] = results_df.groupby("game_id")[score_col].rank(pct=True)
        delta_col = f"{concept_name}_delta"
        if delta_col in results_df.columns:
            results_df[f"{concept_name}_delta_pct"] = results_df.groupby("game_id")[delta_col].rank(pct=True)

    for game_id in results_df["game_id"].unique():
        game_df = results_df[results_df["game_id"] == game_id]
        game_output = output_path / "move_concepts" / game_id
        game_output.mkdir(parents=True, exist_ok=True)

        def py_serialize(o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            return o

        with open(game_output / "concepts.jsonl", "w") as f:
            for _, r in game_df.iterrows():
                f.write(json.dumps(r.to_dict(), default=py_serialize) + "\n")

    results_df.to_parquet(output_path / "move_concepts.parquet", index=False)

    concept_names = sorted(available_concepts)
    return results_df, concept_names


def generate_html_data(
    move_concepts_df: pd.DataFrame,
    concept_names: List[str],
    results: Dict[str, Dict],
    output_dir: str,
):
    """
    Generate JSON data files for HTML visualization.

    Uses within-game percentile ranks for top concepts instead of hard logit cutoffs.
    Surfaces probe quality so weak concepts are not shown confidently.
    """
    output_path = Path(output_dir)

    concepts_meta = {}
    for name in concept_names:
        if name not in results:
            continue
        res = results[name]
        concepts_meta[name] = _json_safe({
            "auc": res.get("cv_auc"),
            "cv_avg_precision": res.get("cv_avg_precision"),
            "cv_f1": res.get("cv_f1"),
            "train_f1": res.get("train_f1"),
            "train_accuracy": res.get("train_accuracy"),
            "pos_rate": res.get("pos_rate"),
            "n_samples": res.get("n_samples"),
            "export_ok": res.get("export_ok", True),
            "calibrated": res.get("calibrated", False),
            "feature_mode": res.get("feature_mode", "pre"),
            "optimal_threshold": res.get("optimal_threshold"),
        })

    with open(output_path / "concepts_meta.json", "w") as f:
        json.dump(concepts_meta, f, indent=2)

    for game_id in move_concepts_df["game_id"].unique():
        game_df = move_concepts_df[move_concepts_df["game_id"] == game_id].sort_values("move_number")

        game_data = {
            "game_id": game_id,
            "concept_names": concept_names,
            "moves": [],
        }

        for _, row in game_df.iterrows():
            move_data = {
                "move_number": int(row["move_number"]),
                "player": row["player"],
                "move_loc": int(row["move_loc"]),
                "scores": {},
                "deltas": {},
                "score_percentiles": {},
                "delta_percentiles": {},
                "probs": {},
            }

            for concept in concept_names:
                score_col = f"{concept}_score"
                delta_col = f"{concept}_delta"
                prob_col = f"{concept}_prob"
                score_pct_col = f"{concept}_score_pct"
                delta_pct_col = f"{concept}_delta_pct"

                if score_col in game_df.columns:
                    move_data["scores"][concept] = (
                        float(row[score_col]) if pd.notna(row[score_col]) else None
                    )
                if delta_col in game_df.columns:
                    move_data["deltas"][concept] = (
                        float(row[delta_col]) if pd.notna(row[delta_col]) else None
                    )
                if prob_col in game_df.columns:
                    move_data["probs"][concept] = (
                        float(row[prob_col]) if pd.notna(row[prob_col]) else None
                    )
                if score_pct_col in game_df.columns:
                    move_data["score_percentiles"][concept] = (
                        float(row[score_pct_col]) if pd.notna(row[score_pct_col]) else None
                    )
                if delta_pct_col in game_df.columns:
                    move_data["delta_percentiles"][concept] = (
                        float(row[delta_pct_col]) if pd.notna(row[delta_pct_col]) else None
                    )

            # Rank by within-game score percentile (top quintile)
            score_pcts = [
                (c, move_data["score_percentiles"].get(c))
                for c in concept_names
                if move_data["score_percentiles"].get(c) is not None
            ]
            score_pcts.sort(key=lambda x: x[1], reverse=True)
            move_data["top_concepts"] = [
                s[0] for s in score_pcts[:5] if s[1] is not None and s[1] >= 0.8
            ]

            delta_pcts = [
                (c, move_data["delta_percentiles"].get(c), move_data["deltas"].get(c))
                for c in concept_names
                if move_data["delta_percentiles"].get(c) is not None
            ]
            delta_pcts.sort(key=lambda x: x[1], reverse=True)
            move_data["top_delta_concepts"] = [
                {"concept": d[0], "delta": d[2], "percentile": d[1]}
                for d in delta_pcts[:5]
                if d[1] is not None and d[1] >= 0.8
            ]

            game_data["moves"].append(move_data)

        game_output = output_path / "html_data" / game_id
        game_output.mkdir(parents=True, exist_ok=True)

        with open(game_output / "concepts.json", "w") as f:
            json.dump(_json_safe(game_data), f, indent=2)

    print(f"Generated HTML data for {len(move_concepts_df['game_id'].unique())} games")


def main():
    """Main pipeline entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Linear Probe Pipeline for Go Concepts")
    parser.add_argument("--games-dir", type=str, default="../games",
                        help="Path to games directory")
    parser.add_argument("--concepts-yaml", type=str, default="concepts.yaml",
                        help="Path to concepts YAML file")
    parser.add_argument("--output-dir", type=str, default="linear_probes",
                        help="Output directory for models and data")
    parser.add_argument("--dataset-path", type=str, default=None,
                        help="Path to existing dataset parquet (overrides default location)")
    parser.add_argument("--rebuild-dataset", action="store_true",
                        help="Force rebuild dataset even if it exists")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, only compute move concepts")
    parser.add_argument("--html-only", action="store_true",
                        help="Only regenerate HTML from existing move_concepts.parquet")
    parser.add_argument("--integrate-html", action="store_true",
                        help="Inject concepts into games/*/viz.html and export sidecar JSON files")
    parser.add_argument("--labeling", type=str, default="quantile",
                        choices=["fixed", "quantile"],
                        help="Labeling method: fixed (numeric thresholds) or quantile (top/bottom q%%)")

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    games_dir = script_dir / args.games_dir
    concepts_yaml = script_dir / args.concepts_yaml
    output_dir = script_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = args.dataset_path or (output_dir / "dataset.parquet")

    print("Loading concept definitions...")
    concepts, config = load_concepts(concepts_yaml)
    print(f"Loaded {len(concepts)} concept definitions")

    results_path = output_dir / "probe_results.json"
    if results_path.exists():
        with open(results_path, "r") as f:
            results = json.load(f)
    else:
        results = {}

    if args.html_only:
        parquet_path = output_dir / "move_concepts.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Missing {parquet_path}; run scoring first")
        print(f"Loading existing move concepts from {parquet_path}")
        move_concepts_df = pd.read_parquet(parquet_path)
        # Prefer currently enabled + export_ok concepts
        enabled = {c.name for c in concepts}
        concept_names = [
            name for name in sorted(enabled)
            if f"{name}_score" in move_concepts_df.columns
            and bool(results.get(name, {}).get("export_ok", True))
        ]
        # Fallback: any score columns present
        if not concept_names:
            concept_names = sorted({
                c[: -len("_score")]
                for c in move_concepts_df.columns
                if c.endswith("_score")
            })
        print("\nGenerating HTML visualization data...")
        generate_html_data(move_concepts_df, concept_names, results, str(output_dir))
        if args.integrate_html:
            from daniele_experiment.integrate_concepts_html import process_all_games
            process_all_games(str(games_dir), str(output_dir))
        print("\nHTML-only pipeline complete!")
        print(f"  HTML data: {output_dir / 'html_data'}")
        return

    if dataset_path.exists() and not args.rebuild_dataset:
        print(f"Loading existing dataset from {dataset_path}")
        df = pd.read_parquet(dataset_path)
    else:
        print("Building dataset...")
        feature_config = config.get("feature_extraction", {})
        df = build_dataset(
            games_dir=str(games_dir),
            concepts=concepts,
            output_path=str(dataset_path),
            aggregation=feature_config.get("aggregation", "global_pool"),
            pool_type=feature_config.get("pool_type", "mean"),
            include_move_location=feature_config.get("include_move_location", True),
        )

    if not args.skip_training:
        print(f"\nTraining linear probes (labeling={args.labeling})...")
        results = train_probes(df, concepts, config, str(output_dir), labeling=args.labeling)
        print(f"\nTrained {len(results)} concept probes")
    elif not results:
        raise FileNotFoundError(f"Missing {results_path}; train probes first")

    print("\nComputing per-move concept scores...")
    move_concepts_df, concept_names = compute_move_concepts(
        df, str(output_dir), concepts=concepts, results=results
    )

    print("\nGenerating HTML visualization data...")
    generate_html_data(move_concepts_df, concept_names, results, str(output_dir))
    if args.integrate_html:
        from daniele_experiment.integrate_concepts_html import process_all_games
        process_all_games(str(games_dir), str(output_dir))

    print("\nPipeline complete!")
    print(f"  Dataset: {dataset_path}")
    print(f"  Models: {output_dir}")
    print(f"  Move concepts: {output_dir / 'move_concepts.parquet'}")
    print(f"  HTML data: {output_dir / 'html_data'}")


if __name__ == "__main__":
    main()
