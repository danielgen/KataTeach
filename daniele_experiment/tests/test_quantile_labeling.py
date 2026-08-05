#!/usr/bin/env python3
"""
Unit test for quantile threshold labeling.

Tests that with q=0.1 on values 0..99:
- direction='high': positives are {90..99}, negatives are {0..9}, middle unlabeled
- direction='low': positives are {0..9}, negatives are {90..99}, middle unlabeled
"""

import numpy as np
import pandas as pd
from linear_probe_pipeline import ConceptDefinition, compute_quantile_thresholds, extract_label

def test_quantile_labeling():
    """Test quantile labeling logic."""
    # Create synthetic data: values 0..99
    n_samples = 100
    df = pd.DataFrame({
        'raw_test_value': np.arange(n_samples),
        'game_id': ['game1'] * n_samples,
    })
    
    # Create concept with direction='high'
    concept_high = ConceptDefinition(
        name='test_high',
        type='quantile',
        source='test_value',
        description='Test high quantile',
        q=0.1,
        direction='high',
        use_abs=False,
    )
    
    # Create concept with direction='low'
    concept_low = ConceptDefinition(
        name='test_low',
        type='quantile',
        source='test_value',
        description='Test low quantile',
        q=0.1,
        direction='low',
        use_abs=False,
    )
    
    # Compute thresholds on all data (simulating train fold)
    train_indices = np.arange(n_samples)
    thresholds_high = compute_quantile_thresholds(df, concept_high, train_indices)
    thresholds_low = compute_quantile_thresholds(df, concept_low, train_indices)
    
    assert thresholds_high is not None, "Should compute thresholds for high direction"
    assert thresholds_low is not None, "Should compute thresholds for low direction"
    
    low_thr_high, high_thr_high = thresholds_high
    low_thr_low, high_thr_low = thresholds_low
    
    # With q=0.1, low_thr should be around 9, high_thr around 90
    assert abs(low_thr_high - 9.0) < 1.0, f"low_thr should be ~9, got {low_thr_high}"
    assert abs(high_thr_high - 90.0) < 1.0, f"high_thr should be ~90, got {high_thr_high}"
    
    # Test labeling for direction='high'
    quantile_thresholds_high = {'test_high': thresholds_high}
    positives_high = []
    negatives_high = []
    unlabeled_high = []
    
    for val in range(n_samples):
        analysis = {'test_value': float(val)}
        label = extract_label(analysis, concept_high, quantile_thresholds_high)
        if label == 1:
            positives_high.append(val)
        elif label == 0:
            negatives_high.append(val)
        else:
            unlabeled_high.append(val)
    
    # For direction='high': positives should be top 10% (90-99), negatives bottom 10% (0-9)
    assert set(positives_high) == set(range(90, 100)), \
        f"High direction positives should be 90-99, got {positives_high}"
    assert set(negatives_high) == set(range(10)), \
        f"High direction negatives should be 0-9, got {negatives_high}"
    assert set(unlabeled_high) == set(range(10, 90)), \
        f"High direction unlabeled should be 10-89, got {unlabeled_high}"
    
    # Test labeling for direction='low'
    quantile_thresholds_low = {'test_low': thresholds_low}
    positives_low = []
    negatives_low = []
    unlabeled_low = []
    
    for val in range(n_samples):
        analysis = {'test_value': float(val)}
        label = extract_label(analysis, concept_low, quantile_thresholds_low)
        if label == 1:
            positives_low.append(val)
        elif label == 0:
            negatives_low.append(val)
        else:
            unlabeled_low.append(val)
    
    # For direction='low': positives should be bottom 10% (0-9), negatives top 10% (90-99)
    assert set(positives_low) == set(range(10)), \
        f"Low direction positives should be 0-9, got {positives_low}"
    assert set(negatives_low) == set(range(90, 100)), \
        f"Low direction negatives should be 90-99, got {negatives_low}"
    assert set(unlabeled_low) == set(range(10, 90)), \
        f"Low direction unlabeled should be 10-89, got {unlabeled_low}"
    
    # Check balance: roughly 10% positive, 10% negative, 80% unlabeled
    assert len(positives_high) == 10, f"Should have 10 positives, got {len(positives_high)}"
    assert len(negatives_high) == 10, f"Should have 10 negatives, got {len(negatives_high)}"
    assert len(unlabeled_high) == 80, f"Should have 80 unlabeled, got {len(unlabeled_high)}"
    
    print("✓ All quantile labeling tests passed!")
    print(f"  High direction: {len(positives_high)} positives, {len(negatives_high)} negatives, {len(unlabeled_high)} unlabeled")
    print(f"  Low direction: {len(positives_low)} positives, {len(negatives_low)} negatives, {len(unlabeled_low)} unlabeled")


if __name__ == "__main__":
    test_quantile_labeling()

