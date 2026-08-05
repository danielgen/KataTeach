import numpy as np
import pytest

from daniele_experiment.linear_probe_pipeline import aggregate_features


def test_local_features_use_explicit_idx361_not_padded_board_location():
    activation = np.arange(2 * 19 * 19, dtype=np.float32).reshape(2, 19, 19)
    idx361 = 2 * 19 + 3

    features = aggregate_features(
        activation, idx361, "global_pool", "mean", include_move_location=True
    )

    np.testing.assert_array_equal(features[-2:], activation[:, 2, 3])
    # KataGo's padded loc for (3, 2) is 4 + 20*3 = 64. Treating 64 as
    # idx361 would sample (3, 7), demonstrating why the types cannot mix.
    assert not np.array_equal(features[-2:], activation[:, 3, 7])


def test_idx361_pass_is_zero_and_invalid_indices_fail_closed():
    activation = np.ones((3, 19, 19), dtype=np.float32)
    passed = aggregate_features(
        activation, 361, "global_pool", "mean", include_move_location=True
    )
    np.testing.assert_array_equal(passed[-3:], np.zeros(3, dtype=np.float32))

    with pytest.raises(ValueError, match="idx361"):
        aggregate_features(
            activation, 399, "global_pool", "mean", include_move_location=True
        )
