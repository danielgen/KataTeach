import numpy as np
import pytest

torch = pytest.importorskip("torch")

from daniele_experiment.tenuki_gradient_analysis import (
    direction_projected_gradient,
    far_mass_and_gradient,
    mask_projected_gradient,
)
from daniele_experiment.tenuki_single_site_analysis import (
    ratio_readout_and_gradient,
    select_candidate_sites,
    single_site_flat_mask,
)
from daniele_experiment.test_tenuki_gradient_analysis import _StubPolicyHead, _random_setup


def test_ratio_readout_generalises_far_mass():
    head, trunk, board_mask, legal, far = _random_setup(seed=23)
    far_value, far_gradient = far_mass_and_gradient(
        head, trunk, board_mask, legal, far, dtype=torch.float64
    )
    ratio_value, ratio_gradient = ratio_readout_and_gradient(
        head, trunk, board_mask, far, legal, dtype=torch.float64
    )
    assert ratio_value == pytest.approx(far_value, abs=1e-14)
    np.testing.assert_allclose(ratio_gradient, far_gradient, rtol=1e-12)


def test_single_site_readout_is_conditional_probability():
    head, trunk, board_mask, legal, _ = _random_setup(seed=29)
    site = int(np.flatnonzero(legal)[1])
    value, gradient = ratio_readout_and_gradient(
        head,
        trunk,
        board_mask,
        single_site_flat_mask(legal.size, site),
        legal,
        dtype=torch.float64,
    )
    logits = trunk.reshape(-1) @ np.asarray(head.weight, dtype=np.float64).T
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    board_probabilities = probabilities[:-1]
    expected = board_probabilities[site] / board_probabilities[legal].sum()
    assert value == pytest.approx(expected, abs=1e-14)
    assert gradient.shape == trunk.shape


def test_single_site_self_effect_matches_finite_difference():
    head, trunk, board_mask, legal, _ = _random_setup(seed=31)
    site = int(np.flatnonzero(legal)[0])
    rng = np.random.default_rng(5)
    delta = rng.normal(size=trunk.shape[0])
    numerator = single_site_flat_mask(legal.size, site)

    _, gradient = ratio_readout_and_gradient(
        head, trunk, board_mask, numerator, legal, dtype=torch.float64
    )
    analytic = float(gradient.reshape(trunk.shape[0], -1)[:, site] @ delta)

    size = trunk.shape[1]
    y, x = divmod(site, size)
    step = 1e-6
    perturbed = np.zeros_like(trunk)
    perturbed[:, y, x] = delta

    def value_at(dose: float) -> float:
        value, _ = ratio_readout_and_gradient(
            head, trunk + dose * perturbed, board_mask, numerator, legal, dtype=torch.float64
        )
        return value

    central = (value_at(step) - value_at(-step)) / (2 * step)
    assert analytic == pytest.approx(central, rel=1e-5, abs=1e-10)


def test_broadcast_decomposition_sums_to_full_derivative():
    head, trunk, board_mask, legal, far = _random_setup(seed=37)
    rng = np.random.default_rng(9)
    delta = rng.normal(size=trunk.shape[0])
    spatial = rng.normal(size=trunk.shape[1:]) * legal.reshape(trunk.shape[1:])

    _, gradient = far_mass_and_gradient(
        head, trunk, board_mask, legal, far, dtype=torch.float64
    )
    full = float(mask_projected_gradient(gradient, spatial) @ delta)
    coupling_map = direction_projected_gradient(gradient, delta).reshape(-1)
    contributions = coupling_map * spatial.reshape(-1)
    far_part = float(contributions[far].sum())
    near_part = float(contributions[legal & ~far].sum())
    assert far_part + near_part == pytest.approx(full, rel=1e-10)


def test_select_candidate_sites_orders_by_probability_within_region():
    probabilities = np.array([0.05, 0.30, 0.01, 0.20, 0.10, 0.34])
    legal = np.array([True, True, False, True, True, True])
    far = np.array([False, True, False, True, False, False])
    candidates = select_candidate_sites(probabilities, legal, far, per_set=2)
    assert candidates["far"] == [1, 3]
    assert candidates["near"] == [5, 4]


def test_single_site_flat_mask_bounds():
    mask = single_site_flat_mask(361, 42)
    assert mask.sum() == 1 and mask[42]
    with pytest.raises(ValueError):
        single_site_flat_mask(361, 361)
