import numpy as np
import pytest

torch = pytest.importorskip("torch")

from daniele_experiment.tenuki_gradient_analysis import (
    direction_projected_gradient,
    far_mass_and_gradient,
    flat_tenuki_masks,
    mask_projected_gradient,
    one_sided_positive_p,
    stratified_mean,
)


class _StubPolicyHead:
    """Minimal callable with KataGo's policy-head signature and output shape."""

    def __init__(self, channels: int, board_size: int, seed: int = 0, dtype=torch.float64):
        rng = np.random.default_rng(seed)
        actions = board_size * board_size + 1
        weight = rng.normal(size=(actions, channels * board_size * board_size)) * 0.05
        self.weight = torch.tensor(weight, dtype=dtype)
        self.board_size = board_size

    def __call__(self, x, mask=None, mask_sum_hw=None, mask_sum=None, extra_outputs=None):
        assert mask is not None and mask_sum_hw is not None and mask_sum is not None
        flat = x.reshape(x.shape[0], -1)
        logits = flat @ self.weight.T
        return logits.unsqueeze(1)


def _random_setup(seed: int = 7, channels: int = 6, size: int = 5):
    rng = np.random.default_rng(seed)
    trunk = rng.normal(size=(channels, size, size))
    board_mask = np.ones((size, size), dtype=np.float32)
    legal = rng.random(size * size) > 0.2
    legal[0] = True
    far = legal & (rng.random(size * size) > 0.5)
    if not far.any():
        far[np.flatnonzero(legal)[0]] = True
    head = _StubPolicyHead(channels, size, seed=seed)
    return head, trunk, board_mask, legal, far


def test_far_mass_matches_numpy_softmax():
    head, trunk, board_mask, legal, far = _random_setup()
    far_mass, gradient = far_mass_and_gradient(
        head, trunk, board_mask, legal, far, dtype=torch.float64
    )
    logits = trunk.reshape(-1) @ np.asarray(head.weight, dtype=np.float64).T
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    board_probabilities = probabilities[:-1]
    expected = board_probabilities[far].sum() / board_probabilities[legal].sum()
    assert far_mass == pytest.approx(expected, abs=1e-12)
    assert gradient.shape == trunk.shape


def test_gradient_matches_central_finite_difference_on_dose():
    head, trunk, board_mask, legal, far = _random_setup(seed=11)
    rng = np.random.default_rng(3)
    delta = rng.normal(size=trunk.shape[0])
    spatial = rng.normal(size=trunk.shape[1:])

    _, gradient = far_mass_and_gradient(
        head, trunk, board_mask, legal, far, dtype=torch.float64
    )
    analytic = float(mask_projected_gradient(gradient, spatial) @ delta)

    step = 1e-6
    perturbation = spatial[None, :, :] * delta[:, None, None]

    def far_mass_at(dose: float) -> float:
        value, _ = far_mass_and_gradient(
            head, trunk + dose * perturbation, board_mask, legal, far, dtype=torch.float64
        )
        return value

    central = (far_mass_at(step) - far_mass_at(-step)) / (2 * step)
    assert analytic == pytest.approx(central, rel=1e-5, abs=1e-10)


def test_projection_identities_agree_with_full_contraction():
    rng = np.random.default_rng(19)
    gradient = rng.normal(size=(4, 3, 3))
    spatial = rng.normal(size=(3, 3))
    delta = rng.normal(size=4)
    full = float(np.einsum("chw,hw,c->", gradient, spatial, delta))
    by_mask = float(mask_projected_gradient(gradient, spatial) @ delta)
    by_direction = float(np.sum(direction_projected_gradient(gradient, delta) * spatial))
    assert by_mask == pytest.approx(full, rel=1e-12)
    assert by_direction == pytest.approx(full, rel=1e-12)


def test_far_mask_must_be_subset_of_legal():
    head, trunk, board_mask, legal, far = _random_setup()
    bad_far = far.copy()
    bad_far[np.flatnonzero(~legal)[0]] = True
    with pytest.raises(ValueError, match="subset"):
        far_mass_and_gradient(head, trunk, board_mask, legal, bad_far)


def test_flat_masks_match_policy_tensor_index_convention():
    from board import Board
    from daniele_experiment.operational_definitions import (
        manhattan_distance,
        tenuki_action_predicate,
    )

    board = Board(19)
    previous_move = board.loc(3, 3)
    board.play(Board.BLACK, previous_move)
    board.play(Board.WHITE, board.loc(15, 15))

    legal_flat, far_flat = flat_tenuki_masks(board, previous_move)
    assert legal_flat.shape == (361,) and far_flat.shape == (361,)
    for tensor_index in range(361):
        x = tensor_index % 19
        y = tensor_index // 19
        loc = board.loc(x, y)
        expected_legal = bool(board.would_be_legal(board.pla, loc))
        assert bool(legal_flat[tensor_index]) == expected_legal
        expected_far = expected_legal and tenuki_action_predicate(
            board, previous_move, loc
        )
        assert bool(far_flat[tensor_index]) == expected_far
    assert manhattan_distance(board, previous_move, board.loc(9, 3)) == 6
    assert far_flat[3 * 19 + 9]


def test_stratified_mean_weights_label_groups_equally():
    values = [1.0, 1.0, 1.0, 5.0]
    labels = [0, 0, 0, 1]
    assert stratified_mean(values, labels) == pytest.approx(3.0)
    assert stratified_mean([2.0, 4.0], [1, 1]) == pytest.approx(3.0)


def test_one_sided_positive_p_matches_report_convention():
    # 70 of 100 controls at least as extreme in the positive direction -> 0.703.
    trained = -0.5
    controls = [1.0] * 70 + [-1.0] * 30
    assert one_sided_positive_p(trained, controls) == pytest.approx(71 / 101)
