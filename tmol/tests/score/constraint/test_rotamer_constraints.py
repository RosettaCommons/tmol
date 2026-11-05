"""A single attached constraint must survive expansion into rotamer pairs."""

import pytest
import torch

from tmol import ConstraintSet
from tmol.score import ConstraintEnergyTerm


@pytest.mark.parametrize("two_body", [False, True])
def test_first_constraint_contributes_to_rotamer_values_and_gradients(
    default_database, two_body
):
    device = torch.device("cpu")
    term = ConstraintEnergyTerm(default_database, device)

    def integer(values):
        return torch.tensor(values, dtype=torch.int32, device=device)

    coords = torch.tensor(
        [
            [0, 0, 0],
            [2, 0, 0],
            [1, 0, 0],
            [4, 0, 0],
            [4, 0, 0],
            [4, 1, 0],
            [6, 0, 0],
            [6, 1, 0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    atoms = integer([[[0, 0, 0], [0, 1, 0] if two_body else [0, 0, 1]]])
    constraints = ConstraintSet.create_empty(device=device, n_poses=1).add_constraints(
        ConstraintEnergyTerm.harmonic, atoms, torch.tensor([[1.25, 0.5]])
    )
    scores, indices = term.constraint_pose_scores(
        coords,
        integer([0, 2, 4, 6]),
        integer([0] * 8),
        integer([[0, 2]]),
        integer([[0, 0]]),
        integer([0, 0, 1, 1]),
        integer([0] * 4),
        integer([0] * 4),
        integer([4]),
        integer([0]),
        integer([[2, 2]]),
        integer([[0, 2]]),
        4,
        constraints,
        True,
    )
    # Independent harmonic model, with all combinations of the two rotamers.
    pairs = [(0, 4), (2, 4), (0, 6), (2, 6)] if two_body else [(0, 1), (2, 3)]
    expected_values = []
    expected_gradient = torch.zeros_like(coords)
    for first, second in pairs:
        distance = float(coords[second, 0].detach() - coords[first, 0].detach())
        expected_values.append(((distance - 1.25) / 0.5) ** 2)
        slope = 2 * (distance - 1.25) / 0.5**2
        expected_gradient[first, 0] -= slope
        expected_gradient[second, 0] += slope
    assert indices.shape == (3, len(pairs))
    torch.testing.assert_close(scores.flatten(), torch.tensor(expected_values))
    gradient = torch.autograd.grad(scores.sum(), coords)[0]
    torch.testing.assert_close(gradient, expected_gradient)
