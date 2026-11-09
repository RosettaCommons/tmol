"""A single attached constraint must survive expansion into rotamer pairs."""

import pytest
import torch

from tmol import ConstraintSet
from tmol.score import ConstraintEnergyTerm


@pytest.mark.parametrize("two_body", [False, True])
@pytest.mark.parametrize("n_conformers", [1, 2])
def test_first_constraint_contributes_to_rotamer_values_and_gradients(
    default_database, two_body, n_conformers
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
    if n_conformers == 1:
        coords = coords.detach()[[0, 1, 4, 5]].requires_grad_()
    n = n_conformers
    atoms = integer([[[0, 0, 0], [0, 1, 0] if two_body else [0, 0, 1]]])
    constraints = ConstraintSet.create_empty(device=device, n_poses=1).add_constraints(
        ConstraintEnergyTerm.harmonic, atoms, torch.tensor([[1.25, 0.5]])
    )
    scores, indices = term.constraint_rotamer_scores(
        coords,
        integer(list(range(0, 4 * n, 2))),
        integer([0] * (4 * n)),
        integer([[0, n]]),
        integer([[0, 0]]),
        integer([0] * n + [1] * n),
        integer([0] * (2 * n)),
        integer([0] * (2 * n)),
        integer([2 * n]),
        integer([0]),
        integer([[n, n]]),
        integer([[0, n]]),
        integer([[-1, -1]]),
        2 * n,
        constraints,
        True,
    )
    # Independent harmonic model, with all combinations of the two rotamers.
    pairs = [(0, 4), (2, 4), (0, 6), (2, 6)] if two_body else [(0, 1), (2, 3)]
    if n == 1:
        pairs = [(0, 2)] if two_body else [(0, 1)]
    assert scores.shape == (1, len(pairs))
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


def test_interleaved_blocks_keep_four_atom_constraint(default_database):
    # A/B/A/B ordering is physical atom order, not four distinct blocks.
    device = torch.device("cpu")
    term = ConstraintEnergyTerm(default_database, device)

    def integer(values):
        return torch.tensor(values, dtype=torch.int32)

    coords = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [3.0, 0.0, 0.0], [3.0, 2.0, 1.0]],
        requires_grad=True,
    )

    def four_atom_potential(atoms, params):
        return ((atoms[:, 0] - atoms[:, 1]) ** 2).sum(-1) + (
            (atoms[:, 2] - atoms[:, 3]) ** 2
        ).sum(-1)

    constraints = ConstraintSet.create_empty(device=device, n_poses=1).add_constraints(
        four_atom_potential,
        integer([[[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]]]),
        torch.zeros((1, 1)),
    )
    scores, indices = term.constraint_rotamer_scores(
        coords,
        integer([0, 2]),
        integer([0] * 4),
        integer([[0, 1]]),
        integer([[0, 0]]),
        integer([0, 1]),
        integer([0, 0]),
        integer([0, 0]),
        integer([2]),
        integer([0]),
        integer([[1, 1]]),
        integer([[0, 1]]),
        integer([[-1, -1]]),
        2,
        constraints,
        True,
    )
    assert scores.shape == (1, 1)
    torch.testing.assert_close(scores, torch.tensor([[15.0]]))
    torch.testing.assert_close(
        indices, torch.tensor([[0], [0], [1]], dtype=indices.dtype)
    )
    torch.testing.assert_close(
        torch.autograd.grad(scores.sum(), coords)[0],
        torch.tensor(
            [[-6.0, 0.0, 0.0], [-4.0, -2.0, -2.0], [6.0, 0.0, 0.0], [4.0, 2.0, 2.0]]
        ),
    )


@pytest.mark.parametrize("attached_empty", [False, True])
def test_empty_constraint_rotamers_keep_sparse_rank(default_database, attached_empty):
    device = torch.device("cpu")
    term = ConstraintEnergyTerm(default_database, device)
    empty = torch.zeros(0, dtype=torch.int32)
    blocks = torch.zeros((0, 0), dtype=torch.int32)
    constraints = (
        ConstraintSet.create_empty(device=device, n_poses=0) if attached_empty else None
    )
    scores, indices = term.constraint_rotamer_scores(
        torch.zeros((0, 3)),
        empty,
        empty,
        blocks,
        blocks,
        empty,
        empty,
        empty,
        empty,
        empty,
        blocks,
        blocks,
        blocks,
        0,
        constraints,
        True,
    )
    assert scores.shape == (1, 0)
    assert indices.shape == (3, 0)
