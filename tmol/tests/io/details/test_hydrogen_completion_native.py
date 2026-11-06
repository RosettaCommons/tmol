"""Connected heavy-neighbor geometry, padding, and native coordinate derivatives."""

import pytest
import torch

from tmol.io.details._hydrogen_completion import complete_hydrogens


def inputs(real=torch.float64, integer=torch.int32):
    coordinates = torch.tensor(
        [[[0, 0, 0], [1, 0, 0], [0, 1, 0], [4, 5, 6], [0, 0, 1], [float("nan"), 0, 0]]],
        dtype=real,
    )
    types = torch.tensor([[0, 1, -1]], dtype=integer)
    offsets = torch.tensor([[0, 4, 5]], dtype=integer)
    connections = torch.tensor([[[[1, 0]], [[0, 0]], [[-1, -1]]]], dtype=integer)
    connection_atoms = torch.zeros((2, 1), dtype=integer)
    rebuilt = torch.zeros((1, 3, 4), dtype=torch.bool)
    rebuilt[0, 0, 3] = True
    eligible = torch.zeros((2, 4), dtype=torch.bool)
    eligible[0, 3] = True
    parent = torch.full((2, 4), -1, dtype=torch.int64)
    parent[0, 3] = 0
    neighbors = torch.full((2, 4, 3), -1, dtype=torch.int64)
    neighbors[0, 3, :2] = torch.tensor([1, 2])
    counts = torch.zeros((2, 4), dtype=torch.int64)
    counts[0, 3] = 2
    distances = torch.zeros((2, 4), dtype=real)
    distances[0, 3] = 0.987654321
    ports = torch.full((2, 4, 3), -1, dtype=torch.int64)
    ports[0, 3, 0] = 0
    port_counts = torch.zeros((2, 4), dtype=torch.int64)
    port_counts[0, 3] = 1
    return [
        coordinates,
        types,
        offsets,
        connections,
        connection_atoms,
        rebuilt,
        eligible,
        parent,
        neighbors,
        counts,
        distances,
        ports,
        port_counts,
    ]


@pytest.mark.parametrize("real", [torch.float32, torch.float64])
@pytest.mark.parametrize("integer", [torch.int32, torch.int64])
def test_connected_hydrogen_bond_and_translation(real, integer):
    data = inputs(real, integer)
    result = complete_hydrogens(*data)
    expected = -data[10][0, 3] * torch.ones(3, dtype=real) / (3**0.5)
    torch.testing.assert_close(result[0, 3], expected)
    observed = torch.tensor([0, 1, 2, 4])
    torch.testing.assert_close(
        result[:, observed], data[0][:, observed], rtol=0, atol=0
    )
    assert torch.isnan(result[0, 5, 0])
    shift = torch.tensor([0.25, -0.5, 0.75], dtype=real)
    moved = complete_hydrogens(data[0] + shift, *data[1:])
    torch.testing.assert_close(moved[:, :5], result[:, :5] + shift)


def test_connected_hydrogen_coordinate_and_distance_gradients_and_hvp():
    data = inputs()
    coordinates, distances = data[0].requires_grad_(), data[10].requires_grad_()

    def build(x, distance):
        return complete_hydrogens(x, *data[1:10], distance, *data[11:])[:, :5]

    assert torch.autograd.gradcheck(build, (coordinates, distances), fast_mode=True)
    assert torch.autograd.gradgradcheck(build, (coordinates, distances), fast_mode=True)
