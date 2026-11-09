"""Framework-neutral NA preparation invariants."""

import numpy as np
import pytest

from tmol_core.scoring.na_torsion_params import _circular_mean
from tmol_core.scoring.na_torsion_topology import prepare_na_torsion_topology


def test_circular_mean_crosses_angle_boundary():
    angles = np.array([[179, 359], [-179, 1]], dtype=np.float32)
    mean = _circular_mean(angles)
    error = (mean - np.array([180, 0]) + 180) % 360 - 180
    np.testing.assert_allclose(error, 0, atol=2e-5)
    assert mean.dtype == np.float32


@pytest.mark.parametrize("padding_offset", [-1, 8])
def test_connected_topology_preserves_batch_padding_and_missing_neighbors(
    padding_offset,
):
    # Two residues in pose zero, a disconnected residue plus padding in pose one.
    types = np.array([[0, 0], [0, -1]], dtype=np.int32)
    offsets = np.array([[0, 4], [0, padding_offset]], dtype=np.int32)
    connections = np.full((2, 2, 1, 2), -1, dtype=np.int32)
    connections[0, 0, 0] = [1, 0]
    connections[0, 1, 0] = [0, 0]
    uaids = np.array([[[[0, -1, -1], [1, -1, -1], [-1, 0, 0], [-1, 0, 1]]]])
    result = prepare_na_torsion_topology(
        types,
        np.array([0]),
        uaids,
        np.array([[0, 1, 2, 3, -1]]),
        np.array([0]),
        np.array([[[2, 3]]]),
        offsets,
        connections,
        8,
    )
    base, mask, torsions, valid, ring, ring_valid, previous = result
    np.testing.assert_array_equal(base, [[0, 0], [0, -1]])
    np.testing.assert_array_equal(mask, [[True, True], [True, False]])
    np.testing.assert_array_equal(torsions[0, :, 0], [[0, 1, 6, 7], [4, 5, 2, 3]])
    np.testing.assert_array_equal(valid[..., 0], [[True, True], [False, False]])
    np.testing.assert_array_equal(ring[1, 0, :4], [8, 9, 10, 11])
    assert not ring_valid[..., -1].any()
    assert not ring_valid[1, 1].any()
    np.testing.assert_array_equal(previous, [[1, 0], [-1, -1]])
    assert torsions.dtype == ring.dtype == previous.dtype == np.int64
    # Padded blocks may store the end offset. All gather placeholders must be
    # safe before a frontend applies the scientific masks.
    assert np.all((torsions >= 0) & (torsions < 16))
    assert np.all((ring >= 0) & (ring < 16))


@pytest.mark.parametrize("shape", [(0, 2), (2, 0)])
def test_empty_topology_retains_declared_batch_dimensions(shape):
    result = prepare_na_torsion_topology(
        np.empty(shape, dtype=np.int32),
        np.array([0]),
        np.full((1, 10, 4, 3), -1, dtype=np.int32),
        np.full((1, 5), -1, dtype=np.int32),
        np.array([-1]),
        np.full((1, 1, 1), -1, dtype=np.int32),
        np.empty(shape, dtype=np.int32),
        np.full((*shape, 1, 2), -1, dtype=np.int32),
        0,
    )
    assert [x.shape for x in result] == [
        shape,
        shape,
        (*shape, 10, 4),
        (*shape, 10),
        (*shape, 5),
        (*shape, 5),
        shape,
    ]
