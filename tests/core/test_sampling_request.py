"""Independent task masks, frozen identity and cross-type atom-resolution checks."""

from types import SimpleNamespace

import numpy as np

from tmol_core.pack.dunbrack import DunSamplerPBTCache
from tmol_core.pack.sampling_request import (
    apply_frozen_chi,
    backbone_dihedral_indices,
    prepare_dunbrack_sampling_request,
)


def metadata():
    uaids = np.full((2, 2, 4, 3), -1, dtype=np.int32)
    uaids[..., 0] = np.arange(4)
    counts = np.zeros((2, 2, 2), dtype=np.int32)
    counts[:, 1] = 2  # Frozen chi1 is a real slot before sampled chi2.
    samples = np.zeros((2, 2, 2, 2), dtype=np.float32)
    samples[0, 0, :, 0] = 0.3
    samples[1, 0, :, 0] = 0.7
    return DunSamplerPBTCache(
        uaids,
        np.array([[1, 2], [1, 2]], dtype=np.int32),
        counts,
        samples,
        np.array([True, True]),
        np.array([2, -1], dtype=np.int32),
        (("chi1",), ("chi1",)),
    )


def test_task_masks_and_frozen_chi_preserve_chemical_identity():
    cache = metadata()
    types = [SimpleNamespace(torsions=[SimpleNamespace(name="chi1")])] * 2
    request = prepare_dunbrack_sampling_request(
        types,
        cache,
        np.array([[0, 1], [1, -1]]),
        np.array([[0, 4], [0, -1]]),
        np.full((2, 2, 0, 2), -1),
        np.empty((2, 0, 0), dtype=np.int32),
        8,
        np.array([0, 0, 0, 1]),
        np.array([0, 0, 1, 0]),
        np.array([0, 1, 1, 1]),
        np.array([0, 1, 0, 0]),
        np.array([True, True, False, True]),
        np.array([[True, True], [False, False]]),
        np.zeros((2, 2, 2, 2), dtype=np.int32),
    )
    np.testing.assert_array_equal(request.bbt_to_gbt, [0, 1])
    np.testing.assert_array_equal(
        request.bubl_and_rottable_set_for_bbt, [[0, 2], [0, -1]]
    )
    np.testing.assert_array_equal(request.n_chi_for_bbt, [2, 2])
    assert request.frozen_requests == ((0, 0, 0, "chi1"),)
    np.testing.assert_array_equal(request.frozen_destinations, [[0, 0]])
    result = apply_frozen_chi(request, [np.nan])
    np.testing.assert_array_equal(
        result[:, 0, 0], np.array([0.3, 0.7], dtype=np.float32)
    )
    result = apply_frozen_chi(request, [0.9])
    np.testing.assert_array_equal(
        result[:, 0, 0], np.array([0.9, 0.7], dtype=np.float32)
    )
    # A request owns its mutable sample values; database metadata stays unchanged.
    assert cache.non_dunbrack_samples[0, 0, 1, 0] == np.float32(0.3)
    assert request.dihedral_atom_inds.dtype == np.int32
    assert result.dtype == np.float32


def test_backbone_uses_destination_type_and_marks_missing_atoms():
    cache = metadata()
    cache.bbdihe_uaids[0, 0, 0] = [-1, 0, 0]
    connections = np.array([[[[1, 0]], [[0, 0]]]], dtype=np.int32)
    downstream = np.array([[[0]], [[3]]], dtype=np.int32)
    args = (cache, np.array([[0, 1]]), np.array([[0, 4]]), connections, downstream, 8)
    indices = backbone_dihedral_indices(*args)
    assert indices[0, 0, 0, 0] == 7
    downstream[1, 0, 0] = -1
    assert backbone_dihedral_indices(*args)[0, 0, 0, 0] == -1
    connections[0, 0, 0] = [-1, -1]
    assert backbone_dihedral_indices(*args)[0, 0, 0, 0] == -1
