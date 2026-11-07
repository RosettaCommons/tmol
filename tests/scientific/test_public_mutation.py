"""Public design preserves frozen molecular identities and parameter context."""

import importlib

import numpy as np
import pytest

from .molecules import BACKBONE_CASES


def test_public_mutation_preserves_mixed_chemistry_and_frozen_atoms(backend):
    b = backend.molecule(BACKBONE_CASES["mixed"], terms=())
    b.use_beta2016()
    api = importlib.import_module("tmol_jax" if b.framework == "jax" else "tmol")
    pose = b.pose
    initial = b.coords().copy()
    result = api.mutate_residue(pose, 0, 2, "ALA", b.score_function, seed=4)
    assert result.block_type(0, 2).name3 == "ALA"
    assert result.packed_block_types.chem_db is pose.packed_block_types.chem_db
    np.testing.assert_array_equal(b.coords(), initial)
    old_offsets, new_offsets = map(
        b.numpy, (pose.block_coord_offset, result.block_coord_offset)
    )
    for p, blocks in ((0, [0, 1, 3]), (1, [0, 1, 2])):
        for r in blocks:
            bt = pose.block_type(p, r)
            assert result.block_type(p, r).name == bt.name
            a, c = int(old_offsets[p, r]), int(new_offsets[p, r])
            np.testing.assert_array_equal(
                b.numpy(result.coords)[p, c : c + bt.n_atoms],
                initial[p, a : a + bt.n_atoms],
            )
    b.pose = result
    if b.framework == "torch":
        b._render_torch()
    values, gradient = b.channel_values_and_gradients(b.coords())
    assert np.isfinite(values).all() and np.isfinite(gradient).all()
    assert np.linalg.norm(gradient) > 1
    with pytest.raises(ValueError, match="valid one-letter"):
        api.mutate_to_sequence(pose, 0, "AXVX", b.score_function)
    with pytest.raises(ValueError, match="4 real blocks"):
        api.mutate_to_sequence(pose, 0, "AV", b.score_function)
