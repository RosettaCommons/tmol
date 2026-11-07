"""Explicit palette construction retains custom chemistry, ordering and forces."""

from contextlib import nullcontext
from importlib import import_module

import attrs
import numpy as np
import pytest

from .molecules import DOMAIN_CASES


@pytest.mark.parametrize(
    "name", ("ligand", "connected_noncanonical", "d_peptide", "mixed_batch")
)
def test_public_palette_context_lookup_and_cpu_transport(backend, name):
    b = backend.molecule(DOMAIN_CASES[name], terms=())
    front = import_module("tmol_jax" if b.framework == "jax" else "tmol")
    chemistry = import_module(front.__name__ + ".chemical")
    original = b.pose.packed_block_types
    collection = chemistry.ResidueTypeSet.from_database(b.parameter_database.chemical)
    active = tuple(original.active_block_types)
    palette = front.PackedBlockTypes.from_restype_list(
        original.chem_db, collection, active, b.device
    )
    assert palette.chem_db is original.chem_db
    assert palette.restype_set is collection
    assert all(a is c for a, c in zip(palette.active_block_types, active))
    assert palette.device == b.device
    np.testing.assert_array_equal(
        palette.inds_for_restypes(active[::-1]), np.arange(len(active) - 1, -1, -1)
    )
    absent = next(
        t for t in collection.residue_types if t.name not in palette.restype_index
    )
    np.testing.assert_array_equal(
        palette.inds_for_restypes([absent, active[0], active[0]]), [-1, 0, 0]
    )
    for field in ("n_atoms", "atom_is_real", "atom_is_hydrogen", "n_bonds", "n_conn"):
        np.testing.assert_array_equal(
            b.numpy(getattr(palette, field)), b.numpy(getattr(original, field))
        )
    copied = palette.cpu()
    assert copied.chem_db is palette.chem_db and copied.restype_set is collection
    assert (
        copied.device == b.module.devices("cpu")[0]
        if b.framework == "jax"
        else copied.device.type == "cpu"
    )
    for field in ("n_atoms", "atom_is_real", "bond_indices", "n_conn"):
        np.testing.assert_array_equal(
            b.numpy(getattr(copied, field)), b.numpy(getattr(palette, field))
        )
    b.use_beta2016()
    precision = b.module.enable_x64() if b.framework == "jax" else nullcontext()
    with precision:
        coords = b.coords().astype(np.float64)
        before = b.channel_values_and_gradients(coords)
        b.pose = (
            b.pose.replace(packed_block_types=palette)
            if b.framework == "jax"
            else attrs.evolve(b.pose, packed_block_types=palette)
        )
        if b.framework == "torch":
            b._render_torch()
        after = b.channel_values_and_gradients(coords)
    for expected, actual in zip(before, after):
        assert np.isfinite(actual).all()
        # Native CUDA atomic reductions need not repeat bitwise even with the
        # identical palette. Use the existing float64 score-repeat tolerance;
        # chemistry, ordering and topology above still compare exactly.
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert np.linalg.norm(after[1]) > 1
    with pytest.raises(ValueError, match="empty residue-type palette"):
        front.PackedBlockTypes.from_restype_list(
            original.chem_db, collection, [], b.device
        )
