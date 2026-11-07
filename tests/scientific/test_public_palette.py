"""Explicit palette construction retains custom chemistry, ordering and forces."""

from contextlib import nullcontext
from importlib import import_module

import attrs
import numpy as np
import pytest

from .molecules import DOMAIN_CASES


@pytest.mark.parametrize(
    "name", ("ligand", "connected_noncanonical", "d_peptide", "dna", "metal_mixed")
)
def test_public_palette_topology_retains_physical_atoms_and_torsions(backend, name):
    b = backend.molecule(DOMAIN_CASES[name], terms=())
    p = b.pose.packed_block_types
    cls, types = type(p), p.active_block_types
    assert cls.count_max_n_atoms(types) == max(len(t.atoms) for t in types)
    counts = cls.count_n_atoms(types, b.device)
    assert counts.device == b.device
    np.testing.assert_array_equal(b.numpy(counts), [len(t.atoms) for t in types])
    real = cls.determine_real_atoms(p.max_n_atoms, counts, b.device)
    hydrogen = cls.determine_h_atoms(p.chem_db, p.max_n_atoms, counts, types, b.device)
    element = {a.name: a.element for a in p.chem_db.atom_types}
    for i, t in enumerate(types):
        assert b.numpy(real)[i].sum() == len(t.atoms)
        np.testing.assert_array_equal(
            b.numpy(hydrogen)[i, : len(t.atoms)],
            [element[a.atom_type] == "H" for a in t.atoms],
        )
        n = int(b.numpy(p.n_torsions)[i])
        assert n == len(t.torsions)
        assert b.numpy(p.torsion_is_real)[i].sum() == n
        mc = b.numpy(p.mc_torsions)[i][b.numpy(p.mc_torsion_is_real)[i].astype(bool)]
        sc = b.numpy(p.sc_torsions)[i][b.numpy(p.sc_torsion_is_real)[i].astype(bool)]
        assert not set(mc) & set(sc)
        np.testing.assert_array_equal(np.sort(np.concatenate((mc, sc))), np.arange(n))
        for j, torsion in enumerate(t.torsions):
            np.testing.assert_array_equal(
                b.numpy(p.torsion_uaids)[i, j], t.torsion_to_uaids[torsion.name]
            )
        atoms = {a.name: j for j, a in enumerate(t.atoms)}
        np.testing.assert_array_equal(
            b.numpy(p.conn_atom)[i, : len(t.connections)],
            [atoms[c.atom] for c in t.connections],
        )
        assert b.numpy(p.conn_is_real)[i].sum() == len(t.connections)
        assert b.numpy(p.bond_is_real)[i].sum() == len(t.bond_indices)
        assert int(b.numpy(p.default_jump_connection_atom_inds)[i]) in range(
            len(t.atoms)
        )
    # Public preparation returns device arrays and all physical rows, even for
    # entirely disconnected palettes whose helper connection axis is empty.
    groups = {
        "atom_downstream_of_conn": ("atom_downstream_of_conn",),
        "atom_paths_from_conn": ("atom_paths_from_conn",),
        "torsion_uaids": ("n_torsions", "torsion_is_real", "torsion_uaids"),
        "is_torsion_mcs": ("is_torsion_mc",),
        "mc_torsion_inds": ("n_mc_torsions", "mc_torsion_is_real", "mc_torsions"),
        "sc_torsion_inds": ("n_sc_torsions", "sc_torsion_is_real", "sc_torsions"),
        "mcsc_torsion_inds": ("which_mcsc_torsions",),
        "bond_indices": ("n_bonds", "bond_is_real", "bond_indices"),
        "conn_indices": ("n_conn", "conn_is_real", "conn_atom"),
        "polymeric_connections": ("down_conn_inds", "up_conn_inds"),
        "default_jump_connection_atom_inds": ("default_jump_connection_atom_inds",),
    }
    for method, fields in groups.items():
        values = getattr(cls, "join_" + method)(types, b.device)
        for value, field in zip(
            values if isinstance(values, tuple) else (values,), fields
        ):
            assert value.device == b.device
            expected = b.numpy(getattr(p, field))
            actual = b.numpy(value)
            # Ignore only storage padding, never physical entries or atom order.
            slices = tuple(slice(0, n) for n in actual.shape)
            if field == "bond_indices":
                mask = b.numpy(p.bond_is_real).astype(bool)
                np.testing.assert_array_equal(actual[mask], expected[mask])
            else:
                np.testing.assert_array_equal(actual, expected[slices])


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
