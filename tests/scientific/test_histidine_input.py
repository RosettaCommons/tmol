"""Shared histidine decisions retain atom aliases, chirality and input ownership."""

from contextlib import nullcontext

import numpy as np
import pytest

from .molecules import MolecularCase


@pytest.mark.parametrize("name", ("HIS", "DHIS"))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_histidine_protonation_and_alias_identity(backend, name, dtype):
    from tmol_core.io.canonical_ordering import CanonicalOrdering

    b = backend.molecule(MolecularCase((), ((name,),), ((1,),)), terms=("ref",))
    co = CanonicalOrdering.from_chemdb(b.parameter_database.chemical)
    residue = b.pose.block_type(0, 0)
    equiv = residue.io_equiv_class
    atom_map = co.restypes_atom_index_mapping[equiv]
    fields = ("ND1", "NE2", "HD1", "HE2", "HN", "NH", "NN", "CG")
    nd, ne, hd, he, hn, nh, nn, cg = indices = [atom_map[a] for a in fields]
    xyz = np.zeros((1, 9, co.max_n_canonical_atoms, 3), dtype)
    present = np.zeros(xyz.shape[:-1], bool)
    for i, atom in enumerate(residue.atoms):
        xyz[0, :, atom_map[atom.name]] = b.coords()[0, i]
        present[0, :, atom_map[atom.name]] = True
    present[0, :, [hd, he, hn, nh, nn]] = False
    present[0, 0, hd] = present[0, 1, he] = True
    present[0, 2, [hd, he]] = True
    direction = xyz[0, 0, nd] - xyz[0, 0, ne]
    direction /= np.linalg.norm(direction)
    xyz[0, 4, hn] = xyz[0, 4, nd] + direction
    present[0, 4, hn] = True
    for row, reverse in ((5, False), (6, True)):
        xyz[0, row, nh] = xyz[0, row, ne if reverse else nd]
        xyz[0, row, nn] = xyz[0, row, nd if reverse else ne]
        xyz[0, row, hn] = xyz[0, row, nh] + (-direction if reverse else direction)
        present[0, row, [nd, ne]] = False
        present[0, row, [nh, nn, hn]] = True
    present[0, 7] = False
    present[0, 7, hn] = True
    present[0, 8] = False
    types = np.full((1, 9), co.restype_io_equiv_classes.index(equiv), np.int32)
    types[0, -1] = -1
    variants = np.zeros((1, 9), np.int32)
    variants[0, -1] = 7
    selected_poses = np.array([0] * 8 + [-1], np.int64)
    selected_residues = np.array(list(range(8)) + [-1], np.int64)
    identity = (
        np.broadcast_to(np.arange(co.max_n_canonical_atoms), present.shape)
        .astype(np.int64)
        .copy()
    )
    expected = identity.copy()
    expected[0, 4, hd] = hn
    expected[0, 5, [nd, hd, ne]] = [nh, hn, nn]
    expected[0, 6, [nd, he, ne]] = [nn, hn, nh]
    precision = (
        b.module.enable_x64(dtype == np.float64)
        if b.framework == "jax"
        else nullcontext()
    )
    with precision:

        def array(value):
            return (
                b.module.device_put(value, b.device)
                if b.framework == "jax"
                else b.module.as_tensor(value, device=b.device)
            )

        if b.framework == "jax":
            from tmol_jax.io._native import histidine_state, histidine_coordinates

            def wide(x):
                return x.view(np.uint8).reshape((*x.shape, 8))

            arguments = tuple(
                map(
                    array,
                    (
                        xyz,
                        types,
                        variants,
                        wide(selected_poses),
                        wide(selected_residues),
                        present,
                        np.array([indices], np.int32),
                    ),
                )
            )
            result = b.module.jit(histidine_state)(*arguments)
            mapped = b.module.jit(b.module.vmap(histidine_state))(
                *(b.module.numpy.stack((x, x)) for x in arguments)
            )
            for a, batched in zip(result, mapped):
                np.testing.assert_array_equal(b.numpy(a), b.numpy(batched[0]))
            actual_variants, mapping, states = map(b.numpy, result)
            mapping = mapping.copy().view(np.int64).reshape(present.shape)
            states = states.copy().view(np.int64).reshape(types.shape)
            np.testing.assert_array_equal(b.numpy(arguments[2]), variants)
            resolved_variants, resolved_coords, resolved_present = (
                histidine_coordinates(*arguments)
            )
            np.testing.assert_array_equal(b.numpy(resolved_variants), actual_variants)
            np.testing.assert_array_equal(
                b.numpy(resolved_coords),
                np.take_along_axis(xyz, expected[..., None], axis=2),
            )
            np.testing.assert_array_equal(
                b.numpy(resolved_present), np.take_along_axis(present, expected, axis=2)
            )
        else:
            from tmol.io.details.compiled._compiled import _resolve_his_taut

            actual_variants = array(variants).clone()
            mapping = array(identity)
            states = _resolve_his_taut(
                array(xyz),
                array(types),
                actual_variants,
                array(selected_poses),
                array(selected_residues),
                array(present),
                array(np.array([indices], np.int32)),
                mapping,
            )
            actual_variants, mapping, states = map(
                b.numpy, (actual_variants, mapping, states)
            )
        np.testing.assert_array_equal(actual_variants, [[1, 0, 2, 0, 1, 1, 0, 3, 7]])
        np.testing.assert_array_equal(states, [[1, 2, 5, 2, 1, 3, 4, 6, 0]])
        np.testing.assert_array_equal(mapping, expected)
