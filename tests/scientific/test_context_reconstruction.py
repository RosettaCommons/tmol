"""Context construction preserves observed atoms and differentiable geometry."""

from contextlib import nullcontext

import numpy as np
import pytest

from .molecules import DOMAIN_CASES


@pytest.mark.parametrize(
    "case", ("ligand", "noncanonical", "connected_noncanonical", "modified_dna")
)
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_context_heavy_atom_coordinates_and_derivatives(backend, case, dtype):
    from tmol_core.io.missing_atoms import prepare_leaf_atom_flags

    b = backend.molecule(DOMAIN_CASES[case], terms=("ref",))
    pbt = b.pose.packed_block_types
    types, offsets, connections = [
        b.numpy(a)
        for a in (
            b.pose.block_type_ind,
            b.pose.block_coord_offset,
            b.pose.inter_residue_connections,
        )
    ]
    is_h = b.numpy(pbt.atom_is_hydrogen).astype(bool)
    leaf = prepare_leaf_atom_flags(
        pbt.active_block_types, is_h, pbt.max_n_atoms
    ).is_leaf_atom
    coords = b.coords().astype(dtype)
    missing = np.zeros(coords.shape[:2], bool)
    bonds = []
    for p, r in np.argwhere(types >= 0):
        t, start = int(types[p, r]), int(offsets[p, r])
        residue = pbt.active_block_types[t]
        # Actual non-leaf heavy atoms require context, not hydrogen-only frames.
        candidates = np.flatnonzero(
            ~is_h[t, : residue.n_atoms] & ~leaf[t, : residue.n_atoms]
        )
        assert len(candidates)
        atom = int(candidates[-1])
        missing[p, start + atom] = True
        ideal = residue.ideal_coords[residue.at_to_icoor_ind].astype(np.float64)
        neighbors = [int(j) for i, j in residue.bond_indices if i == atom]
        bonds.append(
            (
                p,
                start + atom,
                [
                    (start + j, np.linalg.norm(ideal[atom] - ideal[j]))
                    for j in neighbors
                ],
            )
        )
    coords[missing] = 17.25
    precision = (
        b.module.enable_x64(dtype == np.float64)
        if b.framework == "jax"
        else nullcontext()
    )
    with precision:

        def array(x):
            return (
                b.module.device_put(x, b.device)
                if b.framework == "jax"
                else b.module.as_tensor(x, device=b.device)
            )

        x = array(coords)
        if b.framework == "jax":
            from tmol_jax.io._context import (
                prepare_context_completion,
                execute_context_plan,
            )

            out, built, plans = prepare_context_completion(
                b.pose.replace(coords=x), missing, missing
            )

            if dtype == np.float32:
                from .test_canonical_molecular_input import canonical_input
                from tmol_jax.io import pose_stack_from_canonical_form

                co, cf = canonical_input(DOMAIN_CASES[case], b.parameter_database)
                for p, r in np.argwhere(types >= 0):
                    residue = pbt.active_block_types[int(types[p, r])]
                    start = int(offsets[p, r])
                    target_row = int(r + (r >= 1 and types.shape[1] > 1))
                    for atom in np.flatnonzero(
                        missing[p, start : start + residue.n_atoms]
                    ):
                        slot = co.restypes_atom_index_mapping[residue.io_equiv_class][
                            residue.atoms[atom].name
                        ]
                        cf["coords"][p, target_row, slot] = np.nan
                with b.module.default_device(b.device):
                    loaded = pose_stack_from_canonical_form(
                        dict(cf, canonical_ordering=co),
                        pbt,
                        compact_padding=True,
                        build_missing_context=True,
                    )
                np.testing.assert_allclose(
                    b.numpy(loaded.coords), b.numpy(out), rtol=2e-6, atol=2e-6
                )

            def run(value):
                return execute_context_plan(value, plans)

            mapped = b.module.jit(b.module.vmap(run))(b.module.numpy.stack((x, x)))
            np.testing.assert_allclose(
                b.numpy(mapped[0]),
                b.numpy(out),
                rtol=2e-6 if dtype == np.float32 else 2e-12,
                atol=2e-6 if dtype == np.float32 else 2e-12,
            )
        else:
            from tmol_core.io.context import build_missing_by_context
            from tmol.io.details._build_missing_context import _place_differentiably

            out, plans = x.clone(), []

            def place(build):
                plans.append(build)
                positions = _place_differentiably(
                    build.steps,
                    build.chosen,
                    out,
                    build.pose,
                    build.s,
                    build.known_atoms,
                    build.node_global,
                )
                return {a: b.numpy(v).astype(np.float64) for a, v in positions.items()}

            built = build_missing_by_context(
                pbt.active_block_types,
                {a.name: a for a in pbt.chem_db.atom_types},
                b.numpy(pbt.conn_atom),
                coords.astype(np.float64),
                missing,
                missing,
                offsets,
                types,
                connections,
                place,
            )

            def run(value):
                result = value.clone()
                for build in plans:
                    _place_differentiably(
                        build.steps,
                        build.chosen,
                        result,
                        build.pose,
                        build.s,
                        build.known_atoms,
                        build.node_global,
                    )
                return result

        np.testing.assert_array_equal(built, missing)
        result = b.numpy(out)
        assert np.isfinite(result).all() and result.dtype == dtype
        np.testing.assert_array_equal(result[~missing], coords[~missing])
        np.testing.assert_array_equal(b.numpy(x), coords)
        for p, atom, neighbors in bonds:
            errors = [
                abs(np.linalg.norm(result[p, atom] - result[p, other]) - length)
                for other, length in neighbors
            ]
            assert min(errors) < (3e-6 if dtype == np.float32 else 3e-12)
        weights = array(
            np.sin(np.arange(coords.size) * 0.37).reshape(coords.shape).astype(dtype)
        )
        direction = (
            np.cos(np.arange(coords.size) * 0.23).reshape(coords.shape).astype(dtype)
        )
        direction /= np.linalg.norm(direction)
        if b.framework == "jax":
            gradient = b.module.jit(
                b.module.grad(lambda value: (run(value) * weights).sum())
            )
        else:

            def gradient(value):
                value = value.clone().requires_grad_(True)
                return b.module.autograd.grad((run(value) * weights).sum(), value)[0]

        g = b.numpy(gradient(x))
        np.testing.assert_array_equal(g[missing], 0)
        assert np.isfinite(g).all() and np.linalg.norm(g) > 0
        step = 2e-3 if dtype == np.float32 else 1e-5
        plus, minus = [
            b.numpy(run(array(coords + sign * step * direction))).astype(np.float64)
            for sign in (1, -1)
        ]
        finite_difference = (
            (plus - minus) * b.numpy(weights).astype(np.float64)
        ).sum() / (2 * step)
        np.testing.assert_allclose(
            (g * direction).sum(),
            finite_difference,
            rtol=3e-3 if dtype == np.float32 else 2e-6,
            atol=3e-3 if dtype == np.float32 else 2e-6,
        )
