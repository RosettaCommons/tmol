"""Shared generated-H geometry, coordinate/parameter gradients and curvature."""

from contextlib import nullcontext

import numpy as np
import pytest

from .molecules import DOMAIN_CASES, MolecularCase

HYDROGEN_CASES = {
    "ligand": MolecularCase(
        ("ligand_params/opth_propanediol.tmol",), (("OPL",),), ((1,),)
    ),
    "connected_noncanonical": DOMAIN_CASES["connected_noncanonical"],
    "mixed_batch": MolecularCase(
        (
            "ligand_params/opth_propanediol.tmol",
            "ligand_params/noncanonical_nonstandard_aa.tmol",
        ),
        (("GLY", "B3Q", "OPL"), ("OPL",)),
        ((1, 1, 1), (1,)),
    ),
}


@pytest.mark.parametrize("name", ("ligand", "connected_noncanonical", "mixed_batch"))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_generated_hydrogen_geometry_and_curvature(backend, name, dtype):
    b = backend
    precision = (
        b.module.enable_x64(dtype == np.float64)
        if b.framework == "jax"
        else nullcontext()
    )
    with precision:
        _check_geometry(b, name, dtype)


def _check_geometry(b, name, dtype):
    from tmol_core.chemical.packed import PackedBlockTypesData
    from tmol_core.io.missing_atoms import prepare_h_completion_metadata

    b.molecule(HYDROGEN_CASES[name], terms=("ref",))
    pbt = b.pose.packed_block_types
    host = PackedBlockTypesData.from_restypes(
        pbt.chem_db,
        pbt.active_block_types,
        min_n_conn=b.pose.inter_residue_connections.shape[2],
    )
    info = prepare_h_completion_metadata(
        host.active_block_types, host.atom_is_hydrogen, host.max_n_atoms
    )
    types = b.numpy(b.pose.block_type_ind).astype(np.int32)
    offsets = b.numpy(b.pose.block_coord_offset).astype(np.int32)
    missing = info.eligible[np.maximum(types, 0)] & (types >= 0)[..., None]
    assert missing.any(), name
    xyz = b.coords().astype(dtype)
    observed = np.ones(xyz.shape[:2], bool)
    targets = []
    for p, r, a in zip(*np.nonzero(missing)):
        t, start = types[p, r], offsets[p, r]
        observed[p, start + a] = False
        targets.append((p, start + a, start + info.parent[t, a], t, a))
    xyz[~observed] = 17.25
    distances = info.dist.astype(dtype)
    values = (
        types,
        offsets,
        b.numpy(b.pose.inter_residue_connections).astype(np.int32),
        host.conn_atom.astype(np.int32),
        missing,
        info.eligible,
        info.parent,
        info.neigh,
        info.n_neigh,
        info.par_conn,
        info.n_par_conn,
    )
    if b.framework == "jax":
        from tmol_jax.io._hydrogen import complete_hydrogens
        from tmol_jax.io._selection import _wide

        def array(a):
            return b.module.device_put(a, b.device)

        metadata = tuple(array(_wide(a) if a.dtype == np.int64 else a) for a in values)
        run = b.module.jit(lambda x, d: complete_hydrogens(x, d, metadata))
    else:
        from tmol.io.details._hydrogen_completion import complete_hydrogens

        def array(a):
            return b.module.as_tensor(a, device=b.device)

        metadata = tuple(map(array, values))

        def run(x, d):
            return complete_hydrogens(x, *metadata[:9], d, *metadata[9:])

    x, d = array(xyz), array(distances)
    output = b.numpy(run(x, d))
    assert output.dtype == dtype and np.isfinite(output).all()
    np.testing.assert_array_equal(output[observed], xyz[observed])
    tolerance = 2e-6 if dtype == np.float32 else 2e-13
    for p, a, parent, t, local in targets:
        np.testing.assert_allclose(
            np.linalg.norm(output[p, a] - output[p, parent]),
            distances[t, local],
            rtol=tolerance,
            atol=tolerance,
        )
    shift = np.array([0.5, -0.75, 0.25], dtype)
    np.testing.assert_allclose(
        b.numpy(run(array(xyz + shift), d)),
        output + shift,
        rtol=tolerance,
        atol=tolerance,
    )
    if b.framework == "jax":
        batched = b.module.vmap(lambda y: run(y, d))(
            array(np.stack((xyz, xyz + shift)))
        )
        np.testing.assert_allclose(
            b.numpy(batched),
            np.stack((output, output + shift)),
            rtol=tolerance,
            atol=tolerance,
        )
    weights = array(np.cos(np.arange(xyz.size)).reshape(xyz.shape).astype(dtype))
    direction = np.sin(np.arange(xyz.size) * 0.37).reshape(xyz.shape).astype(dtype)
    distance_direction = (info.eligible * 0.13).astype(dtype)
    if b.framework == "jax":
        import jax.numpy as jnp

        gradient = b.module.jit(
            b.module.grad(
                lambda y, z: jnp.sum(run(y, z) ** 2 * weights) / 2, argnums=(0, 1)
            )
        )
        curvature = b.module.grad(
            lambda y, z: sum(
                jnp.sum(g * v)
                for g, v in zip(
                    gradient(y, z), (array(direction), array(distance_direction))
                )
            ),
            argnums=(0, 1),
        )(x, d)
    else:

        def gradient(y, z, create_graph=False):
            y = y.detach().requires_grad_(True)
            z = z.detach().requires_grad_(True)
            g = b.module.autograd.grad(
                (run(y, z) ** 2 * weights).sum() / 2, (y, z), create_graph=create_graph
            )
            return (y, z, g) if create_graph else g

        y, z, g = gradient(x, d, True)
        product = sum(
            (a * v).sum()
            for a, v in zip(g, (array(direction), array(distance_direction)))
        )
        curvature = b.module.autograd.grad(product, (y, z))
    gx, gd = map(b.numpy, gradient(x, d))
    np.testing.assert_array_equal(gx[~observed], 0)
    assert np.linalg.norm(gd) > 0
    eps = 2e-3 if dtype == np.float32 else 1e-5
    plus = (array(xyz + eps * direction), array(distances + eps * distance_direction))
    minus = (array(xyz - eps * direction), array(distances - eps * distance_direction))
    yp, ym = (b.numpy(run(*args)).astype(np.float64) for args in (plus, minus))
    finite = np.sum((yp - ym) * (yp + ym) * b.numpy(weights).astype(np.float64)) / (
        4 * eps
    )
    analytic = np.sum(gx.astype(np.float64) * direction) + np.sum(
        gd.astype(np.float64) * distance_direction
    )
    tol = 3e-3 if dtype == np.float32 else 2e-7
    np.testing.assert_allclose(analytic, finite, rtol=tol, atol=tol)
    for hv, gp, gm in zip(curvature, gradient(*plus), gradient(*minus)):
        finite_hv = (
            b.numpy(gp).astype(np.float64) - b.numpy(gm).astype(np.float64)
        ) / (2 * eps)
        np.testing.assert_allclose(b.numpy(hv), finite_hv, rtol=tol, atol=tol)


@pytest.mark.parametrize("name", ("ligand", "connected_noncanonical"))
def test_canonical_hydrogens_follow_named_parent_and_rebuild_policy(backend, name):
    from tmol_core.io.missing_atoms import prepare_h_completion_metadata
    from .test_canonical_molecular_input import canonical_input, physical_edges

    b = backend.molecule(HYDROGEN_CASES[name], terms=("gen_torsions", "fa_elec"))
    pbt = b.pose.packed_block_types
    expected = b.coords().copy()
    expected_edges = physical_edges(b)
    types = b.numpy(b.pose.block_type_ind)
    offsets = b.numpy(b.pose.block_coord_offset)
    hydrogen = b.numpy(pbt.atom_is_hydrogen)
    info = prepare_h_completion_metadata(
        pbt.active_block_types, hydrogen, pbt.max_n_atoms
    )
    t = int(types[0, 0])
    target = int(np.flatnonzero(info.eligible[t])[0])
    parent = info.parent[t, target]
    co, cf = canonical_input(HYDROGEN_CASES[name], b.parameter_database)
    residue = pbt.active_block_types[t]
    canonical_atom = co.restypes_atom_index_mapping[residue.io_equiv_class][
        residue.atoms[target].name
    ]
    cf["coords"] = cf["coords"].copy()
    cf["coords"][0, 0, canonical_atom] += [8, 0, 0]
    heavy = np.zeros(expected.shape[:2], bool)
    for p, r in zip(*np.nonzero(types >= 0)):
        kind, start = int(types[p, r]), int(offsets[p, r])
        n = pbt.active_block_types[kind].n_atoms
        heavy[p, start : start + n] = ~hydrogen[kind, :n].astype(bool)
    placements = []
    for mode in ("input", "names", "rebuild"):
        if b.framework == "jax":
            from tmol_jax.io import pose_stack_from_canonical_form

            with b.module.default_device(b.device):
                b.pose = pose_stack_from_canonical_form(
                    dict(cf, canonical_ordering=co),
                    pbt,
                    compact_padding=True,
                    ligand_hydrogens=mode,
                )
        else:
            from tmol.io._pose_stack_construction import _pose_stack_from_canonical_form
            from tmol.io._canonical_ordering import CanonicalOrdering

            co = CanonicalOrdering.from_chemdb(b.parameter_database.chemical)
            args = [
                b.module.as_tensor(cf[k], device=b.device)
                for k in ("chain_id", "res_types", "coords")
            ]
            b.pose = _pose_stack_from_canonical_form(
                co,
                pbt,
                *args,
                None,
                None,
                None,
                res_not_connected=b.module.as_tensor(cf["res_gap"], device=b.device),
                find_additional_cyclic_closures=False,
                ligand_hydrogens=mode,
            )
            b._render_torch()
        result = b.coords()
        np.testing.assert_array_equal(result[heavy], expected[heavy])
        assert physical_edges(b) == expected_edges
        atom = int(offsets[0, 0]) + target
        parent_atom = int(offsets[0, 0]) + parent
        np.testing.assert_allclose(
            np.linalg.norm(result[0, atom] - result[0, parent_atom]),
            info.dist[t, target],
            rtol=2e-6,
            atol=2e-6,
        )
        placements.append(result[0, atom])
        energies, gradients = b.channel_values_and_gradients(result)
        assert np.isfinite(energies).all() and np.isfinite(gradients).all()
        assert np.linalg.norm(gradients) > 0
        if mode == "input":
            observed = np.ones(expected.shape[:2], bool)
            observed[0, atom] = False
            np.testing.assert_array_equal(result[observed], expected[observed])
    np.testing.assert_array_equal(placements[0], placements[1])
    np.testing.assert_array_equal(placements[1], placements[2])
