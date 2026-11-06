"""Real ion and cluster virtual sites retain precision and metal translation."""

from contextlib import nullcontext

import numpy as np
import pytest

from .metal_molecules import METAL_CASES
from .test_canonical_molecular_input import canonical_input


@pytest.mark.parametrize("name", ("zinc_site", "cluster_bridge"))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_fixed_metal_sites_preserve_translation_and_coordinate_gradients(
    backend, name, dtype
):
    precision = (
        backend.module.enable_x64(dtype == np.float64)
        if backend.framework == "jax"
        else nullcontext()
    )
    with precision:
        _check_placement(backend, name, dtype)


def _check_placement(b, name, dtype):
    from tmol_core.io.metal_detection import (
        find_metal_geometries,
        metal_virtual_placements,
    )

    b.molecule(METAL_CASES[name], terms=("metal_coordination",))
    pbt = b.pose.packed_block_types
    co, cf = canonical_input(METAL_CASES[name], b.parameter_database)
    types = b.numpy(b.pose.block_type_ind)
    offsets = b.numpy(b.pose.block_coord_offset)
    order = np.argsort(cf["res_types"] < 0, axis=1, kind="stable")[:, : types.shape[1]]
    canonical_types = np.take_along_axis(cf["res_types"], order, axis=1)
    canonical_coords = np.take_along_axis(cf["coords"], order[..., None, None], axis=1)
    _, assignments = find_metal_geometries(
        co, pbt.chem_db, canonical_types, canonical_coords
    )
    missing = np.zeros((*types.shape, pbt.max_n_atoms), bool)
    for p, r in zip(*np.nonzero(types >= 0)):
        bt = pbt.active_block_types[types[p, r]]
        for site in bt.metal_sites:
            for atom in site.site_virts:
                missing[p, r, bt.atom_to_idx[atom]] = True
    plan = list(
        metal_virtual_placements(pbt.active_block_types, types, missing, assignments)
    )
    assert plan and any(wide for *_, wide in plan) == (name == "cluster_bridge")
    xyz = b.coords().astype(dtype)
    observed = np.ones(xyz.shape[:2], bool)
    rows, shifts, wide = [], [], []
    for p, r, a, parent, shift, widened in plan:
        rows.append((p, offsets[p, r] + a, offsets[p, r] + parent))
        shifts.append(shift)
        wide.append(widened)
        observed[p, offsets[p, r] + a] = False
    xyz[~observed] = 17.25
    if b.framework == "jax":
        from tmol_jax.io._metal import execute_metal_placement

        def array(value):
            return b.module.device_put(value, b.device)

        shifts = np.asarray(shifts, np.float64)
        data = tuple(
            map(
                array,
                (
                    np.asarray(rows, np.int32),
                    shifts.view(np.uint8).reshape((-1, 3, 8)),
                    np.asarray(wide, bool),
                ),
            )
        )
        run = b.module.jit(lambda x: execute_metal_placement(x, *data))
    else:
        from tmol.io.details._metal_detection import _place_site_virtuals

        def array(value):
            return b.module.as_tensor(value, device=b.device)

        def run(x):
            out = x.clone()
            for (p, r), assignment in assignments:
                t, start = int(types[p, r]), int(offsets[p, r])
                n = pbt.active_block_types[t].n_atoms
                _place_site_virtuals(
                    pbt,
                    array(np.array([[t]], np.int64)),
                    out[p, start : start + n].reshape(1, 1, n, 3),
                    array(missing[p, r].copy()[None, None]),
                    [((0, 0), assignment)],
                )
            return out

    x = array(xyz)
    output = b.numpy(run(x))
    np.testing.assert_array_equal(output[observed], xyz[observed])
    assert output.dtype == dtype and np.isfinite(output).all()
    tolerance = 2e-6 if dtype == np.float32 else 2e-13
    for (p, target, parent), shift in zip(rows, shifts):
        np.testing.assert_allclose(
            np.linalg.norm(output[p, target] - output[p, parent]),
            np.linalg.norm(shift),
            rtol=tolerance,
            atol=tolerance,
        )
    direction = np.zeros_like(xyz)
    direction[observed] = [0.5, -0.75, 0.25]
    shifted = b.numpy(run(array(xyz + direction)))
    np.testing.assert_allclose(
        shifted, output + [0.5, -0.75, 0.25], rtol=tolerance, atol=tolerance
    )
    weights = array(np.sin(np.arange(xyz.size)).reshape(xyz.shape).astype(dtype))
    if b.framework == "jax":
        import jax.numpy as jnp

        gradient = b.module.grad(lambda y: jnp.sum(run(y) * weights))(x)
        batch = b.module.vmap(run)(array(np.stack((xyz, xyz + direction))))
        np.testing.assert_array_equal(b.numpy(batch), np.stack((output, shifted)))
    else:
        x = x.detach().requires_grad_(True)
        gradient = b.module.autograd.grad((run(x) * weights).sum(), x)[0]
    gradient = b.numpy(gradient)
    np.testing.assert_array_equal(gradient[~observed], 0)
    predicted = np.sum(gradient.astype(np.float64) * direction)
    actual = np.sum((shifted.astype(np.float64) - output) * b.numpy(weights))
    np.testing.assert_allclose(predicted, actual, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("name", ("zinc_site", "cluster_bridge"))
def test_public_metal_input_reconstructs_missing_sites(backend, name):
    from .test_canonical_molecular_input import physical_edges, rebuild_canonical

    b = backend.molecule(METAL_CASES[name], terms=("metal_coordination", "fa_elec"))
    expected = b.coords().copy()
    edges = physical_edges(b)
    co, cf = canonical_input(METAL_CASES[name], b.parameter_database)
    observed = np.ones(expected.shape[:2], bool)
    for r in range(b.pose.block_type_ind.shape[1]):
        residue = b.pose.block_type(0, r)
        start = int(b.numpy(b.pose.block_coord_offset)[0, r])
        canonical_residue = r + (r >= 1)
        index = co.restypes_atom_index_mapping[residue.io_equiv_class]
        for site in residue.metal_sites:
            for atom in site.site_virts:
                observed[0, start + residue.atom_to_idx[atom]] = False
                cf["coords"][0, canonical_residue, index[atom]] = np.nan
    assert not observed.all()
    rebuild_canonical(b, co, cf, unpatched_metals=True)
    rebuilt = b.coords().copy()
    np.testing.assert_array_equal(rebuilt[observed], expected[observed])
    assert np.isfinite(rebuilt).all() and physical_edges(b) == edges
    energy, gradient = b.channel_values_and_gradients(rebuilt)
    assert np.isfinite(energy).all() and np.isfinite(gradient).all()
    assert np.max(np.abs(energy[b.channel_names().index("metal_coordination")])) > 1e-3
    assert np.linalg.norm(gradient) > 0
    shift = np.array([0.25, -0.5, 1.0], np.float32)
    rebuild_canonical(
        b, co, dict(cf, coords=cf["coords"] + shift), unpatched_metals=True
    )
    np.testing.assert_allclose(b.coords(), rebuilt + shift, rtol=2e-6, atol=2e-6)
