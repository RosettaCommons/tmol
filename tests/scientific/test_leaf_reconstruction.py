"""Shared missing-atom reconstruction preserves observed molecular geometry."""

from contextlib import nullcontext

import numpy as np
import pytest

from .molecules import DOMAIN_CASES

LEAF_CASES = (
    "ligand",
    "connected_noncanonical",
    "d_peptide",
    "modified_dna",
    "mixed_batch",
)


def leaf_reconstruction_case(b, name, dtype):
    from tmol_core.chemical.packed import PackedBlockTypesData
    from tmol_core.io.missing_atoms import prepare_leaf_atom_metadata

    b.molecule(DOMAIN_CASES[name], terms=("ref",))
    pbt = b.pose.packed_block_types
    ports = b.pose.inter_residue_connections.shape[2]
    host = PackedBlockTypesData.from_restypes(
        pbt.chem_db, pbt.active_block_types, min_n_conn=ports
    )
    info = prepare_leaf_atom_metadata(
        host.active_block_types, host.atom_is_hydrogen, host.max_n_atoms
    )
    xyz = b.coords().astype(dtype)
    types, offsets = (
        b.numpy(a).astype(np.int32)
        for a in (b.pose.block_type_ind, b.pose.block_coord_offset)
    )
    missing = np.zeros((*types.shape, host.max_n_atoms), bool)
    flat_missing = np.zeros(xyz.shape[:2], bool)
    real = np.zeros(xyz.shape[:2], bool)
    targets = []
    for pose, block in np.argwhere(types >= 0):
        t, offset = types[pose, block], offsets[pose, block]
        real[pose, offset : offset + host.n_atoms[t]] = True
        local = info.icoors.anc_uaids[t, :, :, 0]
        eligible = np.flatnonzero(info.is_leaf_atom[t] & np.all(local >= 0, axis=1))
        if len(eligible):
            atom = eligible[0]
            missing[pose, block, atom] = True
            flat_missing[pose, offset + atom] = True
            parent = offset + local[atom, 0]
            targets.append((pose, offset + atom, parent, info.icoors.geom[t, atom, 2]))
    assert targets, f"{name} must exercise real missing atoms"
    # Unused missing input coordinates cannot influence reconstructed outputs.
    xyz[flat_missing] = 17.25
    metadata = (
        missing,
        flat_missing,
        offsets,
        types,
        b.numpy(b.pose.inter_residue_connections).astype(np.int32),
        host.n_atoms.astype(np.int32),
        host.atom_downstream_of_conn.astype(np.int32),
        info.icoors.anc_uaids,
        info.icoors.geom.astype(dtype),
        info.icoors.anc_uaids_backup,
        info.icoors.geom_backup.astype(dtype),
    )

    def array(value):
        value = np.asarray(value)
        return (
            b.module.device_put(value, b.device)
            if b.framework == "jax"
            else b.module.as_tensor(value, device=b.device)
        )

    arrays = tuple(map(array, metadata))
    if b.framework == "jax":
        from tmol_jax.io._native import generate_leaf_atoms

        run = b.module.jit(lambda x: generate_leaf_atoms(x, arrays))
    else:
        from tmol.io.details.compiled._compiled import _gen_pose_leaf_atoms

        def run(x):
            return _gen_pose_leaf_atoms(x, *arrays)

    return xyz, array, run, real, flat_missing, targets


@pytest.mark.parametrize("name", LEAF_CASES)
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_molecular_leaf_reconstruction_and_gradient(backend, name, dtype):
    b = backend
    precision = (
        b.module.enable_x64(dtype == np.float64)
        if b.framework == "jax"
        else nullcontext()
    )
    with precision:
        xyz, array, run, real, missing, targets = leaf_reconstruction_case(
            b, name, dtype
        )
        x = array(xyz)
        out = b.numpy(run(x))
        assert out.dtype == dtype and np.isfinite(out).all()
        np.testing.assert_array_equal(out[real & ~missing], xyz[real & ~missing])
        np.testing.assert_array_equal(out[~real], 0)
        for pose, atom, parent, distance in targets:
            np.testing.assert_allclose(
                np.linalg.norm(out[pose, atom] - out[pose, parent]),
                distance,
                rtol=2e-6 if dtype == np.float32 else 2e-12,
                atol=2e-6 if dtype == np.float32 else 2e-12,
            )
        weights = np.sin(np.arange(xyz.size) * 0.13).reshape(xyz.shape).astype(dtype)
        w = array(weights)
        if b.framework == "jax":
            gradient = b.numpy(
                b.module.jit(b.module.grad(lambda x: (run(x) * w).sum()))(x)
            )
            mapped = b.module.jit(b.module.vmap(run))(b.module.numpy.stack((x, x)))
            np.testing.assert_array_equal(b.numpy(mapped[0]), out)
        else:
            x = x.detach().requires_grad_()
            gradient = b.numpy(b.module.autograd.grad((run(x) * w).sum(), x)[0])
        np.testing.assert_array_equal(gradient[missing | ~real], 0)
        # Translation covariance fixes the sum of all coordinate derivatives.
        np.testing.assert_allclose(
            gradient.sum(axis=(0, 1)),
            weights[real].sum(axis=0),
            rtol=2e-5 if dtype == np.float32 else 2e-12,
            atol=2e-5 if dtype == np.float32 else 2e-12,
        )
        direction = np.cos(np.arange(xyz.size) * 0.19).reshape(xyz.shape).astype(dtype)
        step = 0.002 if dtype == np.float32 else 1e-5
        numerical = (
            (
                b.numpy(run(array(xyz + step * direction))).astype(np.float64)
                - b.numpy(run(array(xyz - step * direction))).astype(np.float64)
            )
            * weights
        ).sum() / (2 * step)
        np.testing.assert_allclose(
            (gradient * direction).sum(),
            numerical,
            rtol=2e-3 if dtype == np.float32 else 2e-7,
            atol=2e-3 if dtype == np.float32 else 2e-7,
        )
        # An unavailable construction frame must remain visibly unbuildable.
        pose, atom, parent, _ = targets[0]
        broken = xyz.copy()
        broken[pose, parent] = np.nan
        assert np.isnan(b.numpy(run(array(broken)))[pose, atom]).all()
