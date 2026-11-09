"""Packed and residue-major layouts preserve molecular identity and gradients."""

import numpy as np
import pytest

from .molecules import BACKBONE_CASES, DOMAIN_CASES


def test_pose_layout_preserves_atoms_padding_and_coordinate_derivatives(backend):
    b = backend.molecule(BACKBONE_CASES["mixed"], terms=())
    pose = b.pose
    counts = np.zeros(pose.block_type_ind.shape, dtype=np.int32)
    for p, row in enumerate(BACKBONE_CASES["mixed"].blocks(b.parameter_database)):
        for r, block in enumerate(row):
            counts[p, r] = len(block[2])
    np.testing.assert_array_equal(b.numpy(pose.n_ats_per_block), counts)
    np.testing.assert_array_equal(b.numpy(pose.n_res_per_pose), (counts > 0).sum(1))
    assert pose.max_n_pose_atoms == pose.coords.shape[1]
    assert pose.max_n_block_atoms == pose.packed_block_types.max_n_atoms
    assert pose.block_identity_map().device == b.device
    np.testing.assert_array_equal(
        b.numpy(pose.block_identity_map()),
        np.broadcast_to(np.arange(counts.shape[1]), counts.shape),
    )
    assert not bool(pose.is_real_block(1, pose.max_n_blocks - 1))
    real = np.arange(pose.max_n_pose_atoms) < counts.sum(1)[:, None]
    np.testing.assert_array_equal(b.numpy(pose.real_atoms), real)
    xyz = b.coords().copy()
    xyz[~real] = 99
    if b.framework == "torch":
        x = b.module.tensor(xyz, device=b.device, requires_grad=True)
        pose.coords = x
        expanded, mask = pose.expand_coords()
        gradient = b.module.autograd.grad(expanded.square().sum(), x)[0]
    else:
        import jax.numpy as jnp

        x = jnp.asarray(xyz, device=b.device)
        expanded, mask = b.module.jit(lambda c: pose.replace(coords=c).expand_coords())(
            x
        )
        gradient = b.module.jit(
            b.module.grad(
                lambda c: (pose.replace(coords=c).expand_coords()[0] ** 2).sum()
            )
        )(x)
    expanded, mask = b.numpy(expanded), b.numpy(mask)
    assert expanded.dtype == xyz.dtype
    np.testing.assert_array_equal(mask.sum(2), counts)
    np.testing.assert_array_equal(expanded[~mask], 0)
    np.testing.assert_array_equal(expanded[mask], xyz[real])
    np.testing.assert_array_equal(b.numpy(gradient), 2 * xyz * real[..., None])

    # The original sparse/dense bitwise score contract is a CPU contract.
    if backend.framework == "jax" and backend.device.platform == "cpu":
        jax = backend.module
        for name in ("connected_noncanonical", "mixed_batch", "metal_cluster"):
            backend = backend.molecule(
                DOMAIN_CASES[name], terms=("fa_ljatr", "fa_ljrep", "fa_lk", "fa_elec")
            )
            pose = backend.pose
            assert pose._inter_block_bondsep is None
            assert not hasattr(pose, "_dense_connections")
            scorer = backend.score_function.render_whole_pose_scoring_module(pose)
            evaluate = jax.jit(jax.value_and_grad(lambda x: scorer(x).sum()))
            value, gradient = evaluate(pose.coords)
            assert value.device == gradient.device == pose.device
            assert not hasattr(pose, "_dense_connections")
            dense = pose.inter_block_bondsep
            assert pose.inter_block_bondsep is dense
            explicit = pose.replace(inter_block_bondsep=dense)
            other = backend.score_function.render_whole_pose_scoring_module(explicit)
            expected, derivatives = jax.jit(
                jax.value_and_grad(lambda x: other(x).sum())
            )(pose.coords)
            assert expected.device == derivatives.device == pose.device
            np.testing.assert_array_equal(value, expected)
            np.testing.assert_array_equal(gradient, derivatives)
            # Dense replacements override any retained sparse graph, including under jit.
            changed = explicit.replace(inter_block_bondsep=dense + 1)
            np.testing.assert_array_equal(changed.inter_block_bondsep, dense + 1)
            np.testing.assert_array_equal(
                jax.jit(lambda p: p.inter_block_bondsep)(changed), dense + 1
            )


def test_pose_split_and_clone_preserve_custom_context_and_connections(backend):
    b = backend.molecule(BACKBONE_CASES["mixed"], terms=())
    pose = b.pose
    for index in range(pose.n_poses):
        result = pose.split(index).clone_sharing_topology()
        assert result.n_poses == 1
        assert result.packed_block_types is pose.packed_block_types
        assert result.get_constraint_set() is None
        for name in (
            "coords",
            "block_type_ind",
            "block_coord_offset",
            "inter_residue_connections",
        ):
            np.testing.assert_array_equal(
                b.numpy(getattr(result, name)),
                b.numpy(getattr(pose, name))[index : index + 1],
            )

    if backend.framework != "jax":
        return
    jax = backend.module
    pose = backend.molecule(DOMAIN_CASES["mixed_batch"], terms=("ref",)).pose
    expected = np.asarray(pose.inter_block_bondsep)
    selected = pose[[1, 0, 1]]
    assert selected._inter_block_bondsep is None
    assert not hasattr(selected, "_dense_connections")
    np.testing.assert_array_equal(selected.inter_block_bondsep, expected[[1, 0, 1]])
    moved = selected.to(jax.devices("cpu")[0])
    assert not hasattr(moved, "_dense_connections")
    result = jax.jit(lambda p: p.inter_block_bondsep)(moved)
    np.testing.assert_array_equal(result, expected[[1, 0, 1]])
    # Tracing must not cache a tracer on the source pose.
    assert not hasattr(moved, "_dense_connections")
    np.testing.assert_array_equal(moved.inter_block_bondsep, result)
    empty = pose[:0]
    assert empty.inter_block_bondsep.shape[0] == 0
    from tmol_jax.io.write_pose_stack_pdb import atom_records_from_pose_stack

    heterogeneous_pose = heterogeneous_protein_pose(backend)
    for index, expected in (
        (1, [1]),
        (-1, [1]),
        (np.int64(-2), [0]),
        (slice(None, None, -1), [1, 0]),
        ([1, 0, 1], [1, 0, 1]),
        (np.array([False, True]), [1]),
        (slice(0, 0), []),
    ):
        pose = heterogeneous_pose
        sliced = pose[index]
        assert sliced.n_poses == len(expected)
        for i, source in enumerate(expected):

            def identity_records(stack, pose_index):
                return [
                    {
                        key: value
                        for key, value in record.items()
                        if key not in ("model", "modeli")
                    }
                    for record in atom_records_from_pose_stack(
                        stack, pose_index=pose_index
                    )
                ]

            assert identity_records(sliced, i) == identity_records(pose, source)
            np.testing.assert_array_equal(sliced.coords[i], pose.coords[source])
        assert pose.replace(pdb_info=None)[index].pdb_info is None
    for index in (-3, 2):
        with pytest.raises(IndexError):
            heterogeneous_pose[index]


def heterogeneous_protein_pose(backend):
    from tmol_jax.io import (
        canonical_form_from_pdb,
        default_packed_block_types,
        pose_stack_from_canonical_form,
    )
    from .test_canonical_molecular_input import CANONICAL_PROTEIN_PDB

    canonical = canonical_form_from_pdb(CANONICAL_PROTEIN_PDB)
    canonical = {
        key: np.concatenate([value[:, :1], value[:, 1:2]], axis=0)
        for key, value in canonical.items()
    }
    canonical["chain_labels"][:] = [["A"], ["B"]]
    canonical["res_num"][:] = [[11], [22]]
    with backend.module.default_device(backend.device):
        pbt = backend.module.tree.map(
            lambda value: backend.module.device_put(value, backend.device),
            default_packed_block_types(),
        )
        return pose_stack_from_canonical_form(canonical, pbt)
