"""Packed and residue-major layouts preserve molecular identity and gradients."""

import numpy as np

from .molecules import BACKBONE_CASES


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
