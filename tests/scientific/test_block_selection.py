"""Common molecular selection and repetition preserve topology and cotangents."""

from contextlib import nullcontext
from importlib import import_module

import attrs
import numpy as np
import pytest

from .constraint_execution import Constraints
from .molecules import BACKBONE_CASES


def dense(b, pose):
    value = pose.inter_block_bondsep
    return b.numpy(value.to_dense() if b.framework == "torch" else value)


@pytest.mark.parametrize("blocks", ([3, 0, 2], [-1, 1, 2], [3, 3]))
def test_select_repeat_preserves_molecular_identity_paths_and_cotangents(
    backend, blocks
):
    b = backend.molecule(BACKBONE_CASES["mixed"], terms=())
    front = import_module("tmol_jax.pose" if b.framework == "jax" else "tmol.pose")
    pose = b.pose.split(0)
    source = np.concatenate(
        [
            np.arange(
                int(b.numpy(pose.block_coord_offset)[0, block]),
                int(b.numpy(pose.block_coord_offset)[0, block])
                + pose.block_type(0, block).n_atoms,
            )
            for block in blocks
        ]
    )
    selected = front.select_blocks(pose, blocks)
    np.testing.assert_array_equal(
        b.numpy(selected.coords), b.numpy(pose.coords)[:, source]
    )
    np.testing.assert_array_equal(
        dense(b, selected), dense(b, pose)[:, blocks][:, :, blocks]
    )
    np.testing.assert_array_equal(
        b.numpy(selected.chain_id), b.numpy(pose.chain_id)[:, blocks]
    )
    np.testing.assert_array_equal(
        selected.pdb_info.residue_labels, pose.pdb_info.residue_labels[:, blocks]
    )
    repeated = front.repeat_pose_stack(selected, 3)
    np.testing.assert_array_equal(
        b.numpy(repeated.coords), np.repeat(b.numpy(selected.coords), 3, axis=0)
    )
    precision = b.module.enable_x64() if b.framework == "jax" else nullcontext()
    with precision:
        coords = b.numpy(pose.coords).astype(np.float64)
        weights = np.arange(
            repeated.coords.size if b.framework == "jax" else repeated.coords.numel(),
            dtype=np.float64,
        ).reshape(repeated.coords.shape)
        expected = np.zeros_like(coords)
        np.add.at(expected[0], source, weights.sum(axis=0))
        if b.framework == "jax":
            x = b.module.device_put(coords, b.device)
            w = b.module.device_put(weights, b.device)

            def objective(x):
                out = front.repeat_pose_stack(
                    front.select_blocks(pose.replace(coords=x), blocks), 3
                )
                return (out.coords * w).sum()

            gradient = b.module.grad(objective)(x)
        else:
            x = b.module.tensor(coords, device=b.device, requires_grad=True)
            w = b.module.as_tensor(weights, device=b.device)
            out = front.repeat_pose_stack(
                front.select_blocks(attrs.evolve(pose, coords=x), blocks), 3
            )
            gradient = b.module.autograd.grad((out.coords * w).sum(), x)[0]
        np.testing.assert_array_equal(b.numpy(gradient), expected)
    if len(set(blocks)) == len(blocks):
        b.pose = selected
        b.use_beta2016()
        value, gradient = b.channel_values_and_gradients(b.coords())
        assert value.shape[0] == 25
        assert np.isfinite(value).all() and np.isfinite(gradient).all()
        assert np.linalg.norm(gradient) > 0


def test_repeat_constraints_and_selection_limits(backend):
    b = backend.molecule(BACKBONE_CASES["mixed"], terms=("constraint",))
    Constraints(b).attach(
        [("harmonic", np.array([[[0, 0, 0], [0, 0, 1]]]), np.array([[1.3, 0.4]]))]
    )
    front = import_module("tmol_jax.pose" if b.framework == "jax" else "tmol.pose")
    pose = b.pose.split(0)
    repeated = front.repeat_pose_stack(pose, 3)
    atoms = b.numpy(repeated.constraint_set.constraint_atoms)
    np.testing.assert_array_equal(
        atoms[:, :2, 0], np.repeat(np.arange(3)[:, None], 2, axis=1)
    )
    assert front.select_blocks(pose, [0, 1]).constraint_set is None
    original = pose.pdb_info.residue_labels.copy()
    repeated.pdb_info.residue_labels[1] = -99
    np.testing.assert_array_equal(pose.pdb_info.residue_labels, original)
    np.testing.assert_array_equal(repeated.pdb_info.residue_labels[0], original[0])
    with pytest.raises(ValueError, match="single reference"):
        front.repeat_pose_stack(b.pose, 2)
    with pytest.raises(ValueError, match="homogeneous"):
        front.select_blocks(b.pose, [0])


@pytest.mark.parametrize("field", ("residue_labels", "chain_labels"))
def test_repeat_retains_absent_labels_and_independent_present_metadata(backend, field):
    b = backend.molecule(BACKBONE_CASES["mixed"], terms=())
    front = import_module("tmol_jax.pose" if b.framework == "jax" else "tmol.pose")
    pose = b.pose.split(0)
    metadata = attrs.evolve(pose.pdb_info, **{field: None})
    pose = (
        pose.replace(pdb_info=metadata)
        if b.framework == "jax"
        else attrs.evolve(pose, pdb_info=metadata)
    )
    repeated = front.repeat_pose_stack(pose, 3)
    assert getattr(repeated.pdb_info, field) is None
    np.testing.assert_array_equal(
        b.numpy(repeated.coords), np.repeat(b.numpy(pose.coords), 3, axis=0)
    )
    present = "chain_labels" if field == "residue_labels" else "residue_labels"
    original = np.asarray(getattr(pose.pdb_info, present)).copy()
    actual = getattr(repeated.pdb_info, present)
    np.testing.assert_array_equal(actual, np.repeat(original, 3, axis=0))
    actual[1, 0] = "changed" if present == "chain_labels" else -99
    np.testing.assert_array_equal(getattr(pose.pdb_info, present), original)
    np.testing.assert_array_equal(actual[0], original[0])
