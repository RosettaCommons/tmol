"""Mature supported design keeps constrained atom identities and derivatives."""

from importlib import import_module

import numpy as np

from .constraint_execution import Constraints
from .test_pose_imposition import imposition_fixture, impose


def constrained_design_fixture(backend):
    b = backend
    rotamers, graph, choices, selected = imposition_fixture(b, "design", design=True)
    old_types = b.numpy(b.pose.block_type_ind)
    blocks = b.pose.packed_block_types.active_block_types
    atoms = np.array(
        [
            [
                [p, block, blocks[int(old_types[p, block])].atom_to_idx[name]]
                for block, name in ((0, "CA"), (1, "N"))
            ]
            for p in range(b.pose.n_poses)
        ],
        np.int32,
    )
    params = np.array([[2.3, 0.4]] * b.pose.n_poses, np.float32)
    constraints = Constraints(b)
    constraints.attach([("harmonic", atoms, params)])
    return rotamers, graph, choices, selected, params


def test_design_preserves_backbone_constraints_across_atom_layout_change(backend):
    b = backend
    rotamers, graph, choices, selected, params = constrained_design_fixture(b)
    old_types = b.numpy(b.pose.block_type_ind)
    blocks = b.pose.packed_block_types.active_block_types
    original_atoms = b.numpy(b.pose.constraint_set.constraint_atoms).copy()
    old_pose = b.pose
    out = impose(b, rotamers, graph, choices)
    assert np.any(b.numpy(out.block_type_ind) != old_types)
    assert np.any(
        b.numpy(out.block_coord_offset) != b.numpy(old_pose.block_coord_offset)
    )
    np.testing.assert_array_equal(
        b.numpy(old_pose.constraint_set.constraint_atoms), original_atoms
    )
    xyz = b.numpy(out.coords)
    offsets = b.numpy(out.block_coord_offset)
    types = b.numpy(out.block_type_ind)
    expected = np.zeros(out.n_poses)
    expected_grad = np.zeros_like(xyz)
    for p in range(out.n_poses):
        indices = [
            int(offsets[p, block]) + blocks[int(types[p, block])].atom_to_idx[name]
            for block, name in ((0, "CA"), (1, "N"))
        ]
        delta = xyz[p, indices[0]] - xyz[p, indices[1]]
        distance = np.linalg.norm(delta)
        mean, sd = params[p]
        expected[p] = ((distance - mean) / sd) ** 2
        gradient = 2 * (distance - mean) / sd**2 * delta / distance
        expected_grad[p, indices[0]] += gradient
        expected_grad[p, indices[1]] -= gradient
    b.pose = out
    ScoreType = import_module("tmol_jax" if b.framework == "jax" else "tmol").ScoreType
    b.score_function.set_weight(ScoreType.ref, 0.0)
    b.score_function.set_weight(ScoreType.constraint, 1.0)
    if b.framework == "torch":
        b._render_torch()
    values, gradient = b.value_and_grad(xyz)
    np.testing.assert_allclose(values, expected, rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(gradient, expected_grad, rtol=3e-5, atol=3e-5)
    assert np.linalg.norm(gradient) > 1
