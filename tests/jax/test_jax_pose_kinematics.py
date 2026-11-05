"""Pose-layout transformations and explicit native capability boundaries."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.scientific.execution import Execution
from tests.scientific.molecules import DOMAIN_CASES
from tmol_jax import PoseStackKinematics


def test_pose_jit_vmap_translation_and_cpu_placement():
    # This explicitly targets CPU even in a process whose default device is CUDA.
    backend = Execution("jax-cpu").molecule(DOMAIN_CASES["mixed_batch"], terms=("ref",))
    pose = backend.pose
    module = PoseStackKinematics(pose)
    mapping = module.bind(np.asarray(pose.coords))
    initial = mapping.initial_dofs
    ids = np.asarray(module.atom_ids)
    parent = module.forest._indices.parent
    kinds = module.forest._indices.doftype
    roots = np.flatnonzero(
        (parent == 0) & (kinds == 1) & (ids // pose.coords.shape[1] == 1)
    )
    assert len(roots) == 1
    root = roots[0]
    shift = jax.device_put(np.array([1.5, -0.25, 0.5], np.float32), pose.device)
    shifted = initial.at[root, :3].add(shift)
    batch = jnp.stack((initial, shifted))
    result = jax.jit(jax.vmap(mapping))(batch)
    assert result.device == initial.device == pose.device
    np.testing.assert_allclose(result[0], pose.coords, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(result[1, 0], result[0, 0], atol=2e-5)
    real = np.asarray(pose.atom_is_real[1])
    np.testing.assert_allclose(
        result[1, 1, real] - result[0, 1, real],
        np.broadcast_to(shift, (real.sum(), 3)),
        atol=2e-5,
    )
    np.testing.assert_array_equal(result[1, 1, ~real], result[0, 1, ~real])
    gradient = jax.jit(jax.vmap(jax.grad(lambda x: mapping(x).sum())))(batch)
    np.testing.assert_allclose(gradient[:, root, :3], real.sum(), atol=2e-5)
    with pytest.raises(NotImplementedError, match="initializes independent"):
        jax.grad(lambda x: module.inverse(x).sum())(pose.coords)
    with pytest.raises(ValueError, match="coordinate shape"):
        module.bind(pose.coords[:, :-1])
    with jax.enable_x64(False):
        with pytest.raises(ValueError, match="enable JAX float64"):
            module.bind(np.asarray(pose.coords, dtype=np.float64))


def test_pose_assembly_rejects_empty_poses_and_snapshots_topology():
    backend = Execution("jax-cpu").molecule(DOMAIN_CASES["ligand"], terms=("ref",))
    pose = backend.pose
    with pytest.raises(ValueError, match="at least one block"):
        PoseStackKinematics(pose[:0])
    module = PoseStackKinematics(pose)
    mapping = module.bind()
    before = mapping(mapping.initial_dofs)
    gradient = jax.grad(lambda x: mapping(x).sum())(mapping.initial_dofs)
    assert before.device == gradient.device == pose.device
    batched = jax.vmap(mapping)(jnp.stack((mapping.initial_dofs,) * 2))
    np.testing.assert_allclose(batched[1], before, atol=2e-5)
    # A different pose view requires its own preparation; the old map is stable.
    reordered = pose[[0, 0]]
    other = PoseStackKinematics(reordered).bind()
    np.testing.assert_allclose(other(other.initial_dofs)[1], before[0], atol=2e-5)
    np.testing.assert_array_equal(mapping(mapping.initial_dofs), before)
