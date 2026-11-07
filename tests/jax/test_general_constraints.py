"""JAX transformations and topology changes for the common constraint model."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tmol_jax.pose import ConstraintSet
from tmol_jax.score.constraint import ConstraintEnergyTerm
from tests.scientific.constraint_execution import Constraints
from tests.scientific.execution import Execution
from tests.scientific.molecules import CASES


def _callback(atoms, parameters):
    return (atoms[:, 0, 0] - parameters[:, 0]) ** 2


def test_constraint_parameter_assembly_preserves_two_derivatives():
    def build(parameters):
        constraints = ConstraintSet.create_empty(n_poses=3).add_constraints(
            _callback, np.array([[[0, 0]]], np.int32), parameters
        )
        return ConstraintSet.concatenate([constraints.split(2), constraints.split(0)])

    parameters = jnp.array([[2.0, 3.0]])

    def objective(p):
        return jnp.sum(build(p).constraint_params ** 2)

    np.testing.assert_array_equal(jax.grad(objective)(parameters), 4 * parameters)
    np.testing.assert_array_equal(
        jax.hessian(objective)(parameters).reshape(2, 2), np.eye(2) * 4
    )
    constraints = build(parameters)
    np.testing.assert_array_equal(
        jax.jit(lambda cs: cs.constraint_params)(constraints), [[2, 3], [2, 3]]
    )
    np.testing.assert_array_equal(constraints.constraint_atoms[:, 0, 0], [0, 1])


def test_pose_reorder_repeat_and_empty_preserve_constraints():
    backend = Execution("jax-cpu").molecule(CASES["mixed_batch"], terms=("constraint",))
    Constraints(backend).attach([("harmonic", [[[0, 0], [0, 1]]], [[1.0, 0.5]])])
    scores = backend.scores(backend.coords())
    selected = backend.pose[[1, 0, 1]].to(backend.device)
    score = backend.score_function.render_whole_pose_scoring_module(selected)
    np.testing.assert_allclose(jax.jit(score)(selected.coords), scores[[1, 0, 1]])
    empty = backend.pose[:0]
    assert empty.constraint_set.n_poses == 0
    assert empty.constraint_set.n_constraints == 0
    score = backend.score_function.render_whole_pose_scoring_module(empty)
    assert jax.jit(score)(empty.coords).shape == (0,)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_constraint_callback_jvp_vmap_and_parameter_hessian(dtype):
    with jax.enable_x64():
        xyz = jnp.array([[[0, 0, 0], [3, 4, 0]]], dtype)
        params = jnp.array([[3, 2]], dtype)
        fn = ConstraintEnergyTerm.harmonic
        batched = jax.jit(jax.vmap(fn, in_axes=(0, None)))(
            jnp.stack((xyz, xyz)), params
        )
        np.testing.assert_allclose(batched, [[1], [1]])
        _, tangent = jax.jit(lambda x: jax.jvp(lambda z: fn(z, params), (x,), (x,)))(
            xyz
        )
        np.testing.assert_allclose(tangent, [5])
        hessian = jax.jit(jax.hessian(lambda p: fn(xyz, p).sum()))(params)
        np.testing.assert_allclose(hessian.reshape(2, 2), [[0.5, 1], [1, 1.5]])


def test_general_constraints_reject_missing_callbacks_and_molecular_padding():
    from attrs import evolve

    backend = Execution("jax-cpu").molecule(CASES["mixed_batch"], terms=("constraint",))
    adapter = Constraints(backend)
    cs = adapter.attach([("harmonic", [[[0, 0], [0, 1]]], [[1.0, 0.5]])])
    invalid = evolve(
        cs, constraint_function_inds=jnp.ones_like(cs.constraint_function_inds)
    )
    backend.pose = backend.pose.replace(constraint_set=invalid)
    with pytest.raises(ValueError, match="registry"):
        backend.scores(backend.coords())
    adapter.attach([("harmonic", [[[1, 2, 0], [1, 2, 1]]], [[1.0, 0.5]])])
    with pytest.raises(ValueError, match="padding"):
        backend.scores(backend.coords())


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_native_constraint_torsion_transforms(dtype):
    with jax.enable_x64():
        xyz = jnp.array([[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 0, 1]]], dtype)
        fn = ConstraintEnergyTerm.get_torsion_angle_test
        values = jax.jit(jax.vmap(fn))(jnp.stack((xyz, xyz)))
        np.testing.assert_allclose(values, np.pi / 2, rtol=1e-6)
        direction = jnp.arange(12, dtype=dtype).reshape(xyz.shape) / 12
        value, tangent = jax.jit(lambda x, v: jax.jvp(fn, (x,), (v,)))(xyz, direction)
        gradient = jax.jit(jax.grad(lambda x: fn(x).sum()))(xyz)
        np.testing.assert_allclose(tangent, jnp.sum(gradient * direction), atol=1e-7)
        np.testing.assert_allclose(value, np.pi / 2, rtol=1e-6)
        assert jax.jit(fn)(xyz[:0]).shape == (0,)


def test_multi_block_constraints_transform_and_reject_sparse_expansion():
    from tests.scientific.test_constraints import attach_multi_block_constraint
    from tests.scientific.rotamer_execution import RotamerExecution

    device = "cuda" if jax.devices()[0].platform == "gpu" else "cpu"
    backend = Execution(f"jax-{device}")
    attach_multi_block_constraint(backend)
    scorer = backend.score_function.render_whole_pose_scoring_module(backend.pose)
    coords = backend.pose.coords
    expected = scorer(coords)
    batched = jax.jit(jax.vmap(scorer))(jnp.stack((coords, coords + 0.25)))
    np.testing.assert_allclose(batched, jnp.stack((expected, expected)), rtol=2e-6)
    gradient = jax.jit(jax.grad(lambda x: scorer(x).sum()))(coords)
    np.testing.assert_allclose(gradient.sum(axis=1), 0, atol=3e-6)
    assert np.any(np.abs(gradient) > 1)
    with pytest.raises(NotImplementedError, match="sparse constraints"):
        RotamerExecution(backend, family="constraint")
