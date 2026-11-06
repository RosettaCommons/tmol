"""Sparse callbacks retain parameter AD and native molecular atom identities."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.scientific.execution import Execution
from tests.scientific.rotamer_execution import RotamerExecution
from tests.scientific.test_rotamer_constraints import constrained_rotamers
from tests.scientific.molecules import CASES
from tmol_jax.pack.constraint_scoring import (
    prepare_rotamer_constraints,
    sparse_constraint_scores,
)
from tmol_jax.pose import ConstraintSet


@pytest.mark.parametrize("platform", ("cpu", "cuda"))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_sparse_constraint_jvp_vmap_coordinate_and_parameter_ad(platform, dtype):
    b = Execution("jax-" + platform)
    with jax.enable_x64(dtype == np.float64), jax.default_device(b.device):
        e, cs = constrained_rotamers(b, "mixed_batch", True)
        x = jnp.asarray(e.rotamers.coords, dtype)
        p = jnp.asarray(cs.constraint_params, dtype)

        def loss(x, p):
            values = sparse_constraint_scores(x, e.state, p)[0]
            return jnp.sum(values * jnp.cos(jnp.arange(values.size) * 0.21))

        value, (dx, dp) = jax.jit(jax.value_and_grad(loss, (0, 1)))(x, p)
        direction = jnp.sin(jnp.arange(x.size).reshape(x.shape)).astype(dtype)
        pdirection = jnp.cos(jnp.arange(p.size).reshape(p.shape)).astype(dtype)
        _, tangent = jax.jit(
            lambda x, p: jax.jvp(loss, (x, p), (direction, pdirection))
        )(x, p)
        tolerance = 2e-4 if dtype == np.float32 else 2e-11
        np.testing.assert_allclose(
            tangent,
            jnp.sum(dx * direction) + jnp.sum(dp * pdirection),
            rtol=tolerance,
            atol=tolerance,
        )
        bv, (bx, bp) = jax.jit(jax.vmap(jax.value_and_grad(loss, (0, 1))))(
            jnp.stack((x, x + jnp.array([2.0, -3.0, 4.0], dtype))),
            jnp.stack((p, p)),
        )
        for actual, expected in ((bv, value), (bx, dx), (bp, dp)):
            np.testing.assert_allclose(
                actual,
                jnp.stack((expected, expected)),
                rtol=tolerance,
                atol=tolerance,
            )
            assert actual.dtype == dtype and actual.device == b.device
        # Native torsion provides its first coordinate derivative. Parameter
        # curvature remains differentiable without a second native derivative.
        h = jax.jit(jax.hessian(lambda params: loss(x, params)))(p)
        np.testing.assert_allclose(
            h.reshape(p.size, p.size),
            h.reshape(p.size, p.size).T,
            rtol=tolerance,
            atol=tolerance,
        )
        assert jnp.isfinite(h).all() and jnp.any(h != 0)


def test_sparse_custom_callback_curvature_legacy_and_empty():
    import attrs
    from tmol_jax.score.constraint import ConstraintEnergyTerm
    from tmol_jax.score.constraint.constraint_energy_term import (
        ConstraintSet as Coordinates,
    )

    b = Execution("jax-cpu").molecule(CASES["mixed_batch"], terms=("constraint",))
    e = RotamerExecution(b, family="constraint")
    values, indices = sparse_constraint_scores(e.rotamers.coords, e.state)
    assert values.shape == (1, 0) and indices.shape == (3, 0)
    np.testing.assert_array_equal(
        jax.grad(lambda x: sparse_constraint_scores(x, e.state)[0].sum())(
            e.rotamers.coords
        ),
        jnp.zeros_like(e.rotamers.coords),
    )

    def callback(atoms, params):
        return params[:, 0] * jnp.sum(atoms[:, 0] ** 2, axis=-1)

    cs = ConstraintSet.create_empty(b.device, b.pose.n_poses).add_constraints(
        callback, np.array([[[0, 0, 0]]], np.int32), jnp.array([[3.0]])
    )
    state = prepare_rotamer_constraints(b.pose.replace(constraint_set=cs), e.rotamers)

    def loss(x):
        return sparse_constraint_scores(x, state)[0].sum()

    x = e.rotamers.coords
    _, hvp = jax.jit(lambda x: jax.jvp(jax.grad(loss), (x,), (jnp.ones_like(x),)))(x)
    expected = np.zeros(x.shape, x.dtype)
    expected[0] = 6
    np.testing.assert_array_equal(hvp, expected)

    legacy = Coordinates(
        jnp.array([[2.0, 3.0, 4.0]]),
        jnp.array([[0, 0]]),
        jnp.array([2.0]),
        jnp.array([0.1]),
    )
    pose = b.pose.replace(constraint_set=legacy)
    state = prepare_rotamer_constraints(pose, e.rotamers)
    term = ConstraintEnergyTerm()
    expected = sum(term(pose, term.setup_for_pose_stack(pose)).values()).sum()
    np.testing.assert_allclose(sparse_constraint_scores(x, state)[0].sum(), expected)
    with pytest.raises(ValueError, match="prepared shape"):
        sparse_constraint_scores(x, state, jnp.zeros((2, 5)))
    # A flattened padded coordinate is not a molecular atom.
    padded = attrs.evolve(
        legacy, atom_indices=jnp.array([[1, pose.coords.shape[1] - 1]])
    )
    with pytest.raises(ValueError, match="padding"):
        prepare_rotamer_constraints(pose.replace(constraint_set=padded), e.rotamers)


def test_sparse_constraint_explicit_capability_boundaries():
    import attrs
    from tmol_core.pack.correlation import CorrelatedBlockGroup
    from tmol_jax.score.constraint import ConstraintEnergyTerm

    b = Execution("jax-cpu")
    e, _ = constrained_rotamers(b, "mixed_batch", False)
    with pytest.raises(NotImplementedError, match="lockstep"):
        prepare_rotamer_constraints(
            b.pose,
            attrs.evolve(
                e.rotamers, correlated_groups=(CorrelatedBlockGroup(0, (0, 1)),)
            ),
        )
    cs = ConstraintSet.create_empty(b.device, b.pose.n_poses).add_constraints(
        ConstraintEnergyTerm.circularharmonic,
        np.array([[[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 0, 1]]], np.int32),
        jnp.array([[0.0, 1.0, 0.0]]),
    )
    with pytest.raises(NotImplementedError, match="more than two blocks"):
        prepare_rotamer_constraints(b.pose.replace(constraint_set=cs), e.rotamers)
