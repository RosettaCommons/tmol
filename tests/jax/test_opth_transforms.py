"""Prepared OptH table/flip AD and one physical NHQ measurement per residue."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.scientific.execution import Execution
from tests.scientific.test_opth_sampling import OPTH_CASES, opth_task


@pytest.mark.parametrize("platform", ("cpu", "cuda"))
def test_opth_prepared_parameter_transforms_and_coordinate_gradient(platform):
    b = Execution("jax-" + platform).molecule(OPTH_CASES["protein"], terms=("ref",))
    task, sampler, concrete = opth_task(b)
    plan = sampler.prepare(b.pose, concrete(task))
    proton, flips = plan.inputs[13:15]

    def loss(p, f):
        return jnp.sin(plan.chi_values(p, f)).sum()

    value, (gp, gf) = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))(proton, flips)
    dp, df = jnp.ones_like(proton) * 0.2, jnp.ones_like(flips) * 0.3
    _, hvp = jax.jit(
        lambda p, f: jax.jvp(jax.grad(loss, argnums=(0, 1)), (p, f), (dp, df))
    )(proton, flips)
    for step in (1e-2, 3e-3):
        numerical = (
            loss(proton + step * dp, flips + step * df)
            - loss(proton - step * dp, flips - step * df)
        ) / (2 * step)
        np.testing.assert_allclose(
            numerical, (gp * dp).sum() + (gf * df).sum(), rtol=2e-3, atol=3e-4
        )
        plus = jax.grad(loss, argnums=(0, 1))(proton + step * dp, flips + step * df)
        minus = jax.grad(loss, argnums=(0, 1))(proton - step * dp, flips - step * df)
        for a, z, h in zip(plus, minus, hvp):
            np.testing.assert_allclose((a - z) / (2 * step), h, rtol=2e-3, atol=3e-4)
    assert np.isfinite(value) and all(np.isfinite(np.asarray(g)).all() for g in hvp)
    batched = jax.jit(jax.vmap(plan.chi_values))(
        jnp.stack((proton, proton)), jnp.stack((flips, flips))
    )
    np.testing.assert_array_equal(batched[0], plan.samples()[3])
    np.testing.assert_array_equal(batched[1], batched[0])

    def coordinate_loss(x):
        return jnp.sin(plan.measure_flip_values(x)).sum()

    gradient = jax.jit(jax.grad(coordinate_loss))(b.pose.coords)
    direction = (
        np.random.default_rng(811).normal(size=b.pose.coords.shape).astype(np.float32)
    )
    direction /= np.linalg.norm(direction)
    direction = jax.device_put(direction, b.device)
    for step in (1e-2, 3e-3):
        numerical = (
            coordinate_loss(b.pose.coords + step * direction)
            - coordinate_loss(b.pose.coords - step * direction)
        ) / (2 * step)
        np.testing.assert_allclose(
            numerical, (gradient * direction).sum(), rtol=3e-3, atol=3e-4
        )
    assert np.linalg.norm(gradient) > 0
