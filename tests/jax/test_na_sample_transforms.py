"""Fixed NA sample plans expose linear parameter AD through native values."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.scientific.execution import Execution
from tests.scientific.na_molecules import NA_CASES
from tests.scientific.test_na_sampling import na_sampling_task


@pytest.mark.parametrize("platform", ("cpu", "cuda"))
@pytest.mark.parametrize("syn", (False, True))
def test_native_na_sample_parameter_jvp_vmap_and_curvature(platform, syn):
    b = Execution("jax-" + platform).molecule(NA_CASES["rna"], terms=("ref",))
    task, sampler, concrete = na_sampling_task(b, level=1, sample_syn=syn)
    prepared = sampler.prepare_chi_for_poses(b.pose, concrete(task))
    params = tuple(prepared.inputs[i] for i in (6, 8, 9))
    with jax.enable_x64():
        direction = tuple(
            jnp.cos(jnp.arange(p.size, dtype=p.dtype)).reshape(p.shape) * scale
            for p, scale in zip(params, (1.0, 180 / np.pi, 180 / np.pi))
        )

        def loss(*params):
            return jnp.sin(prepared.chi_values(*params).astype(jnp.float64)).sum()

        value, tangent = jax.jit(lambda *p: jax.jvp(loss, p, direction))(*params)
        gradient = jax.jit(jax.grad(loss, (0, 1, 2)))(*params)
        np.testing.assert_allclose(
            tangent,
            sum(jnp.sum(g * d) for g, d in zip(gradient, direction)),
            rtol=2e-6,
            atol=2e-5,
        )
        _, hvp = jax.jit(lambda *p: jax.jvp(jax.grad(loss, (0, 1, 2)), p, direction))(
            *params
        )
        curvature = sum(jnp.sum(h * d) for h, d in zip(hvp, direction))
        chi = prepared.chi_values(*params)
        for step in (0.01, 0.003):
            plus = prepared.chi_values(
                *(p + step * d for p, d in zip(params, direction))
            )
            minus = prepared.chi_values(
                *(p - step * d for p, d in zip(params, direction))
            )
            numeric = (
                jnp.sin(plus.astype(jnp.float64)).sum()
                - jnp.sin(minus.astype(jnp.float64)).sum()
            ) / (2 * step)
            np.testing.assert_allclose(tangent, numeric, rtol=8e-4, atol=3e-4)
            numeric_direction = (plus - minus).astype(jnp.float64) / (2 * step)
            expected_curvature = (
                -jnp.sin(chi.astype(jnp.float64)) * numeric_direction**2
            ).sum()
            np.testing.assert_allclose(
                curvature, expected_curvature, rtol=8e-4, atol=3e-4
            )
        batched = jax.jit(jax.vmap(loss))(*(jnp.stack((p, p)) for p in params))
        np.testing.assert_allclose(batched, jnp.stack((value, value)), atol=2e-10)
        assert chi.device == b.device and chi.dtype == jnp.float32
        assert all(jnp.isfinite(g).all() and jnp.any(g != 0) for g in gradient)


def test_na_preparation_context_and_precision_boundaries():
    from tmol_jax.pack import build_rotamers

    b = Execution("jax-cpu").molecule(NA_CASES["dna"], terms=("ref",))
    task, sampler, concrete = na_sampling_task(b)
    fixed = concrete(task)
    with pytest.raises(TypeError, match="eager"):
        jax.jit(
            lambda xyz: sampler.sample_chi_for_poses(b.pose.replace(coords=xyz), fixed)
        )(b.pose.coords)
    prepared = sampler.prepare_chi_for_poses(b.pose, fixed)
    with pytest.raises(ValueError, match="float32 table shapes"):
        prepared.chi_values(means=jnp.ones((1, 1), jnp.float32))
    with jax.enable_x64():
        pose = b.pose.replace(coords=b.pose.coords.astype(jnp.float64))
        _, rotamers = build_rotamers(pose, fixed, b.parameter_database.chemical)
        assert rotamers.coords.dtype == jnp.float64
        assert sampler.sample_chi_for_poses(pose, fixed)[3].dtype == jnp.float32
