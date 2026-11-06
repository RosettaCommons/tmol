"""Native transition transport preserves typed independent segment batches."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tmol_jax.optimization._armijo_native import classify, start


@pytest.mark.parametrize("real", (np.float32, np.float64))
@pytest.mark.parametrize("integer", (np.int32, np.int64))
def test_armijo_native_jit_vmap_precision_and_status(real, integer):
    with jax.enable_x64(True):
        alpha = jnp.asarray([[1, 0.1, 4], [0.5, 0.25, 2]], dtype=real)
        searching = jnp.ones(3, bool)

        def run(row):
            selected = start(searching, row)
            return classify(
                searching,
                selected,
                0.5 * (2 - 2 * selected) ** 2,
                jnp.full(3, 2, dtype=real),
                jnp.full(3, -4, dtype=real),
                0.8,
                0.1,
                status_dtype=integer,
            )

        accepted, values, status, active = jax.jit(jax.vmap(run))(alpha)
        assert accepted.dtype == values.dtype == real
        assert status.dtype == integer
        np.testing.assert_allclose(
            accepted, [[1, 0.1, 0], [0.5, 0.25, 0]], rtol=0, atol=1e-7
        )
        np.testing.assert_array_equal(status, [[0, 1, 2], [0, 1, 2]])
        np.testing.assert_array_equal(active, [[False, True, True]] * 2)
        np.testing.assert_allclose(
            values, [[0, 1.62, 2], [0.5, 1.125, 2]], rtol=0, atol=2e-7
        )


def test_armijo_float32_preserves_disabled_global_x64():
    with jax.enable_x64(False):
        alpha = start(
            jnp.asarray([True, False]), jnp.asarray([1.0, 7.0], dtype=jnp.float32)
        )
        np.testing.assert_array_equal(alpha, [1, 0])
        assert not jax.config.x64_enabled


def test_armijo_native_rejects_mismatched_segment_extents():
    with pytest.raises(RuntimeError, match="shape|extent|Armijo"):
        start(jnp.asarray([True]), jnp.ones(2, jnp.float32)).block_until_ready()
