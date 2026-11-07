"""CUDA search preserves independent graph minima, seed bits and transformations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.scientific.execution import Execution
from tests.jax.test_native_annealing import independent_graph
from tmol_jax.pack.annealing import anneal_cuda


@pytest.mark.parametrize("empty", (False, True))
def test_cuda_annealer_minima_seed_bits_and_transforms(empty):
    device = Execution("jax-cuda").device
    arguments = list(jax.device_put(independent_graph(empty), device))
    if empty:
        arguments[0] = jnp.zeros_like(arguments[0])
    original = tuple(np.asarray(a).copy() for a in arguments)
    for bits in ((0, 0, 0), (42, 1 << 63, (1 << 64) - 1)):
        phase = jax.device_put(
            np.array(bits, np.uint64).view(np.uint8).reshape(3, 8), device
        )

        def run(onebody):
            return anneal_cuda(
                *arguments[:-2],
                onebody,
                arguments[-1],
                chunk_size=2,
                max_rotamers=0 if empty else 5,
                phase_seeds=phase,
            )

        scores, assignments = jax.jit(run)(arguments[-2])
        np.testing.assert_array_equal(scores[0], 0)
        np.testing.assert_array_equal(scores[1], 0 if empty else 10)
        np.testing.assert_array_equal(assignments, 0)
        assert scores.shape == (2, 312) and assignments.shape == (2, 312, 2)
        assert scores.dtype == np.float32 and assignments.dtype == np.int32
        assert scores.device == assignments.device == device
        for a, b in zip((scores, assignments), run(arguments[-2])):
            np.testing.assert_array_equal(a, b)
        v, choices = jax.jit(jax.vmap(run))(
            jnp.stack((arguments[-2], arguments[-2] + 1))
        )
        np.testing.assert_array_equal(v[0], scores)
        np.testing.assert_array_equal(v[1, 0], 0)
        np.testing.assert_array_equal(v[1, 1], 0 if empty else 12)
        np.testing.assert_array_equal(choices, np.stack((assignments,) * 2))
        with pytest.raises(ValueError, match="differentiated"):
            jax.grad(lambda x: run(x)[0].sum())(arguments[-2])
    for a, b in zip(original, arguments):
        np.testing.assert_array_equal(a, b)
