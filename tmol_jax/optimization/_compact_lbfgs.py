"""Compiled JAX linear algebra for the common compact L-BFGS direction."""

from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.linalg

from tmol_core.optimization.lbfgs import compact_lbfgs_direction
from tmol_jax._compilation import jit_without_autotuning


class _Execution:
    float64 = staticmethod(lambda value: value.astype(jnp.float64))
    cast = staticmethod(lambda value, dtype: value.astype(dtype))
    mv = mm = bmm = staticmethod(jnp.matmul)
    einsum = staticmethod(jnp.einsum)
    triu = staticmethod(jnp.triu)

    @staticmethod
    def roll(value, shifts, dims):
        return jnp.roll(value, shifts, axis=dims)

    @staticmethod
    def diagonal(value):
        return jnp.diagonal(value, axis1=-2, axis2=-1)

    @staticmethod
    def fill_diagonal_where(value, mask, fill):
        indices = jnp.arange(value.shape[-1])
        diagonal = jnp.where(mask, fill, _Execution.diagonal(value))
        return value.at[..., indices, indices].set(diagonal)

    @staticmethod
    def solve_triangular(matrix, value, upper):
        return jax.scipy.linalg.solve_triangular(matrix, value, lower=not upper)


@partial(jit_without_autotuning, static_argnames=("history_start",))
def compact_direction(gradient, differences, steps, history_start=0):
    """Retain mature H0=I, float64 solves and circular history ordering."""
    unbatched = gradient.ndim == 1
    poses = 1 if unbatched else gradient.shape[0]
    with jax.enable_x64(True):
        return compact_lbfgs_direction(
            gradient,
            differences,
            steps,
            history_start,
            execution=_Execution,
            single_segment=poses == 1 and gradient.dtype == jnp.float32,
            use_bmm=not unbatched and poses > 1,
        )
