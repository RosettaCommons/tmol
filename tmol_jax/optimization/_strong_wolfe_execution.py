"""Compiled JAX array primitives for the common host strong-Wolfe policy."""

import jax
import jax.numpy as jnp
import numpy as np

from tmol_core.optimization.lbfgs import two_loop_direction
from tmol_jax._compilation import jit_without_autotuning


class JaxStrongWolfeExecution:
    # Immutable arrays can share snapshots without a device copy.
    clone = staticmethod(lambda x: x)
    # Scalar reductions determine trial steps and line-search decisions. Keep
    # their compilation stable without promoting the molecular coordinate dtype.
    dot = staticmethod(jit_without_autotuning(jnp.dot))
    abs_max = staticmethod(jax.jit(lambda x: jnp.max(jnp.abs(x))))
    abs_sum = staticmethod(jit_without_autotuning(lambda x: jnp.sum(jnp.abs(x))))
    sqrt = staticmethod(jnp.sqrt)
    add_scaled = staticmethod(
        jax.jit(lambda x, y, alpha: x + jnp.asarray(alpha, dtype=x.dtype) * y)
    )


class _LoopExecution(JaxStrongWolfeExecution):
    zero_coefficients = staticmethod(jnp.zeros_like)
    set_coefficient = staticmethod(
        lambda values, index, value: values.at[index].set(value)
    )
    fori_loop = staticmethod(jax.lax.fori_loop)


@jax.jit
def _direction(gradient, steps, differences, rho, scale, count):
    # Cast inside the compiled call; its array operands determine placement.
    scale = jnp.asarray(scale, gradient.dtype)
    count = jnp.asarray(count, jnp.int32)
    return two_loop_direction(
        gradient, steps, differences, rho, scale, count, _LoopExecution()
    )


@jax.jit
def _append_history(steps, differences, rho, step, difference, inverse, count):
    inverse = jnp.asarray(inverse, rho.dtype)
    count = jnp.asarray(count, jnp.int32)

    def append(values):
        old_steps, old_differences, old_rho = values
        return (
            old_steps.at[count].set(step),
            old_differences.at[count].set(difference),
            old_rho.at[count].set(inverse),
        )

    def shift(values):
        return tuple(
            jnp.concatenate((old[1:], new[None]), axis=0)
            for old, new in zip(values, (step, difference, inverse))
        )

    return jax.lax.cond(
        count < steps.shape[0], append, shift, (steps, differences, rho)
    )


def _host_reduction(reduce):
    def apply(*args):
        # The host policy branches on these values. Transfer once, retaining the
        # reduction dtype for its subsequent scalar arithmetic (NumPy >= 2.1,
        # required by the supported JAX versions).
        return np.asarray(reduce(*args))[()]

    return apply


class JaxLBFGSExecution(JaxStrongWolfeExecution):
    """Fixed device storage mirroring histories accepted by the common policy."""

    def __init__(self, x, capacity):
        # Host scalar arithmetic matches CPU execution. CUDA scalar division
        # and interpolation retain their device rounding and trial trajectory.
        if x.device.platform == "cpu":
            for name in ("dot", "abs_max", "abs_sum"):
                setattr(
                    self, name, _host_reduction(getattr(JaxStrongWolfeExecution, name))
                )
            self.sqrt = np.sqrt
        self.steps = jnp.zeros((capacity, x.size), x.dtype, device=x.device)
        self.differences = jnp.zeros_like(self.steps)
        self.rho = jnp.zeros((capacity,), x.dtype, device=x.device)
        self.count = 0
        self.last_step = None

    def two_loop(self, gradient, steps, differences, rho, scale):
        if steps and steps[-1] is not self.last_step:
            self.steps, self.differences, self.rho = _append_history(
                self.steps,
                self.differences,
                self.rho,
                steps[-1],
                differences[-1],
                rho[-1],
                self.count,
            )
            self.count = len(steps)
            self.last_step = steps[-1]
        return _direction(
            gradient,
            self.steps,
            self.differences,
            self.rho,
            scale,
            self.count,
        )
