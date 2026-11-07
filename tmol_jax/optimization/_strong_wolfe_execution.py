"""Compiled JAX array primitives for the common host strong-Wolfe policy."""

import jax
import jax.numpy as jnp

from tmol_core.optimization.lbfgs import two_loop_direction
from tmol_jax._compilation import jit_without_autotuning


class JaxStrongWolfeExecution:
    # Immutable arrays can share snapshots without a device copy.
    clone = staticmethod(lambda x: x)
    dot = staticmethod(jax.jit(jnp.dot))
    abs_max = staticmethod(jax.jit(lambda x: jnp.max(jnp.abs(x))))
    # The initial step depends on this reduction. Avoid autotuning-dependent
    # reduction order while preserving the input dtype and common policy.
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
    return two_loop_direction(
        gradient, steps, differences, rho, scale, count, _LoopExecution()
    )


@jax.jit
def _append_history(steps, differences, rho, step, difference, inverse, count):
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


class JaxLBFGSExecution(JaxStrongWolfeExecution):
    """Fixed device storage mirroring histories accepted by the common policy."""

    def __init__(self, x, capacity):
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
                jnp.asarray(rho[-1], gradient.dtype, device=gradient.device),
                jnp.asarray(self.count, jnp.int32, device=gradient.device),
            )
            self.count = len(steps)
            self.last_step = steps[-1]
        return _direction(
            gradient,
            self.steps,
            self.differences,
            self.rho,
            jnp.asarray(scale, gradient.dtype, device=gradient.device),
            jnp.asarray(self.count, jnp.int32, device=gradient.device),
        )
