"""Compiled JAX array primitives for the common host strong-Wolfe policy."""

import jax
import jax.numpy as jnp


class JaxStrongWolfeExecution:
    # Immutable arrays can share snapshots without a device copy.
    clone = staticmethod(lambda x: x)
    dot = staticmethod(jax.jit(jnp.dot))
    abs_max = staticmethod(jax.jit(lambda x: jnp.max(jnp.abs(x))))
    abs_sum = staticmethod(jax.jit(lambda x: jnp.sum(jnp.abs(x))))
    sqrt = staticmethod(jnp.sqrt)
    add_scaled = staticmethod(
        jax.jit(lambda x, y, alpha: x + jnp.asarray(alpha, dtype=x.dtype) * y)
    )
