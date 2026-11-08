"""Compilation contracts shared by JAX execution adapters."""

from functools import wraps

import jax


def jit_without_autotuning(function, *, static_argnames=()):
    """Compile without GPU autotuning; enclosing transforms own compilation."""
    compiled = jax.jit(
        function,
        static_argnames=static_argnames,
        compiler_options={"xla_gpu_autotune_level": 0},
    )

    @wraps(function)
    def execute(*args, **kwargs):
        # Compiler options belong to the outermost jit. An enclosing user
        # transformation owns compilation and receives the ordinary array graph.
        if any(
            isinstance(value, jax.core.Tracer)
            for value in jax.tree_util.tree_leaves((args, kwargs))
        ):
            return function(*args, **kwargs)
        return compiled(*args, **kwargs)

    return execute
