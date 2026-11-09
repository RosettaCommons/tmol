"""Compilation contracts shared by JAX execution adapters."""

from functools import lru_cache, wraps

import jax


def _jit(function, *, compiler_options, static_argnames=()):
    compiled = jax.jit(
        function,
        static_argnames=static_argnames,
        compiler_options=compiler_options,
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

    execute._tmol_jit_options = dict(
        compiler_options=compiler_options, static_argnames=static_argnames
    )
    return execute


def jit_without_autotuning(function, *, static_argnames=()):
    """Compile without GPU autotuning; enclosing transforms own compilation."""
    return _jit(
        function,
        static_argnames=static_argnames,
        compiler_options={"xla_gpu_autotune_level": 0},
    )


@lru_cache(maxsize=64)
def with_cuda_graph(function, enabled):
    """Select XLA command-buffer execution without caching numeric operands."""
    settings = dict(getattr(function, "_tmol_jit_options", {}))
    options = dict(settings.pop("compiler_options", {}))
    options["xla_gpu_enable_command_buffer"] = (
        "FUSION,CUSTOM_CALL,CUBLAS,CUDNN,WHILE,CONDITIONAL" if enabled else ""
    )
    if enabled:
        options["xla_gpu_graph_min_graph_size"] = 1
    return _jit(
        getattr(function, "__wrapped__", function),
        compiler_options=options,
        **settings,
    )
