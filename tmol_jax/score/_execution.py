"""Reusable compiled execution with fresh topology and parameter operands."""

from functools import partial, wraps

import jax
import jax.numpy as jnp
import numpy as np


class RenderedScore:
    def __init__(self, executor, arrays, shape):
        self.executor = executor
        self.arrays = arrays
        self.shape = shape

    def _apply(self, function, coords):
        if coords.shape != self.shape or not jnp.issubdtype(coords.dtype, jnp.floating):
            raise ValueError(f"expected floating coordinates with shape {self.shape}")
        return function(coords, self.arrays)

    def __call__(self, coords):
        return self._apply(self.executor[0], coords)

    def jitted(self):
        return partial(self._apply, self.executor[1])

    def value_and_grad(self, *, coordinate_map=None, coordinate_dtype=None):
        """Return energies and source gradients, optionally through a prepared map.

        An Atom37 binding supplies fixed chemistry and fresh numeric operands.
        Executable reuse never caches its coordinates or score parameters.
        ``coordinate_dtype`` casts mapped coordinates before scoring while source
        gradients retain the input precision, as with an explicit JAX cast.
        """
        if coordinate_map is None:
            if coordinate_dtype is not None:
                raise ValueError("coordinate_dtype requires a coordinate_map")
            return partial(self._apply, self.executor[2])
        if coordinate_dtype is not None:
            coordinate_dtype = np.dtype(coordinate_dtype)
            if coordinate_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
                raise TypeError("mapped score coordinates require float32/64")
        transform, state = coordinate_map._coordinate_execution()
        evaluate = self.executor[0]

        def score(source, operands):
            mapping, parameters = operands
            coords = transform(source, mapping)
            if coordinate_dtype is not None:
                coords = coords.astype(coordinate_dtype)
            return evaluate(coords, parameters)

        rendered = render_execution(
            self.executor[5],
            (transform, coordinate_dtype),
            score,
            (state, self.arrays),
            coordinate_map.input_shape,
            native_only=False,
        )
        gradient = rendered.value_and_grad()

        def execute(source):
            coordinate_map._validate(source)
            if coordinate_dtype == np.dtype(np.float64) and not jax.config.x64_enabled:
                raise ValueError("float64 mapped scoring requires JAX x64 enabled")
            return gradient(source)

        return execute

    def compact_objective(self, reference, selected, active, coordinate_map=None):
        """Bind fresh coordinate/parameter operands to reusable minimization code."""
        operands = dict(
            reference=reference,
            selected=selected,
            active=active,
            operands=self.arrays,
            coordinate_map=coordinate_map,
        )
        return partial(self.executor[3], **operands), partial(
            self.executor[4], **operands
        )


class WholePoseScorer(RenderedScore):
    """Reuse prepared operands when callers select a different score reduction."""

    def __init__(self, render, sum_terms, apply_weights):
        default = render(sum_terms, apply_weights)
        super().__init__(default.executor, default.arrays, default.shape)
        self._default_options = (sum_terms, apply_weights)
        self._render_variant = render
        self._variants = {}

    def __call__(self, coords, sum_terms=None, apply_weights=None):
        options = tuple(
            default if value is None else value
            for default, value in zip(self._default_options, (sum_terms, apply_weights))
        )
        if options == self._default_options:
            return super().__call__(coords)
        if options not in self._variants:
            self._variants[options] = self._render_variant(*options)
        return self._variants[options](coords)

    def unweighted_scores(self, coords):
        """Return all retained channels in the rendered score-type order."""
        return self(coords, sum_terms=False, apply_weights=False)


def _jit_native(function):
    # These graphs dispatch fixed native physics kernels. GPU autotuning adds
    # large compilation-only allocations without choosing their algorithms.
    compiled = jax.jit(function, compiler_options={"xla_gpu_autotune_level": 0})

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


def render_execution(
    cache,
    configuration,
    evaluate,
    states,
    shape,
    *,
    native_only,
    compile_gradient=True,
):
    # Numeric arrays are runtime operands. Reach constants and constraint
    # callbacks are static, so changing either obtains a distinct executable.
    leaves, tree = jax.tree_util.tree_flatten(states)
    positions = tuple(
        i for i, leaf in enumerate(leaves) if isinstance(leaf, (jax.Array, np.ndarray))
    )
    arrays = tuple(leaves[i] for i in positions)
    template = tuple(None if i in positions else leaf for i, leaf in enumerate(leaves))
    key = (configuration, tree, template, native_only, compile_gradient)
    executor = cache.get(key) if cache is not None else None
    if executor is None:

        def execute(coords, operands):
            restored = list(template)
            for i, operand in zip(positions, operands):
                restored[i] = operand
            return evaluate(coords, tree.unflatten(restored))

        def value(coords, operands):
            energies = execute(coords, operands)
            return energies.sum(), energies

        differentiated = jax.value_and_grad(value, has_aux=True)

        def value_and_grad(coords, operands):
            (_, energies), gradient = differentiated(coords, operands)
            return energies, gradient

        def compact_value(
            values, reference, selected, active, operands, coordinate_map
        ):
            coords = (
                reference.reshape(-1).at[selected].set(values).reshape(reference.shape)
            )
            if coordinate_map is not None:
                coords = coordinate_map(coords)
            return execute(coords, operands)[active]

        def compact_total(
            values, reference, selected, active, operands, coordinate_map
        ):
            energies = compact_value(
                values, reference, selected, active, operands, coordinate_map
            )
            return energies.sum(), energies

        compact_gradient = jax.value_and_grad(compact_total, has_aux=True)

        def compact_value_and_grad(
            values, reference, selected, active, operands, coordinate_map
        ):
            (_, energies), gradient = compact_gradient(
                values, reference, selected, active, operands, coordinate_map
            )
            return energies, gradient

        compile_execution = _jit_native if native_only else jax.jit
        executor = (
            execute,
            compile_execution(execute),
            compile_execution(value_and_grad) if compile_gradient else value_and_grad,
            compile_execution(compact_value),
            compile_execution(compact_value_and_grad),
            {},  # Coordinate-map executables share this scorer's bounded lifetime.
        )
        if cache is not None:
            # Bound retained executable configurations, including callback sets.
            if len(cache) == 8:
                del cache[next(iter(cache))]
            cache[key] = executor
    return RenderedScore(executor, arrays, shape)
