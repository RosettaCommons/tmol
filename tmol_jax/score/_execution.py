"""Reusable compiled execution with fresh topology and parameter operands."""

from functools import partial

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

    def value_and_grad(self):
        return partial(self._apply, self.executor[2])

    def compact_objective(self, reference, selected, active):
        """Bind fresh coordinate/parameter operands to reusable minimization code."""
        operands = dict(
            reference=reference,
            selected=selected,
            active=active,
            operands=self.arrays,
        )
        return partial(self.executor[3], **operands), partial(
            self.executor[4], **operands
        )


def render_execution(cache, configuration, evaluate, states, shape):
    # Numeric arrays are runtime operands. Reach constants and constraint
    # callbacks are static, so changing either obtains a distinct executable.
    leaves, tree = jax.tree_util.tree_flatten(states)
    positions = tuple(
        i for i, leaf in enumerate(leaves) if isinstance(leaf, (jax.Array, np.ndarray))
    )
    arrays = tuple(leaves[i] for i in positions)
    template = tuple(None if i in positions else leaf for i, leaf in enumerate(leaves))
    key = (configuration, tree, template)
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

        def compact_value(values, reference, selected, active, operands):
            coords = (
                reference.reshape(-1).at[selected].set(values).reshape(reference.shape)
            )
            return execute(coords, operands)[active]

        def compact_total(values, reference, selected, active, operands):
            energies = compact_value(values, reference, selected, active, operands)
            return energies.sum(), energies

        compact_gradient = jax.value_and_grad(compact_total, has_aux=True)

        def compact_value_and_grad(values, reference, selected, active, operands):
            (_, energies), gradient = compact_gradient(
                values, reference, selected, active, operands
            )
            return energies, gradient

        executor = (
            execute,
            jax.jit(execute),
            jax.jit(value_and_grad),
            jax.jit(compact_value),
            jax.jit(compact_value_and_grad),
        )
        if cache is not None:
            # Bound retained executable configurations, including callback sets.
            if len(cache) == 8:
                del cache[next(iter(cache))]
            cache[key] = executor
    return RenderedScore(executor, arrays, shape)
