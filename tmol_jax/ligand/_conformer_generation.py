"""JAX execution of the shared ligand distance-geometry objectives."""

from contextlib import nullcontext
from functools import partial, wraps

import jax
import jax.numpy as jnp
import numpy as np

from tmol_core.ligand import refinement
from tmol_core.ligand._conformer_generation import (
    REFINE_ITERS,
    STAGE_A_ANNEAL,
    STAGE_A_ITERS,
)
from tmol_core.optimization.strong_wolfe import lbfgs_minimize
from tmol_jax.optimization._strong_wolfe_execution import JaxStrongWolfeExecution


class _JaxRefinement:
    gather = staticmethod(lambda coordinates, indices: coordinates[indices])
    clamp_min = staticmethod(
        lambda values, minimum: jnp.where(values < minimum, minimum, values)
    )
    relu = staticmethod(jax.nn.relu)
    sqrt = staticmethod(jnp.sqrt)
    cross = staticmethod(jnp.cross)
    eigvalsh = staticmethod(jnp.linalg.eigvalsh)


jax.tree_util.register_dataclass(
    refinement.RefinementParameters,
    data_fields=[
        "pair_i",
        "pair_j",
        "target",
        "lower",
        "upper",
        "components",
        "chiral_indices",
        "chiral_sign",
    ],
    meta_fields=["n_exact", "n_lower"],
)


_OPERATIONS = _JaxRefinement()
_OPTIMIZER = JaxStrongWolfeExecution()
flush_denormals = nullcontext


def _parameters(L0, U0, L, U, components, chirals):
    return refinement.prepare_refinement(
        L0,
        U0,
        L,
        U,
        components,
        chirals,
        index_array=lambda value: jnp.asarray(value, dtype=jnp.int32),
        float_array=lambda value: jnp.asarray(value, dtype=jnp.float32),
    )


@partial(jax.jit, static_argnames=("shape", "fourth_dimension"))
def _refinement_value_and_grad(flat, parameters, *, shape, fourth_dimension):
    return jax.value_and_grad(
        lambda x: refinement.refinement_loss(
            x.reshape(shape),
            parameters,
            _OPERATIONS,
            fourth_dimension=fourth_dimension,
        )
    )(flat)


@partial(jax.jit, static_argnames=("shape",))
def _planar_value_and_grad(flat, first, second, target, *, shape):
    return jax.value_and_grad(
        lambda x: refinement.planar_loss(
            x.reshape(shape),
            first,
            second,
            target,
            _OPERATIONS,
        )
    )(flat)


def _minimize(evaluate, coordinates, iterations):
    shape = coordinates.shape
    result = lbfgs_minimize(
        evaluate,
        coordinates.reshape(-1),
        _OPTIMIZER,
        max_iter=iterations,
        history_size=50,
    )
    output = result["x"].reshape(shape)
    if not np.isfinite(np.asarray(output)).all():
        raise FloatingPointError("ligand refinement produced nonfinite coordinates")
    return output


def _host_refinement(function):
    @wraps(function)
    def execute(*args, **kwargs):
        # These coordinates become shared chemical parameters, including ideal
        # hydrogen geometry. They must not depend on the eventual scoring device
        # or the device active when a seeded conformer cache is first populated.
        with jax.default_device(jax.devices("cpu")[0]):
            return function(*args, **kwargs)

    return execute


@_host_refinement
def planar_refine(coordinates, pair_targets, max_iters):
    # Planar target construction in the mature pipeline explicitly uses doubles,
    # even when the eventual molecular coordinates and scoring use float32.
    with jax.enable_x64():
        first = jnp.asarray([p[0] for p in pair_targets], dtype=jnp.int32)
        second = jnp.asarray([p[1] for p in pair_targets], dtype=jnp.int32)
        target = jnp.asarray([p[2] for p in pair_targets], dtype=jnp.float64)
        result = _minimize(
            lambda flat: _planar_value_and_grad(
                flat, first, second, target, shape=coordinates.shape
            ),
            jnp.asarray(coordinates, dtype=jnp.float64),
            max_iters,
        )
        return np.array(result, copy=True)


@_host_refinement
def chiral_anneal(coordinates, L0, U0, L, U, components, chirals, rng_seed):
    parameters = _parameters(L0, U0, L, U, components, chirals)
    fourth = np.random.default_rng(rng_seed).normal(0, 0.5, (len(coordinates), 1))
    x = jnp.asarray(np.hstack([coordinates, fourth]), dtype=jnp.float32)
    for weight in STAGE_A_ANNEAL:
        x = _minimize(
            lambda flat: _refinement_value_and_grad(
                flat, parameters, shape=x.shape, fourth_dimension=weight
            ),
            x,
            STAGE_A_ITERS,
        )
    return np.array(x[:, :3], copy=True)


@_host_refinement
def stress_refine(coordinates, L0, U0, L, U, components, chirals):
    parameters = _parameters(L0, U0, L, U, components, chirals)
    result = _minimize(
        lambda flat: _refinement_value_and_grad(
            flat, parameters, shape=coordinates.shape, fourth_dimension=None
        ),
        jnp.asarray(coordinates, dtype=jnp.float32),
        REFINE_ITERS,
    )
    return np.array(result, copy=True)
