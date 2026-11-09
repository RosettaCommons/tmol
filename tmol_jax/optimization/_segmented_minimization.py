"""Bind compact movable variables and per-pose objectives to mature Armijo."""

import jax
import jax.numpy as jnp
import numpy as np

from .armijo import armijo_minimize


@jax.jit
def _finite_fields(*values):
    return jnp.stack([jnp.isfinite(value).all() for value in values])


def minimize_pose_variables(
    energy,
    initial,
    mask,
    pose_for_variable,
    settings,
    coordinate_scorer=None,
    coordinate_map=None,
    minimize=armijo_minimize,
):
    """Optimize only movable elements, retaining fixed poses and padding."""
    shape = initial.shape
    selected = np.flatnonzero(np.asarray(mask).reshape(-1))
    flat = initial.reshape(-1)
    if not len(selected):
        values = energy(initial)
        return dict(
            x=initial,
            initial_fun=values.sum(),
            fun=values.sum(),
            grad=jnp.zeros_like(initial),
            n_iter=0,
            func_evals=1,
            converged=jnp.ones(values.shape, bool),
            stalled=jnp.zeros(values.shape, bool),
        )
    poses = np.asarray(pose_for_variable).reshape(-1)[selected]
    if np.any(poses < 0):
        raise ValueError("movable variables require a physical pose identity")
    active, ids = np.unique(poses, return_inverse=True)
    with jax.default_device(initial.device):
        selected = jax.device_put(selected.astype(np.int32), initial.device)
        active = jax.device_put(active.astype(np.int32), initial.device)

        def expand(values):
            return flat.at[selected].set(values).reshape(shape)

        def fallback_objective(values):
            return energy(expand(values))[active]

        objective = fallback_objective
        execution = {}
        if coordinate_scorer is not None:
            objective, derivative = coordinate_scorer.compact_objective(
                initial.reshape(
                    coordinate_scorer.shape
                    if coordinate_map is None
                    else coordinate_map.initial_dofs.shape
                ),
                selected,
                active,
                coordinate_map,
            )
            execution = dict(value_only=objective, value_and_grad=derivative)
        result = minimize(
            objective,
            flat[selected],
            segment_ids=jnp.asarray(ids, jnp.int32),
            **execution,
            **settings,
        )
        if minimize is not armijo_minimize:
            result = dict(result)
            checked = []
            for name in ("initial_fun", "fun"):
                value = jnp.asarray(result[name])
                if value.shape:
                    raise ValueError(f"optimizer {name} must be a scalar")
                if value.device != initial.device:
                    value = jax.device_put(value, initial.device)
                checked.append(value)
            for name in ("x", "grad"):
                value = result[name]
                if not isinstance(value, jax.Array):
                    raise TypeError(f"optimizer {name} must be a JAX array")
                if value.shape != selected.shape or value.dtype != initial.dtype:
                    raise ValueError(
                        f"optimizer {name} must preserve variable shape and dtype"
                    )
                if value.device != initial.device:
                    raise ValueError(
                        f"optimizer {name} must remain on the input device"
                    )
                checked.append(value)
            # Transfer only the four validation flags together. Coordinate and
            # gradient arrays stay on their original device and retain dtype.
            for name, finite in zip(
                ("initial_fun", "fun", "x", "grad"),
                np.asarray(_finite_fields(*checked)),
            ):
                if not finite:
                    raise FloatingPointError(f"optimizer {name} must be finite")
        result["x"] = expand(result["x"])
        result["grad"] = (
            jnp.zeros_like(flat).at[selected].set(result["grad"]).reshape(shape)
        )
        result["active_poses"] = active
        return result
