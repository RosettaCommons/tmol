"""Bind compact movable variables and per-pose objectives to mature Armijo."""

import jax
import jax.numpy as jnp
import numpy as np

from .armijo import armijo_minimize


def minimize_pose_variables(
    energy, initial, mask, pose_for_variable, settings, coordinate_scorer=None
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
                initial.reshape(coordinate_scorer.shape), selected, active
            )
            execution = dict(value_only=objective, value_and_grad=derivative)
        result = armijo_minimize(
            objective,
            flat[selected],
            segment_ids=jnp.asarray(ids, jnp.int32),
            **execution,
            **settings,
        )
        result["x"] = expand(result["x"])
        result["grad"] = (
            jnp.zeros_like(flat).at[selected].set(result["grad"]).reshape(shape)
        )
        result["active_poses"] = active
        return result
