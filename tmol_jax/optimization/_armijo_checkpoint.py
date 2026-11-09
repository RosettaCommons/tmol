"""Validate and move immutable JAX storage around the shared optimizer state."""

import jax
import numpy as np


def checkpoint_settings(checkpoint, settings):
    if checkpoint is None:
        return settings
    if not isinstance(checkpoint, dict) or checkpoint.get("version") != 1:
        raise ValueError("unsupported Armijo checkpoint version")
    required = {"x", "segment_ids", "grad_mask", "settings", "state"}
    if not required.issubset(checkpoint):
        raise ValueError("incomplete Armijo checkpoint")
    return dict(checkpoint["settings"], **settings)


def restore_checkpoint(checkpoint, x, ids, mask, settings):
    if checkpoint is None:
        return {}
    for name, value in (("x", x), ("segment_ids", ids), ("grad_mask", mask)):
        saved = np.asarray(checkpoint[name])
        if saved.shape != value.shape or not np.array_equal(saved, np.asarray(value)):
            raise ValueError(f"Armijo checkpoint {name} does not match this execution")
    if np.asarray(checkpoint["x"]).dtype != x.dtype:
        raise ValueError("Armijo checkpoint coordinate dtype does not match")
    state = checkpoint["state"]
    capacity = min(settings["history_size"], max(1, settings["max_iter"] - 1))
    if state["old_dirs_mat"].shape[0] != capacity:
        raise ValueError("Armijo checkpoint history capacity does not match settings")
    return jax.tree.map(
        lambda value: (
            jax.device_put(value, x.device)
            if isinstance(value, (jax.Array, np.ndarray))
            else value
        ),
        state,
    )


def make_checkpoint(x, ids, mask, settings, state):
    return dict(
        version=1,
        x=x,
        segment_ids=ids,
        grad_mask=mask,
        settings=dict(settings),
        state=dict(state),
    )
