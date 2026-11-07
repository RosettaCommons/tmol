"""Checkpoint storage roundtrips without changing the optimizer's parameter context."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.scientific.execution import Execution
from tmol_jax.optimization import armijo_minimize


def test_checkpoint_host_roundtrip_and_context_mismatch_rejection():
    name = (
        "jax-cuda"
        if "jax-cuda" in os.environ.get("TMOL_TEST_BACKENDS", "")
        else "jax-cpu"
    )
    device = Execution(name).device
    with jax.default_device(device), jax.enable_x64():

        def objective(x):
            return jnp.sum((x - 0.4) ** 4 + 0.3 * x**2)

        x = jax.device_put(np.array([2.0, -1.0, 3.0], np.float64), device)
        first = armijo_minimize(
            objective, x, max_iter=3, history_size=2, fixed_iterations=True
        )
        checkpoint = first["checkpoint"]
        host = jax.tree.map(
            lambda value: np.asarray(value) if isinstance(value, jax.Array) else value,
            checkpoint,
        )
        copied = armijo_minimize(objective, first["x"], checkpoint=host)
        direct = armijo_minimize(objective, first["x"], checkpoint=checkpoint)
        np.testing.assert_array_equal(copied["x"], direct["x"])
        assert copied["x"].dtype == jnp.float64 and copied["x"].device == device
        for value in jax.tree.leaves(copied["state"]):
            if isinstance(value, jax.Array):
                assert value.device == device
        for altered, options, message in (
            (first["x"] + 0.1, {}, "checkpoint x"),
            (first["x"], {"segment_ids": jnp.array([0, 1, 0])}, "segment_ids"),
            (first["x"], {"grad_mask": jnp.array([True, False, True])}, "grad_mask"),
            (first["x"], {"history_size": 1}, "history capacity"),
        ):
            with pytest.raises(ValueError, match=message):
                armijo_minimize(objective, altered, checkpoint=host, **options)
