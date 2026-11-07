"""Prepared Cartesian execution keeps live molecular operands out of code caches."""

import os

import jax
import jax.numpy as jnp
import numpy as np

from tests.scientific.execution import Execution
from tests.scientific.molecules import DOMAIN_CASES
from tmol_core.scoring.elec import partial_charge_table, partial_charges_for_block
from tmol_jax.optimization import armijo_minimize


def device_name():
    return (
        "jax-cuda"
        if "jax-cuda" in os.environ.get("TMOL_TEST_BACKENDS", "")
        else "jax-cpu"
    )


def test_compact_execution_uses_fresh_coordinate_masks_and_parameter_snapshots():
    b = Execution(device_name()).molecule(
        DOMAIN_CASES["mixed_batch"], terms=("fa_elec",)
    )
    score, pose = b.score_function, b.pose
    types = pose.packed_block_types.active_block_types
    previous = [getattr(bt, "elec_partial_charges", None) for bt in types]
    table = partial_charge_table(b.parameter_database.scoring.elec)
    with jax.default_device(b.device), jax.enable_x64():
        x = pose.coords.astype(jnp.float64)
        active = jax.device_put(np.array([0], np.int32), b.device)
        original = score.render_whole_pose_scoring_module(pose)
        selected = jax.device_put(
            np.arange(0, x.shape[1] * 3, 3, dtype=np.int32), b.device
        )
        value, derivative = original.compact_objective(x, selected, active)
        expected = derivative(x.reshape(-1)[selected])
        assert np.linalg.norm(expected[0]) > 1e-3 and np.linalg.norm(expected[1]) > 1e-3
        try:
            for bt, override in zip(types, previous):
                charges = partial_charges_for_block(bt, table, overrides=override)
                bt.elec_partial_charges = {
                    atom.name: float(2 * q) for atom, q in zip(bt.atoms, charges)
                }
            changed = score.render_whole_pose_scoring_module(pose)
            _, updated = changed.compact_objective(x, selected, active)
            for actual, before in zip(updated(x.reshape(-1)[selected]), expected):
                np.testing.assert_allclose(actual, 4 * before, rtol=2e-10, atol=2e-10)
            for actual, before in zip(derivative(x.reshape(-1)[selected]), expected):
                np.testing.assert_allclose(actual, before, rtol=2e-10, atol=2e-10)
            # Same cardinality, different components and a fresh non-rigid reference.
            selected = selected + 1
            reference = x * 1.01
            function, gradient = changed.compact_objective(reference, selected, active)
            compact = reference.reshape(-1)[selected]
            direction = jax.device_put(
                np.random.default_rng(127).normal(size=compact.shape), b.device
            )
            h = 1e-5
            numeric = (
                function(compact + h * direction).sum()
                - function(compact - h * direction).sum()
            ) / (2 * h)
            analytic = (gradient(compact)[1] * direction).sum()
            np.testing.assert_allclose(numeric, analytic, rtol=2e-6, atol=2e-6)
        finally:
            for bt, override in zip(types, previous):
                if override is None:
                    del bt.elec_partial_charges
                else:
                    bt.elec_partial_charges = override


def test_prepared_objective_callbacks_receive_the_original_parameter_shape():
    device = Execution(device_name()).device
    with jax.default_device(device):
        x = jax.device_put(
            np.array([[2.0, -1.0, 3.0], [-3.0, 1.0, 2.0]], np.float32), device
        )
        ids = jax.device_put(np.array([[0] * 3, [1] * 3], np.int32), device)
        observed = []

        def objective(value):
            return ((value - 0.2) ** 2).sum(1)

        compiled = jax.jit(objective)

        def value_only(value):
            observed.append(value.shape)
            return compiled(value)

        result = armijo_minimize(
            objective,
            x,
            segment_ids=ids,
            value_only=value_only,
            value_only_line_search=True,
            max_iter=3,
            history_size=2,
            fixed_iterations=True,
        )
        assert observed and all(shape == x.shape for shape in observed)
        assert np.isfinite(result["x"]).all()
        assert objective(result["x"]).sum() < objective(x).sum()
