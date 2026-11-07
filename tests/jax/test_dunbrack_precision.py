"""Rendered rotamer objectives retain precision across mixed coordinate calls."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.scientific.dunbrack_molecules import DUNBRACK_CASES
from tests.scientific.execution import Execution
from tests.scientific.test_dunbrack import TERMS


@pytest.mark.parametrize("name", ("d_peptide", "mixed_batch"))
def test_rotamer_preparation_preserves_cross_dtype_values_and_gradients(name):
    backend = (
        "jax-cuda"
        if "jax-cuda" in os.environ.get("TMOL_TEST_BACKENDS", "")
        else "jax-cpu"
    )
    b = Execution(backend).molecule(DUNBRACK_CASES[name], terms=TERMS)
    with jax.default_device(b.device), jax.enable_x64():
        x32 = b.pose.coords.astype(jnp.float32)
        x64 = x32.astype(jnp.float64)
        original = b.score_function.jitted_value_and_grad(b.pose.replace(coords=x32))
        expected = original(x64)
        assert expected[0].dtype == expected[1].dtype == jnp.float64
        assert np.linalg.norm(expected[1]) > 1e-3
        prepared = b.score_function.jitted_value_and_grad(b.pose.replace(coords=x64))
        for actual, reference in zip(prepared(x64), expected):
            np.testing.assert_allclose(actual, reference, rtol=2e-10, atol=2e-10)
        # A rendered scorer can still execute either coordinate dtype. Returning
        # to single precision must retain the source tables and old snapshots.
        for actual, reference in zip(prepared(x32), original(x32)):
            assert actual.dtype == reference.dtype == jnp.float32
            np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-5)
        restored = b.score_function.jitted_value_and_grad(b.pose.replace(coords=x32))
        for actual, reference in zip(restored(x64), expected):
            np.testing.assert_allclose(actual, reference, rtol=2e-10, atol=2e-10)
