"""The shared dynamic sampler preserves explicit chemical chi choices."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tmol_jax.database import ParameterDatabase
from tmol_jax.pack import DunbrackChiSampler
from tmol_jax.pack._native import sample_chi_cpu


@pytest.fixture(scope="module")
def tables():
    return DunbrackChiSampler.from_database(ParameterDatabase.get_default()).tables


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_libraryless_chi_choices_counts_identity_and_budget(tables, dtype):
    coordinates = np.zeros((4, 3), dtype)
    angles = np.asarray([-np.pi / 3, np.pi / 3, np.pi], dtype)
    request = (
        np.array([2], np.int32),
        np.array([0], np.int32),
        np.full((2, 4), -1, np.int32),
        np.array([[0, -1]], np.int32),
        np.zeros((1, 1), np.int32),
        angles.reshape(1, 1, 3),
        np.array([[3]], np.int32),
        np.array([0.98], dtype),
        np.array([1], np.int32),
    )
    with jax.enable_x64():
        counts, offsets, identities, chi = sample_chi_cpu(tables, coordinates, *request)
        np.testing.assert_array_equal(counts, [3])
        np.testing.assert_array_equal(offsets, [0])
        np.testing.assert_array_equal(identities, [0, 0, 0])
        np.testing.assert_array_equal(chi[:, 0], angles)
        assert chi.dtype == dtype
        assert all(
            value.device.platform == "cpu"
            for value in (counts, offsets, identities, chi)
        )
        with pytest.raises(ValueError, match="budget|limit"):
            sample_chi_cpu(tables, coordinates, *request, max_samples_per_restype=2)
        # Dynamic output sizes cannot be silently traced as a fixed-size kernel.
        with pytest.raises(TypeError, match="eager"):
            jax.jit(lambda x: sample_chi_cpu(tables, x, *request))(
                jnp.asarray(coordinates)
            )
