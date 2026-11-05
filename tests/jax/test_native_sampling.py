"""The shared dynamic sampler preserves explicit chemical chi choices."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tmol_jax.database import ParameterDatabase
from tmol_jax.pack import DunbrackChiSampler
from tmol_jax.pack._native import sample_chi, sample_chi_cpu


@pytest.fixture(scope="module")
def tables():
    return DunbrackChiSampler.from_database(ParameterDatabase.get_default()).tables


def test_sampling_does_not_silently_downcast_coordinates(tables):
    with jax.enable_x64(False):
        with pytest.raises(ValueError, match="preserve output dtype"):
            sample_chi_cpu(tables, np.zeros((4, 3), np.float64))
    with pytest.raises(TypeError, match="float32 or float64"):
        sample_chi_cpu(tables, np.zeros((4, 3), np.float16))


@pytest.fixture(params=("cpu", "gpu"))
def sampling_device(request):
    try:
        return jax.devices(request.param)[0]
    except RuntimeError:
        pytest.skip(f"{request.param} device unavailable")


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_libraryless_chi_choices_counts_identity_and_budget(
    tables, dtype, sampling_device
):
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
        coordinates = jax.device_put(coordinates, sampling_device)
        counts, offsets, identities, chi = sample_chi(tables, coordinates, *request)
        np.testing.assert_array_equal(counts, [3])
        np.testing.assert_array_equal(offsets, [0])
        np.testing.assert_array_equal(identities, [0, 0, 0])
        np.testing.assert_array_equal(chi[:, 0], angles)
        assert chi.dtype == dtype
        assert all(
            value.device == sampling_device
            for value in (counts, offsets, identities, chi)
        )
        with pytest.raises(ValueError, match="budget|limit"):
            sample_chi(tables, coordinates, *request, max_samples_per_restype=2)
        # Dynamic output sizes cannot be silently traced as a fixed-size kernel.
        with pytest.raises(TypeError, match="eager"):
            jax.jit(lambda x: sample_chi(tables, x, *request))(jnp.asarray(coordinates))


def test_real_library_mirrors_and_int64_transport(tables, sampling_device):
    # Two target chiralities share missing backbone coordinates. The library
    # orientation resolves the defaults independently for the same physical row.
    indices = [tables.all_table_indices[name] for name in ("ARG", "DARG")]
    request = (
        np.array([2], np.int32),
        np.array([0], np.int32),
        np.full((2, 4), -1, np.int32),
        np.array([[0, index] for index in indices], np.int32),
        np.zeros((2, 4), np.int32),
        np.zeros((2, 4, 1), np.float32),
        np.ones((2, 4), np.int32),
        np.full(2, 0.95, np.float32),
        np.full(2, 4, np.int32),
    )
    # Metadata remain int64 without changing the user's floating precision mode.
    with jax.enable_x64(False):
        coordinates = jax.device_put(np.zeros((1, 3), np.float32), sampling_device)
        count, offsets, identities, chi = sample_chi(tables, coordinates, *request)
        count, offsets, identities, chi = map(
            np.asarray, (count, offsets, identities, chi)
        )
        assert count[0] == count[1] and count[0] > 1
        np.testing.assert_array_equal(offsets, [0, count[0]])
        np.testing.assert_array_equal(identities, np.repeat([0, 1], count))
        np.testing.assert_allclose(chi[: count[0]], -chi[count[0] :], rtol=0, atol=2e-6)
        assert chi.dtype == np.float32 and not jax.config.x64_enabled


def test_empty_sampling_request_preserves_shapes(tables, sampling_device):
    request = (
        np.empty(0, np.int32),
        np.empty(0, np.int32),
        np.empty((0, 4), np.int32),
        np.empty((0, 2), np.int32),
        np.empty((0, 0), np.int32),
        np.empty((0, 0, 0), np.float32),
        np.empty((0, 0), np.int32),
        np.empty(0, np.float32),
        np.empty(0, np.int32),
    )
    result = sample_chi(
        tables, jax.device_put(np.empty((0, 3), np.float32), sampling_device), *request
    )
    assert tuple(x.shape for x in result) == ((0,), (0,), (0,), (0, 0))
    assert all(x.device == sampling_device for x in result)
