"""The host ABI preserves data and reconstructs periodic interpolation nodes."""

import numpy as np
import pytest

from tmol_jax.score._native import fit_bspline_coefficients


@pytest.mark.parametrize("shape", [(7, 11), (3, 5, 7), (3, 4, 5, 6), (1, 7)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_periodic_coefficients_reconstruct_nodes_and_preserve_input(shape, dtype):
    source = np.random.default_rng(934).normal(size=shape).astype(dtype)
    source = source[..., ::-1]  # Noncontiguous host records are accepted.
    original = source.copy()
    coefficients = fit_bspline_coefficients(source)
    assert coefficients.dtype == dtype and coefficients.flags.c_contiguous
    assert not np.shares_memory(source, coefficients)
    np.testing.assert_array_equal(source, original)
    reconstructed = coefficients
    for axis in range(source.ndim):
        reconstructed = (
            np.roll(reconstructed, 1, axis)
            + 4 * reconstructed
            + np.roll(reconstructed, -1, axis)
        ) / 6
    tolerance = 3e-6 if dtype == np.float32 else 3e-13
    np.testing.assert_allclose(reconstructed, source, atol=tolerance, rtol=tolerance)


def test_coefficient_abi_rejects_unsupported_storage():
    for value in (np.ones((3, 3), np.int32), np.ones((3, 3), np.float16)):
        with pytest.raises(TypeError, match="float32 or float64"):
            fit_bspline_coefficients(value)
    for shape in ((2,), (2, 0), (2, 2, 2, 2, 2)):
        with pytest.raises(ValueError, match="2-4 nonempty"):
            fit_bspline_coefficients(np.ones(shape, np.float32))
