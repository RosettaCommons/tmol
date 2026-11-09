"""Periodic splines obey the cardinal basis for every supported degree."""

from fractions import Fraction
from math import comb, factorial

import numpy as np
import pytest


@pytest.fixture
def spline_backend(backend):
    if backend.framework != "jax":
        pytest.skip("General-degree interpolation transforms require the JAX API")
    from tmol_jax.numeric.bspline import bspline_coefficients, bspline_interpolate

    with backend.module.default_device(backend.device):
        yield backend.module, backend.module.numpy, bspline_coefficients, bspline_interpolate


def _cardinal(degree, x):
    # Independent truncated-power definition, evaluated exactly at rational
    # coordinates to avoid cancellation in this numerical oracle.
    if degree == 0:
        return Fraction(abs(x) < Fraction(1, 2))
    return sum(
        (-1) ** k
        * comb(degree + 1, k)
        * max(x + Fraction(degree + 1, 2) - k, 0) ** degree
        for k in range(degree + 2)
    ) / factorial(degree)


def _periodic_basis(degree, x, size):
    x = x % size
    return np.array(
        [
            float(
                sum(_cardinal(degree, x - i + image * size) for image in range(-2, 3))
            )
            for i in range(size)
        ]
    )


@pytest.mark.parametrize("degree", (2, 3, 4, 5))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_splines_match_cardinal_values_and_derivatives(spline_backend, degree, dtype):
    jax, jnp, bspline_coefficients, bspline_interpolate = spline_backend
    with jax.enable_x64():
        coefficients = np.random.default_rng(614).normal(size=(7, 8)).astype(dtype)
        points = (
            (Fraction(1, 4), Fraction(3, 8)),
            (Fraction(3, 4), Fraction(13, 8)),
            (Fraction(25, 8), Fraction(-3, 4)),
            (Fraction(-1, 8), Fraction(33, 8)),
        )
        expected_values, expected_gradients, expected_weights = [], [], []
        for point in points:
            weights, derivatives = [], []
            for x, size in zip(point, coefficients.shape):
                weights.append(_periodic_basis(degree, x, size))
                derivatives.append(
                    _periodic_basis(degree - 1, x + Fraction(1, 2), size)
                    - _periodic_basis(degree - 1, x - Fraction(1, 2), size)
                )
            expected_weights.append(np.outer(*weights))
            expected_values.append(np.sum(coefficients * expected_weights[-1]))
            expected_gradients.append(
                [
                    np.sum(coefficients * np.outer(derivatives[0], weights[1])),
                    np.sum(coefficients * np.outer(weights[0], derivatives[1])),
                ]
            )
        grid = jnp.asarray(coefficients)
        coords = jnp.asarray([[float(x) for x in row] for row in points], dtype)

        def interpolate(x):
            return bspline_interpolate(grid, x, degree)

        values, gradients = jax.jit(jax.vmap(jax.value_and_grad(interpolate)))(coords)
        tolerance = 2e-6 if dtype == np.float32 else 2e-13
        for actual, expected in (
            (values, expected_values),
            (gradients, expected_gradients),
            (interpolate(coords + jnp.array(coefficients.shape)), expected_values),
        ):
            assert actual.device == grid.device and actual.dtype == grid.dtype
            np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
        weights = jax.jit(
            jax.vmap(
                jax.grad(lambda c, x: bspline_interpolate(c, x, degree)),
                in_axes=(None, 0),
            )
        )(grid, coords)
        np.testing.assert_allclose(
            weights, expected_weights, rtol=tolerance, atol=tolerance
        )


@pytest.mark.parametrize(
    "shape,degree,seed",
    [((4,) * rank, degree, 882 + rank) for rank in (2, 3, 4) for degree in (4, 5)]
    + [(shape, 3, 712) for shape in ((2, 2), (4, 4), (8, 9), (36, 36))],
)
def test_high_degree_prefilter_reconstructs_its_periodic_grid(
    spline_backend, shape, degree, seed
):
    jax, jnp, bspline_coefficients, bspline_interpolate = spline_backend
    with jax.enable_x64():
        values = jnp.asarray(np.random.default_rng(seed).normal(size=shape))
        points = jnp.asarray(
            np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1)
        )
        restored = jax.jit(
            lambda x: bspline_interpolate(
                bspline_coefficients(x, degree), points, degree
            )
        )(values)
        np.testing.assert_allclose(restored, values, rtol=0, atol=3e-14)


@pytest.mark.parametrize("shape", [(7, 11), (3, 5, 7), (3, 4, 5, 6), (1, 7)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_periodic_coefficients_reconstruct_nodes_and_preserve_input(
    backend, shape, dtype
):
    if backend.name not in ("torch-cpu", "jax-cpu"):
        pytest.skip("Coefficient fitting is a host operation")
    if backend.framework == "jax":
        from tmol_jax.score._native import fit_bspline_coefficients
    else:
        from tmol.numeric._bspline import BSplineInterpolation

        def fit_bspline_coefficients(source):
            return BSplineInterpolation._coefficients_from_coordinates(
                backend.module.from_numpy(np.asfortranarray(source))
            ).numpy()

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


def test_coefficient_abi_rejects_unsupported_storage(spline_backend):
    from tmol_jax.score._native import fit_bspline_coefficients

    for value in (np.ones((3, 3), np.int32), np.ones((3, 3), np.float16)):
        with pytest.raises(TypeError, match="float32 or float64"):
            fit_bspline_coefficients(value)
    for shape in ((2,), (2, 0), (2, 2, 2, 2, 2)):
        with pytest.raises(ValueError, match="2-4 nonempty"):
            fit_bspline_coefficients(np.ones(shape, np.float32))
