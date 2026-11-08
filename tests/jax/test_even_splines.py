"""Periodic splines obey the cardinal basis for every supported degree."""

from fractions import Fraction
from math import comb, factorial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tmol_jax.numeric.bspline import bspline_coefficients, bspline_interpolate


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
def test_splines_match_cardinal_values_and_derivatives(degree, dtype):
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


@pytest.mark.parametrize("rank", (2, 3, 4))
@pytest.mark.parametrize("degree", (4, 5))
def test_high_degree_prefilter_reconstructs_its_periodic_grid(rank, degree):
    with jax.enable_x64():
        shape = (4,) * rank
        values = jnp.asarray(np.random.default_rng(882 + rank).normal(size=shape))
        points = jnp.asarray(
            np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1)
        )
        restored = jax.jit(
            lambda x: bspline_interpolate(
                bspline_coefficients(x, degree), points, degree
            )
        )(values)
        np.testing.assert_allclose(restored, values, rtol=0, atol=3e-14)
