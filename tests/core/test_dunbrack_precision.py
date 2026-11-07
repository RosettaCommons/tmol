"""Probability table precision is independent of the framework's vector log."""

from decimal import Decimal, localcontext

import numpy as np

from tmol_core.scoring.dunbrack import _negative_log_probabilities


def test_negative_log_rounds_once_to_library_dtype():
    probabilities = np.array(
        [1e-6, 0.03, 0.2, 0.4, 0.97, 0.99999994, 1.0], dtype=np.float32
    )
    original = probabilities.copy()
    with localcontext() as context:
        context.prec = 60
        expected = np.array(
            [-float(Decimal(float(p)).ln()) for p in probabilities],
            dtype=probabilities.dtype,
        )
    actual = _negative_log_probabilities(probabilities)
    assert actual.dtype == probabilities.dtype
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(probabilities, original)
    # Preserve explicit double-precision custom libraries as well.
    double = probabilities.astype(np.float64)
    actual_double = _negative_log_probabilities(double)
    assert actual_double.dtype == double.dtype
    with localcontext() as context:
        context.prec = 60
        expected_double = np.array([-float(Decimal(p).ln()) for p in double])
    np.testing.assert_array_max_ulp(actual_double, expected_double, maxulp=1)
