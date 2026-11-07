"""Framework-neutral NumPy metadata operations."""

import numpy
from typing import Optional


def invert_mapping(
    a_2_b: numpy.ndarray,
    n_elements_b: Optional[int] = None,
    sentinel: Optional[int] = -1,
):
    """Create the inverse mapping, b_2_a, given the input mapping, a_2_b"""
    if n_elements_b is None:
        n_elements_b = numpy.max(a_2_b) + 1

    b_2_a = numpy.full((n_elements_b,), sentinel, dtype=a_2_b.dtype)

    b_2_a[a_2_b] = numpy.arange(a_2_b.shape[0], dtype=a_2_b.dtype)
    return b_2_a
