"""Pure-JAX port of the ``tmol.numeric`` B-spline interpolation.

:mod:`tmol_jax.numeric.bspline` implements Thevenaz/Blu/Unser B-spline
interpolation on regular periodic grids, as used by the backbone-torsion and
Dunbrack terms. It is jit-safe and differentiable via :func:`jax.grad`.
"""

from tmol_jax.numeric.bspline import (
    bspline_coefficients,
    bspline_interpolate,
)

__all__ = ["bspline_coefficients", "bspline_interpolate"]
