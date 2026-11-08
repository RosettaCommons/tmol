"""Standalone differentiable periodic B-spline utilities.

:mod:`tmol_jax.numeric.bspline` implements Thevenaz/Blu/Unser B-spline
interpolation on regular periodic grids. Molecular backbone-torsion and
Dunbrack terms use shared native splines. These utilities support JIT and AD.
"""

from tmol_jax.numeric.bspline import (
    bspline_coefficients,
    bspline_interpolate,
)

__all__ = ["bspline_coefficients", "bspline_interpolate"]
