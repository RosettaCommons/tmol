"""Compatibility entry points for the shared strong-Wolfe search policy."""

import jax.numpy as jnp

from tmol_core.optimization.strong_wolfe import _cubic_interpolate, strong_wolfe
from ._strong_wolfe_execution import JaxStrongWolfeExecution

_EXECUTION = JaxStrongWolfeExecution()


def _cubic_interp(a_lo, f_lo, g_lo, a_hi, f_hi, g_hi):
    return float(_cubic_interpolate(_EXECUTION, a_lo, f_lo, g_lo, a_hi, f_hi, g_hi))


def strong_wolfe_line_search(
    f_and_grad, x, direction, f0, g0, alpha0=1.0, c1=1e-4, c2=0.9, max_iter=25
):
    """Return (step, value, gradient) with the shared bounded Wolfe search."""
    if max_iter < 1 or not 0 < c1 < c2 < 1:
        raise ValueError("expected a positive search budget and 0 < c1 < c2 < 1")
    derivative = _EXECUTION.dot(g0, direction)
    if not derivative < 0:
        return 0.0, f0, g0

    def evaluate(base, step, vector):
        value, gradient = f_and_grad(_EXECUTION.add_scaled(base, vector, step))
        finite = bool(jnp.isfinite(value) & jnp.isfinite(gradient).all())
        return float(value) if finite else float("inf"), gradient

    value, gradient, step, _ = strong_wolfe(
        _EXECUTION,
        evaluate,
        x,
        alpha0,
        direction,
        float(f0),
        g0,
        derivative,
        c1=c1,
        c2=c2,
        max_ls=max_iter - 1,
    )
    return (
        float(step),
        jnp.asarray(value, jnp.asarray(f0).dtype, device=x.device),
        gradient,
    )


__all__ = ["strong_wolfe_line_search"]
