"""Strong-Wolfe line search with safeguarded cubic interpolation.

The host loop brackets and zooms a step along a fixed direction. Values and
gradients may come from a compiled JAX callable. A finite iteration budget
returns the best accepted trial; optimizer diagnostics determine convergence."""

from __future__ import annotations

from typing import Callable, Tuple

import jax.numpy as jnp

# Cubic interpolant floor: if the bracket length drops below this we
# bail out and accept the best point seen so far.
_MIN_BRACKET_LEN = 1e-12


def _cubic_interp(
    a_lo: float,
    f_lo: float,
    g_lo: float,
    a_hi: float,
    f_hi: float,
    g_hi: float,
) -> float:
    """Safeguarded cubic interpolation between two bracket endpoints.

    Uses the formula from Nocedal & Wright eq. (3.59). Falls back to
    bisection when the interpolant would place the new point outside
    the bracket (or when the numerator / discriminant is non-finite).
    """
    d1 = g_lo + g_hi - 3.0 * (f_lo - f_hi) / (a_lo - a_hi)
    disc = d1 * d1 - g_lo * g_hi
    if disc < 0.0:
        return 0.5 * (a_lo + a_hi)
    d2 = jnp.sqrt(disc)
    d2 = float(d2) if jnp.isfinite(d2) else 0.0
    if a_hi < a_lo:
        d2 = -d2
    denom = g_hi - g_lo + 2.0 * d2
    if denom == 0.0:
        return 0.5 * (a_lo + a_hi)
    a_new = a_hi - (a_hi - a_lo) * (g_hi + d2 - d1) / denom
    # Safeguard: keep the new point strictly inside the bracket.
    lo, hi = (a_lo, a_hi) if a_lo < a_hi else (a_hi, a_lo)
    margin = 0.1 * (hi - lo)
    if not jnp.isfinite(a_new):
        return 0.5 * (lo + hi)
    a_new = float(a_new)
    if a_new < lo + margin:
        a_new = lo + margin
    if a_new > hi - margin:
        a_new = hi - margin
    return a_new


def strong_wolfe_line_search(
    f_and_grad: Callable[[jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]],
    x: jnp.ndarray,
    direction: jnp.ndarray,
    f0: jnp.ndarray,
    g0: jnp.ndarray,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    c2: float = 0.9,
    max_iter: int = 25,
) -> Tuple[float, jnp.ndarray, jnp.ndarray]:
    """Return (step, value, gradient) along direction from x.

    f0 and g0 are the initial value and gradient. Bracketing and safeguarded zoom
    seek sufficient decrease (c1) and strong curvature (c2), bounded by max_iter.
    The callable evaluates value and gradient together; this routine controls the
    search on the host and accepts JAX arrays."""
    # Directional derivative at the start.
    dphi0 = float(jnp.sum(g0 * direction))
    phi0 = float(f0)

    # If the provided direction is not a descent direction, bail out.
    if not (dphi0 < 0.0):
        return 0.0, f0, g0

    # Bracket state.
    alpha_prev = 0.0
    phi_prev = phi0
    dphi_prev = dphi0
    alpha = float(alpha0)
    evals = 0

    # Best-seen fallback if we exhaust the budget without strict success.
    best_alpha = 0.0
    best_phi = phi0
    best_f = f0
    best_g = g0

    def _eval(alpha_t: float):
        y = x + alpha_t * direction
        f_y, g_y = f_and_grad(y)
        if not bool(jnp.isfinite(f_y)) or not bool(jnp.isfinite(g_y).all()):
            return float("inf"), f_y, g_y, float("nan")
        return float(f_y), f_y, g_y, float(jnp.sum(g_y * direction))

    bracket_lo = None
    bracket_hi = None
    bracket_phi_lo = 0.0
    bracket_phi_hi = 0.0
    bracket_dphi_lo = 0.0
    bracket_dphi_hi = 0.0

    for _ in range(max_iter):
        phi_a, f_a, g_a, dphi_a = _eval(alpha)
        evals += 1

        # Track the best-seen (lowest f) admissible point.
        if phi_a < best_phi:
            best_phi = phi_a
            best_alpha = alpha
            best_f = f_a
            best_g = g_a

        # Armijo violation (or non-monotone on subsequent trials) -> bracket.
        if (phi_a > phi0 + c1 * alpha * dphi0) or (evals > 1 and phi_a >= phi_prev):
            bracket_lo, bracket_hi = alpha_prev, alpha
            bracket_phi_lo, bracket_phi_hi = phi_prev, phi_a
            bracket_dphi_lo, bracket_dphi_hi = dphi_prev, dphi_a
            break

        # Strong-Wolfe curvature test.
        if abs(dphi_a) <= -c2 * dphi0:
            return alpha, f_a, g_a

        # Derivative flipped sign (and is non-negative) -> minimizer lies in
        # [alpha, alpha_prev]; reverse endpoints to keep phi_lo <= phi_hi.
        if dphi_a >= 0.0:
            bracket_lo, bracket_hi = alpha, alpha_prev
            bracket_phi_lo, bracket_phi_hi = phi_a, phi_prev
            bracket_dphi_lo, bracket_dphi_hi = dphi_a, dphi_prev
            break

        # Otherwise expand the step and try again.
        alpha_prev = alpha
        phi_prev = phi_a
        dphi_prev = dphi_a
        alpha = min(alpha * 2.0, 1e10)
    else:
        # Exhausted budget without bracketing — take the best we have.
        if best_alpha > 0.0:
            return best_alpha, best_f, best_g
        return 0.0, f0, g0

    # Invariants:
    #   phi_lo is the best Armijo-satisfying point on the bracket;
    #   dphi_lo * (hi - lo) < 0 (minimizer is bracketed).
    for _ in range(max_iter - evals):
        if abs(bracket_hi - bracket_lo) < _MIN_BRACKET_LEN:
            break

        # Cubic interpolation, safeguarded inside the bracket.
        alpha_new = _cubic_interp(
            bracket_lo,
            bracket_phi_lo,
            bracket_dphi_lo,
            bracket_hi,
            bracket_phi_hi,
            bracket_dphi_hi,
        )

        phi_new, f_new, g_new, dphi_new = _eval(alpha_new)
        evals += 1
        if phi_new < best_phi:
            best_phi = phi_new
            best_alpha = alpha_new
            best_f = f_new
            best_g = g_new

        # Shrink bracket depending on where the new point falls.
        if (phi_new > phi0 + c1 * alpha_new * dphi0) or (phi_new >= bracket_phi_lo):
            bracket_hi = alpha_new
            bracket_phi_hi = phi_new
            bracket_dphi_hi = dphi_new
        else:
            if abs(dphi_new) <= -c2 * dphi0:
                return alpha_new, f_new, g_new
            # Move lo towards the new point; keep the bracket spanning a
            # minimizer by flipping hi/lo when the sign of dphi demands it.
            if dphi_new * (bracket_hi - bracket_lo) >= 0.0:
                bracket_hi = bracket_lo
                bracket_phi_hi = bracket_phi_lo
                bracket_dphi_hi = bracket_dphi_lo
            bracket_lo = alpha_new
            bracket_phi_lo = phi_new
            bracket_dphi_lo = dphi_new

    # Exhausted budget or bracket collapsed — fall back to best-seen.
    if best_alpha > 0.0:
        return best_alpha, best_f, best_g
    return 0.0, f0, g0


__all__ = ["strong_wolfe_line_search"]
