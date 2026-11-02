"""Thevenaz/Blu/Unser B-spline interpolation in pure JAX.

A line-for-line port of ``tmol/numeric/bspline_compiled/bspline.hh`` used by
upstream tmol's Ramachandran and Dunbrack potentials (:func:`bspline_coefficients`,
:func:`bspline_interpolate`). The causal /
anti-causal IIR filters that produce spline coefficients are expressed as
:func:`jax.lax.scan` recurrences and vectorised with :func:`jax.vmap`. Periodic
boundary conditions; degree 2..5 in 2/3/4 spatial dimensions.

Reference
---------
Thevenaz, Blu, and Unser. *Interpolation revisited [medical images
application]*. IEEE TMI 19(7), 2000.
http://bigwww.epfl.ch/publications/thevenaz0002.pdf
"""

from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Helpers: poles for B-splines of various degrees
# ---------------------------------------------------------------------------


def _get_poles(degree: int) -> Tuple[float, ...]:
    """Return the tuple of real poles for a B-spline of the given degree.

    Degrees 2..5 supported; matches the upstream C++ ``_get_poles``.
    """
    if degree == 2:
        return (np.sqrt(8.0) - 3.0,)
    if degree == 3:
        return (np.sqrt(3.0) - 2.0,)
    if degree == 4:
        return (
            np.sqrt(664.0 - np.sqrt(438976.0)) + np.sqrt(304.0) - 19.0,
            np.sqrt(664.0 + np.sqrt(438976.0)) - np.sqrt(304.0) - 19.0,
        )
    if degree == 5:
        return (
            np.sqrt(135.0 / 2.0 - np.sqrt(17745.0 / 4.0))
            + np.sqrt(105.0 / 4.0)
            - 13.0 / 2.0,
            np.sqrt(135.0 / 2.0 + np.sqrt(17745.0 / 4.0))
            - np.sqrt(105.0 / 4.0)
            - 13.0 / 2.0,
        )
    raise ValueError(f"B-spline degree must be in 2..5 (got {degree}).")


# ---------------------------------------------------------------------------
# 1D causal / anti-causal coefficient conversion (Thevenaz eqns 2-3)
# ---------------------------------------------------------------------------


def _init_causal_coeff(line: jax.Array, pole: float) -> jax.Array:
    """Periodic boundary init for the causal recursion."""
    N = line.shape[0]
    tol = 4.0 * np.finfo(np.float64).eps
    horiz = int(np.ceil(np.log(tol) / np.log(np.abs(pole))))

    if horiz < N:
        # line[0] += sum_{i=1}^{horiz-1} pole^i * line[N-i]
        i = jnp.arange(1, horiz)  # static length -> static shape
        zn = pole**i
        contributions = zn * line[N - i]
        return line.at[0].add(jnp.sum(contributions))
    else:
        # Closed-form for short signals, line[0] / (1 - pole^N)
        i = jnp.arange(1, N)
        zn = pole**i
        zn_N = pole**N
        contributions = zn * line[N - i]
        new0 = (line[0] + jnp.sum(contributions)) / (1.0 - zn_N)
        return line.at[0].set(new0)


def _init_anticausal_coeff(line: jax.Array, pole: float) -> jax.Array:
    """Periodic boundary init for the anti-causal recursion."""
    N = line.shape[0]
    tol = 4.0 * np.finfo(np.float64).eps
    horiz = int(np.ceil(np.log(tol) / np.log(np.abs(pole))))

    if horiz < N:
        i = jnp.arange(0, horiz)
        zn = pole ** (i + 1)
        contributions = zn * line[i]
        new_last = (line[N - 1] + jnp.sum(contributions)) * (-pole)
        return line.at[N - 1].set(new_last)
    else:
        i = jnp.arange(0, N)
        zn = pole ** (i + 1)
        zn_N = pole**N
        contributions = zn * line[i]
        new_last = (line[N - 1] + jnp.sum(contributions)) * (-pole) / (1.0 - zn_N)
        return line.at[N - 1].set(new_last)


def _convert_interp_coeffs_1d(line: jax.Array, poles: Tuple[float, ...]) -> jax.Array:
    """In-place (functional) conversion of a 1-D stripe of samples to
    B-spline coefficients. Matches ``_convert_interp_coeffs`` in
    ``bspline.hh``.
    """

    N = line.shape[0]
    if N == 1:
        return line

    # Overall gain lambda = prod_i (1 - p_i) * (1 - 1/p_i)
    lam = 1.0
    for p in poles:
        lam *= (1.0 - p) * (1.0 - 1.0 / p)
    line = line * lam

    def apply_pole(line, pole):
        # Causal filter: c[j] = s[j] + pole * c[j-1]
        line = _init_causal_coeff(line, pole)

        def causal_step(prev, cur):
            out = cur + pole * prev
            return out, out

        _, tail = jax.lax.scan(causal_step, line[0], line[1:])
        line = jnp.concatenate([line[:1], tail])

        # Anti-causal filter: c[j] = pole * (c[j+1] - c[j])
        line = _init_anticausal_coeff(line, pole)

        def anticausal_step(prev, cur):
            out = pole * (prev - cur)
            return out, out

        rev = line[:-1][::-1]
        _, tail_rev = jax.lax.scan(anticausal_step, line[-1], rev)
        line = jnp.concatenate([tail_rev[::-1], line[-1:]])
        return line

    for p in poles:
        line = apply_pole(line, p)
    return line


def _apply_along_axis(
    coeffs: jax.Array, axis: int, poles: Tuple[float, ...]
) -> jax.Array:
    """Run :func:`_convert_interp_coeffs_1d` along ``axis`` of ``coeffs``.

    Achieved by moving the target axis to the front, vmapping over all
    remaining axes, and restoring the original layout.
    """

    coeffs = jnp.moveaxis(coeffs, axis, 0)
    shape = coeffs.shape
    flat = coeffs.reshape(shape[0], -1)

    # One call per stripe -- vmap over the flattened trailing dim.
    stripe_fn = partial(_convert_interp_coeffs_1d, poles=poles)
    flat_out = jax.vmap(stripe_fn, in_axes=1, out_axes=1)(flat)

    out = flat_out.reshape(shape)
    return jnp.moveaxis(out, 0, axis)


# ---------------------------------------------------------------------------
# Public: coefficient precomputation
# ---------------------------------------------------------------------------


def bspline_coefficients(samples: jax.Array, degree: int = 3) -> jax.Array:
    """Convert an N-dim tensor of samples to B-spline coefficients.

    The output has the same shape as ``samples`` (B-splines do not
    expand the grid). Periodic boundary conditions are applied along
    every axis.

    Parameters
    ----------
    samples : jax.Array
        Tensor of shape ``[N_0, N_1, ..., N_{D-1}]`` with ``D in 2..4``
        spatial dimensions. ``float32`` or ``float64``.
    degree : int, default 3
        B-spline order (2, 3, 4, or 5). Degree 3 matches the upstream
        default and is the canonical choice for Rama/Dunbrack.

    Returns
    -------
    jax.Array
        Coefficient tensor, same shape as ``samples``.
    """

    coeffs = jnp.asarray(samples)
    if coeffs.ndim < 2 or coeffs.ndim > 4:
        raise ValueError(
            "B-spline coefficient computation supports 2-4 dims "
            f"(got ndim={coeffs.ndim})."
        )
    poles = _get_poles(degree)
    for axis in range(coeffs.ndim):
        coeffs = _apply_along_axis(coeffs, axis, poles)
    return coeffs


# ---------------------------------------------------------------------------
# Weights and derivatives for interpolation
# ---------------------------------------------------------------------------


def _weights_1d(frac: jax.Array, degree: int) -> jax.Array:
    """Per-dimension spline weights for the (DEGREE+1) support cells.

    ``frac`` is the fractional offset (shape ``[...]``); the returned
    array has shape ``[..., DEGREE+1]``. Directly mirrors the
    ``_get_weights`` function in ``bspline.hh``.
    """

    f = jnp.asarray(frac)
    if degree == 2:
        w1 = 0.75 - f * f
        w2 = 0.5 * (f - w1 + 1.0)
        w0 = 1.0 - w1 - w2
        return jnp.stack([w0, w1, w2], axis=-1)
    if degree == 3:
        w3 = (1.0 / 6.0) * f * f * f
        w0 = (1.0 / 6.0) + 0.5 * f * (f - 1.0) - w3
        w2 = f + w0 - 2.0 * w3
        w1 = 1.0 - w0 - w2 - w3
        return jnp.stack([w0, w1, w2, w3], axis=-1)
    if degree == 4:
        t = (1.0 / 6.0) * f * f
        t0 = f * (t - 11.0 / 24.0)
        t1 = 19.0 / 96.0 + f * f * (0.25 - t)
        w0 = (1.0 / 24.0) * (0.5 - f) ** 3
        w1 = t1 + t0
        w3 = t1 - t0
        w4 = w0 + t0 + 0.5 * f
        w2 = 1.0 - w0 - w1 - w3 - w4
        return jnp.stack([w0, w1, w2, w3, w4], axis=-1)
    if degree == 5:
        f2 = f * f
        w5 = (1.0 / 120.0) * f2 * f * f2
        w4d = (f2 - f) ** 2
        w_ = f - 0.5
        t = f2 * (f2 - 3.0)
        w0 = (1.0 / 24.0) * (0.2 + f2 + w4d) - w5
        t0 = (1.0 / 24.0) * (f2 * (f2 - 5.0) + 46.0 / 5.0)
        t1 = (-1.0 / 12.0) * w_ * (t + 4.0)
        w2 = t0 + t1
        w3 = t0 - t1
        t0 = (1.0 / 16.0) * (9.0 / 5.0 - t)
        t1 = (1.0 / 24.0) * w_ * (w4d - f2 - 5.0)
        w1 = t0 + t1
        w4 = t0 - t1
        return jnp.stack([w0, w1, w2, w3, w4, w5], axis=-1)
    raise ValueError(f"Unsupported B-spline degree {degree}.")


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------


def _interpolate_single(coeffs: jax.Array, X: jax.Array, degree: int) -> jax.Array:
    """Interpolate ``coeffs`` at a single N-D coordinate ``X``.

    Returns a scalar. Gradient w.r.t. ``X`` is available through
    :func:`jax.grad`; we don't return it explicitly because a manual
    hand-derived gradient is unnecessary in JAX.
    """

    ndim = coeffs.ndim
    # Integer base index in each dim and fractional offset.
    idx0 = jnp.floor(X).astype(jnp.int32)  # [ndim]
    frac = X - idx0.astype(X.dtype)
    idx_base = idx0 - (degree // 2)  # support starts here

    # Per-dim weights: [ndim, DEGREE+1]
    wts = jax.vmap(lambda fi: _weights_1d(fi, degree))(frac)

    # Outer product of weights -> full [DEGREE+1]^ndim tensor.
    w = wts[0]
    for d in range(1, ndim):
        w = w[..., None] * wts[d]

    # Gather coefficient neighbourhood via periodic wrap-around.
    k = degree + 1
    offsets = np.arange(k)  # static python-range -> static shape

    # Build a broadcast of [..., ndim] indices covering the ndim-D window.
    per_dim_indices = []
    for d in range(ndim):
        # shape: broadcastable to ndim axes of length k each
        shape = [1] * ndim
        shape[d] = k
        per_dim_indices.append(
            jnp.reshape(
                jnp.mod(idx_base[d] + offsets, coeffs.shape[d]),
                shape,
            )
        )

    neighbourhood = coeffs[tuple(per_dim_indices)]
    # Weighted sum.
    return jnp.sum(w * neighbourhood)


def bspline_interpolate(coeffs: jax.Array, X: jax.Array, degree: int = 3) -> jax.Array:
    """Interpolate (pre-computed) B-spline coefficients at points.

    Parameters
    ----------
    coeffs : jax.Array
        Coefficient tensor produced by :func:`bspline_coefficients`.
        Shape ``[N_0, ..., N_{D-1}]`` with ``D = coeffs.ndim``.
    X : jax.Array
        Query coordinates. Either a single point of shape ``[D]`` or a
        batch of shape ``[..., D]``. Each coordinate is in grid-cell
        units (periodic): ``X[..., d] in R`` with integer values
        aligning on grid nodes.
    degree : int, default 3

    Returns
    -------
    jax.Array
        Interpolated values with shape ``X.shape[:-1]``.
    """
    X = jnp.asarray(X)
    if X.ndim == 1:
        return _interpolate_single(coeffs, X, degree)

    # Batched: vmap over all leading axes.
    fn = partial(_interpolate_single, coeffs, degree=degree)
    # Collapse the leading dims to a single batch, then restore.
    lead = X.shape[:-1]
    flat = X.reshape(-1, X.shape[-1])
    out = jax.vmap(fn)(flat)
    return out.reshape(lead)


__all__ = [
    "bspline_coefficients",
    "bspline_interpolate",
]
