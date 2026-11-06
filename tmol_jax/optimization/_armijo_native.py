"""Typed JAX launches for the shared native Armijo transitions."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.pack._ffi import stage


def _transition(arguments, operation, types, first=0.0, second=0.0):
    output = [jax.ShapeDtypeStruct(arguments[1].shape, dtype) for dtype in types]
    return stage(
        arguments,
        "tmol_armijo",
        output,
        operation,
        first=np.float64(first),
        second=np.float64(second),
    )


@jax.jit
def start(searching, alpha0):
    return _transition((searching, alpha0), 0, (alpha0.dtype,))[0]


@partial(jax.jit, static_argnames=("sigma_increase", "sigma_decrease", "status_dtype"))
def classify(
    searching,
    alpha,
    phi,
    phi0,
    derphi0,
    sigma_increase,
    sigma_decrease,
    *,
    status_dtype=np.int32,
):
    return _transition(
        (searching, alpha, phi, phi0, derphi0),
        1,
        (alpha.dtype, alpha.dtype, status_dtype, np.bool_),
        sigma_increase,
        sigma_decrease,
    )


@partial(jax.jit, static_argnames=("factor",))
def trial(status, alpha, accepted, factor):
    return _transition((status, alpha, accepted), 2, (alpha.dtype,), factor)[0]


@partial(jax.jit, static_argnames=("sigma_decrease", "minstep"))
def update(
    status,
    trial,
    accepted,
    phi_accepted,
    phi,
    phi_trial,
    phi0,
    derphi0,
    sigma_decrease,
    minstep,
):
    return _transition(
        (status, trial, accepted, phi_accepted, phi, phi_trial, phi0, derphi0),
        3,
        (trial.dtype, trial.dtype, status.dtype, np.bool_, np.bool_),
        sigma_decrease,
        minstep,
    )


@partial(jax.jit, static_argnames=("minstep",))
def finalize(status, derphi0, accepted, start, searching, minstep):
    return _transition(
        (status, derphi0, accepted, start, searching),
        4,
        (accepted.dtype, np.bool_),
        minstep,
    )


class ArmijoExecution:
    start = staticmethod(start)
    classify = staticmethod(classify)
    trial = staticmethod(trial)
    update = staticmethod(update)

    @staticmethod
    def any_true(mask):
        return bool(np.asarray(jnp.any(mask)))

    @staticmethod
    def equal(left, right):
        return bool(np.asarray(jnp.array_equal(left, right)))

    @staticmethod
    def decisions(failed, active):
        return np.asarray(jnp.stack((failed.any(), active.any()))).tolist()

    @staticmethod
    def failed_indices(failed):
        return np.flatnonzero(np.asarray(failed)).tolist()
