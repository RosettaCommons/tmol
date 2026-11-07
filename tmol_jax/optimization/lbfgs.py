"""JAX movement masks and diagnostics around the common L-BFGS policy."""

import jax
import jax.numpy as jnp

from tmol_core.optimization.strong_wolfe import lbfgs_minimize as _minimize
from ._strong_wolfe_execution import JaxLBFGSExecution


@jax.jit
def _masked_finite_gradient(value, gradient, mask):
    if mask is not None:
        gradient = gradient * mask
    finite = jnp.isfinite(value) & jnp.isfinite(gradient).all()
    return jnp.where(finite, value, jnp.inf), gradient


def lbfgs_minimize(
    fn,
    x0,
    max_iter=100,
    history=5,
    gtol=1e-5,
    ftol=1e-6,
    grad_mask=None,
    line_search_max_iter=25,
    c1=1e-4,
    c2=0.9,
    verbose=False,
    value_and_grad=None,
):
    """Minimize a flat real objective with shared strong-Wolfe/history policy.

    ``grad_mask`` freezes zero entries. ``ftol`` retains the relative loss-change
    criterion, and ``line_search_max_iter`` bounds evaluations per search. The
    mature shared interpolation can choose different trials from older JAX
    releases. Objective and gradient execution stays in JAX; this host-controlled
    optimizer itself is eager. Reaching ``max_iter`` alone is not convergence.
    """
    if max_iter < 0 or history < 1 or line_search_max_iter < 1:
        raise ValueError(
            "iteration limits must be nonnegative and history/search positive"
        )
    if gtol < 0 or ftol < 0:
        raise ValueError("gtol and ftol must be nonnegative")
    if not 0 < c1 < c2 < 1:
        raise ValueError("expected 0 < c1 < c2 < 1")
    x = jnp.asarray(x0)
    if x.ndim != 1 or not jnp.issubdtype(x.dtype, jnp.floating):
        raise TypeError("x0 must be a one-dimensional floating array")
    mask = (
        None if grad_mask is None else jnp.asarray(grad_mask, x.dtype, device=x.device)
    )
    if mask is not None and mask.shape != x.shape:
        raise ValueError("grad_mask must have the same shape as x0")
    evaluate = value_and_grad or jax.jit(jax.value_and_grad(fn))

    def masked(y):
        return _masked_finite_gradient(*evaluate(y), mask)

    initial_value, initial_gradient = masked(x)
    if not bool(jnp.isfinite(initial_value)):
        raise FloatingPointError("initial objective value and gradient must be finite")
    first = True

    def objective(y):
        nonlocal first
        if first:
            first = False
            return initial_value, initial_gradient
        return masked(y)

    def accepted(iteration, coordinates, value, gradient):
        print(
            f"  iter {iteration:4d}  E={value:12.4f}"
            f"  max_grad={float(jnp.max(jnp.abs(gradient))):10.4e}"
        )

    result = _minimize(
        objective,
        x,
        JaxLBFGSExecution(x, history),
        max_iter=max_iter,
        history_size=history,
        max_eval=float("inf"),
        tolerance_grad=gtol,
        tolerance_change=None,
        relative_loss_tolerance=ftol,
        line_search_max_iter=line_search_max_iter,
        c1=c1,
        c2=c2,
        accepted_step=accepted if verbose else None,
    )
    gradient_convergence = float(jnp.max(jnp.abs(result["grad"]))) <= gtol
    relative_convergence = result["reason"] == "relative_loss_tolerance"
    status = result["reason"]
    if gradient_convergence:
        status = "initial_gtol" if result["n_iter"] == 0 else "gtol"
    elif relative_convergence:
        status = "ftol"
    return dict(
        x=result["x"],
        fun=jnp.asarray(result["fun"], x.dtype, device=x.device),
        initial_fun=initial_value,
        grad=result["grad"],
        n_iter=result["n_iter"],
        converged=gradient_convergence or relative_convergence,
        status=status,
    )


__all__ = ["lbfgs_minimize"]
