"""Coordinate minimization against rendered, topology-stable JAX scorers."""

from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp

from .lbfgs import lbfgs_minimize


def _coordinate_mask(pose_stack: Any, move_mask: Optional[Any]) -> jax.Array:
    """Return a strict xyz movement mask, always excluding padding atoms."""
    coords_shape = pose_stack.coords.shape
    atom_shape = coords_shape[:2]
    if move_mask is None:
        mask = jnp.broadcast_to(pose_stack.atom_is_real[..., None], coords_shape)
    else:
        mask = jnp.asarray(move_mask, dtype=jnp.bool_)
        if mask.shape == atom_shape:
            mask = jnp.broadcast_to(mask[..., None], coords_shape)
        elif mask.shape != coords_shape:
            raise ValueError(
                "move_mask must have shape "
                f"{atom_shape} or {coords_shape}, got {mask.shape}"
            )
        mask &= pose_stack.atom_is_real[..., None]
    return mask


def run_cart_min(
    pose_stack: Any,
    score_function: Any,
    move_mask: Optional[Any] = None,
    *,
    max_iter: Optional[int] = None,
    history: Optional[int] = None,
    gtol: float = 1e-5,
    ftol: float = 1e-6,
    line_search_max_iter: int = 25,
    verbose: bool = False,
    return_result: bool = False,
    constraint_set: Optional[Any] = None,
    optimizer="armijo",
    optimizer_kwargs=None,
    coord_mask=None,
    cuda_graph=None,
):
    """Minimize Cartesian coordinates with the shared segmented L-BFGS/Armijo policy.

    The score function is rendered once because topology and atom layout stay
    fixed. ``move_mask`` may select atoms (``[poses, atoms]``) or individual
    xyz components (``[poses, atoms, 3]``); padding is always frozen. The
    returned pose is new and the input is unchanged. ``coord_mask`` is the mature
    API spelling for ``move_mask``; supply only one non-None mask.

    Set ``return_result=True`` to receive ``(pose, diagnostics)`` where the
    default diagnostics mapping includes the final value, gradient, iteration count,
    per-pose convergence and stalled flags. The default ``optimizer="armijo"`` uses the
    shared mature segmented policy on compact movable coordinates. Its convergence
    settings come from ``optimizer_kwargs`` (shared Torch names/defaults); the
    legacy ``gtol``, ``ftol`` and line-search limit apply only to explicit
    ``optimizer="strong_wolfe"``. Omitted iteration/history settings use the
    common Armijo defaults (200 iterations, 200 history entries). A callable
    ``optimizer`` uses the compact functional protocol documented in the JAX guide.
    ``cuda_graph=True`` captures native Armijo scoring through XLA command buffers;
    False disables capture and None preserves automatic compilation. Explicit modes
    require a rendered TMol scorer; capture requires CUDA coordinates.
    """
    if coord_mask is not None:
        if move_mask is not None:
            raise TypeError("supply only one of move_mask and coord_mask")
        move_mask = coord_mask
    coords, scorer = _prepare_minimization(
        pose_stack, score_function, constraint_set, cuda_graph, optimizer
    )
    mask = _coordinate_mask(pose_stack, move_mask)
    shape = coords.shape
    x0 = coords.reshape(-1)
    flat_mask = mask.reshape(-1)

    def energy(x_flat):
        values = scorer(x_flat.reshape(shape))
        return values.sum() if optimizer == "strong_wolfe" else values

    result = _minimize_selected(
        energy,
        x0,
        flat_mask,
        jnp.repeat(jnp.arange(shape[0]), shape[1] * shape[2]),
        max_iter,
        history,
        gtol,
        ftol,
        line_search_max_iter,
        verbose,
        optimizer,
        optimizer_kwargs,
        coordinate_scorer=scorer if hasattr(scorer, "compact_objective") else None,
    )
    final_coords = result["x"].reshape(shape).astype(coords.dtype)
    final_coords = jnp.where(mask, final_coords, coords)
    minimized = pose_stack.replace(coords=final_coords)
    if return_result:
        diagnostics = dict(result)
        diagnostics["initial_fun"] = float(result["initial_fun"])
        return minimized, diagnostics
    return minimized


def _minimize_selected(
    energy,
    initial,
    mask,
    pose_ids,
    max_iter,
    history,
    gtol,
    ftol,
    line_search_max_iter,
    verbose,
    optimizer,
    optimizer_kwargs,
    coordinate_scorer=None,
    coordinate_map=None,
):
    if optimizer == "strong_wolfe":
        if optimizer_kwargs:
            raise ValueError("optimizer_kwargs requires the Armijo execution path")
        return _minimize(
            energy,
            initial,
            mask,
            100 if max_iter is None else max_iter,
            5 if history is None else history,
            gtol,
            ftol,
            line_search_max_iter,
            verbose,
            value_and_grad=_rendered_wolfe_gradient(
                coordinate_scorer, initial, coordinate_map
            ),
        )
    if optimizer != "armijo" and not callable(optimizer):
        raise ValueError("optimizer must be 'armijo', 'strong_wolfe' or a callable")
    from ._segmented_minimization import minimize_pose_variables

    settings = dict(verbose=verbose)
    if max_iter is not None:
        settings["max_iter"] = max_iter
    if history is not None:
        settings["history_size"] = history
    settings.update(optimizer_kwargs or {})
    return _validate_minimum(
        minimize_pose_variables(
            energy,
            initial,
            mask,
            pose_ids,
            settings,
            coordinate_scorer=coordinate_scorer,
            coordinate_map=coordinate_map,
            **({"minimize": optimizer} if callable(optimizer) else {}),
        )
    )


def _prepare_minimization(
    pose_stack, score_function, constraint_set, cuda_graph=None, optimizer="armijo"
):
    if cuda_graph is not None and not isinstance(cuda_graph, bool):
        raise TypeError("cuda_graph must be True, False or None")
    if cuda_graph is not None and optimizer != "armijo":
        raise NotImplementedError("explicit CUDA graph mode requires Armijo execution")
    coords = jnp.asarray(pose_stack.coords)
    if cuda_graph and coords.device.platform != "gpu":
        raise ValueError("CUDA graph capture requires a CUDA pose")
    if not jnp.issubdtype(coords.dtype, jnp.floating):
        raise TypeError("pose coordinates must be floating point")
    if not bool(jnp.isfinite(coords).all()):
        raise ValueError("pose coordinates must be finite")
    if any(
        getattr(term, "use_neighbor_list", False)
        for term in getattr(score_function, "terms", ())
    ):
        raise ValueError("minimization requires a dense score function")

    try:
        scorer = score_function.render_whole_pose_scoring_module(
            pose_stack, constraint_set=constraint_set
        )
    except AttributeError as error:
        raise TypeError(
            "score_function must provide the TMol JAX render API"
        ) from error

    if cuda_graph is not None and coords.device.platform == "gpu":
        if not hasattr(scorer, "_with_cuda_graph"):
            raise TypeError("explicit CUDA graph mode requires a rendered TMol scorer")
        scorer = scorer._with_cuda_graph(cuda_graph)
    return coords, scorer


@jax.jit
def _sum_flat_gradient(values, gradient):
    return values.sum(), gradient.reshape(-1)


def _rendered_wolfe_gradient(scorer, initial, coordinate_map):
    """Bind fresh scorer operands to its existing reusable gradient executable."""
    if scorer is None:
        return None
    if coordinate_map is None:
        evaluate = scorer.value_and_grad()
        shape = scorer.shape
    else:
        shape = initial.shape
        _, evaluate = scorer.compact_objective(
            initial.reshape(coordinate_map.initial_dofs.shape),
            jnp.arange(initial.size, dtype=jnp.int32, device=initial.device),
            jnp.arange(scorer.shape[0], dtype=jnp.int32, device=initial.device),
            coordinate_map,
        )

    def value_and_grad(flat):
        return _sum_flat_gradient(*evaluate(flat.reshape(shape)))

    return value_and_grad


def _minimize(
    energy,
    x0,
    mask,
    max_iter,
    history,
    gtol,
    ftol,
    line_search_max_iter,
    verbose,
    value_and_grad=None,
):
    with jax.default_device(x0.device):
        result = lbfgs_minimize(
            energy,
            x0,
            max_iter=max_iter,
            history=history,
            gtol=gtol,
            ftol=ftol,
            grad_mask=mask,
            line_search_max_iter=line_search_max_iter,
            verbose=verbose,
            value_and_grad=value_and_grad or jax.jit(jax.value_and_grad(energy)),
        )
    return _validate_minimum(result)


def _validate_minimum(result):
    if not bool(jnp.isfinite(result["x"]).all()) or not bool(
        jnp.isfinite(result["fun"])
    ):
        raise FloatingPointError("minimization produced non-finite output")
    initial_energy, final_energy = float(result["initial_fun"]), float(result["fun"])
    if final_energy > initial_energy + 1e-6 * max(abs(initial_energy), 1.0):
        raise RuntimeError(
            "minimization increased the energy: "
            f"{initial_energy:.7g} -> {final_energy:.7g}"
        )
    return result


@jax.jit
def _kinematic_coordinates(mapping, flat, mask):
    initial = mapping.initial_dofs
    return mapping(jnp.where(mask, flat.reshape(initial.shape), initial))


def run_kin_min(
    pose_stack,
    score_function,
    move_map,
    *,
    fold_forest=None,
    kin_dtype=None,
    max_iter=None,
    history=None,
    gtol=1e-5,
    ftol=1e-6,
    line_search_max_iter=25,
    verbose=False,
    return_result=False,
    constraint_set=None,
    optimizer="armijo",
    optimizer_kwargs=None,
    cuda_graph=None,
):
    """Minimize selected internal DOFs through shared native molecular kinematics.

    ``move_map`` uses the same hierarchical selection rules as Torch. Its default
    disables every DOF. Padding and unselected DOFs remain fixed; atoms downstream
    of selected bonds or jumps move. The original pose is unchanged. Diagnostics
    contain flat DOFs and their masked gradient, rather than Cartesian coordinates.
    ``kin_dtype`` optionally selects float32/64 internal DOFs while preserving
    the stored coordinate dtype; float64 requires JAX x64 enabled.
    This host-controlled optimizer requires first reverse derivatives only.
    The default ``optimizer="armijo"`` uses the shared mature segmented policy, with
    ``optimizer_kwargs`` using Torch setting names. Only movable DOFs enter its
    history; independent poses retain independent line search and convergence.
    A callable ``optimizer`` receives the same compact variables and movement
    policy; its functional protocol is documented in the JAX guide.
    ``cuda_graph`` selects the same explicit Armijo scoring execution as Cartesian
    minimization; None preserves automatic compilation.
    """
    from tmol_jax.kinematics import PoseStackKinematics

    coords, scorer = _prepare_minimization(
        pose_stack, score_function, constraint_set, cuda_graph, optimizer
    )
    kinematics = PoseStackKinematics(pose_stack, fold_forest)
    mapping = kinematics.bind(coords, kin_dtype=kin_dtype)
    mask = kinematics.resolve_move_map(move_map)
    initial = mapping.initial_dofs
    movable = bool(jnp.any(mask))

    def coordinates(flat):
        if not movable:
            return coords
        return _kinematic_coordinates(mapping, flat, mask)

    def energy(flat):
        values = scorer(coordinates(flat))
        return values.sum() if optimizer == "strong_wolfe" else values

    node_pose = kinematics.atom_ids // pose_stack.max_n_atoms_per_pose
    result = _minimize_selected(
        energy,
        initial.reshape(-1),
        mask.reshape(-1),
        jnp.repeat(node_pose, initial.shape[1]),
        max_iter,
        history,
        gtol,
        ftol,
        line_search_max_iter,
        verbose,
        optimizer,
        optimizer_kwargs,
        coordinate_scorer=(
            scorer if movable and hasattr(scorer, "compact_objective") else None
        ),
        coordinate_map=mapping,
    )
    final_coords = coordinates(result["x"])
    if not bool(jnp.isfinite(final_coords).all()):
        raise FloatingPointError("minimization produced non-finite coordinates")
    minimized = pose_stack.replace(coords=final_coords)
    if return_result:
        return minimized, dict(result, initial_fun=float(result["initial_fun"]))
    return minimized


__all__ = ["run_cart_min", "run_kin_min"]
