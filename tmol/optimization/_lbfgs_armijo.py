from contextlib import nullcontext
from math import isfinite
from numbers import Integral, Real
from types import SimpleNamespace
from typing import Any

import torch
from torch.optim import Optimizer

from tmol_core.optimization.armijo import armijo_linesearch_segmented
from tmol_core.optimization.lbfgs import compact_lbfgs_direction
from tmol_core.optimization.segmented import SegmentedLBFGSPolicy

from tmol.optimization._armijo_compiled import (
    _armijo_classify,
    _armijo_finalize,
    _armijo_start,
    _armijo_trial,
    _armijo_update,
)


def _lbfgs_two_loop(grad, dirs, stps):
    """L-BFGS search direction -H_k @ grad via the compact
    representation of Byrd, Nocedal & Schnabel, (Math. Prog. 63 (1994)):
        H_0 = I
        M = [[ R^-T (D + Y^T Y) R^-1, -R^-T ], [ -R^-1, 0 ]]
        H_k g = g + [S Y] M [S^T g ; Y^T g]

    Algebraically identical to the classic two-loop recursion, but all O(N*m)
    ops are parallelized.

    grad is (N) or (k, N) and dirs/stps are (m, N) or (m, k, N); in the latter
    case each of the k batches gets its own inverse-Hessian estimate.
    """
    return _lbfgs_two_loop_impl(grad, dirs, stps)


class _TorchCompactLBFGS:
    float64 = staticmethod(torch.Tensor.double)
    cast = staticmethod(torch.Tensor.to)
    mv = staticmethod(torch.mv)
    mm = staticmethod(torch.mm)
    bmm = staticmethod(torch.bmm)
    einsum = staticmethod(torch.einsum)
    roll = staticmethod(torch.roll)
    triu = staticmethod(torch.triu)
    solve_triangular = staticmethod(torch.linalg.solve_triangular)

    @staticmethod
    def diagonal(matrix):
        return matrix.diagonal(dim1=-2, dim2=-1)

    @staticmethod
    def fill_diagonal_where(matrix, mask, value):
        matrix.diagonal(dim1=-2, dim2=-1).masked_fill_(mask, value)
        return matrix


def _lbfgs_two_loop_impl(grad, dirs, stps, history_start=0):
    unbatched = grad.dim() == 1
    poses = 1 if unbatched else grad.shape[0]
    return compact_lbfgs_direction(
        grad,
        dirs,
        stps,
        history_start,
        execution=_TorchCompactLBFGS,
        single_segment=poses == 1 and grad.dtype == torch.float32,
        use_bmm=(
            not unbatched
            and poses > 1
            and stps.shape[1] == dirs.shape[1] == poses
            and stps.shape[2] == dirs.shape[2] == grad.shape[1]
            and not (
                torch.is_grad_enabled()
                and (grad.requires_grad or dirs.requires_grad or stps.requires_grad)
            )
        ),
    )


# per-segment line search states
_LS_DONE = 0
_LS_INCREASE = 1  # the step looked linear; trying a longer one
_LS_BACKTRACK = 2
_LS_FAILED = 3


def _any_true(mask):
    """Synchronize a predicate without launching a reduction for one segment."""
    return bool(mask) if mask.numel() == 1 else bool(mask.any())


class _TorchArmijoExecution:
    start = staticmethod(_armijo_start)
    classify = staticmethod(_armijo_classify)
    trial = staticmethod(_armijo_trial)
    update = staticmethod(_armijo_update)
    any_true = staticmethod(_any_true)
    equal = staticmethod(torch.equal)

    @staticmethod
    def decisions(failed, active):
        return torch.stack((failed, active)).any(dim=1).tolist()

    @staticmethod
    def failed_indices(failed):
        return failed.nonzero(as_tuple=False).flatten().tolist()


def _armijo_linesearch_segmented(
    func,
    derphi0,
    old_fval,
    alpha0,
    searching,
    factor=0.5,
    sigma_decrease=0.1,
    sigma_increase=0.8,
    minstep=1e-6,
):
    return armijo_linesearch_segmented(
        func,
        derphi0,
        old_fval,
        alpha0,
        searching,
        factor,
        sigma_decrease,
        sigma_increase,
        minstep,
        execution=_TorchArmijoExecution,
        combine_checks=searching.is_cuda and searching.numel() > 1,
    )


class LBFGS_Armijo(SegmentedLBFGSPolicy, Optimizer):
    """
    Implements L-BFGS algorithm with Armijo line search.
    All scaling and parameters taken directly from Rosetta

    Parameters:
        lr (float): learning rate (default: 1)
        max_iter (int): maximal number of iterations (default: 200)
        atol (float): an iteration whose energy change is at most atol counts
            as small (default: 1e-2, energy units)
        patience (int): a segment converges after this many consecutive small
            iterations (default: 5)
        rtol (float): if set, an iteration whose relative energy change is at
            most rtol also counts as small (default: None)
        gradtol (float): if set, a segment also converges once
            max_i |df/dx_i| <= gradtol, in the units of the minimized degrees
            of freedom (default: None)
        history_size (int): update history size (default: 128).
        segment_ids (Tensor): the segment (e.g. pose) each parameter element
            belongs to; each segment is minimized independently (default: one)
        value_only_line_search (bool): evaluate Armijo trial points with a
            separate value-only closure and refresh the gradient only at the
            accepted point (default: False)
        fixed_iterations (bool): run exactly ``max_iter`` iterations instead of
            evaluating convergence criteria (default: False).
    """

    supports_segments = True

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """Add the optimizer's sole parameter group without compiler imports.

        PyTorch's generic implementation is Dynamo-disabled and initializes
        Dynamo, DTensor, and distributed operator registries on first use.
        This optimizer supports exactly one group and one tensor, so that
        general machinery adds substantial cold-start latency without adding
        functionality.
        """
        if self.param_groups:
            raise ValueError("LBFGS doesn't support per-parameter options")
        if not isinstance(param_group, dict):
            raise TypeError(f"param_group must be a dict, got {type(param_group)}")

        params = param_group["params"]
        if isinstance(params, torch.Tensor):
            params = [params]
        elif isinstance(params, set):
            raise TypeError("optimizer parameters must use an ordered collection")
        else:
            params = list(params)
        if len(params) != 1:
            raise ValueError("LBFGS requires exactly one parameter tensor")
        param = params[0]
        param_name = None
        if isinstance(param, tuple):
            if len(param) != 2 or not isinstance(param[0], str):
                raise TypeError(
                    "named optimizer parameters must be (name, Tensor) pairs"
                )
            param_name, param = param
        if not isinstance(param, torch.Tensor):
            raise TypeError(f"optimizer parameter must be a Tensor, got {type(param)}")
        if not param.is_leaf and not param.retains_grad:
            raise ValueError("can't optimize a non-leaf Tensor")

        param_group = dict(param_group)
        param_group["params"] = [param]
        if param_name is not None:
            param_group["param_names"] = [param_name]
        for name, default in self.defaults.items():
            param_group.setdefault(name, default)
        self.param_groups.append(param_group)

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Reset the single optimized tensor's gradient."""
        param = self._params[0]
        if param.grad is None:
            return
        if set_to_none:
            param.grad = None
        else:
            if param.grad.grad_fn is not None:
                param.grad.detach_()
            else:
                param.grad.requires_grad_(False)
            param.grad.zero_()

    def __init__(
        self,
        params,
        lr=1,
        max_iter=200,
        rtol=None,
        atol=1e-2,
        gradtol=None,
        history_size=128,
        minstep=1e-12,
        verbose=False,
        segment_ids=None,
        fixed_iterations=False,
        *,
        patience=5,
        value_only_line_search: bool = False,
    ):
        defaults = dict(
            lr=lr,
            max_iter=max_iter,
            atol=atol,
            patience=patience,
            rtol=rtol,
            gradtol=gradtol,
            history_size=history_size,
            fixed_iterations=fixed_iterations,
        )
        super(LBFGS_Armijo, self).__init__(params, defaults)

        group = self.param_groups[0]
        patience = group["patience"]
        if (
            isinstance(patience, bool)
            or not isinstance(patience, Integral)
            or patience < 1
        ):
            raise ValueError("patience must be a positive integer")
        for name in ("rtol", "atol", "gradtol"):
            value = group[name]
            if value is not None and (
                not isinstance(value, Real) or not isfinite(value) or value < 0
            ):
                raise ValueError(f"{name} must be None or a finite nonnegative number")

        if len(self.param_groups) != 1:
            raise ValueError(
                "LBFGS doesn't support per-parameter options " "(parameter groups)"
            )

        self._params = self.param_groups[0]["params"]
        self._minstep = minstep
        self.verbose = verbose
        if not isinstance(value_only_line_search, bool):
            raise TypeError("value_only_line_search must be bool")
        self.value_only_line_search = value_only_line_search

        self._last_loss_vec = None
        self._sum_scratch = None
        if segment_ids is None:
            # the whole parameter as one segment: the same code path as a stack
            segment_ids = torch.zeros(
                self._params[0].numel(),
                dtype=torch.int64,
                device=self._params[0].device,
            )
        self._init_segments(segment_ids)

    def __setstate__(self, state):
        super().__setstate__(state)
        group = self.param_groups[0]
        if "patience" not in group:
            # Older checkpoints stopped after one small step and used dtype
            # tolerances for None; new checkpoints use None to disable them.
            group["patience"] = 1
            tolerance = float(torch.finfo(group["params"][0].dtype).eps ** 0.5)
            for name in ("rtol", "atol"):
                if group[name] is None:
                    group[name] = tolerance

    def reset(self) -> None:
        """Reset trajectory state while retaining shape-compatible scratch.

        This is useful when repeatedly minimizing new coordinates with the
        same parameter tensor and optimizer configuration. History contents
        are ignored until overwritten, so large buffers do not need clearing.
        """
        param = self._params[0]
        param.grad = None
        state = self.state.get(param)
        if not state:
            return

        state["func_evals"] = 0
        state["n_iter"] = 0
        state["history_start"] = 0
        state["history_count"] = 0
        for name in ("t", "prev_flat_grad", "prev_loss_vec"):
            state.pop(name, None)
        if state.get("x_ref") is not None:
            state["x_ref"].copy_(param.data.view(-1))
        for name in ("converged", "stalled", "needs_reset", "was_reset", "small_steps"):
            if state.get(name) is not None:
                state[name].zero_()
        state["any_needs_reset"] = False
        state["any_was_reset"] = False
        state["any_inactive"] = False
        self._last_loss_vec = None

    def _init_segments(self, segment_ids):
        """Set up the mapping from parameter elements to independent blocks.

        Builds the index that scatters a parameter-shaped vector into a dense
        (n_segments, segment_size) layout, so all per-segment reductions are
        batched tensor ops rather than a loop over segments.
        """
        assert len(self._params) == 1, "segment_ids requires a single tensor"
        param = self._params[0]
        segment_ids = segment_ids.reshape(-1).to(torch.int64)
        assert (
            segment_ids.numel() == param.numel()
        ), "segment_ids needs one entry per parameter element"
        assert bool((segment_ids >= 0).all()), "segment_ids must be non-negative"

        n_segments = int(segment_ids.max().item()) + 1
        if n_segments == 1:
            self._segment_ids = segment_ids
            self._n_segments = 1
            self._segment_size = segment_ids.numel()
            self._segments_are_dense = True
            self._pad_index = None
            return

        counts = torch.bincount(segment_ids, minlength=n_segments)
        assert bool((counts > 0).all()), "segment_ids must not skip a segment"
        segment_size = int(counts.max().item())

        # Avoid sorting the overwhelmingly common dense layout: one pose, or
        # equal-sized poses stored contiguously. Dense reductions only reshape
        # the parameter vector and never consult ``_pad_index``.
        segments_are_dense = False
        if bool((counts == segment_size).all()):
            expected = torch.arange(n_segments, device=segment_ids.device)
            expected = expected.repeat_interleave(segment_size)
            segments_are_dense = torch.equal(segment_ids, expected)

        self._segment_ids = segment_ids
        self._n_segments = n_segments
        self._segment_size = segment_size
        self._segments_are_dense = segments_are_dense
        if segments_are_dense:
            self._pad_index = None
            return

        # rank of each element within its own segment
        order = torch.argsort(segment_ids, stable=True)
        offsets = torch.cumsum(counts, 0) - counts
        rank = torch.empty_like(order)
        arange = torch.arange(segment_ids.numel(), device=order.device)
        rank[order] = arange - offsets[segment_ids[order]]

        self._pad_index = segment_ids * segment_size + rank

    def _seg_sum(self, values):
        """Sum a parameter-shaped vector within each segment."""
        if self._segments_are_dense:
            return values.view(self._n_segments, self._segment_size).sum(-1)
        if self._sum_scratch is None or self._sum_scratch.dtype != values.dtype:
            self._sum_scratch = torch.zeros(
                self._n_segments * self._segment_size,
                dtype=values.dtype,
                device=values.device,
            )
        return self._pad(values, out=self._sum_scratch).sum(-1)

    def _seg_amax(self, values):
        """Maximum of a parameter-shaped vector within each segment."""
        if self._segments_are_dense:
            return values.view(self._n_segments, self._segment_size).amax(-1)
        out = torch.empty(self._n_segments, dtype=values.dtype, device=values.device)
        return out.scatter_reduce_(
            0, self._segment_ids, values, "amax", include_self=False
        )

    def _pad(self, values, out=None):
        """Scatter a parameter-shaped vector to (n_segments, segment_size)."""
        if self._segments_are_dense:
            padded = values.view(self._n_segments, self._segment_size)
            if out is not None:
                out.copy_(padded)
                return out.view(self._n_segments, self._segment_size)
            return padded
        if out is None:
            out = torch.zeros(
                self._n_segments * self._segment_size,
                dtype=values.dtype,
                device=values.device,
            )
        else:
            out = out.view(-1)
            out.zero_()
        out[self._pad_index] = values
        return out.view(self._n_segments, self._segment_size)

    def _unpad(self, padded):
        """Gather a (n_segments, segment_size) tensor back to parameter shape."""
        if self._segments_are_dense:
            return padded.reshape(-1)
        return padded.reshape(-1)[self._pad_index]

    def _per_element(self, segment_values: torch.Tensor) -> torch.Tensor:
        """Broadcast per-segment values to parameters without indexing one segment."""
        if self._n_segments == 1:
            return segment_values[0]
        # Repeated CPU copies avoid reading an index for every parameter.
        # Keep differentiable broadcasts on their original reduction path.
        if (
            self._segments_are_dense
            and segment_values.device.type == "cpu"
            and segment_values.shape == (self._n_segments,)
            and not (torch.is_grad_enabled() and segment_values.requires_grad)
        ):
            return segment_values.repeat_interleave(self._segment_size)
        return segment_values[self._segment_ids]

    def _wrap_closure(self, closure):
        """Reduce a per-segment closure to a scalar, keeping the vector around.

        A closure may return one energy per segment; the line search needs the
        total, while the convergence tests want the per-segment values.
        """

        def wrapped():
            loss = closure()
            if (
                self._segment_ids is not None
                and torch.is_tensor(loss)
                and loss.numel() == self._n_segments
            ):
                self._last_loss_vec = loss.detach().reshape(self._n_segments)
                return loss.sum()
            self._last_loss_vec = None
            return loss

        return wrapped

    def _step_setup(self, closure):
        """Prepare for L-BFGS:
        read config
        initialize state (history buffers, x_ref, preallocate d and x_backup)
        evaluate initial loss and gradient
        returns a SimpleNamespace ctx"""
        # lbfgs only works w/ single parameter group
        assert len(self.param_groups) == 1
        assert len(self._params) == 1, "This version requires single tensor"

        group = self.param_groups[0]
        lr = group["lr"]
        max_iter = group["max_iter"]
        rtol = group["rtol"]
        atol = group["atol"]
        patience = group["patience"]
        gradtol = group["gradtol"]
        history_size = group["history_size"]
        fixed_iterations = group["fixed_iterations"]
        # At most one curvature pair is created after each iteration beyond
        # the first. Short protocols should not allocate the default 128 slots.
        history_size = min(history_size, max(1, max_iter - 1))

        # tolerances below sqrt(eps) of the parameter dtype cannot be resolved
        #   float32 : eps~3.45e-4
        #   float64 : eps~1.49e-8
        dtype_based_tol = float(torch.finfo(self._params[0].dtype).eps ** 0.5)
        if rtol is not None and 0 < rtol < dtype_based_tol:
            print(f"  WARNING: rtol ({rtol}) is too low for dtype! ({dtype_based_tol})")
        if atol is not None and 0 < atol < dtype_based_tol:
            print(f"  WARNING: atol ({atol}) is too low for dtype! ({dtype_based_tol})")

        param = self._params[0]

        # NOTE: LBFGS has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        state = self.state[param]
        state.setdefault("func_evals", 0)
        state.setdefault("n_iter", 0)

        # evaluate initial f(x)
        orig_loss = closure()
        # The optimizer uses the per-segment tensor below for all line-search
        # and convergence decisions. Materializing a Python scalar here forces
        # a device synchronization and is only useful for verbose reporting.
        loss = float(orig_loss.detach()) if self.verbose else None
        loss_vec = self._last_loss_vec
        state["func_evals"] += 1

        x = param.data.view(-1)
        flat_grad = param.grad.data.view(-1)

        # Preallocate on first call. `d` and `x_backup` must have stable
        # tensor identity so linefn (defined once in step) can capture them.
        L = x.numel()
        if state.get("d") is None:
            state["d"] = torch.empty_like(x)
            state["x_backup"] = torch.empty_like(x)
            if self._segment_ids is None:
                hist_shape = (history_size, L)
            else:
                hist_shape = (history_size, self._n_segments, self._segment_size)
                if not self._segments_are_dense:
                    state["grad_pad"] = torch.zeros(
                        hist_shape[1:], device=x.device, dtype=x.dtype
                    )
            # Only ``history_count`` written slots are consumed. Non-dense
            # segmented writes explicitly zero their own padding.
            state["old_dirs_mat"] = torch.empty(
                hist_shape, device=x.device, dtype=x.dtype
            )
            state["old_stps_mat"] = torch.empty(
                hist_shape, device=x.device, dtype=x.dtype
            )
            state["history_start"] = 0  # Circular buffer start index
            state["history_count"] = 0  # Number of items in history
            state["x_ref"] = x.clone()  # reference position for s computation
            flags = dict(dtype=torch.bool, device=x.device)
            # stationary segments; segments that gave up; segments owed a
            # steepest-descent restart after a failed line search
            state["converged"] = torch.zeros(self._n_segments, **flags)
            state["stalled"] = torch.zeros(self._n_segments, **flags)
            state["needs_reset"] = torch.zeros(self._n_segments, **flags)
            state["was_reset"] = torch.zeros(self._n_segments, **flags)
            state["any_was_reset"] = False
        # consecutive small iterations per segment; old state dictionaries lack it
        if state.get("small_steps") is None:
            state["small_steps"] = torch.zeros(
                self._n_segments, dtype=torch.int64, device=x.device
            )

        # State loading casts tensor masks to the parameter's floating dtype.
        for key in ("converged", "stalled", "needs_reset", "was_reset"):
            state[key] = state[key].bool()
        state["small_steps"] = state["small_steps"].long()

        return SimpleNamespace(
            # Keep the wrapped closure local to this step. Storing it on the
            # optimizer creates a cycle through the wrapper's self reference.
            closure=closure,
            # config
            max_iter=max_iter,
            lr=lr,
            rtol=rtol,
            atol=atol,
            patience=patience,
            gradtol=gradtol,
            history_size=history_size,
            fixed_iterations=fixed_iterations,
            # torch / state
            state=state,
            param=param,
            x=x,
            flat_grad=flat_grad,
            # preallocated scratch (stable refs captured by linefn)
            d=state["d"],
            x_backup=state["x_backup"],
            # cached across steps
            t=state.get("t"),
            prev_flat_grad=state.get("prev_flat_grad"),
            prev_loss_vec=state.get("prev_loss_vec"),
            # history
            old_dirs_mat=state["old_dirs_mat"],
            old_stps_mat=state["old_stps_mat"],
            history_start=state["history_start"],
            history_count=state["history_count"],
            x_ref=state["x_ref"],
            converged=state["converged"],
            stalled=state["stalled"],
            needs_reset=state["needs_reset"],
            was_reset=state["was_reset"],
            small_steps=state["small_steps"],
            # Host mirrors of predicates already synchronized in the previous
            # iteration. Reusing them avoids repeating identical device-to-host
            # checks at the start of the next iteration.
            any_needs_reset=state.get("any_needs_reset", False),
            # Old state dictionaries lack this host mirror. Clear their mask
            # once conservatively instead of trusting its contents.
            any_was_reset=state.get("any_was_reset", "was_reset" in state),
            any_inactive=state.get("any_inactive", False),
            gtd_seg=None,
            # current eval
            orig_loss=orig_loss,
            loss=loss,
            loss_vec=loss_vec,
            # line-search accounting
            ls_evals=0,
        )

    # Storage and execution hooks for the shared scientific policy.
    _copy = staticmethod(torch.Tensor.copy_)
    _clone = staticmethod(torch.Tensor.clone)
    _zero = staticmethod(torch.Tensor.zero_)
    _where = staticmethod(torch.where)
    _abs = staticmethod(torch.abs)
    _sign = staticmethod(torch.sign)
    _zeros_like = staticmethod(torch.zeros_like)
    _any_true = staticmethod(_any_true)
    _line_search = staticmethod(_armijo_linesearch_segmented)
    _finalize_search = staticmethod(_armijo_finalize)

    @staticmethod
    def _steepest_direction(direction, gradient):
        return direction.copy_(gradient).neg_()

    @staticmethod
    def _count_true(mask):
        return int(mask.sum().item())

    @staticmethod
    def _total_loss(loss):
        return float(loss.sum())

    @staticmethod
    def _cap_step(step, maximum):
        return step.clamp(max=maximum)

    @staticmethod
    def _freeze_direction(direction, active):
        return direction.mul_(active.to(direction.dtype))

    def _initial_step(self, x, value):
        return torch.full((self._n_segments,), value, dtype=x.dtype, device=x.device)

    def _update_trial(self, ctx, step):
        torch.addcmul(ctx.x_backup, ctx.d, self._per_element(step), out=ctx.x)

    @staticmethod
    def _current_gradient(ctx):
        return ctx.param.grad.data.view(-1)

    def _write_history(self, ctx, idx, y, s, keep):
        # Preserve the mature dense/scattered writes and storage identity.
        if self._n_segments == 1:
            ctx.old_dirs_mat[idx, 0].copy_(y)
            ctx.old_stps_mat[idx, 0].copy_(s)
            ctx.x_ref.copy_(ctx.x)
        else:
            keep_elem = self._per_element(keep)
            zero = torch.zeros((), dtype=y.dtype, device=y.device)
            if self._segments_are_dense:
                torch.where(keep_elem, y, zero, out=ctx.old_dirs_mat[idx].view(-1))
                torch.where(keep_elem, s, zero, out=ctx.old_stps_mat[idx].view(-1))
            else:
                self._pad(torch.where(keep_elem, y, zero), out=ctx.old_dirs_mat[idx])
                self._pad(torch.where(keep_elem, s, zero), out=ctx.old_stps_mat[idx])
            ctx.x_ref = torch.where(keep_elem, ctx.x, ctx.x_ref)

    def _history_direction(self, ctx, old_dirs_view, old_stps_view):
        grad_pad = self._pad(
            ctx.flat_grad,
            out=None if self._segments_are_dense else ctx.state["grad_pad"],
        )
        with torch.inference_mode() if grad_pad.is_cuda else nullcontext():
            return ctx.d.copy_(
                self._unpad(
                    _lbfgs_two_loop_impl(
                        grad_pad, old_dirs_view, old_stps_view, ctx.history_start
                    )
                )
            )

    @staticmethod
    def _reset_history(ctx):
        reset = ctx.needs_reset.nonzero(as_tuple=False).squeeze(-1)
        ctx.old_dirs_mat[:, reset, :] = 0.0
        ctx.old_stps_mat[:, reset, :] = 0.0
