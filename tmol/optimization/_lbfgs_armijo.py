from contextlib import nullcontext
from typing import Any

import torch
from torch.optim import Optimizer

from tmol_core.optimization.armijo import armijo_linesearch_segmented
from tmol_core.optimization.lbfgs import compact_lbfgs_direction
from tmol_core.optimization.segmented import SegmentedLBFGSPolicy
from tmol_core.optimization.setup import (
    LBFGS_DEFAULTS,
    migrate_settings,
    validate_settings,
)

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


def _lbfgs_two_loop_impl(grad, dirs, stps, history_start=0, double_scratch=None):
    promoted_history = None
    if double_scratch is not None and stps.dtype != torch.float64:
        m = stps.shape[0]
        promoted_history = (
            double_scratch[0][:m].view_as(dirs).copy_(dirs),
            double_scratch[1][:m].view_as(stps).copy_(stps),
        )
    unbatched = grad.dim() == 1
    poses = 1 if unbatched else grad.shape[0]
    return compact_lbfgs_direction(
        grad,
        dirs,
        stps,
        history_start,
        execution=_TorchCompactLBFGS,
        promoted_history=promoted_history,
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
        gradtol (float): if set, energy convergence also requires
            max_i |df/dx_i| <= gradtol. A failed line search below this
            threshold converges immediately (default: None).
        history_size (int): update history size (default: 200).
        segment_ids (Tensor): the segment (e.g. pose) each parameter element
            belongs to; each segment is minimized independently (default: one)
        n_segments (int): explicit pose count, retaining empty/frozen segments.
            If omitted, segment IDs must cover every segment.
        value_only_line_search (bool): evaluate Armijo trial points with a
            separate value-only closure and refresh the gradient only at the
            accepted point (default: False)
        fixed_iterations (bool): run exactly ``max_iter`` iterations instead of
            evaluating convergence criteria (default: False).
    """

    supports_segments = True
    supports_segment_count = True

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
        lr=LBFGS_DEFAULTS["lr"],
        max_iter=LBFGS_DEFAULTS["max_iter"],
        rtol=LBFGS_DEFAULTS["rtol"],
        atol=LBFGS_DEFAULTS["atol"],
        gradtol=LBFGS_DEFAULTS["gradtol"],
        history_size=LBFGS_DEFAULTS["history_size"],
        minstep=LBFGS_DEFAULTS["minstep"],
        verbose=LBFGS_DEFAULTS["verbose"],
        segment_ids=None,
        fixed_iterations=LBFGS_DEFAULTS["fixed_iterations"],
        *,
        n_segments=None,
        patience=LBFGS_DEFAULTS["patience"],
        value_only_line_search: bool = LBFGS_DEFAULTS["value_only_line_search"],
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
        validate_settings(group, value_only_line_search)

        if len(self.param_groups) != 1:
            raise ValueError(
                "LBFGS doesn't support per-parameter options " "(parameter groups)"
            )

        self._params = self.param_groups[0]["params"]
        self._minstep = minstep
        self.verbose = verbose
        self.value_only_line_search = value_only_line_search

        self._last_loss_vec = None
        self._sum_scratch = None
        self._double_history_scratch = None
        if segment_ids is None:
            # the whole parameter as one segment: the same code path as a stack
            segment_ids = torch.zeros(
                self._params[0].numel(),
                dtype=torch.int64,
                device=self._params[0].device,
            )
        self._init_segments(segment_ids, n_segments)

    def __setstate__(self, state):
        super().__setstate__(state)
        group = self.param_groups[0]
        migrate_settings(group, float(torch.finfo(group["params"][0].dtype).eps ** 0.5))

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

    def _init_segments(self, segment_ids, n_segments=None):
        assert len(self._params) == 1, "segment_ids requires a single tensor"
        super()._init_segments(segment_ids, n_segments)

    _parameter_size = staticmethod(torch.Tensor.numel)

    @staticmethod
    def _segment_vector(segment_ids):
        return segment_ids.reshape(-1).to(torch.int64)

    @staticmethod
    def _all_true(value):
        return bool(value.all())

    @staticmethod
    def _max_index(value):
        return int(value.max().item())

    @staticmethod
    def _segment_counts(segment_ids, n_segments):
        return torch.bincount(segment_ids, minlength=n_segments)

    @staticmethod
    def _dense_segment_order(segment_ids, n_segments, segment_size):
        expected = torch.arange(n_segments, device=segment_ids.device)
        return torch.equal(segment_ids, expected.repeat_interleave(segment_size))

    @staticmethod
    def _segment_ranks(segment_ids, counts):
        order = torch.argsort(segment_ids, stable=True)
        offsets = torch.cumsum(counts, 0) - counts
        rank = torch.empty_like(order)
        arange = torch.arange(segment_ids.numel(), device=order.device)
        rank[order] = arange - offsets[segment_ids[order]]
        return rank

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

    def _double_history(self, hist):
        """Reused float64 (dirs, stps) buffers shaped like the history matrices."""
        if hist.dtype == torch.float64:
            return None
        scratch = getattr(self, "_double_history_scratch", None)
        if (
            scratch is None
            or scratch[0].shape != hist.shape
            or scratch[0].device != hist.device
        ):
            scratch = tuple(
                torch.empty(hist.shape, dtype=torch.float64, device=hist.device)
                for _ in range(2)
            )
            self._double_history_scratch = scratch
        return scratch

    def _seg_amax(self, values):
        """Maximum of a parameter-shaped vector within each segment."""
        if self._segments_are_dense:
            return values.view(self._n_segments, self._segment_size).amax(-1)
        out = torch.zeros(self._n_segments, dtype=values.dtype, device=values.device)
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
        Promote lower-precision objectives to parameter precision for the native
        line search; never lower the precision of an objective.
        """

        def wrapped():
            loss = closure()
            dtype = self._params[0].dtype
            if torch.is_tensor(loss) and loss.dtype != dtype:
                if torch.promote_types(loss.dtype, dtype) != dtype:
                    raise TypeError(
                        "objective precision exceeds optimizer parameter precision"
                    )
                loss = loss.to(dtype=dtype)
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

    def _setup_group(self):
        assert len(self.param_groups) == 1
        assert len(self._params) == 1, "This version requires single tensor"
        return self.param_groups[0]

    def _setup_parameter(self):
        return self._params[0]

    def _setup_state(self, param):
        return self.state[param]

    def _dtype_tolerance(self):
        return float(torch.finfo(self._params[0].dtype).eps ** 0.5)

    @staticmethod
    def _setup_vectors(param):
        return param.data.view(-1), param.grad.data.view(-1)

    _empty_like = staticmethod(torch.empty_like)

    @staticmethod
    def _allocate(shape, reference, kind, *, zero):
        dtype = {"real": reference.dtype, "bool": torch.bool, "index": torch.int64}[
            kind
        ]
        allocate = torch.zeros if zero else torch.empty
        return allocate(shape, dtype=dtype, device=reference.device)

    @staticmethod
    def _state_cast(value, kind):
        return value.bool() if kind == "bool" else value.long()

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
                        grad_pad,
                        old_dirs_view,
                        old_stps_view,
                        ctx.history_start,
                        self._double_history(ctx.old_dirs_mat),
                    )
                )
            )

    @staticmethod
    def _reset_history(ctx):
        reset = ctx.needs_reset.nonzero(as_tuple=False).squeeze(-1)
        ctx.old_dirs_mat[:, reset, :] = 0.0
        ctx.old_stps_mat[:, reset, :] = 0.0
