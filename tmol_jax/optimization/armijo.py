"""JAX execution of the common mature segmented L-BFGS/Armijo policy."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from tmol_core.optimization.armijo import armijo_linesearch_segmented
from tmol_core.optimization.segmented import SegmentedLBFGSPolicy
from tmol_core.optimization.setup import LBFGS_DEFAULTS, validate_settings
from ._armijo_native import ArmijoExecution, finalize
from ._compact_lbfgs import compact_direction
from ._armijo_checkpoint import checkpoint_settings, restore_checkpoint, make_checkpoint


class _Optimizer(SegmentedLBFGSPolicy):
    """Array allocation, objective binding and functional storage hooks only."""

    def __init__(self, x, segment_ids, settings):
        self.x = x.reshape(-1)
        self.settings = settings
        validate_settings(settings, settings["value_only_line_search"])
        self._minstep = settings["minstep"]
        self.verbose = settings["verbose"]
        self.value_only_line_search = settings["value_only_line_search"]
        self.state = {}
        self._last_loss_vec = None
        self._init_segments(segment_ids)

    def _setup_group(self):
        return self.settings

    def _setup_parameter(self):
        return self.x

    def _setup_state(self, parameter):
        return self.state

    def _dtype_tolerance(self):
        return float(np.finfo(self.x.dtype).eps ** 0.5)

    def _setup_vectors(self, parameter):
        return self.x, self.gradient

    _segment_vector = staticmethod(lambda ids: jnp.asarray(ids, jnp.int32).reshape(-1))
    _parameter_size = staticmethod(lambda value: value.size)
    _all_true = staticmethod(lambda value: bool(np.asarray(value.all())))
    _max_index = staticmethod(lambda value: int(np.asarray(value.max())))
    _segment_counts = staticmethod(lambda ids, size: jnp.bincount(ids, length=size))

    @staticmethod
    def _dense_segment_order(ids, size, width):
        return bool(
            np.asarray(jnp.array_equal(ids, jnp.repeat(jnp.arange(size), width)))
        )

    @staticmethod
    def _segment_ranks(ids, counts):
        order = jnp.argsort(ids, stable=True)
        starts = jnp.cumsum(counts) - counts
        ranks = jnp.arange(ids.size) - jnp.repeat(
            starts, counts, total_repeat_length=ids.size
        )
        return jnp.empty_like(order).at[order].set(ranks)

    @staticmethod
    def _allocate(shape, reference, kind, *, zero):
        dtype = {"real": reference.dtype, "bool": jnp.bool_, "index": jnp.int32}[kind]
        return jnp.zeros(shape, dtype) if zero else jnp.empty(shape, dtype)

    _state_cast = staticmethod(
        lambda value, kind: value.astype(jnp.bool_ if kind == "bool" else jnp.int32)
    )
    _empty_like = staticmethod(jnp.empty_like)
    _copy = staticmethod(lambda target, source: source)
    _clone = staticmethod(lambda value: value)
    _zero = _zeros_like = staticmethod(jnp.zeros_like)
    _where = staticmethod(jnp.where)
    _abs = staticmethod(jnp.abs)
    _sign = staticmethod(jnp.sign)
    _any_true = staticmethod(ArmijoExecution.any_true)
    _count_true = staticmethod(lambda value: int(np.asarray(value.sum())))
    _total_loss = staticmethod(lambda value: float(np.asarray(value.sum())))
    _cap_step = staticmethod(lambda value, maximum: jnp.minimum(value, maximum))
    _steepest_direction = staticmethod(lambda direction, gradient: -gradient)
    _freeze_direction = staticmethod(lambda direction, active: direction * active)
    _finalize_search = staticmethod(finalize)
    _line_search = staticmethod(
        partial(
            armijo_linesearch_segmented, execution=ArmijoExecution, combine_checks=True
        )
    )

    def _wrap_closure(self, closure):
        def wrapped():
            values = closure()
            if values.size != self._n_segments:
                raise ValueError("objective must return one energy per segment")
            self._last_loss_vec = values.reshape(self._n_segments)
            return values.sum()

        return wrapped

    def _pad(self, values):
        shape = (self._n_segments, self._segment_size)
        if self._segments_are_dense:
            return values.reshape(shape)
        return (
            jnp.zeros(np.prod(shape), values.dtype)
            .at[self._pad_index]
            .set(values)
            .reshape(shape)
        )

    def _unpad(self, values):
        return (
            values.reshape(-1)
            if self._segments_are_dense
            else values.reshape(-1)[self._pad_index]
        )

    def _seg_sum(self, values):
        return self._pad(values).sum(-1)

    def _seg_amax(self, values):
        return self._pad(values).max(-1)

    def _per_element(self, values):
        return values[0] if self._n_segments == 1 else values[self._segment_ids]

    def _initial_step(self, x, value):
        return jnp.full((self._n_segments,), value, x.dtype)

    def _update_trial(self, ctx, step):
        self.x = ctx.x = _trial(ctx.x_backup, ctx.d, self._per_element(step))

    def _current_gradient(self, ctx):
        return self.gradient

    def _write_history(self, ctx, index, y, s, keep):
        keep_element = self._per_element(keep)
        ctx.old_dirs_mat = ctx.old_dirs_mat.at[index].set(
            self._pad(jnp.where(keep_element, y, 0))
        )
        ctx.old_stps_mat = ctx.old_stps_mat.at[index].set(
            self._pad(jnp.where(keep_element, s, 0))
        )
        ctx.x_ref = jnp.where(keep_element, ctx.x, ctx.x_ref)

    def _history_direction(self, ctx, differences, steps):
        return self._unpad(
            compact_direction(
                self._pad(ctx.flat_grad), differences, steps, ctx.history_start
            )
        )

    @staticmethod
    def _reset_history(ctx):
        keep = ~ctx.needs_reset[None, :, None]
        ctx.old_dirs_mat = jnp.where(keep, ctx.old_dirs_mat, 0)
        ctx.old_stps_mat = jnp.where(keep, ctx.old_stps_mat, 0)


@jax.jit
def _trial(x, direction, step):
    return x + direction * step


def armijo_minimize(
    fn,
    x0,
    *,
    segment_ids=None,
    grad_mask=None,
    value_and_grad=None,
    value_only=None,
    checkpoint=None,
    **settings,
):
    """Minimize independent segments using mature TMol settings and native search.

    ``fn(x)`` returns one energy per segment (a scalar for one segment).
    ``segment_ids`` maps every parameter element to its independent objective;
    IDs must cover consecutive integers from zero. A boolean movement mask
    freezes elements. The input is unchanged. Settings use the shared Torch
    names/defaults, including ``max_iter``, ``history_size``, ``rtol``, ``atol``,
    ``gradtol``, ``patience``, ``fixed_iterations`` and ``value_only_line_search``.
    Preparation and convergence control are eager; numerical objectives execute
    on the input device. This API does not differentiate optimizer trajectories.
    Pass a returned ``checkpoint`` with its ``x`` to resume the common history.
    Segment identities, movement mask, coordinate dtype and history capacity must
    match. The caller must retain the same objective and parameter context.
    Prepared ``value_and_grad`` and ``value_only`` callbacks may reuse compiled
    execution; both receive the original parameter shape.
    """
    settings = checkpoint_settings(checkpoint, settings)
    if checkpoint is not None:
        segment_ids = checkpoint["segment_ids"] if segment_ids is None else segment_ids
        grad_mask = checkpoint["grad_mask"] if grad_mask is None else grad_mask
    unknown = settings.keys() - LBFGS_DEFAULTS.keys()
    if unknown:
        raise TypeError("unknown optimizer settings: " + ", ".join(sorted(unknown)))
    settings = dict(LBFGS_DEFAULTS, **settings)
    x = jnp.asarray(x0)
    if x.dtype not in (jnp.float32, jnp.float64) or not x.size:
        raise TypeError("x0 must be a nonempty float32/float64 array")
    if settings["max_iter"] < 0 or settings["history_size"] < 1:
        raise ValueError("max_iter must be nonnegative and history_size positive")
    ids = (
        jnp.zeros(x.shape, jnp.int32)
        if segment_ids is None
        else jnp.asarray(segment_ids)
    )
    if ids.shape != x.shape or not jnp.issubdtype(ids.dtype, jnp.integer):
        raise ValueError("segment_ids must be integers with the parameter shape")
    mask = (
        jnp.ones(x.shape, bool) if grad_mask is None else jnp.asarray(grad_mask, bool)
    )
    if mask.shape != x.shape:
        raise ValueError("grad_mask must have the parameter shape")
    with jax.default_device(x.device):
        ids = jax.device_put(ids, x.device)
        mask = jax.device_put(mask, x.device)
        optimizer = _Optimizer(x, ids, settings)
        optimizer.state = restore_checkpoint(checkpoint, x, ids, mask, settings)

        def total(flat):
            values = fn(flat.reshape(x.shape))
            return values.sum(), values

        evaluate = jax.jit(jax.value_and_grad(total, has_aux=True))
        if value_only is None:
            value_only = jax.jit(lambda flat: fn(flat.reshape(x.shape)))

        def closure():
            if value_and_grad is None:
                (_, values), gradient = evaluate(optimizer.x)
            else:
                values, gradient = value_and_grad(optimizer.x.reshape(x.shape))
                gradient = gradient.reshape(-1)
            optimizer.gradient = gradient * mask.reshape(-1)
            return values

        initial = optimizer.step(
            closure, lambda: value_only(optimizer.x.reshape(x.shape))
        )
        return {
            "x": optimizer.x.reshape(x.shape),
            "initial_fun": initial,
            "fun": optimizer._last_loss_vec.sum(),
            "per_segment_fun": optimizer._last_loss_vec,
            "grad": optimizer.gradient.reshape(x.shape),
            "n_iter": optimizer.state["n_iter"],
            "func_evals": optimizer.state["func_evals"],
            "converged": optimizer.state["converged"],
            "stalled": optimizer.state["stalled"],
            "state": optimizer.state,
            "checkpoint": make_checkpoint(
                optimizer.x.reshape(x.shape), ids, mask, settings, optimizer.state
            ),
        }
