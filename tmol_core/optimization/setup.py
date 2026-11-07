"""Shared segmented optimizer configuration and persistent-state preparation.

Executors own array storage and objective binding. The common policy owns buffer
shapes, lazy initialization, legacy checkpoint migration and context fields.
Allocation hooks distinguish real, boolean and index storage; all hooks return
arrays so immutable executors and stable Torch storage use the same policy.
"""

from math import isfinite
from numbers import Integral, Real
from types import SimpleNamespace

LBFGS_DEFAULTS = dict(
    lr=1,
    max_iter=200,
    rtol=None,
    atol=1e-2,
    gradtol=None,
    history_size=128,
    minstep=1e-12,
    verbose=False,
    fixed_iterations=False,
    patience=5,
    value_only_line_search=False,
)


def validate_settings(group, value_only_line_search):
    patience = group["patience"]
    if isinstance(patience, bool) or not isinstance(patience, Integral) or patience < 1:
        raise ValueError("patience must be a positive integer")
    for name in ("rtol", "atol", "gradtol"):
        value = group[name]
        if value is not None and (
            not isinstance(value, Real) or not isfinite(value) or value < 0
        ):
            raise ValueError(f"{name} must be None or a finite nonnegative number")

    if not isinstance(value_only_line_search, bool):
        raise TypeError("value_only_line_search must be bool")


def migrate_settings(group, dtype_tolerance):
    """Restore pre-patience checkpoint semantics without changing new defaults."""
    if "patience" not in group:
        group["patience"] = 1
        for name in ("rtol", "atol"):
            if group[name] is None:
                group[name] = dtype_tolerance


class SegmentedLBFGSSetup:
    def _init_segments(self, segment_ids, n_segments=None):
        """Set up the mapping from parameter elements to independent blocks.

        Builds the index that scatters a parameter-shaped vector into a dense
        (n_segments, segment_size) layout, so all per-segment reductions are
        batched tensor ops rather than a loop over segments. An explicit count
        preserves empty segments, including trailing fully frozen poses.
        """
        param = self._setup_parameter()
        segment_ids = self._segment_vector(segment_ids)
        assert segment_ids.shape[0] == self._parameter_size(
            param
        ), "segment_ids needs one entry per parameter element"
        assert self._all_true(segment_ids >= 0), "segment_ids must be non-negative"

        explicit_count = n_segments is not None
        if explicit_count:
            if (
                isinstance(n_segments, bool)
                or not isinstance(n_segments, Integral)
                or n_segments < 1
            ):
                raise ValueError("n_segments must be a positive integer")
            if not self._all_true(segment_ids < n_segments):
                raise ValueError("segment_ids must be smaller than n_segments")
        else:
            n_segments = self._max_index(segment_ids) + 1
        if n_segments == 1 and segment_ids.shape[0]:
            self._segment_ids = segment_ids
            self._n_segments = 1
            self._segment_size = segment_ids.shape[0]
            self._segments_are_dense = True
            self._pad_index = None
            return

        counts = self._segment_counts(segment_ids, n_segments)
        if not explicit_count:
            assert self._all_true(counts > 0), "segment_ids must not skip a segment"
        # Empty/frozen poses have zero gradients. One padding slot also makes
        # reductions well-defined when every pose is frozen.
        segment_size = max(1, self._max_index(counts))

        # Avoid sorting the overwhelmingly common dense layout: one pose, or
        # equal-sized poses stored contiguously. Dense reductions only reshape
        # the parameter vector and never consult ``_pad_index``.
        segments_are_dense = False
        if self._all_true(counts == segment_size):
            segments_are_dense = self._dense_segment_order(
                segment_ids, n_segments, segment_size
            )

        self._segment_ids = segment_ids
        self._n_segments = n_segments
        self._segment_size = segment_size
        self._segments_are_dense = segments_are_dense
        if segments_are_dense:
            self._pad_index = None
            return

        # rank of each element within its own segment
        rank = self._segment_ranks(segment_ids, counts)

        self._pad_index = segment_ids * segment_size + rank

    def _step_setup(self, closure):
        """Prepare for L-BFGS:
        read config
        initialize state (history buffers, x_ref, preallocate d and x_backup)
        evaluate initial loss and gradient
        returns a SimpleNamespace ctx"""
        # lbfgs only works w/ single parameter group
        group = self._setup_group()
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
        dtype_based_tol = self._dtype_tolerance()
        if rtol is not None and 0 < rtol < dtype_based_tol:
            print(f"  WARNING: rtol ({rtol}) is too low for dtype! ({dtype_based_tol})")
        if atol is not None and 0 < atol < dtype_based_tol:
            print(f"  WARNING: atol ({atol}) is too low for dtype! ({dtype_based_tol})")

        param = self._setup_parameter()

        # NOTE: LBFGS has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        state = self._setup_state(param)
        state.setdefault("func_evals", 0)
        state.setdefault("n_iter", 0)

        # evaluate initial f(x)
        orig_loss = closure()
        # The optimizer uses the per-segment tensor below for all line-search
        # and convergence decisions. Materializing a Python scalar here forces
        # a device synchronization and is only useful for verbose reporting.
        loss = self._total_loss(orig_loss) if self.verbose else None
        loss_vec = self._last_loss_vec
        state["func_evals"] += 1

        x, flat_grad = self._setup_vectors(param)

        # Preallocate on first call. `d` and `x_backup` must have stable
        # tensor identity so linefn (defined once in step) can capture them.
        L = x.shape[0]
        if state.get("d") is None:
            state["d"] = self._empty_like(x)
            state["x_backup"] = self._empty_like(x)
            if self._segment_ids is None:
                hist_shape = (history_size, L)
            else:
                hist_shape = (history_size, self._n_segments, self._segment_size)
                if not self._segments_are_dense:
                    state["grad_pad"] = self._allocate(
                        hist_shape[1:], x, "real", zero=True
                    )
            # Only ``history_count`` written slots are consumed. Non-dense
            # segmented writes explicitly zero their own padding.
            state["old_dirs_mat"] = self._allocate(hist_shape, x, "real", zero=False)
            state["old_stps_mat"] = self._allocate(hist_shape, x, "real", zero=False)
            state["history_start"] = 0  # Circular buffer start index
            state["history_count"] = 0  # Number of items in history
            state["x_ref"] = self._clone(x)  # reference position for s computation
            # stationary segments; segments that gave up; segments owed a
            # steepest-descent restart after a failed line search
            state["converged"] = self._allocate(
                (self._n_segments,), x, "bool", zero=True
            )
            state["stalled"] = self._allocate((self._n_segments,), x, "bool", zero=True)
            state["needs_reset"] = self._allocate(
                (self._n_segments,), x, "bool", zero=True
            )
            state["was_reset"] = self._allocate(
                (self._n_segments,), x, "bool", zero=True
            )
            state["any_was_reset"] = False
        # consecutive small iterations per segment; old state dictionaries lack it
        if state.get("small_steps") is None:
            state["small_steps"] = self._allocate(
                (self._n_segments,), x, "index", zero=True
            )

        # State loading casts tensor masks to the parameter's floating dtype.
        for key in ("converged", "stalled", "needs_reset", "was_reset"):
            state[key] = self._state_cast(state[key], "bool")
        state["small_steps"] = self._state_cast(state["small_steps"], "index")

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
