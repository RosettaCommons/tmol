"""Mature per-segment L-BFGS policy shared by numerical executors.

The executor owns parameter/gradient storage, segmented reductions, native
Armijo launches and compact linear algebra. This policy owns iteration order,
curvature acceptance, circular history, restart/retirement, independent
convergence and objective evaluation. Storage hooks return their resulting
array; an executor may preserve storage identity or replace functional arrays.
``_step_setup`` supplies a host context and ``_wrap_closure`` binds objectives.
"""


class SegmentedLBFGSPolicy:
    """Execution-independent optimizer orchestration; not a public optimizer."""

    def _compute_search_direction(self, ctx):
        """L-BFGS update + two-loop recursion."""
        flat_grad = ctx.flat_grad
        d = ctx.d
        x = ctx.x

        if ctx.state["n_iter"] == 1:
            # initialize
            ctx.d = self._steepest_direction(d, flat_grad)
            ctx.history_count = 0
        else:
            # do lbfgs update (update memory)
            y = flat_grad - ctx.prev_flat_grad
            s = x - ctx.x_ref  # cumulative displacement since last good step
            # a segment with no curvature of its own contributes no history
            keep = self._seg_sum(y * s) > 1e-6

            if self._any_true(keep):
                # updating memory - write directly into circular buffer
                if ctx.history_count < ctx.history_size:
                    # Still filling up the buffer
                    idx = ctx.history_count
                    ctx.history_count += 1
                else:
                    # Buffer full, overwrite oldest entry
                    idx = ctx.history_start
                    ctx.history_start = (ctx.history_start + 1) % ctx.history_size

                self._write_history(ctx, idx, y, s, keep)

            # compute the approximate (L-BFGS) inverse Hessian
            if ctx.history_count == 0:
                # No history: use steepest descent direction
                ctx.d = self._steepest_direction(d, flat_grad)
            else:
                old_dirs_view = ctx.old_dirs_mat[: ctx.history_count]
                old_stps_view = ctx.old_stps_mat[: ctx.history_count]

                ctx.d = self._history_direction(ctx, old_dirs_view, old_stps_view)

        self._restart_failed_segments(ctx)
        self._freeze_converged(ctx)

    def _restart_failed_segments(self, ctx):
        """Give a segment whose line search failed a fresh steepest descent.

        Its history is dropped (zeroed slots read as absent) and it is marked as
        restarted, so a second failure retires it instead of searching again.
        """
        if not ctx.any_needs_reset:
            if ctx.any_was_reset:
                ctx.was_reset = self._zero(ctx.was_reset)
                ctx.any_was_reset = False
            return
        ctx.was_reset = self._copy(ctx.was_reset, ctx.needs_reset)
        ctx.any_was_reset = True
        self._reset_history(ctx)
        reset_elem = self._per_element(ctx.needs_reset)
        ctx.d = self._copy(ctx.d, self._where(reset_elem, -ctx.flat_grad, ctx.d))
        ctx.x_ref = self._where(reset_elem, ctx.x, ctx.x_ref)
        ctx.needs_reset = self._zero(ctx.needs_reset)
        ctx.any_needs_reset = False

    def _inactive(self, ctx):
        """Segments that no longer move: converged or retired."""
        return ctx.converged | ctx.stalled

    def _freeze_converged(self, ctx):
        """Hold converged or retired segments still by zeroing their direction."""
        if not ctx.any_inactive:
            return
        inactive = self._inactive(ctx)
        ctx.d = self._freeze_direction(ctx.d, self._per_element(~inactive))

    def _directional_derivative(self, ctx, correct=True):
        """Compute and cache the directional derivative g . d per segment."""
        flat_grad, d = ctx.flat_grad, ctx.d
        gtd_seg = self._seg_sum(flat_grad * d)
        if correct:
            for check in (1, 2):
                bad = (gtd_seg > -1e-5) & ~self._inactive(ctx)
                if not self._any_true(bad):
                    break
                bad_elem = self._per_element(bad)
                if check == 1:
                    repaired = d * -self._sign(flat_grad * d)
                else:
                    repaired = -flat_grad
                d = ctx.d = self._copy(d, self._where(bad_elem, repaired, d))
                gtd_seg = self._seg_sum(flat_grad * d)
        ctx.gtd_seg = gtd_seg

    def _segment_line_search(self, ctx, linefn_vec):
        """Line search with an independent step size per segment."""
        # a segment with no downhill direction cannot search; any negative
        # slope is worth searching, however small, since the Armijo test
        # scales with it
        searching = ~self._inactive(ctx) & (ctx.gtd_seg < 0)
        # match Rosetta: start at 2x prev accepted step, capped at 1.0
        start_t = self._cap_step(ctx.t / 0.5, 1.0)

        accepted, phi_accepted, ls_evals, status, trial_is_accepted = self._line_search(
            linefn_vec,
            ctx.gtd_seg,
            ctx.prev_loss_vec,  # energies at x_backup
            start_t,
            searching,
            factor=0.5,
            sigma_decrease=0.1,
            sigma_increase=0.8,
            minstep=self._minstep,
        )
        ctx.ls_evals = ls_evals

        if self.value_only_line_search:
            self._update_trial(ctx, accepted)
            # Refresh only the gradient; keep Armijo energies on one evaluator.
            ctx.closure()
            ctx.loss_vec = phi_accepted
        else:
            if not trial_is_accepted:
                self._update_trial(ctx, accepted)
                ctx.closure()
            ctx.loss_vec = self._last_loss_vec
        # Keep the normal optimization path asynchronous. The scalar total is
        # only needed for human-readable progress output.
        ctx.loss = self._total_loss(ctx.loss_vec) if self.verbose else None

        # a failed search gets one steepest-descent restart, applied on the next
        # iteration so the other segments are not held up; then it is retired
        ctx.t, failed = self._finalize_search(
            status,
            ctx.gtd_seg,
            accepted,
            start_t,
            searching,
            self._minstep,
        )
        any_failed = self._any_true(failed)
        if any_failed:
            ctx.stalled |= failed & ctx.was_reset
            ctx.needs_reset |= failed & ~ctx.was_reset
        # A failure is the only way needs_reset can become true. It is safe to
        # be conservative when all failed segments were already restarted;
        # the next iteration will process an empty mask and clear this flag.
        ctx.any_needs_reset = any_failed

    def _check_segment_convergence(self, ctx, n_iter):
        """Freeze independently converged segments; stop when all are done.

        A segmented run must stop each segment at the same iteration where an
        equivalent one-segment run would stop. Otherwise an early-converged pose
        keeps moving while another pose finishes, making batch minimization
        depend on which other structures happen to share the stack.
        """
        if ctx.fixed_iterations:
            # Benchmarking and truncated inference protocols often require an
            # exact iteration count. Skip convergence reductions and their
            # device-to-host synchronization while preserving failure resets.
            return False

        newly_converged = self._zeros_like(ctx.converged)
        if ctx.gradtol is not None:
            newly_converged |= self._seg_amax(self._abs(ctx.flat_grad)) <= ctx.gradtol
        if ctx.prev_loss_vec is not None:
            dE = self._abs(ctx.loss_vec - ctx.prev_loss_vec)
            small = self._zeros_like(ctx.converged)
            if ctx.atol is not None:
                small |= dE <= ctx.atol
            if ctx.rtol is not None:
                rdiff = (
                    2
                    * dE
                    / (self._abs(ctx.loss_vec) + self._abs(ctx.prev_loss_vec) + 1e-10)
                )
                small |= rdiff <= ctx.rtol
            # A rejected zero step also has dE == 0, but it is not progress or
            # its absence: the next iteration must try the scheduled
            # steepest-descent reset, so it neither counts nor resets the run.
            counted = self._where(small, ctx.small_steps + 1, 0)
            ctx.small_steps = self._copy(
                ctx.small_steps, self._where(ctx.needs_reset, ctx.small_steps, counted)
            )
            newly_converged |= ctx.small_steps >= ctx.patience
        ctx.converged |= newly_converged
        done = self._inactive(ctx)

        n_done = self._count_true(done)
        ctx.any_inactive = n_done != 0
        if self.verbose:
            n_stalled = self._count_true(ctx.stalled)
            print(
                f"  iter {n_iter:4d}  E={ctx.loss:.6f}"
                f"  evals={ctx.ls_evals}"
                f"  done={n_done}/{self._n_segments}"
                + (f"  stalled={n_stalled}" if n_stalled else "")
            )
        if n_done == self._n_segments:
            if self.verbose:
                print(
                    f"  finished: {self._n_segments - n_stalled} converged,"
                    f" {n_stalled} stalled"
                )
            return True
        return False

    def step(self, closure, value_only_closure=None):
        """
        The LBFGS minimization algorithm. Despite the name, this performs the full
        LBFGS minimization trajectory.

        Arguments:
            closure (callable): a function that evaluates energy and gradient
            value_only_closure (callable): a function that evaluates only
                energy. Required when ``value_only_line_search`` is enabled.

        Returns:
            The initial loss, matching the ``Optimizer.step`` convention.
        """
        closure = self._wrap_closure(closure)
        if self.value_only_line_search:
            if value_only_closure is None:
                raise ValueError(
                    "value_only_closure is required when value_only_line_search "
                    "is enabled"
                )
            line_search_closure = self._wrap_closure(value_only_closure)
        else:
            line_search_closure = closure
        ctx = self._step_setup(closure)

        def linefn(alpha_vec):
            """Evaluate every segment at its own step size."""
            self.ls_func_evals += 1
            # Direct parameter update - eliminates _set_x_from_flat overhead
            self._update_trial(ctx, alpha_vec)
            line_search_closure()
            return self._last_loss_vec

        current_evals = 1
        n_iter = 0
        while n_iter < ctx.max_iter:
            n_iter += 1
            ctx.state["n_iter"] += 1

            self._compute_search_direction(ctx)

            if ctx.prev_flat_grad is None:
                ctx.prev_flat_grad = self._clone(ctx.flat_grad)
            else:
                ctx.prev_flat_grad = self._copy(ctx.prev_flat_grad, ctx.flat_grad)
            ctx.prev_loss_vec = ctx.loss_vec

            # Armijo updates will track step length during optimization
            # thus, "learning rate" is only applied for the initial step
            if ctx.state["n_iter"] == 1:
                ctx.t = self._initial_step(ctx.x, ctx.lr)

            # directional derivative
            # (fd) this is some hacky stuff I put in R3 that is not typically part
            # (fd)   of lbfgs because the bfgs update had us frequently searching
            # (fd)   in positive grad directions
            # check 1: if dir. deriv. is positive, flip signs of positive components
            # check 2: if derivative is still positive, reset Hessian
            self._directional_derivative(ctx)

            # define the line search function
            # we do not need to compute gradients in here
            self.ls_func_evals = 0

            # Optimization: save original position and work directly with param.data
            ctx.x_backup = self._copy(ctx.x_backup, ctx.x)

            self._segment_line_search(ctx, linefn)

            ctx.flat_grad = self._current_gradient(ctx)

            # update func eval
            current_evals += self.ls_func_evals
            ctx.state["func_evals"] += self.ls_func_evals
            if self.value_only_line_search:
                current_evals += 1
                ctx.state["func_evals"] += 1

            if self._check_segment_convergence(ctx, n_iter):
                break

        if self.verbose:
            print(
                f"  LBFGS_Armijo done: {n_iter} iters,"
                f" {current_evals} func evals,"
                f" E={ctx.loss:.4f}"
            )

        # d and x_backup persist via preallocation in state; save the rest.
        ctx.state["d"] = ctx.d
        ctx.state["x_backup"] = ctx.x_backup
        ctx.state["converged"] = ctx.converged
        ctx.state["stalled"] = ctx.stalled
        ctx.state["t"] = ctx.t
        ctx.state["history_start"] = ctx.history_start
        ctx.state["history_count"] = ctx.history_count
        ctx.state["old_dirs_mat"] = ctx.old_dirs_mat
        ctx.state["old_stps_mat"] = ctx.old_stps_mat
        ctx.state["prev_flat_grad"] = ctx.prev_flat_grad
        ctx.state["prev_loss_vec"] = ctx.prev_loss_vec
        ctx.state["x_ref"] = ctx.x_ref
        ctx.state["needs_reset"] = ctx.needs_reset
        ctx.state["was_reset"] = ctx.was_reset
        ctx.state["small_steps"] = ctx.small_steps
        ctx.state["any_needs_reset"] = ctx.any_needs_reset
        ctx.state["any_was_reset"] = ctx.any_was_reset
        ctx.state["any_inactive"] = ctx.any_inactive

        return ctx.orig_loss
