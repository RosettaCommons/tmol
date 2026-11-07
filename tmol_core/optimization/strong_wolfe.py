"""Host L-BFGS/strong-Wolfe policy shared by ligand refinement executors.

Adapted from PyTorch torch/optim/lbfgs.py (BSD-3-Clause); copyright notices and
license are retained in PYTORCH_LICENSE. Scalar control, history updates and
stopping rules match the mature optimizer. Array operations remain compiled
backend primitives; this module imports neither framework nor NumPy.
"""

from .lbfgs import two_loop_direction


def _cubic_interpolate(execution, x1, f1, g1, x2, f2, g2, bounds=None):
    # ported from https://github.com/torch/optim/blob/master/polyinterp.lua
    # Compute bounds of interpolation area
    if bounds is not None:
        xmin_bound, xmax_bound = bounds
    else:
        xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)

    # Code for most common case: cubic interpolation of 2 points
    #   w/ function and derivative values for both
    # Solution in this case (where x2 is the farthest point):
    #   d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
    #   d2 = sqrt(d1^2 - g1*g2);
    #   min_pos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
    #   t_new = min(max(min_pos,xmin_bound),xmax_bound);
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1**2 - g1 * g2
    if d2_square >= 0:
        d2 = execution.sqrt(d2_square)
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        return min(max(min_pos, xmin_bound), xmax_bound)
    else:
        return (xmin_bound + xmax_bound) / 2.0


def strong_wolfe(
    execution,
    obj_func,
    x,
    t,
    d,
    f,
    g,
    gtd,
    c1=1e-4,
    c2=0.9,
    tolerance_change=1e-9,
    max_ls=25,
):
    """Return value, gradient, step and evaluation count from bracket/zoom.

    ``obj_func(x, step, direction)`` evaluates one trial. ``max_ls`` counts
    refinement iterations after the initial evaluation, as in the mature
    optimizer. Frontends own objective validation and coordinate masks.
    """
    # ported from https://github.com/torch/optim/blob/master/lswolfe.lua
    d_norm = execution.abs_max(d)
    g = execution.clone(g)
    # evaluate objective and gradient using initial step
    f_new, g_new = obj_func(x, t, d)
    ls_func_evals = 1
    gtd_new = execution.dot(g_new, d)

    # bracket an interval containing a point satisfying the Wolfe criteria
    t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
    done = False
    ls_iter = 0
    while ls_iter < max_ls:
        # check conditions
        if f_new > (f + c1 * t * gtd) or (ls_iter > 1 and f_new >= f_prev):
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, execution.clone(g_new)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        if abs(gtd_new) <= -c2 * gtd:
            bracket = [t]
            bracket_f = [f_new]
            bracket_g = [g_new]
            done = True
            break

        if gtd_new >= 0:
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, execution.clone(g_new)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        # interpolate
        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10
        tmp = t
        t = _cubic_interpolate(
            execution,
            t_prev,
            f_prev,
            gtd_prev,
            t,
            f_new,
            gtd_new,
            bounds=(min_step, max_step),
        )

        # next step
        t_prev = tmp
        f_prev = f_new
        g_prev = execution.clone(g_new)
        gtd_prev = gtd_new
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = execution.dot(g_new, d)
        ls_iter += 1

    # reached max number of iterations?
    if ls_iter == max_ls:
        bracket = [0, t]
        bracket_f = [f, f_new]
        bracket_g = [g, g_new]

    # zoom phase: we now have a point satisfying the criteria, or
    # a bracket around it. We refine the bracket until we find the
    # exact point satisfying the criteria
    insuf_progress = False
    # find high and low points in bracket
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)
    while not done and ls_iter < max_ls:
        # line-search bracket is so small
        if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:
            break

        # compute new trial value
        t = _cubic_interpolate(
            execution,
            bracket[0],
            bracket_f[0],
            bracket_gtd[0],
            bracket[1],
            bracket_f[1],
            bracket_gtd[1],
        )

        # test that we are making sufficient progress:
        # in case `t` is so close to boundary, we mark that we are making
        # insufficient progress, and if
        #   + we have made insufficient progress in the last step, or
        #   + `t` is at one of the boundary,
        # we will move `t` to a position which is `0.1 * len(bracket)`
        # away from the nearest boundary point.
        eps = 0.1 * (max(bracket) - min(bracket))
        if min(max(bracket) - t, t - min(bracket)) < eps:
            # interpolation close to boundary
            if insuf_progress or t >= max(bracket) or t <= min(bracket):
                # evaluate at 0.1 away from boundary
                if abs(t - max(bracket)) < abs(t - min(bracket)):
                    t = max(bracket) - eps
                else:
                    t = min(bracket) + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        # Evaluate new point
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = execution.dot(g_new, d)
        ls_iter += 1

        if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos]:
            # Armijo condition not satisfied or not lower than lowest point
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = execution.clone(g_new)
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if abs(gtd_new) <= -c2 * gtd:
                # Wolfe conditions satisfied
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                # old high becomes new low
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]

            # new point becomes new low
            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
            bracket_g[low_pos] = execution.clone(g_new)
            bracket_gtd[low_pos] = gtd_new

    # return stuff
    t = bracket[low_pos]
    f_new = bracket_f[low_pos]
    g_new = bracket_g[low_pos]
    return f_new, g_new, t, ls_func_evals


# Kept for callers of the initial shared implementation.
_strong_wolfe = strong_wolfe


class _HostTwoLoop:
    def __init__(self, execution):
        self.dot = execution.dot
        self.add_scaled = execution.add_scaled

    @staticmethod
    def zero_coefficients(inverse_curvatures):
        return [None] * len(inverse_curvatures)

    @staticmethod
    def set_coefficient(coefficients, index, value):
        coefficients[index] = value
        return coefficients

    @staticmethod
    def fori_loop(start, stop, body, state):
        for index in range(start, stop):
            state = body(index, state)
        return state


def _validate_wolfe_settings(c1, c2, line_search_max_iter, relative_loss_tolerance):
    if not 0 < c1 < c2 < 1:
        raise ValueError("expected 0 < c1 < c2 < 1")
    if line_search_max_iter is not None and line_search_max_iter < 1:
        raise ValueError("line_search_max_iter must be positive")
    if relative_loss_tolerance is not None and relative_loss_tolerance < 0:
        raise ValueError("relative_loss_tolerance must be nonnegative")


def lbfgs_minimize(
    value_and_grad,
    x,
    execution,
    *,
    max_iter,
    history_size=50,
    lr=1,
    max_eval=None,
    tolerance_grad=1e-7,
    tolerance_change=1e-9,
    c1=1e-4,
    c2=0.9,
    line_search_max_iter=None,
    relative_loss_tolerance=None,
    accepted_step=None,
):
    """Minimize one flat real coordinate array with mature strong-Wolfe policy.

    ``value_and_grad(x)`` returns a scalar objective and a same-shaped gradient.
    Execution supplies clone, dot, add_scaled(x,y,alpha), abs_max, abs_sum and
    sqrt. Reductions return backend scalars to preserve input precision. Neither
    execution nor the callback may mutate supplied arrays; add_scaled returns
    x + alpha*y with the backend's native scaled-add semantics.

    Defaults retain the mature stopping order and evaluation budget. Pass
    ``max_eval=float("inf")`` for no total evaluation limit, and
    ``tolerance_change=None`` to disable absolute loss/step/directional stops.
    ``line_search_max_iter`` caps objective evaluations per line search.
    ``relative_loss_tolerance`` optionally tests twice the absolute loss change
    against the tolerance times the sum of absolute losses plus 1e-10.
    ``accepted_step(iteration, x, value, gradient)`` observes
    each accepted update before stopping checks; it must not mutate arrays.
    An optional execution.two_loop hook accepts gradient, chronological steps,
    differences, inverse curvatures and scale; compiled adapters should call
    the shared two_loop_direction recurrence with their storage/loop executor.
    """
    if max_iter < 0 or history_size < 1 or lr < 0:
        raise ValueError("invalid L-BFGS iteration, history or learning-rate setting")
    _validate_wolfe_settings(c1, c2, line_search_max_iter, relative_loss_tolerance)
    if max_eval is None:
        max_eval = max_iter * 5 // 4
    x = execution.clone(x)
    loss, gradient = value_and_grad(x)
    loss = float(loss)
    n_eval, n_iter = 1, 0
    reason = "max_iter"
    old_dirs, old_steps, rho = [], [], []
    previous_gradient, direction, step = None, None, None
    scale = 1
    two_loop = getattr(execution, "two_loop", None)
    host_two_loop = _HostTwoLoop(execution)
    if execution.abs_max(gradient) <= tolerance_grad:
        reason = "gradient_tolerance"
    else:
        while n_iter < max_iter:
            n_iter += 1
            if n_iter == 1:
                direction = -gradient
            else:
                y, s = gradient - previous_gradient, direction * step
                ys = execution.dot(y, s)
                if ys > 1e-10:
                    if len(old_dirs) == history_size:
                        old_dirs.pop(0)
                        old_steps.pop(0)
                        rho.pop(0)
                    old_dirs.append(y)
                    old_steps.append(s)
                    rho.append(1.0 / ys)
                    scale = ys / execution.dot(y, y)
                if two_loop is None:
                    direction = two_loop_direction(
                        gradient,
                        old_steps,
                        old_dirs,
                        rho,
                        scale,
                        len(old_dirs),
                        host_two_loop,
                    )
                else:
                    direction = two_loop(gradient, old_steps, old_dirs, rho, scale)
            previous_gradient = execution.clone(gradient)
            previous_loss = loss
            step = (
                min(1.0, 1.0 / execution.abs_sum(gradient)) * lr if n_iter == 1 else lr
            )
            gtd = execution.dot(gradient, direction)
            if tolerance_change is not None and gtd > -tolerance_change:
                reason = "directional_derivative"
                break

            def evaluate(base, step, direction):
                value, grad = value_and_grad(
                    execution.add_scaled(base, direction, step)
                )
                return float(value), grad

            loss, gradient, step, evaluations = strong_wolfe(
                execution,
                evaluate,
                x,
                step,
                direction,
                loss,
                gradient,
                gtd,
                c1=c1,
                c2=c2,
                max_ls=(
                    max_eval - n_eval
                    if line_search_max_iter is None
                    else min(max_eval - n_eval, line_search_max_iter - 1)
                ),
            )
            x = execution.add_scaled(x, direction, step)
            n_eval += evaluations
            if accepted_step is not None:
                accepted_step(n_iter, x, loss, gradient)
            # Preserve the mature stopping order, including the evaluation
            # budget checked after a completed line search.
            if n_iter == max_iter:
                reason = "max_iter"
                break
            if n_eval >= max_eval:
                reason = "max_eval"
                break
            if execution.abs_max(gradient) <= tolerance_grad:
                reason = "gradient_tolerance"
                break
            if (
                tolerance_change is not None
                and execution.abs_max(direction * step) <= tolerance_change
            ):
                reason = "step_tolerance"
                break
            if (
                tolerance_change is not None
                and abs(loss - previous_loss) < tolerance_change
            ):
                reason = "loss_tolerance"
                break
            if relative_loss_tolerance is not None and 2 * abs(
                loss - previous_loss
            ) <= relative_loss_tolerance * (abs(loss) + abs(previous_loss) + 1e-10):
                reason = "relative_loss_tolerance"
                break
    return dict(
        x=x, fun=loss, grad=gradient, n_iter=n_iter, n_eval=n_eval, reason=reason
    )
