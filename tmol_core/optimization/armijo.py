"""Mature segmented Armijo search with shared native state transitions.

The executor supplies native transition launches and device decision transfer;
objective evaluation, accepted-state selection and search policy stay common.
"""


def armijo_linesearch_segmented(
    func,
    derphi0,
    old_fval,
    alpha0,
    searching,
    factor=0.5,
    sigma_decrease=0.1,
    sigma_increase=0.8,
    minstep=1e-6,
    *,
    execution,
    combine_checks=False,
):
    """Minimize over alpha, the function ``f(xk+alpha pk)``.

    Each segment gets its own alpha; one call to f evaluates all of them.

    Arguments:
        f (callable): Function to be minimized, f(step)
        derphi0 : (Tensor) directional derivative, per segment
        fval0 : (Tensor) func(0), the value of the function at the origin
        alpha0 : (Tensor) the initial stepsize, per segment
        searching : (Tensor) which segments take part in the search
        sigma_increase : (float) initial stepsize
                         [must be in (0,1) and >=sigma_decrease]
        sigma_decrease : (float) initial stepsize
                         [must be in (0,1) and <=sigma_increase]
        factor : (float) scalefactor in modifying stepsize [must be in (0,1)]
        minstep : (float) minimum stepsize to take

    Returns:
        stepsize - accepted stepsize, per segment
        f_val - final function value, per segment
        n_evals - number of calls to f
        status - per segment, one of the _LS_ codes
        trial_is_accepted - whether the last trial evaluated the accepted steps

    Notes
        See D.P. Bertsekas, Nonlinear Programming, 2nd ed, 1999, page 29.

        (fd) A few notes about this specific implementation:
        0) I believe this method was originally from Jim Havranek
        1) 'factor' corresponds roughly to 'beta', BUT on a successful initial step,
           factor is used to increase the stepsize.  When factor is used to decrease
           stepsize, factor^2 is used
        2) The stopping critera used is that in the paper, the first integer m>=0 s.t.:
             f(x_k) - f(x_k+beta^m*s*d_k) >= -sigma * beta^m * s * grad{f}(x_k) * d_k
           however, the two different values of sigma are used:
              * sigma_increase (0.8) is used to trigger an _increased_ stepsize
              * sigma_decrease (0.1) is _required_ or the step size is decreased
        3) in the code
              * 'alpha' corresponds to 's' in the text
              * 'factor' corresponds roughly to 'beta' in the text (see point 1)
    """
    phi0 = old_fval
    alpha = execution.start(searching, alpha0)

    phi = func(alpha)
    n_evals = 1

    # sigma_increase > sigma_decrease, so a linear-looking step also has
    # sufficient decrease; it is the one case where a longer step is tried
    accepted, phi_accepted, status, active = execution.classify(
        searching,
        alpha,
        phi,
        phi0,
        derphi0,
        sigma_increase,
        sigma_decrease,
    )

    any_active = execution.any_true(active)
    while any_active:
        trial = execution.trial(status, alpha, accepted, factor)
        phi_trial = func(trial)
        n_evals += 1

        accepted, phi_accepted, status, failed, active = execution.update(
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
        )
        if combine_checks:
            # Transfer both decisions together and reuse the active flag on
            # the next iteration instead of synchronizing its mask again.
            any_failed, any_active = execution.decisions(failed, active)
        else:
            any_failed = execution.any_true(failed)
        if any_failed:
            for p in execution.failed_indices(failed):
                step = float(trial[p])
                finite = (
                    (float(phi_trial[p]) - float(phi0[p])) / step
                    if step
                    else float("inf")
                )
                print(
                    "Inaccurate G! Segment=",
                    p,
                    " Step=",
                    step,
                    " Deriv=",
                    float(derphi0[p]),
                    " Finite=",
                    finite,
                )

        alpha = trial
        phi = phi_trial
        if not combine_checks:
            any_active = execution.any_true(active)

    return accepted, phi_accepted, n_evals, status, execution.equal(alpha, accepted)
