"""Compact per-segment L-BFGS direction with the mature H0 = I model.

Only array casting, native linear algebra and diagonal storage differ between
executors. Chronological/circular history ordering and the inverse-Hessian
formula live here once. Flags select the mature reduction layout explicitly.
"""


def compact_lbfgs_direction(
    grad, dirs, stps, history_start=0, *, execution, single_segment=False, use_bmm=False
):
    """Compact solve with histories kept in their physical circular slots."""
    unbatched = grad.ndim == 1
    if unbatched:
        grad = grad[None, :]
        dirs = dirs[:, None, :]
        stps = stps[:, None, :]

    out_dtype = grad.dtype
    # Triangular solves are:
    #  a) imprecise on CUDA at float32
    #  b) small matrices (M x M) where M is history length (typically 128)
    # promote to float64
    S = execution.float64(stps)
    Y = execution.float64(dirs)
    q = -execution.float64(grad)
    # Explicit batched products avoid einsum's repeated layout planning. Keep
    # the single-segment, broadcast, and differentiable paths: their reduction
    # layouts can differ at tight double-precision tolerances.
    if single_segment:
        S_one, Y_one = S[:, 0], Y[:, 0]
        a = execution.mv(S_one, q[0])[None, ...]  # a_i = s_i . g
        SY = execution.mm(S_one, Y_one.T)[None, ...]  # SY_ij = s_i . y_j
    elif use_bmm:
        S_by_pose, Y_by_pose = S.swapaxes(0, 1), Y.swapaxes(0, 1)
        a = execution.bmm(S_by_pose, q[..., None]).squeeze(-1)
        SY = execution.bmm(S_by_pose, Y_by_pose.swapaxes(-2, -1))
    else:
        a = execution.einsum("ipk,pk->pi", S, q)
        SY = execution.einsum("ipk,jpk->pij", S, Y)
    # Only the small history-space products need chronological ordering.
    # Moving entire (history, pose, coordinate) arrays after every wrap would
    # duplicate both histories. Coefficients return to physical slot order
    # before multiplying by S or Y below.
    if history_start:
        a = execution.roll(a, -history_start, dims=-1)
        SY = execution.roll(SY, (-history_start, -history_start), dims=(-2, -1))
    R = execution.triu(SY)  # upper-triangular incl. diagonal
    D = execution.diagonal(SY)  # D_i = s_i . y_i

    # An absent slot has an all-zero row and column; a unit diagonal there keeps
    # R invertible and leaves the search direction unchanged.
    R = execution.fill_diagonal_where(R, D == 0, 1)

    # u = R^-1 a
    u = execution.solve_triangular(R, a[..., None], upper=True).squeeze(-1)
    # q = g - Y u; v = D u - Y^T q avoids forming Y^T Y.
    Du = D * u
    if history_start:
        u = execution.roll(u, history_start, dims=-1)
    if single_segment:
        q = (q[0] - execution.mv(Y_one.T, u[0]))[None, ...]
        v = execution.mv(Y_one, q[0])[None, ...]
    elif use_bmm:
        q = q - execution.bmm(u[:, None, :], Y_by_pose).squeeze(1)
        v = execution.bmm(Y_by_pose, q[..., None]).squeeze(-1)
    else:
        q = q - execution.einsum("pi,ipk->pk", u, Y)
        v = execution.einsum("ipk,pk->pi", Y, q)
    if history_start:
        v = execution.roll(v, -history_start, dims=-1)
    v = Du - v
    # p = R^-T v
    p = execution.solve_triangular(
        R.swapaxes(-2, -1), v[..., None], upper=False
    ).squeeze(-1)
    if history_start:
        p = execution.roll(p, history_start, dims=-1)
    # result = q + S p
    if single_segment:
        result = (q[0] + execution.mv(S_one.T, p[0]))[None, ...]
    elif use_bmm:
        result = q + execution.bmm(p[:, None, :], S_by_pose).squeeze(1)
    else:
        result = q + execution.einsum("pi,ipk->pk", p, S)
    result = execution.cast(result, out_dtype)
    return result.squeeze(0) if unbatched else result
