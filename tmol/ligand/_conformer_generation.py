"""Torch numerical execution for shared ligand distance geometry."""

import contextlib
import sys
import numpy as np
import torch
from tmol_core.ligand import _conformer_generation as _core
from typing import Optional
from tmol_core.ligand._conformer_generation import (
    CHIRAL_MARGIN,
    REFINE_ITERS,
    STAGE_A_ANNEAL,
    STAGE_A_ITERS,
    W_BOUND,
    W_CHIRAL,
    W_EXACT,
    W_PLANE,
)


def __getattr__(name):
    return getattr(_core, name)


TORCH_DTYPE = torch.float32
# Optimizer construction normally enters ``torch._disable_dynamo`` through
# ``add_param_group`` and imports the full TorchDynamo stack on first use. This
# ligand-only optimizer is never compiled; bypassing that wrapper saves seconds
# of cold-start work while retaining the same validation and state setup.
_ADD_PARAM_GROUP = getattr(
    torch.optim.Optimizer.add_param_group,
    "__wrapped__",
    torch.optim.Optimizer.add_param_group,
)


@contextlib.contextmanager
def _flush_denormals():
    """Flush subnormal floats to zero for the enclosed torch minimization only."""
    torch.set_flush_denormal(True)
    try:
        yield
    finally:
        torch.set_flush_denormal(False)


class _LigandLBFGS(torch.optim.LBFGS):
    def add_param_group(self, param_group: dict[str, object]) -> None:
        _ADD_PARAM_GROUP(self, param_group)


def _atom_distances(coords, i, j):
    # index_select gives deterministic CPU gradients for repeated atom indices.
    delta = coords.index_select(0, i) - coords.index_select(0, j)
    return (delta**2).sum(1).clamp_min(1e-12).sqrt()


def _chiral_volumes(coords, c0, c1, c2, c3):
    origin = coords.index_select(0, c0)
    return (
        (coords.index_select(0, c1) - origin)
        * torch.linalg.cross(
            coords.index_select(0, c2) - origin, coords.index_select(0, c3) - origin
        )
    ).sum(1)


def _pair_masks(L0, U0, L, U):
    tri = np.triu(np.ones_like(L0, bool), 1)
    exact = tri & np.isclose(L0, U0)
    nonex = tri & ~exact
    finite_u = nonex & (U < 1e4)
    T = lambda a: torch.as_tensor(a, dtype=torch.long)  # noqa: E731
    ei, ej = np.nonzero(exact)
    ni, nj = np.nonzero(nonex)
    fi, fj = np.nonzero(finite_u)
    return (
        T(ei),
        T(ej),
        T(ni),
        T(nj),
        T(fi),
        T(fj),
        torch.as_tensor(L0[exact], dtype=TORCH_DTYPE),
        torch.as_tensor(L[nonex], dtype=TORCH_DTYPE),
        torch.as_tensor(U[finite_u], dtype=TORCH_DTYPE),
    )


def _chiral_anneal(X0, L0, U0, L, U, components, chirals, rng_seed: int) -> np.ndarray:
    """The first stage of minimization.

    Following EKTDG, run in 4D to allow easier chirality flipping.
    Ramping constraint tethers 4th dim to 0.
    Return the 3D projection."""
    ei, ej, ni, nj, fi, fj, tgt, lo, hi = _pair_masks(L0, U0, L, U)
    comps = [torch.as_tensor(c, dtype=torch.long) for c in components if len(c) >= 4]
    ch = np.array(chirals, float).reshape(-1, 5)
    c0, c1, c2, c3 = (
        torch.as_tensor(ch[:, k].astype(int), dtype=torch.long) for k in range(4)
    )
    csgn = torch.as_tensor(ch[:, 4], dtype=TORCH_DTYPE)
    w0 = np.random.default_rng(rng_seed).normal(0, 0.5, (X0.shape[0], 1))
    X = torch.tensor(
        np.ascontiguousarray(np.hstack([X0, w0])), dtype=TORCH_DTYPE, requires_grad=True
    )

    def pd(i, j):
        return _atom_distances(X, i, j)

    def chiral_penalty():
        v = _chiral_volumes(X[:, :3], c0, c1, c2, c3)
        return (torch.relu(CHIRAL_MARGIN - csgn * v) ** 2).sum()

    def run(iters, w_dim4):
        opt = _LigandLBFGS(
            [X], max_iter=iters, history_size=50, line_search_fn="strong_wolfe"
        )

        def closure():
            # This optimizer owns only X. Clearing it directly is equivalent to
            # zero_grad(set_to_none=True) and avoids importing TorchDynamo just
            # to reset one gradient during ligand preparation.
            X.grad = None
            loss = W_EXACT * ((pd(ei, ej) - tgt) ** 2).sum()
            loss = loss + W_BOUND * (torch.relu(lo - pd(ni, nj)) ** 2).sum()
            loss = loss + W_BOUND * (torch.relu(pd(fi, fj) - hi) ** 2).sum()
            for c in comps:
                P = X.index_select(0, c)[:, :3]
                P = P - P.mean(0)
                loss = loss + W_PLANE * torch.linalg.eigvalsh(P.t() @ P)[0]
            loss = loss + W_CHIRAL * chiral_penalty()
            if w_dim4 > 0:
                loss = loss + w_dim4 * (X[:, 3] ** 2).sum()
            loss.backward()
            return loss

        opt.step(closure)

    for w_dim4 in STAGE_A_ANNEAL:
        run(STAGE_A_ITERS, w_dim4)
    return X.detach().numpy()[:, :3]


def _stress_refine(X0, L0, U0, L, U, components, chirals) -> np.ndarray:
    """The 2nd stage of minimization.

    Full-weight 3D refine."""
    ei, ej, ni, nj, fi, fj, tgt, lo, hi = _pair_masks(L0, U0, L, U)
    n_exact = len(ei)
    n_lower = len(ni)
    pair_i = torch.cat((ei, ni, fi))
    pair_j = torch.cat((ej, nj, fj))
    X = torch.tensor(np.ascontiguousarray(X0), dtype=TORCH_DTYPE, requires_grad=True)
    comps = [torch.as_tensor(c, dtype=torch.long) for c in components if len(c) >= 4]
    ch = np.array(chirals, float).reshape(-1, 5)
    c0, c1, c2, c3 = (
        torch.as_tensor(ch[:, k].astype(int), dtype=torch.long) for k in range(4)
    )
    csgn = torch.as_tensor(ch[:, 4], dtype=TORCH_DTYPE)

    def pair_distances():
        return _atom_distances(X, pair_i, pair_j)

    def chiral_penalty():
        if len(csgn) == 0:
            return X.sum() * 0.0
        v = _chiral_volumes(X, c0, c1, c2, c3)
        return (torch.relu(CHIRAL_MARGIN - csgn * v) ** 2).sum()

    opt = _LigandLBFGS(
        [X], max_iter=REFINE_ITERS, history_size=50, line_search_fn="strong_wolfe"
    )

    def closure():
        # See the stage-A closure above: X is the optimizer's sole parameter.
        X.grad = None
        distances = pair_distances()
        exact = distances[:n_exact]
        lower = distances[n_exact : n_exact + n_lower]
        upper = distances[n_exact + n_lower :]
        loss = W_EXACT * ((exact - tgt) ** 2).sum()
        loss = loss + W_BOUND * (torch.relu(lo - lower) ** 2).sum()
        loss = loss + W_BOUND * (torch.relu(upper - hi) ** 2).sum()
        for c in comps:
            P = X.index_select(0, c)
            P = P - P.mean(0)
            loss = loss + W_PLANE * torch.linalg.eigvalsh(P.t() @ P)[0]
        loss = loss + W_CHIRAL * chiral_penalty()
        loss.backward()
        return loss

    opt.step(closure)
    return X.detach().numpy()


def planar_refine(Y0, tp, max_iters):
    pair_i = torch.tensor([p[0] for p in tp])
    pair_j = torch.tensor([p[1] for p in tp])
    target_distance = torch.tensor([p[2] for p in tp], dtype=torch.double)
    Y = torch.tensor(np.ascontiguousarray(Y0), dtype=torch.double, requires_grad=True)
    opt = _LigandLBFGS(
        [Y], max_iter=max_iters, history_size=50, line_search_fn="strong_wolfe"
    )

    def closure():
        Y.grad = None
        d = _atom_distances(Y, pair_i, pair_j)
        loss = ((d - target_distance) ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    Yc = Y.detach().numpy()
    return Yc


flush_denormals = _flush_denormals
chiral_anneal = _chiral_anneal
stress_refine = _stress_refine


def _generate_conformer(
    smiles: str, *, minimize_steps: int = 50, seed: Optional[int] = None, execution=None
):
    return _core._generate_conformer(
        smiles=smiles,
        minimize_steps=minimize_steps,
        seed=seed,
        execution=execution or sys.modules[__name__],
    )


def _generate_conformer_impl(
    smiles: str, minimize_steps: int, seed: Optional[int], execution=None
):
    return _core._generate_conformer_impl(
        smiles=smiles,
        minimize_steps=minimize_steps,
        seed=seed,
        execution=execution or sys.modules[__name__],
    )


def _seeded_conformer(smiles: str, minimize_steps: int, seed: int, execution=None):
    return _core._seeded_conformer(
        smiles=smiles,
        minimize_steps=minimize_steps,
        seed=seed,
        execution=execution or sys.modules[__name__],
    )


def _planar_component_distances(
    atoms, r0, ang, max_iters: int = 150, *, execution=None
):
    return _core._planar_component_distances(
        atoms=atoms,
        r0=r0,
        ang=ang,
        max_iters=max_iters,
        execution=execution or sys.modules[__name__],
    )


def _bounds_matrix(targets: dict, *, execution=None):
    return _core._bounds_matrix(
        targets=targets, execution=execution or sys.modules[__name__]
    )


def _embed_coordinates(
    targets: dict, *, seed: Optional[int] = None, execution=None
) -> np.ndarray:
    return _core._embed_coordinates(
        targets=targets, seed=seed, execution=execution or sys.modules[__name__]
    )
