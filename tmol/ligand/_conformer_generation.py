"""Torch numerical execution for shared ligand distance geometry."""

import contextlib
import sys
import numpy as np
import torch
from tmol_core.ligand import _conformer_generation as _core
from typing import Optional
from tmol_core.ligand._conformer_generation import (
    REFINE_ITERS,
    STAGE_A_ANNEAL,
    STAGE_A_ITERS,
)
from tmol_core.ligand import refinement as _refinement


class _TorchRefinement:
    gather = staticmethod(
        lambda coordinates, indices: coordinates.index_select(0, indices)
    )
    clamp_min = staticmethod(torch.clamp_min)
    relu = staticmethod(torch.relu)
    sqrt = staticmethod(torch.sqrt)
    cross = staticmethod(torch.linalg.cross)
    eigvalsh = staticmethod(torch.linalg.eigvalsh)


_OPERATIONS = _TorchRefinement()


def _parameters(L0, U0, L, U, components, chirals):
    return _refinement.prepare_refinement(
        L0,
        U0,
        L,
        U,
        components,
        chirals,
        index_array=lambda a: torch.as_tensor(a, dtype=torch.long),
        float_array=lambda a: torch.as_tensor(a, dtype=TORCH_DTYPE),
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
    return _refinement.atom_distances(coords, i, j, _OPERATIONS)


def _chiral_volumes(coords, c0, c1, c2, c3):
    return _refinement.chiral_volumes(coords, (c0, c1, c2, c3), _OPERATIONS)


def _pair_masks(L0, U0, L, U):
    masks = _refinement.pair_masks(L0, U0, L, U)
    return tuple(
        torch.as_tensor(a, dtype=torch.long if i < 6 else TORCH_DTYPE)
        for i, a in enumerate(masks)
    )


def _chiral_anneal(X0, L0, U0, L, U, components, chirals, rng_seed: int) -> np.ndarray:
    """The first stage of minimization.

    Following EKTDG, run in 4D to allow easier chirality flipping.
    Ramping constraint tethers 4th dim to 0.
    Return the 3D projection."""
    parameters = _parameters(L0, U0, L, U, components, chirals)
    w0 = np.random.default_rng(rng_seed).normal(0, 0.5, (X0.shape[0], 1))
    X = torch.tensor(
        np.ascontiguousarray(np.hstack([X0, w0])), dtype=TORCH_DTYPE, requires_grad=True
    )

    def run(iters, w_dim4):
        opt = _LigandLBFGS(
            [X], max_iter=iters, history_size=50, line_search_fn="strong_wolfe"
        )

        def closure():
            # This optimizer owns only X. Clearing it directly is equivalent to
            # zero_grad(set_to_none=True) and avoids importing TorchDynamo just
            # to reset one gradient during ligand preparation.
            X.grad = None
            loss = _refinement.refinement_loss(
                X, parameters, _OPERATIONS, fourth_dimension=w_dim4
            )
            loss.backward()
            return loss

        opt.step(closure)

    for w_dim4 in STAGE_A_ANNEAL:
        run(STAGE_A_ITERS, w_dim4)
    return X.detach().numpy()[:, :3]


def _stress_refine(X0, L0, U0, L, U, components, chirals) -> np.ndarray:
    """The 2nd stage of minimization.

    Full-weight 3D refine."""
    parameters = _parameters(L0, U0, L, U, components, chirals)
    X = torch.tensor(np.ascontiguousarray(X0), dtype=TORCH_DTYPE, requires_grad=True)

    opt = _LigandLBFGS(
        [X], max_iter=REFINE_ITERS, history_size=50, line_search_fn="strong_wolfe"
    )

    def closure():
        # See the stage-A closure above: X is the optimizer's sole parameter.
        X.grad = None
        loss = _refinement.refinement_loss(X, parameters, _OPERATIONS)
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
        loss = _refinement.planar_loss(Y, pair_i, pair_j, target_distance, _OPERATIONS)
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
