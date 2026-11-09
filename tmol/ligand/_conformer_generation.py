"""Torch numerical execution for shared ligand distance geometry."""

import contextlib
import sys
import numpy as np
import torch
from tmol_core.ligand import _conformer_generation as _core
from typing import Optional
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


@contextlib.contextmanager
def _flush_denormals():
    """Flush subnormal floats to zero for the enclosed torch minimization only."""
    torch.set_flush_denormal(True)
    try:
        yield
    finally:
        torch.set_flush_denormal(False)


class _TorchLBFGSExecution:
    clone = staticmethod(torch.clone)
    dot = staticmethod(torch.dot)
    abs_max = staticmethod(lambda x: x.abs().max())
    abs_sum = staticmethod(lambda x: x.abs().sum())
    sqrt = staticmethod(torch.sqrt)

    @staticmethod
    def add_scaled(x, y, alpha):
        return torch.add(x, y, alpha=alpha)


class _LigandLBFGS:
    """Bind the shared host optimizer to one Torch ligand coordinate tensor."""

    def __init__(self, parameters, *, max_iter, history_size, line_search_fn):
        if len(parameters) != 1 or line_search_fn != "strong_wolfe":
            raise ValueError("ligand refinement requires one strong-Wolfe parameter")
        self.coordinates = parameters[0]
        self.max_iter = max_iter
        self.history_size = history_size

    @torch.no_grad()
    def step(self, closure):
        from tmol_core.optimization.strong_wolfe import lbfgs_minimize

        coordinates = self.coordinates

        def evaluate(values):
            coordinates.copy_(values.reshape_as(coordinates))
            with torch.enable_grad():
                loss = closure()
            return float(loss), coordinates.grad.reshape(-1).clone()

        result = lbfgs_minimize(
            evaluate,
            coordinates.detach().reshape(-1),
            _TorchLBFGSExecution,
            max_iter=self.max_iter,
            history_size=self.history_size,
        )
        coordinates.copy_(result["x"].reshape_as(coordinates))
        return result


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


class _TorchConformerExecution:
    prepare = staticmethod(_parameters)
    to_numpy = staticmethod(lambda coordinates: coordinates.detach().numpy())

    @staticmethod
    def coordinates(values, dtype):
        return torch.tensor(
            np.ascontiguousarray(values),
            dtype=torch.float32 if dtype == np.float32 else torch.float64,
            requires_grad=True,
        )

    @staticmethod
    def prepare_planar(target_pairs):
        return (
            torch.tensor([p[0] for p in target_pairs]),
            torch.tensor([p[1] for p in target_pairs]),
            torch.tensor([p[2] for p in target_pairs], dtype=torch.double),
        )

    @staticmethod
    def minimize(
        coordinates, parameters, max_iter, *, fourth_dimension=None, planar=False
    ):
        opt = _LigandLBFGS(
            [coordinates],
            max_iter=max_iter,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def closure():
            # The optimizer owns only this coordinate tensor. Avoid importing
            # TorchDynamo merely to reset its gradient through zero_grad().
            coordinates.grad = None
            loss = (
                _refinement.planar_loss(coordinates, *parameters, _OPERATIONS)
                if planar
                else _refinement.refinement_loss(
                    coordinates,
                    parameters,
                    _OPERATIONS,
                    fourth_dimension=fourth_dimension,
                )
            )
            loss.backward()
            return loss

        opt.step(closure)
        return coordinates


_EXECUTION = _TorchConformerExecution()


def _chiral_anneal(X0, L0, U0, L, U, components, chirals, rng_seed: int) -> np.ndarray:
    return _refinement.chiral_anneal(
        X0, L0, U0, L, U, components, chirals, rng_seed, execution=_EXECUTION
    )


def _stress_refine(X0, L0, U0, L, U, components, chirals) -> np.ndarray:
    return _refinement.stress_refine(
        X0, L0, U0, L, U, components, chirals, execution=_EXECUTION
    )


def planar_refine(Y0, tp, max_iters):
    return _refinement.planar_refine(Y0, tp, max_iters, execution=_EXECUTION)


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
