"""CPU distance-geometry execution over common typed native value/gradients.

This executor is an explicit candidate until molecular and workflow gates pass.
The binding owns native-library loading and denormal state; chemistry, array
preparation, stage scheduling and optimizer policy remain framework-independent.
"""

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from tmol_core.optimization.strong_wolfe import lbfgs_minimize
from . import refinement
from ._conformer_generation import (
    CHIRAL_MARGIN,
    W_BOUND,
    W_CHIRAL,
    W_EXACT,
    W_PLANE,
)


@dataclass(frozen=True)
class NativeRefinementParameters:
    pairs: np.ndarray
    distances: np.ndarray
    n_exact: int
    n_lower: int
    n_upper: int
    component_offsets: np.ndarray
    component_atoms: np.ndarray
    chiral_atoms: np.ndarray
    chiral_sign: np.ndarray
    weights: np.ndarray


def prepare_native_refinement(
    initial_lower, initial_upper, lower, upper, components, chirals, dtype
):
    """Pack the existing shared chemical plan for the contiguous CPU ABI."""
    parameters = refinement.prepare_refinement(
        initial_lower,
        initial_upper,
        lower,
        upper,
        components,
        chirals,
        index_array=lambda a: np.asarray(a, dtype=np.int32),
        float_array=lambda a: np.ascontiguousarray(a, dtype=dtype),
    )
    offsets = np.asarray(
        [0, *np.cumsum([len(c) for c in parameters.components])], dtype=np.int32
    )
    atoms = (
        np.concatenate(parameters.components)
        if parameters.components
        else np.empty(0, dtype=np.int32)
    )
    return NativeRefinementParameters(
        np.stack((parameters.pair_i, parameters.pair_j), axis=1),
        np.concatenate((parameters.target, parameters.lower, parameters.upper)),
        parameters.n_exact,
        parameters.n_lower,
        len(parameters.upper),
        offsets,
        atoms,
        np.stack(parameters.chiral_indices, axis=1),
        parameters.chiral_sign,
        np.asarray([W_EXACT, W_BOUND, W_PLANE, W_CHIRAL, CHIRAL_MARGIN], dtype=dtype),
    )


def prepare_native_planar(target_pairs):
    """Use the same typed distance kernel for double-precision 2D embedding."""
    targets = np.asarray(target_pairs, dtype=np.float64).reshape(-1, 3)
    return NativeRefinementParameters(
        np.ascontiguousarray(targets[:, :2], dtype=np.int32),
        np.ascontiguousarray(targets[:, 2]),
        len(targets),
        0,
        0,
        np.zeros(1, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty((0, 4), dtype=np.int32),
        np.empty(0, dtype=np.float64),
        np.asarray([1, 0, 0, 0, CHIRAL_MARGIN], dtype=np.float64),
    )


class _NumpyLBFGS:
    clone = staticmethod(np.copy)
    dot = staticmethod(np.dot)
    abs_max = staticmethod(lambda x: np.abs(x).max())
    abs_sum = staticmethod(lambda x: np.abs(x).sum())
    sqrt = staticmethod(np.sqrt)

    @staticmethod
    def add_scaled(x, y, alpha):
        return x + np.asarray(alpha, dtype=x.dtype) * y


class NativeConformerExecution:
    """Bind CPU array execution to one common native value/gradient callback.

    The binding supplies ``denormal_mode()`` / ``set_denormal_mode(mode)`` and
    ``value_and_grad(coordinates, parameters, four_dimensional, fourth_weight)``.
    Coordinates and returned gradients are contiguous NumPy arrays with the same
    shape and float32/float64 dtype. Each returned gradient owns its storage.
    """

    def __init__(self, native):
        self.native = native

    @contextmanager
    def flush_denormals(self):
        previous = self.native.denormal_mode()
        if previous >= 0 and not self.native.set_denormal_mode(3):
            raise RuntimeError("native CPU cannot enable its reported denormal mode")
        try:
            yield
        finally:
            if previous >= 0:
                self.native.set_denormal_mode(previous)

    def _minimize(self, coordinates, parameters, max_iter, fourth_dimension=None):
        shape = coordinates.shape

        def evaluate(flat):
            value, gradient = self.native.value_and_grad(
                flat.reshape(shape),
                parameters,
                four_dimensional=fourth_dimension is not None,
                fourth_weight=0 if fourth_dimension is None else fourth_dimension,
            )
            if gradient.shape != shape or gradient.dtype != coordinates.dtype:
                raise TypeError("native refinement must preserve gradient shape/dtype")
            return value, gradient.reshape(-1)

        result = lbfgs_minimize(
            evaluate,
            coordinates.reshape(-1),
            _NumpyLBFGS,
            max_iter=max_iter,
            history_size=50,
        )
        return result["x"].reshape(shape)

    @staticmethod
    def prepare(L0, U0, L, U, components, chirals):
        return prepare_native_refinement(L0, U0, L, U, components, chirals, np.float32)

    prepare_planar = staticmethod(prepare_native_planar)
    coordinates = staticmethod(np.ascontiguousarray)
    to_numpy = staticmethod(lambda coordinates: coordinates)

    def minimize(
        self, coordinates, parameters, max_iter, *, fourth_dimension=None, planar=False
    ):
        # Native parameters encode the planar objective's weights and pairs.
        return self._minimize(coordinates, parameters, max_iter, fourth_dimension)

    def chiral_anneal(self, X0, L0, U0, L, U, components, chirals, rng_seed):
        return refinement.chiral_anneal(
            X0, L0, U0, L, U, components, chirals, rng_seed, execution=self
        )

    def stress_refine(self, X0, L0, U0, L, U, components, chirals):
        return refinement.stress_refine(
            X0, L0, U0, L, U, components, chirals, execution=self
        )

    def planar_refine(self, Y0, target_pairs, max_iters):
        return refinement.planar_refine(Y0, target_pairs, max_iters, execution=self)
