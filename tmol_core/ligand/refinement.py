"""Shared distance-geometry objectives; executors supply array primitives.

Chemistry, bounds and stage settings come from the common conformer pipeline.
The six primitives below only express differences in array execution. In
particular, gathering must accumulate gradients for repeated atom indices.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ._conformer_generation import (
    CHIRAL_MARGIN,
    REFINE_ITERS,
    STAGE_A_ANNEAL,
    STAGE_A_ITERS,
    W_BOUND,
    W_CHIRAL,
    W_EXACT,
    W_PLANE,
)


class RefinementOperations(Protocol):
    def gather(self, coordinates, indices):
        """Gather rows with repeated-index gradient accumulation."""
        ...

    def clamp_min(self, values, minimum):
        """Clamp squared distances away from zero."""
        ...

    def relu(self, values):
        """Apply the bound and chirality hinge."""
        ...

    def sqrt(self, values):
        """Compute nonnegative distances."""
        ...

    def cross(self, first, second):
        """Cross products along the last coordinate axis."""
        ...

    def eigvalsh(self, matrix):
        """Ascending eigenvalues of a symmetric covariance matrix."""
        ...


@dataclass(frozen=True)
class RefinementParameters:
    pair_i: object
    pair_j: object
    n_exact: int
    n_lower: int
    target: object
    lower: object
    upper: object
    components: tuple
    chiral_indices: tuple
    chiral_sign: object


def pair_masks(initial_lower, initial_upper, lower, upper):
    triangular = np.triu(np.ones_like(initial_lower, bool), 1)
    exact = triangular & np.isclose(initial_lower, initial_upper)
    nonexact = triangular & ~exact
    finite_upper = nonexact & (upper < 1e4)
    return (
        *np.nonzero(exact),
        *np.nonzero(nonexact),
        *np.nonzero(finite_upper),
        initial_lower[exact],
        lower[nonexact],
        upper[finite_upper],
    )


def prepare_refinement(
    initial_lower,
    initial_upper,
    lower,
    upper,
    components,
    chirals,
    *,
    index_array,
    float_array,
):
    """Prepare common pair/chirality metadata in the executor's array storage."""
    ei, ej, ni, nj, fi, fj, target, low, high = pair_masks(
        initial_lower, initial_upper, lower, upper
    )
    chiral = np.array(chirals, float).reshape(-1, 5)
    return RefinementParameters(
        index_array(np.concatenate((ei, ni, fi))),
        index_array(np.concatenate((ej, nj, fj))),
        len(ei),
        len(ni),
        float_array(target),
        float_array(low),
        float_array(high),
        tuple(index_array(c) for c in components if len(c) >= 4),
        tuple(index_array(chiral[:, k].astype(int)) for k in range(4)),
        float_array(chiral[:, 4]),
    )


def atom_distances(coordinates, first, second, operations):
    delta = operations.gather(coordinates, first) - operations.gather(
        coordinates, second
    )
    return operations.sqrt(operations.clamp_min((delta**2).sum(1), 1e-12))


def chiral_volumes(coordinates, indices, operations):
    origin = operations.gather(coordinates, indices[0])
    return (
        (operations.gather(coordinates, indices[1]) - origin)
        * operations.cross(
            operations.gather(coordinates, indices[2]) - origin,
            operations.gather(coordinates, indices[3]) - origin,
        )
    ).sum(1)


def refinement_loss(coordinates, parameters, operations, *, fourth_dimension=None):
    """Mature exact-distance, bound, planar and chiral objective.

    Stage A keeps separate gathers, preserving Torch's accumulation order; the
    final stage gathers all distances together. ``fourth_dimension`` supplies
    the annealing tether and selects the stage-A four-dimensional distances.
    """
    p = parameters
    ne, nl = p.n_exact, p.n_lower
    if fourth_dimension is None:
        distances = atom_distances(coordinates, p.pair_i, p.pair_j, operations)
        exact, lower, upper = (
            distances[:ne],
            distances[ne : ne + nl],
            distances[ne + nl :],
        )
    else:
        exact, lower, upper = (
            atom_distances(coordinates, p.pair_i[s], p.pair_j[s], operations)
            for s in (slice(None, ne), slice(ne, ne + nl), slice(ne + nl, None))
        )
    loss = W_EXACT * ((exact - p.target) ** 2).sum()
    loss = loss + W_BOUND * (operations.relu(p.lower - lower) ** 2).sum()
    loss = loss + W_BOUND * (operations.relu(upper - p.upper) ** 2).sum()
    xyz = coordinates[:, :3]
    for component in p.components:
        points = operations.gather(xyz, component)
        points = points - points.mean(0)
        loss = loss + W_PLANE * operations.eigvalsh(points.T @ points)[0]
    if len(p.chiral_sign) == 0 and fourth_dimension is None:
        chiral_penalty = coordinates.sum() * 0.0
    else:
        volume = chiral_volumes(xyz, p.chiral_indices, operations)
        chiral_penalty = (
            operations.relu(CHIRAL_MARGIN - p.chiral_sign * volume) ** 2
        ).sum()
    loss = loss + W_CHIRAL * chiral_penalty
    if fourth_dimension is not None and fourth_dimension > 0:
        loss = loss + fourth_dimension * (coordinates[:, 3] ** 2).sum()
    return loss


def planar_loss(coordinates, pair_i, pair_j, target, operations):
    return (
        (atom_distances(coordinates, pair_i, pair_j, operations) - target) ** 2
    ).sum()


class ConformerExecution(Protocol):
    """Array storage and minimization for the common host conformer stages.

    ``minimize`` returns coordinates in the same framework and dtype. Parameters
    belong to that executor; ``planar`` selects the existing planar objective.
    Only the final result returns to NumPy, preserving native stage execution.
    """

    def prepare(self, L0, U0, L, U, components, chirals):
        """Prepare exact, bound, plane and chiral parameters."""
        ...

    def prepare_planar(self, target_pairs):
        """Prepare double-precision planar pair targets."""
        ...

    def coordinates(self, values, dtype):
        """Create optimizer coordinates without mutating the caller's input."""
        ...

    def minimize(
        self, coordinates, parameters, max_iter, *, fourth_dimension=None, planar=False
    ):
        """Run the shared optimizer with this executor's objective primitives."""
        ...

    def to_numpy(self, coordinates):
        """Return host coordinates with the frontend's existing copy semantics."""
        ...


def chiral_anneal(X0, L0, U0, L, U, components, chirals, rng_seed, *, execution):
    """Anneal in four dimensions with the shared seeded tether schedule."""
    parameters = execution.prepare(L0, U0, L, U, components, chirals)
    fourth = np.random.default_rng(rng_seed).normal(0, 0.5, (X0.shape[0], 1))
    coordinates = execution.coordinates(np.hstack((X0, fourth)), np.float32)
    for weight in STAGE_A_ANNEAL:
        coordinates = execution.minimize(
            coordinates, parameters, STAGE_A_ITERS, fourth_dimension=weight
        )
    return execution.to_numpy(coordinates[:, :3])


def stress_refine(X0, L0, U0, L, U, components, chirals, *, execution):
    """Run the full-weight three-dimensional final refinement."""
    parameters = execution.prepare(L0, U0, L, U, components, chirals)
    coordinates = execution.coordinates(X0, np.float32)
    return execution.to_numpy(execution.minimize(coordinates, parameters, REFINE_ITERS))


def planar_refine(Y0, target_pairs, max_iters, *, execution):
    """Refine planar target distances in double precision before embedding."""
    parameters = execution.prepare_planar(target_pairs)
    coordinates = execution.coordinates(Y0, np.float64)
    return execution.to_numpy(
        execution.minimize(coordinates, parameters, max_iters, planar=True)
    )
