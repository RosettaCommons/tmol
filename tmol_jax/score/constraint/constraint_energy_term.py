"""Coordinate restraints and callable geometric constraints on molecular atoms."""

from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.score.energy_term import EnergyTerm
from tmol_jax.score.score_types import ScoreType
from tmol_jax.types import pytree_dataclass
from . import functions

# ConstraintSet pytree                                                  #


@pytree_dataclass
class ConstraintSet:
    """Legacy flat-bottom coordinate restraints as differentiable JAX arrays."""

    target_coords: jax.Array
    atom_indices: jax.Array
    k: jax.Array
    tol: jax.Array

    # Convenience constructors                                      #

    @classmethod
    def empty(cls) -> "ConstraintSet":
        """An empty constraint set (no penalties, any pose stack)."""
        return cls(
            target_coords=jnp.zeros((0, 3), dtype=jnp.float32),
            atom_indices=jnp.zeros((0, 2), dtype=jnp.int32),
            k=jnp.zeros((0,), dtype=jnp.float32),
            tol=jnp.zeros((0,), dtype=jnp.float32),
        )

    @property
    def n_constraints(self) -> int:
        return int(self.atom_indices.shape[0])


# ConstraintEnergyTerm                                                  #


class ConstraintEnergyTerm(EnergyTerm):
    """Score geometric callbacks or legacy coordinate restraints per pose."""

    def __init__(self, param_db: Any = None):
        super().__init__(param_db=param_db)

    harmonic = staticmethod(functions.harmonic)
    harmonic_coordinate = staticmethod(functions.harmonic_coordinate)
    bounded = staticmethod(functions.bounded)
    circularharmonic = staticmethod(functions.circularharmonic)
    coordinate_flatbottom = staticmethod(functions.coordinate_flatbottom)
    get_torsion_angle_test = staticmethod(functions.torsion)

    def score_types_provided(self) -> list[ScoreType]:
        return [ScoreType.constraint]

    def setup_for_pose_stack(
        self,
        pose_stack: Any,
        pbt: Any = None,
        constraint_set: Optional[ConstraintSet] = None,
    ) -> ConstraintSet:
        """Validate host atom topology and resolve callbacks before tracing."""
        del pbt
        if constraint_set is not None:
            cs = constraint_set
        else:
            cs = getattr(pose_stack, "constraint_set", None)
        if cs is None:
            cs = ConstraintSet.empty()
        if not isinstance(cs, ConstraintSet):
            from tmol_jax.pose.constraint_set import (
                ConstraintSet as GeometricConstraints,
            )

            if isinstance(cs, GeometricConstraints):
                return _prepare_geometric_constraints(pose_stack, cs)
            raise TypeError("coordinate constraints require a JAX ConstraintSet")

        target = np.asarray(cs.target_coords)
        indices = np.asarray(cs.atom_indices)
        k = np.asarray(cs.k)
        tol = np.asarray(cs.tol)
        n_constraints = indices.shape[0] if indices.ndim == 2 else -1
        if target.shape != (n_constraints, 3):
            raise ValueError("constraint target_coords must have shape [n, 3]")
        if indices.shape != (n_constraints, 2):
            raise ValueError("constraint atom_indices must have shape [n, 2]")
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("constraint atom_indices must contain integers")
        if k.shape != (n_constraints,) or tol.shape != (n_constraints,):
            raise ValueError("constraint k and tol must have shape [n]")
        if not (
            np.isfinite(target).all()
            and np.isfinite(k).all()
            and np.isfinite(tol).all()
        ):
            raise ValueError("constraint parameters must be finite")
        if np.any(k < 0) or np.any(tol < 0):
            raise ValueError("constraint k and tol must be nonnegative")
        if n_constraints:
            shape = pose_stack.coords.shape
            if (
                np.any(indices[:, 0] < 0)
                or np.any(indices[:, 0] >= shape[0])
                or np.any(indices[:, 1] < 0)
                or np.any(indices[:, 1] >= shape[1])
            ):
                raise ValueError("constraint atom index is outside the pose")
        return cs

    def __call__(
        self,
        pose_stack: Any,
        term_state: ConstraintSet,
        coords_override: Optional[jax.Array] = None,
    ) -> dict[ScoreType, jax.Array]:
        """Evaluate prepared constraints and sum their energies within each pose."""
        coords = pose_stack.coords if coords_override is None else coords_override
        if isinstance(term_state, tuple):
            values = jnp.zeros(coords.shape[0], coords.dtype)
            flat = coords.reshape(-1, 3)
            for function, atom_indices, params, pose_indices in term_state:
                penalties = function(flat[atom_indices], params.astype(coords.dtype))
                if penalties.shape != pose_indices.shape:
                    raise ValueError(
                        "constraint callback must return one energy per constraint"
                    )
                values += jax.ops.segment_sum(
                    penalties, pose_indices, num_segments=coords.shape[0]
                )
            return {ScoreType.constraint: values}

        # ``coords`` is [n_poses, max_n_atoms_per_pose, 3].
        n_poses = coords.shape[0]

        atom_indices = term_state.atom_indices.astype(jnp.int32)
        target = term_state.target_coords.astype(coords.dtype)
        k = term_state.k.astype(coords.dtype)
        tol = term_state.tol.astype(coords.dtype)

        # Fast-path the empty case: return a zero vector without trying
        # to gather from a zero-length index array (which jax handles
        # correctly, but this keeps the trace tidy).
        if atom_indices.shape[0] == 0:
            return {ScoreType.constraint: jnp.zeros((n_poses,), dtype=coords.dtype)}

        pose_ind = atom_indices[:, 0]
        atom_ind = atom_indices[:, 1]

        # Gather the atom coordinate for each constraint.
        # ``coords[pose_ind, atom_ind]`` is [n_constraints, 3].
        atom_xyz = coords[pose_ind, atom_ind]

        penalty = functions.coordinate_flatbottom(
            atom_xyz[:, None, :], jnp.column_stack((target, k, tol))
        )

        # Scatter-sum into the per-pose total. ``segment_sum`` is the
        # standard JAX primitive for this and is differentiable w.r.t.
        # its ``data`` argument.
        per_pose = jax.ops.segment_sum(penalty, pose_ind, num_segments=n_poses).astype(
            coords.dtype
        )

        return {ScoreType.constraint: per_pose}


def _prepare_geometric_constraints(pose, constraints):
    """Resolve named block-local topology to native coordinate storage once."""
    if constraints.n_poses != pose.coords.shape[0]:
        raise ValueError("constraint pose count does not match the molecular batch")
    atoms = np.asarray(constraints.constraint_atoms)
    functions = np.asarray(constraints.constraint_function_inds)
    parameters = constraints.constraint_params
    if atoms.ndim != 3 or atoms.shape[1:] != (4, 3):
        raise ValueError("constraint atoms must have shape [constraint, 4, 3]")
    if not np.issubdtype(atoms.dtype, np.integer):
        raise TypeError("constraint atoms must contain integer indices")
    if (
        functions.shape != atoms.shape[:1]
        or not np.issubdtype(functions.dtype, np.integer)
        or np.any(functions < 0)
        or np.any(functions >= len(constraints.constraint_functions))
    ):
        raise ValueError("constraint function indices do not match the registry")
    if parameters.ndim != 2 or parameters.shape[0] != len(atoms):
        raise ValueError("constraint parameters must have one row per constraint")
    if not jnp.issubdtype(parameters.dtype, jnp.floating):
        raise TypeError("constraint parameters must be floating point")
    actual = atoms[..., 0] >= 0
    if not np.all(actual[:, 0]) or np.any(atoms[~actual] != -1):
        raise ValueError("constraint atoms contain invalid padding")
    if np.any(
        atoms[..., 0][actual] != np.broadcast_to(atoms[:, :1, 0], actual.shape)[actual]
    ):
        raise ValueError("a constraint cannot contain atoms from multiple poses")
    p, b, a = np.moveaxis(atoms, -1, 0)
    if (
        np.any(p[actual] >= pose.n_poses)
        or np.any(b[actual] < 0)
        or np.any(b[actual] >= pose.max_n_blocks)
        or np.any(a[actual] < 0)
    ):
        raise ValueError("constraint atom index is outside the pose")
    bti = np.asarray(pose.block_type_ind)
    real_types = bti[p[actual], b[actual]]
    if np.any(real_types < 0) or np.any(
        a[actual]
        >= np.asarray(pose.packed_block_types.n_atoms)[np.maximum(real_types, 0)]
    ):
        raise ValueError("constraint atom refers to molecular padding")
    offsets = np.asarray(pose.block_coord_offset)
    flat_atoms = np.full(atoms.shape[:2], -1, np.int32)
    flat_atoms[actual] = (
        p[actual] * pose.coords.shape[1] + offsets[p[actual], b[actual]] + a[actual]
    )
    groups = []
    for index, function in enumerate(constraints.constraint_functions):
        rows = np.flatnonzero(functions == index)
        if not rows.size:
            continue
        required_atoms = getattr(function, "_tmol_constraint_atoms", None)
        if required_atoms is not None and not np.all(actual[rows, :required_atoms]):
            raise ValueError("constraint callback is missing required atoms")
        required_params = getattr(function, "_tmol_constraint_parameters", 0)
        if parameters.shape[1] < required_params:
            raise ValueError("constraint callback is missing required parameters")
        groups.append(
            (
                function,
                jnp.asarray(flat_atoms[rows]),
                constraints.constraint_params[rows],
                jnp.asarray(p[rows, 0]),
            )
        )
    return tuple(groups)


__all__ = ["ConstraintSet", "ConstraintEnergyTerm"]
