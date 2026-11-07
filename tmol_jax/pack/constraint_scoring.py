"""Common native constraint identities with differentiable JAX callbacks."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.pose.constraint_set import ConstraintSet
from tmol_jax.score.constraint.constraint_energy_term import (
    ConstraintSet as CoordinateConstraints,
    ConstraintEnergyTerm,
    _prepare_geometric_constraints,
)
from .graph import prefix_counts, load_balance
from .scoring import validate_rotamer_context

from ._ffi import stage


@partial(jax.jit, static_argnames=("operation", "outputs"))
def _stage(arguments, *, operation, outputs):
    target = "tmol_constraint_rotamer_topology"
    return stage(arguments, target, outputs, operation)


def _geometric_constraints(pose):
    constraints = pose.constraint_set
    if constraints is None:
        return ConstraintSet.create_empty(pose.device, pose.n_poses)
    if not isinstance(constraints, CoordinateConstraints):
        return constraints
    ConstraintEnergyTerm().setup_for_pose_stack(pose)
    # Legacy flat coordinate identities become explicit block-local identities;
    # targets, spring constants and tolerances remain differentiable leaves.
    atoms = np.asarray(constraints.atom_indices)
    offsets = np.asarray(pose.block_coord_offset)
    types = np.asarray(pose.block_type_ind)
    local = np.empty((len(atoms), 1, 3), np.int32)
    for row, (p, atom) in enumerate(atoms):
        real = np.flatnonzero(types[p] >= 0)
        position = np.searchsorted(offsets[p, real], atom, side="right") - 1
        if position < 0:
            raise ValueError("constraint atom refers to molecular padding")
        block = real[position]
        local[row, 0] = p, block, atom - offsets[p, block]
    params = jnp.column_stack(
        (constraints.target_coords, constraints.k, constraints.tol)
    )
    return ConstraintSet.create_empty(pose.device, pose.n_poses).add_constraints(
        ConstraintEnergyTerm.coordinate_flatbottom, local, params
    )


def prepare_rotamer_constraints(pose, rotamers, database=None):
    """Prepare eager candidate identities while preserving callback parameter AD."""
    del database
    validate_rotamer_context(pose, rotamers)
    with jax.default_device(pose.device):
        cs = _geometric_constraints(pose)
        if not isinstance(cs, ConstraintSet):
            raise TypeError("sparse constraints require a JAX ConstraintSet")
        _prepare_geometric_constraints(pose, cs)
        if np.any(np.asarray(cs.constraint_num_unique_blocks) > 2):
            raise NotImplementedError(
                "sparse constraints spanning more than two blocks are not supported"
            )
        arguments = (
            cs.constraint_atoms,
            cs.constraint_num_unique_blocks,
            cs.constraint_unique_blocks,
            rotamers.coord_offset_for_rot,
            rotamers.n_rots_for_block,
            rotamers.rot_offset_for_block,
        )
        shape = jax.ShapeDtypeStruct
        counts = _stage(
            (*arguments, rotamers.group_for_block),
            operation=0,
            outputs=shape((2 * cs.n_constraints, 8), np.uint8),
        )
        offsets = prefix_counts(counts)
        total = int(np.asarray(offsets[-1]).copy().view(np.int64)[0])
        owners = load_balance(offsets[:-1], total=total)
        sources, atoms, indices = _stage(
            (*arguments, offsets[:-1], owners, rotamers.group_for_block),
            operation=1,
            outputs=(
                shape((total,), np.int32),
                shape((total, 4), np.int32),
                shape((total, 3), np.int32),
            ),
        )
        # Callback groups have static sizes in the compiled evaluator. This
        # eager routing step transfers discrete source rows, never coordinates
        # or floating parameter values, and preserves native sparse row order.
        functions = np.asarray(cs.constraint_function_inds)[np.asarray(sources)]
        groups = tuple(
            (fn, jnp.asarray(np.flatnonzero(functions == i), jnp.int32))
            for i, fn in enumerate(cs.constraint_functions)
            if np.any(functions == i)
        )
    return (groups, sources, atoms, cs.constraint_params), indices.T


def sparse_constraint_scores(coords, state, parameters=None):
    """Values support coordinate/parameter AD, jit and vmap for JAX callbacks."""
    (groups, sources, atoms, stored), indices = state
    parameters = stored if parameters is None else parameters
    if parameters.shape != stored.shape:
        raise ValueError("constraint parameters must retain the prepared shape")
    values = jnp.zeros(sources.shape, coords.dtype)
    for fn, rows in groups:
        penalties = fn(
            coords[atoms[rows]], parameters[sources[rows]].astype(coords.dtype)
        )
        if penalties.shape != rows.shape:
            raise ValueError(
                "constraint callback must return one energy per constraint"
            )
        values = values.at[rows].set(penalties)
    return values[None, :], indices
