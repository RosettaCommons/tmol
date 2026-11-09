"""Nucleic-acid candidate topology and weighted native coordinate execution."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.score.na_torsion.na_torsion_energy_term import (
    na_packed_state,
)
from ._ffi import stage
from .scoring import coordinate_vjp, validate_rotamer_context


@jax.jit
def _topology(arguments):
    r = arguments[0].shape[0]
    shape = jax.ShapeDtypeStruct
    outputs = (
        shape((r, 8), np.uint8),
        shape((r, 8), np.uint8),
        shape((r,), np.bool_),
        shape((r, 10, 4, 8), np.uint8),
        shape((r, 10, 4), np.bool_),
        shape((r, 10), np.bool_),
        shape((r, 5, 8), np.uint8),
        shape((r, 5), np.bool_),
    )
    return stage(arguments, "tmol_na_rotamer_topology", outputs, 0)


def prepare_rotamer_na_torsion(pose, rotamers, database):
    validate_rotamer_context(pose, rotamers)
    pbt = pose.packed_block_types
    with jax.default_device(pose.device):
        metadata, tables = na_packed_state(pbt, database, pose.device)
        arguments = (
            rotamers.block_type_ind_for_rot,
            rotamers.pose_for_rot,
            rotamers.block_ind_for_rot,
            rotamers.coord_offset_for_rot,
            rotamers.rot_offset_for_block,
            rotamers.first_rot_block_type,
            pose.inter_residue_connections,
            *metadata,
        )
        base, previous, is_na, torsions, _, torsion_ok, rings, ring_ok = _topology(
            arguments
        )
    return (
        (
            base,
            is_na,
            torsions,
            torsion_ok,
            rings,
            ring_ok,
            previous,
            *tables.parameters,
        ),
        tuple(tables.attributes.items()),
        rotamers.pose_for_rot,
    )


@partial(jax.jit, static_argnames=("attributes", "backward"))
def _evaluate(coords, parameters, weights=None, *, attributes, backward=False):
    output = jax.ShapeDtypeStruct(
        coords.shape if backward else (2, parameters[0].shape[0]), coords.dtype
    )
    return stage(
        (coords, *parameters, *((weights,) if backward else ())),
        "tmol_rotamer_na_torsion",
        output,
        backward,
        **dict(attributes),
    )


def sparse_na_torsion_scores(coords, state):
    """Return NA channels per candidate with a weighted coordinate reverse rule."""
    parameters, attributes, poses = state
    rot = jnp.arange(poses.shape[0], dtype=jnp.int32)
    return (
        coordinate_vjp(coords, parameters, partial(_evaluate, attributes=attributes)),
        jnp.stack((poses, rot, rot)),
    )
