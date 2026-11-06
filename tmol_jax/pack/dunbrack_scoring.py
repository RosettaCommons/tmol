"""Sparse Dunbrack execution using the same tables and kernels as pose scoring."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.score._native import register, workspace_sizes
from tmol_jax.score.dunbrack.dunbrack_energy_term import DunbrackEnergyTerm
from .scoring import coordinate_vjp, validate_rotamer_context


def prepare_rotamer_dunbrack(pose, rotamers, database):
    validate_rotamer_context(pose, rotamers)
    with jax.default_device(pose.device):
        parameters = DunbrackEnergyTerm(database).setup_for_pose_stack(pose).parameters
    if parameters:
        parameters = (
            rotamers.coord_offset_for_rot,
            rotamers.rot_offset_for_block,
            rotamers.first_rot_block_type,
            *parameters[3:],
            rotamers.pose_for_rot,
            rotamers.block_ind_for_rot,
            rotamers.block_type_ind_for_rot,
        )
    return parameters, rotamers.pose_for_rot


@partial(jax.jit, static_argnames=("workspace", "backward"))
def _evaluate(coords, parameters, weights=None, *, workspace, backward=False):
    register("tmol_rotamer_dunbrack")
    rots = parameters[0].shape[0]
    shapes = (
        jax.ShapeDtypeStruct(
            (3, *coords.shape) if backward else (3, rots), coords.dtype
        ),
        jax.ShapeDtypeStruct((workspace[backward],), jnp.uint8),
        jax.ShapeDtypeStruct((3, 0 if backward else rots), jnp.int32),
    )
    operands = (coords, *parameters, *((weights,) if backward else ()))
    values, _, _ = jax.ffi.ffi_call(
        "tmol_rotamer_dunbrack", shapes, vmap_method="sequential"
    )(*operands, operation=np.int64(backward), extent=np.int64(0), width=np.int64(0))
    return values.sum(axis=0) if backward else values


def sparse_dunbrack_scores(coords, state):
    """Return conformer channels and diagonal identities; supports jit/vmap/VJP."""
    parameters, poses = state
    rot = jnp.arange(poses.shape[0], dtype=jnp.int32)
    indices = jnp.stack((poses, rot, rot))
    if not parameters:
        return jnp.zeros((3, poses.shape[0]), coords.dtype), indices
    workspace = workspace_sizes(
        "tmol_dunbrack",
        (parameters[0].size, parameters[24].shape[1]),
        coords.dtype.itemsize * 8,
    )
    return (
        coordinate_vjp(coords, parameters, partial(_evaluate, workspace=workspace)),
        indices,
    )
