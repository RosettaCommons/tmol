"""Sparse LK-ball execution with shared water generation and coordinate VJP."""

from tmol_jax.score.lk_ball.native import prepare_lk_ball
from tmol_jax.score._native import workspace_sizes
from .scoring import sparse_nonbonded_values, validate_rotamer_context


def prepare_rotamer_lk_ball(pose, rotamers, database):
    validate_rotamer_context(pose, rotamers)
    parameters, reach = prepare_lk_ball(pose, database)
    parameters = (
        rotamers.coord_offset_for_rot,
        rotamers.pose_for_rot,
        rotamers.block_ind_for_rot,
        rotamers.first_rot_block_type,
        rotamers.rot_offset_for_block,
        *parameters[5:],
        rotamers.block_type_ind_for_rot,
        rotamers.group_for_block,
    )
    return "tmol_rotamer_lk_ball", 4, parameters, reach


def sparse_lk_ball_values(coords, state, indices):
    parameters = state[2]
    poses, blocks = parameters[3].shape
    workspace = workspace_sizes(
        state[0], (poses, blocks, coords.shape[0]), coords.dtype.itemsize * 8
    )
    return sparse_nonbonded_values(coords, state, indices, workspace=workspace)
