"""Sparse hydrogen bonds from shared topology, base geometry and native physics."""

import ctypes
import jax

from tmol_jax.score.hbond.native import prepare_hbond
from tmol_jax.score._native import libraries, workspace_sizes
from .scoring import validate_rotamer_context, sparse_nonbonded_values


def prepare_rotamer_hbond(pose, rotamers, database):
    validate_rotamer_context(pose, rotamers)
    with jax.default_device(pose.device):
        parameters = prepare_hbond(pose, database, rotamers=rotamers)
    query = libraries()["cpu"].tmol_hbond_neighbor_reach
    query.argtypes, query.restype = [], ctypes.c_double
    return "tmol_rotamer_hbond", 1, parameters, query()


def sparse_hbond_values(coords, state, indices):
    parameters = state[2]
    poses, blocks = parameters[2].shape
    workspace = workspace_sizes(
        state[0], (poses, blocks, coords.shape[0]), coords.dtype.itemsize * 8
    )
    return sparse_nonbonded_values(coords, state, indices, workspace=workspace)
