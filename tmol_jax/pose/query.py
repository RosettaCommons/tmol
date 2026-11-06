"""Named torsion queries using common identity selection and native geometry."""

import jax
import jax.numpy as jnp
import numpy as np

from tmol_core.pose.query import torsion_requests, format_named_torsions
from tmol_core.chemical.atom_resolution import torsion_atom_indices
from tmol_jax.score._native import atom_downstream_of_conn
from tmol_jax.score.constraint.functions import torsion


def get_torsion_names(pose_stack, pose, block):
    """Names defined by this residue's actual type, in database order."""
    index = int(pose_stack.block_type_ind[pose, block])
    if index == -1:
        raise ValueError(f"block {block} of pose {pose} is not a real block")
    return [
        t.name for t in pose_stack.packed_block_types.active_block_types[index].torsions
    ]


def _torsion_requests(pose_stack, poses, blocks, names):
    return torsion_requests(
        pose_stack.packed_block_types.active_block_types,
        np.asarray(pose_stack.block_type_ind),
        poses,
        blocks,
        names,
    )


def _measure_torsions(pose_stack, requests, degrees):
    pbt = pose_stack.packed_block_types
    indices = torsion_atom_indices(
        pbt.active_block_types,
        requests,
        np.asarray(pose_stack.block_type_ind),
        np.asarray(pose_stack.block_coord_offset),
        np.asarray(pose_stack.inter_residue_connections),
        atom_downstream_of_conn(pbt),
        pose_stack.max_n_atoms_per_pose,
    )
    values = np.full(len(indices), np.nan)
    valid = np.all(indices >= 0, axis=1)
    if np.any(valid):
        # Match mature query precision: measure in double, return coordinate
        # dtype before converting to host values. Never change global x64 mode.
        with jax.enable_x64(), jax.default_device(pose_stack.device):
            atoms = pose_stack.coords.reshape(-1, 3)[
                jax.device_put(indices[valid], pose_stack.device)
            ]
            measured = torsion(atoms.astype(jnp.float64)).astype(
                pose_stack.coords.dtype
            )
        values[valid] = np.asarray(measured).reshape(-1)
    return np.degrees(values) if degrees else values


def get_named_torsions(pose_stack, poses=None, blocks=None, names=None, degrees=True):
    """Return a scalar, a name dictionary, or pose/block dictionaries.

    Missing connected atoms produce NaN. This eager host query retains physical
    atom identity; use scoring primitives for differentiable geometry.
    """
    requests = _torsion_requests(pose_stack, poses, blocks, names)
    values = _measure_torsions(pose_stack, requests, degrees)
    return format_named_torsions(
        requests, values, pose_stack.block_type_ind.shape, poses, blocks, names
    )
