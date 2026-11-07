"""Named torsion queries using common identity selection and native geometry."""

import jax
import jax.numpy as jnp
import numpy as np

from tmol_core.pose.query import torsion_requests, format_named_torsions
from tmol_core.pose import query as common
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


def get_sequence(pose_stack, pose=0):
    return common.sequence_from_types(
        pose_stack.packed_block_types.active_block_types,
        np.asarray(pose_stack.block_type_ind[pose]),
    )


def _type_mask(pose_stack, flags):
    flags = jax.device_put(np.asarray(flags, bool), pose_stack.device)
    types = pose_stack.block_type_ind
    return (types >= 0) & flags[jnp.maximum(types, 0)]


def get_ligand_mask(pose_stack):
    return _type_mask(
        pose_stack,
        common.ligand_flags(pose_stack.packed_block_types.active_block_types),
    )


def get_polymer_type_mask(pose_stack, polymer_type):
    return _type_mask(
        pose_stack,
        common.polymer_type_flags(
            pose_stack.packed_block_types.active_block_types, polymer_type
        ),
    )


def metal_blocks(pose_stack, pose):
    return common.metal_blocks(
        pose_stack.packed_block_types.active_block_types,
        np.asarray(pose_stack.block_type_ind[pose]),
    )


def residue_label(pose_stack, pose, block):
    return common.residue_label(
        pose_stack.block_type(pose, block), pose_stack.pdb_info, pose, block
    )


def atom_label(pose_stack, pose, atom):
    return common.atom_label(
        pose_stack.packed_block_types.active_block_types,
        np.asarray(pose_stack.block_type_ind[pose]),
        np.asarray(pose_stack.block_coord_offset[pose]),
        pose_stack.pdb_info,
        pose,
        atom,
    )


def block_for_residue_label(pose_stack, pose_index, chain, residue_id):
    return common.block_for_residue_label(
        pose_stack.pdb_info, pose_index, chain, residue_id
    )


def blocks_for_residue_labels(pose_stack, pose_index, chain, residue_numbers):
    return jax.device_put(
        common.blocks_for_residue_labels(
            pose_stack.pdb_info, pose_index, chain, np.asarray(residue_numbers)
        ).astype(np.int32),
        pose_stack.device,
    )


def atom_for_residue_label(pose_stack, pose_ind, chain_id, res_id, atom_id):
    return jax.device_put(
        common.atom_for_residue_label(
            pose_stack.packed_block_types.active_block_types,
            np.asarray(pose_stack.block_type_ind),
            pose_stack.pdb_info,
            pose_ind,
            chain_id,
            res_id,
            atom_id,
        ),
        pose_stack.device,
    )


def constraint_atoms_for_labels(pose_stack, atoms):
    return jax.device_put(
        common.constraint_atoms_for_labels(
            pose_stack.packed_block_types.active_block_types,
            np.asarray(pose_stack.block_type_ind),
            pose_stack.pdb_info,
            atoms,
        ),
        pose_stack.device,
    )
