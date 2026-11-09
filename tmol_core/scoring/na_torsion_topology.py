"""Resolve static nucleic-acid atom indices from framework-neutral topology."""

import numpy as np
from tmol_core.chemical.atom_resolution import resolve_uaids as _resolve_uaids


def prepare_na_torsion_topology(
    block_type_ind_for_rot,
    bt_base,
    bt_uaids,
    bt_ring,
    bt_down,
    atom_downstream_of_conn,
    block_coord_offset,
    inter_residue_connections,
    max_n_atoms,
):
    """Return base, masks, flattened torsion/ring indices and previous blocks.

    Arrays retain padded pose/block dimensions. Missing atoms and connections
    are masked; their clamped indices are safe gather placeholders, not atoms.
    """
    n_poses, max_n_blocks = block_coord_offset.shape
    block_type = block_type_ind_for_rot.reshape(n_poses, max_n_blocks).astype(np.int64)
    real = block_type >= 0
    bt_safe = np.maximum(block_type, 0)
    base = np.where(real, bt_base.astype(np.int64)[bt_safe], -1)
    is_na = base >= 0
    pose, index, ok = _resolve_uaids(
        bt_uaids[bt_safe],
        block_type,
        block_coord_offset,
        inter_residue_connections,
        atom_downstream_of_conn,
    )
    torsion_indices = np.where(
        ok & is_na[..., None, None], pose * max_n_atoms + index, 0
    )
    torsion_ok = ok.all(-1) & is_na[..., None]
    ring_local = bt_ring.astype(np.int64)[bt_safe]
    ring_pose = np.arange(n_poses).reshape(n_poses, 1, 1)
    ring_index = block_coord_offset[..., None] + ring_local
    ring_ok = (ring_local >= 0) & is_na[..., None]
    ring_indices = np.where(ring_ok, ring_pose * max_n_atoms + ring_index, 0)
    down = bt_down.astype(np.int64)[bt_safe]
    prev_block = np.take_along_axis(
        inter_residue_connections[..., 0], np.maximum(down, 0)[..., None], axis=2
    )[..., 0]
    has_prev = (down >= 0) & (prev_block >= 0)
    pose_ind = np.arange(n_poses).reshape(n_poses, 1)
    prev = np.where(has_prev, pose_ind * max_n_blocks + prev_block, -1)
    return base, is_na, torsion_indices, torsion_ok, ring_indices, ring_ok, prev
