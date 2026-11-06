"""Resolve local and cross-connection atom IDs from structural NumPy arrays."""

import numpy as np


def resolve_uaids(
    uaids, block_type, block_coord_offset, inter_res_conn, atom_downstream_of_conn
):
    n_poses, max_n_blocks = block_type.shape
    trailing = uaids.shape[2:-1]
    shape = (n_poses, max_n_blocks) + trailing
    n_indices = int(np.prod(trailing))
    atom, conn, sep = (uaids[..., i].astype(np.int64) for i in range(3))
    real = (block_type >= 0).reshape(n_poses, max_n_blocks, *([1] * len(trailing)))
    is_inter = (conn >= 0) & real
    pose = np.broadcast_to(
        np.arange(n_poses).reshape(n_poses, *([1] * (atom.ndim - 1))), atom.shape
    )
    if not np.any(is_inter):
        offset = block_coord_offset.reshape(
            n_poses, max_n_blocks, *([1] * len(trailing))
        )
        return pose, offset + atom, (atom >= 0) & real
    if inter_res_conn.shape[2] == 0:
        raise ValueError(
            "Unresolved atom requests a connection absent from the topology"
        )
    self_block = np.broadcast_to(
        np.arange(max_n_blocks).reshape(1, max_n_blocks, *([1] * len(trailing))),
        atom.shape,
    )
    conn_idx = np.where(is_inter, conn, 0).reshape(n_poses, max_n_blocks, n_indices)
    other_block, other_conn = (
        np.take_along_axis(inter_res_conn[..., i], conn_idx, axis=2).reshape(atom.shape)
        for i in range(2)
    )
    target_block = np.where(is_inter, other_block, self_block)
    ok = real & np.where(is_inter, (other_block >= 0) & (other_conn >= 0), atom >= 0)
    target_safe = np.maximum(target_block, 0).reshape(n_poses, max_n_blocks * n_indices)
    target_bt = np.take_along_axis(block_type, target_safe, axis=1).reshape(shape)
    ok &= target_bt >= 0
    local_inter = atom_downstream_of_conn[
        np.maximum(target_bt, 0), np.maximum(other_conn, 0), np.maximum(sep, 0)
    ]
    local = np.where(is_inter, local_inter, atom)
    ok &= local >= 0
    offset = np.take_along_axis(block_coord_offset, target_safe, axis=1).reshape(shape)
    return pose, offset + local, ok
