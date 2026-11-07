"""Resolve local and cross-connection atom IDs from structural NumPy arrays."""

import numpy as np


def resolve_selected_uaids(
    uaids,
    pose,
    block,
    block_type,
    block_coord_offset,
    inter_res_conn,
    atom_downstream_of_conn,
):
    """Resolve UAIDs at explicit broadcastable pose/block positions."""
    uaids = np.asarray(uaids, dtype=np.int64)
    atom, conn, sep = (uaids[..., i] for i in range(3))
    real = block_type[pose, block] >= 0
    is_inter = (conn >= 0) & real
    if not np.any(is_inter):
        return pose, block_coord_offset[pose, block] + atom, (atom >= 0) & real
    if inter_res_conn.shape[2] == 0:
        raise ValueError(
            "Unresolved atom requests a connection absent from the topology"
        )
    partner = inter_res_conn[pose, block, np.where(is_inter, conn, 0)]
    other_block, other_conn = partner[..., 0], partner[..., 1]
    target = np.where(is_inter, other_block, block)
    ok = real & np.where(is_inter, (other_block >= 0) & (other_conn >= 0), atom >= 0)
    target = np.maximum(target, 0)
    target_bt = block_type[pose, target]
    ok &= target_bt >= 0
    local_inter = atom_downstream_of_conn[
        np.maximum(target_bt, 0), np.maximum(other_conn, 0), np.maximum(sep, 0)
    ]
    local = np.where(is_inter, local_inter, atom)
    ok &= local >= 0
    return pose, block_coord_offset[pose, target] + local, ok


def resolve_uaids(
    uaids, block_type, block_coord_offset, inter_res_conn, atom_downstream_of_conn
):
    """Resolve padded UAIDs with leading pose/block axes."""
    trailing = (1,) * (uaids.ndim - 3)
    pose = np.arange(block_type.shape[0]).reshape((-1, 1) + trailing)
    block = np.arange(block_type.shape[1]).reshape((1, -1) + trailing)
    return resolve_selected_uaids(
        uaids,
        pose,
        block,
        block_type,
        block_coord_offset,
        inter_res_conn,
        atom_downstream_of_conn,
    )


def resolve_atom_id(
    pose,
    block,
    uaid,
    block_type,
    block_coord_offset,
    inter_res_conn,
    atom_downstream_of_conn,
    max_n_pose_atoms,
):
    """Resolve one UAID using structural scalar indexing, or return -1."""
    atom, conn, sep = (int(x) for x in uaid)
    if int(block_type[pose, block]) < 0:
        return -1
    if conn < 0:
        target, local = block, atom
    else:
        partner = inter_res_conn[pose, block, conn]
        target, target_conn = int(partner[0]), int(partner[1])
        if target < 0 or target_conn < 0:
            return -1
        target_type = int(block_type[pose, target])
        if target_type < 0:
            return -1
        local = int(atom_downstream_of_conn[target_type, target_conn, sep])
    if local < 0:
        return -1
    return pose * max_n_pose_atoms + int(block_coord_offset[pose, target]) + local


def torsion_atom_indices(
    block_types,
    requests,
    block_type,
    block_coord_offset,
    inter_res_conn,
    atom_downstream_of_conn,
    max_n_pose_atoms,
):
    """Flat coordinate indices for requested (pose, block, torsion, name) rows."""
    if not requests:
        return np.empty((0, 4), dtype=np.int64)
    # Scalar queries avoid the fixed setup cost of broadcasted array indexing.
    # This is also the resolver used by the legacy scalar pose-query adapter.
    if len(requests) <= 2:
        return np.asarray(
            [
                [
                    resolve_atom_id(
                        p,
                        b,
                        u,
                        block_type,
                        block_coord_offset,
                        inter_res_conn,
                        atom_downstream_of_conn,
                        max_n_pose_atoms,
                    )
                    for u in block_types[int(block_type[p, b])].ordered_torsions[t]
                ]
                for p, b, t, _ in requests
            ],
            dtype=np.int64,
        )
    pose, block, torsion, _ = zip(*requests)
    pose, block = np.asarray(pose), np.asarray(block)
    types = block_type[pose, block]
    uaids = np.asarray(
        [block_types[t].ordered_torsions[tor] for t, tor in zip(types, torsion)]
    )
    pose, local, valid = resolve_selected_uaids(
        uaids,
        pose[:, None],
        block[:, None],
        block_type,
        block_coord_offset,
        inter_res_conn,
        atom_downstream_of_conn,
    )
    return np.where(valid, pose * max_n_pose_atoms + local, -1).astype(np.int64)
