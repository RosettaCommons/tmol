"""Ideal extended-chain geometry from chemical types and host topology arrays.

Coordinate assembly and junction geometry are shared; frontends execute torsion
measurement/updates and place the resulting arrays on their own devices.
"""

import math
from typing import Dict

import numpy
from atomworks.experimental.protonation.geometry import build_coordinates
from tmol_core.chemical.icoor_tree import _vertex_angle

EXTENDED_BACKBONE_TORSIONS = {
    "alpha_aa": {"phi": -135.0, "psi": 135.0, "omega": 180.0},
    "dna": {
        "alpha": -30.0,
        "beta": 136.0,
        "gamma": 31.0,
        "epsilon": -141.0,
        "zeta": -161.0,
        "chi1": -98.0,
    },
    "rna": {
        "alpha": -68.0,
        "beta": 178.0,
        "gamma": 54.0,
        "epsilon": -153.0,
        "zeta": -71.0,
        "chi1": -158.0,
    },
}


def ideal_chained_coords(
    active_types, block_type_ind, offsets, connections, coord_shape
):
    coords = numpy.zeros(coord_shape, dtype=numpy.float32)
    for pose in range(block_type_ind.shape[0]):
        placed: Dict[int, numpy.ndarray] = {}
        for block in range(block_type_ind.shape[1]):
            bt_ind = int(block_type_ind[pose, block])
            if bt_ind == -1:
                continue
            bt = active_types[bt_ind]
            local = numpy.asarray(bt.ideal_coords, dtype=numpy.float64)

            prev = _bonded_predecessor(
                active_types, block_type_ind, connections, pose, block
            )
            if prev is None:
                placed[block] = local
            else:
                rot, trans = _junction_transform(
                    active_types, block_type_ind, pose, block, bt, local, prev, placed
                )
                placed[block] = local @ rot.T + trans

        for block, block_coords in placed.items():
            bt_ind = int(block_type_ind[pose, block])
            bt = active_types[bt_ind]
            offset = int(offsets[pose, block])
            coords[pose, offset : offset + bt.n_atoms] = block_coords[
                bt.at_to_icoor_ind
            ]
    return coords


def _bonded_predecessor(active_types, block_type_ind, connections, pose, block):
    bt_ind = int(block_type_ind[pose, block])
    bt = active_types[bt_ind]
    if bt.down_connection_ind == -1:
        return None
    partner = connections[pose, block, bt.down_connection_ind]
    prev_block, prev_conn = int(partner[0]), int(partner[1])
    if prev_block == -1 or prev_block >= block:
        return None
    prev_bt_ind = int(block_type_ind[pose, prev_block])
    prev_bt = active_types[prev_bt_ind]
    if prev_conn != prev_bt.up_connection_ind:
        return None
    return prev_block, prev_conn


def _junction_transform(
    active_types, block_type_ind, pose, block, bt, local, prev, placed
):
    prev_block, prev_conn = prev
    prev_bt = active_types[int(block_type_ind[pose, prev_block])]
    prev_coords = placed[prev_block]

    prev_conn_name = prev_bt.connections[prev_conn].name
    prev_mainchain = prev_bt.properties.polymer.mainchain_atoms
    conn_atom = prev_bt.connections[prev_conn].atom
    ref_atom = prev_mainchain[prev_mainchain.index(conn_atom) - 1]

    ref = prev_coords[prev_bt.icoors_index[ref_atom]]
    anchor = prev_coords[prev_bt.icoors_index[conn_atom]]
    hinge = prev_coords[prev_bt.icoors_index[prev_conn_name]]

    mainchain = bt.properties.polymer.mainchain_atoms
    src = numpy.array(
        [
            local[bt.icoors_index[bt.connections[bt.down_connection_ind].name]],
            local[bt.icoors_index[mainchain[0]]],
            local[bt.icoors_index[mainchain[1]]],
        ]
    )
    dist = numpy.linalg.norm(src[2] - src[1])
    angle = _vertex_angle(src[0], src[1], src[2])
    torsion = numpy.radians(_junction_torsion_value(prev_bt, prev_conn))

    dst = numpy.array(
        [
            anchor,
            hinge,
            build_coordinates(hinge, anchor, ref, (dist, math.pi - angle, torsion)),
        ]
    )
    return _rigid_transform(src, dst)


def _junction_torsion_value(prev_bt, prev_conn):
    targets = EXTENDED_BACKBONE_TORSIONS.get(
        prev_bt.properties.polymer.backbone_type, {}
    )
    for tor in prev_bt.torsions:
        uaids = prev_bt.torsion_to_uaids[tor.name]
        spans = (
            uaids[0][1] == -1
            and uaids[1][1] == -1
            and uaids[2][1] == prev_conn
            and uaids[3][1] == prev_conn
        )
        if spans:
            return targets.get(tor.name, 180.0)
    return 180.0


def _rigid_transform(src, dst):
    src_center = src.mean(axis=0)
    dst_center = dst.mean(axis=0)
    h = (src - src_center).T @ (dst - dst_center)
    u, _, vt = numpy.linalg.svd(h)
    d = numpy.sign(numpy.linalg.det(vt.T @ u.T))
    rot = vt.T @ numpy.diag([1.0, 1.0, d]) @ u.T
    return rot, dst_center - rot @ src_center


def extended_torsion_requests(active_types, block_type_ind):
    requests, values = [], []
    for pose in range(block_type_ind.shape[0]):
        for block in range(block_type_ind.shape[1]):
            bt_ind = int(block_type_ind[pose, block])
            if bt_ind == -1:
                continue
            bt = active_types[bt_ind]
            targets = EXTENDED_BACKBONE_TORSIONS.get(
                bt.properties.polymer.backbone_type
            )
            if targets is None:
                continue
            for j, tor in enumerate(bt.torsions):
                if tor.name in targets:
                    requests.append((pose, block, j, tor.name))
                    values.append(targets[tor.name])
    return requests, values


def chain_separation_shifts(active_types, block_type_ind, offsets, chain_ids, coords):
    """Pose/atom-index/x-shift updates that separate chains by six Angstroms.

    Shifts are host scalars, as in the mature builder; coordinate addition stays
    in the frontend so existing coordinate derivatives are preserved.
    """
    shifts = []
    for pose in range(block_type_ind.shape[0]):
        atoms_for_chain = {}
        for block in range(block_type_ind.shape[1]):
            bt_ind = int(block_type_ind[pose, block])
            if bt_ind == -1:
                continue
            offset = int(offsets[pose, block])
            n_atoms = active_types[bt_ind].n_atoms
            chain = int(chain_ids[pose, block])
            atoms_for_chain.setdefault(chain, []).append(
                numpy.arange(offset, offset + n_atoms)
            )
        if len(atoms_for_chain) < 2:
            continue
        next_x = 0.0
        for chain in sorted(atoms_for_chain):
            inds = numpy.concatenate(atoms_for_chain[chain])
            xs = coords[pose, inds, 0]
            shift = next_x - float(xs.min())
            shifts.append((pose, inds, shift))
            next_x = float(xs.max()) + shift + 6.0
    return shifts
