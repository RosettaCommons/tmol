"""Named torsion ownership and target validation shared by coordinate executors."""

import numpy as np

from tmol_core.chemical.atom_resolution import resolve_atom_id
from tmol_core.kinematics import NodeType


def controlling_node(
    block_type_ind,
    block_in_and_first_out,
    uaid_for_torsion_by_inconn,
    offsets,
    connections,
    downstream_atoms,
    max_n_pose_atoms,
    kfo_for_atom,
    pose,
    block,
    torsion,
):
    """Forest node whose bond phi_c controls a named torsion, or -1.

    Phi_c and the requested torsion have slope +1: reversing the rotation axis
    and the controlled dihedral end cancels. Arrays may be structural views;
    executors own their placement and any device-to-host transfer.
    """
    block_type = int(block_type_ind[pose, block])
    in_conn = int(block_in_and_first_out[pose, block, 0])
    uaid = uaid_for_torsion_by_inconn[block_type, in_conn, torsion]
    atom = resolve_atom_id(
        pose,
        block,
        uaid,
        block_type_ind,
        offsets,
        connections,
        downstream_atoms,
        max_n_pose_atoms,
    )
    return -1 if atom == -1 else int(kfo_for_atom[atom])


def is_bonded_node(node, node_types):
    return node != -1 and int(node_types[node]) == NodeType.bond


def require_bonded_node(node, node_types, request):
    if not is_bonded_node(node, node_types):
        pose, block, _, name = request
        raise ValueError(
            f"torsion {name!r} on block {block} of pose {pose} is not a bonded "
            "degree of freedom under this fold forest"
        )


def torsion_targets(requests, values, degrees=True):
    """Broadcast a scalar target, retaining mature units and length validation."""
    targets = np.atleast_1d(np.asarray(values, dtype=np.float64))
    if len(targets) == 1 and len(requests) > 1:
        targets = np.repeat(targets, len(requests))
    if len(targets) != len(requests):
        raise ValueError("values must be a scalar or match the number of torsions")
    if degrees:
        targets = np.radians(targets)
    return targets


def torsion_deltas(requests, targets, current):
    """Reject undefined torsions before converting targets to angle updates."""
    if np.isnan(current).any():
        bad = requests[int(np.argmax(np.isnan(current)))]
        raise ValueError(
            f"torsion {bad[3]!r} on block {bad[1]} of pose {bad[0]} is undefined; "
            "it reaches a residue that is not present"
        )
    return targets - current
