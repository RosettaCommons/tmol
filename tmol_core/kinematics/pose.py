"""Cross-array validation for shared native pose assembly."""

import numpy as np
import numba

from .check_fold_forest import validate_fold_forest


def validate_pose_kinematics_inputs(
    block_type,
    block_coord_offset,
    inter_residue_connections,
    ff_edges,
    n_atoms,
    n_conn,
    polymeric_conn,
    max_n_pose_atoms,
):
    """Validate physical indexing before calling the typed native planner.

    Arrays are host metadata, independent of the coordinate execution device.
    Palette scan stencils must come from ``prepare_packed_kinematic_stencils``.
    Fold-forest validation retains the established polymer/jump/chemical rules.
    """
    values = []
    for name, value, rank in (
        ("block_type", block_type, 2),
        ("block_coord_offset", block_coord_offset, 2),
        ("inter_residue_connections", inter_residue_connections, 4),
        ("ff_edges", ff_edges, 3),
        ("n_atoms", n_atoms, 1),
        ("n_conn", n_conn, 1),
        ("polymeric_conn", polymeric_conn, 2),
    ):
        array = np.asarray(value)
        if array.ndim != rank or array.dtype.kind not in "iu":
            raise ValueError(f"{name} must be a rank-{rank} integer array")
        needs_bounds_check = array.dtype.itemsize > 4 or (
            array.dtype.kind == "u" and array.dtype.itemsize == 4
        )
        if (
            needs_bounds_check
            and array.size
            and (
                array.min() < np.iinfo(np.int32).min
                or array.max() > np.iinfo(np.int32).max
            )
        ):
            raise ValueError(f"{name} exceeds native int32 indexing")
        values.append(array)
    block_type, offsets, connections, edges, n_atoms, n_conn, polymeric = values
    if (
        offsets.shape != block_type.shape
        or connections.shape[:2] != block_type.shape
        or connections.shape[3] != 2
        or edges.shape[0] != block_type.shape[0]
        or edges.shape[2] != 4
        or n_atoms.shape != n_conn.shape
        or polymeric.shape != (len(n_atoms), 2)
    ):
        raise ValueError("Incompatible pose kinematic metadata shapes")
    if not isinstance(max_n_pose_atoms, (int, np.integer)) or not (
        0 <= max_n_pose_atoms <= np.iinfo(np.int32).max
    ):
        raise ValueError("Coordinate capacity must be a nonnegative int32 extent")
    if block_type.size == 0 or np.any(np.all(block_type == -1, axis=1)):
        raise ValueError("Pose kinematics requires at least one block per pose")
    if np.any(block_type < -1) or np.any(block_type >= len(n_atoms)):
        raise ValueError("Block type index is outside the prepared palette")
    if (
        np.any(n_atoms < 0)
        or np.any(n_conn < 0)
        or np.any(n_conn > connections.shape[2])
    ):
        raise ValueError("Palette atom/connection counts exceed pose capacities")
    real = block_type != -1
    if np.any(real[:, 1:] & ~real[:, :-1]):
        raise ValueError("Real blocks must precede padded blocks")
    pose, block = np.nonzero(real)
    types = block_type[pose, block]
    starts = offsets[pose, block].astype(np.int64)
    ends = starts + n_atoms[types]
    if np.any(starts < 0) or np.any(ends > max_n_pose_atoms):
        raise ValueError("Physical atom range exceeds pose coordinate capacity")
    order = np.lexsort((starts, pose))
    if np.any(
        (pose[order[1:]] == pose[order[:-1]]) & (starts[order[1:]] < ends[order[:-1]])
    ):
        raise ValueError("Physical block atom ranges overlap")
    if 1 + n_atoms[types].sum(dtype=np.int64) > np.iinfo(np.int32).max:
        raise ValueError("Kinematic atom count exceeds native int32 indexing")

    validate_fold_forest(real.sum(axis=1), edges)
    _validate_connections_and_fold_edges(
        block_type, connections, edges, n_conn, polymeric
    )


@numba.njit
def _validate_connections_and_fold_edges(
    block_type, connections, edges, n_conn, polymeric
):
    # Like the mature fold-forest validator, traverse topology in compiled
    # loops. Validation runs during preparation, never coordinate execution.
    for p in range(block_type.shape[0]):
        for b in range(block_type.shape[1]):
            bt = block_type[p, b]
            for c in range(connections.shape[2]):
                target, port = connections[p, b, c]
                # Legacy termini retain an unused opposite-port number.
                if target == -1:
                    continue
                if target < -1 or port < 0:
                    raise ValueError(
                        "Active connections require nonnegative endpoint ports"
                    )
                if target >= block_type.shape[1] or bt < 0:
                    raise ValueError("Connection endpoint is outside physical blocks")
                dest_type = block_type[p, target]
                if dest_type < 0 or c >= n_conn[bt] or port >= n_conn[dest_type]:
                    raise ValueError("Connection endpoint is outside physical ports")
                if (
                    connections[p, target, port, 0] != b
                    or connections[p, target, port, 1] != c
                ):
                    raise ValueError("Chemical connections must be reciprocal")
        for e in range(edges.shape[1]):
            kind, start, stop, index = edges[p, e]
            if kind == 3:
                if not 0 <= index < n_conn[block_type[p, start]]:
                    raise ValueError("Chemical fold edge uses an invalid source port")
                if connections[p, start, index, 0] != stop:
                    raise ValueError("Chemical fold edge disagrees with connectivity")
            elif kind == 0:
                direction = 1 if stop > start else -1
                outgoing = 1 if direction == 1 else 0
                incoming = 1 - outgoing
                for b in range(start, stop, direction):
                    left_type, right_type = (
                        block_type[p, b],
                        block_type[p, b + direction],
                    )
                    left = polymeric[left_type, outgoing]
                    right = polymeric[right_type, incoming]
                    if (
                        not 0 <= left < n_conn[left_type]
                        or not 0 <= right < n_conn[right_type]
                    ):
                        raise ValueError("Polymer fold edge requires polymeric ports")
                    if (
                        connections[p, b, left, 0] != b + direction
                        or connections[p, b, left, 1] != right
                    ):
                        raise ValueError(
                            "Polymer fold edge crosses a chemical chain break"
                        )
