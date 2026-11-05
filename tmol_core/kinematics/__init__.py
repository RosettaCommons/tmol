"""Framework-neutral kinematic forest metadata in kinematic coordinate order."""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class NodeType(IntEnum):
    root = 0
    jump = 1
    bond = 2


@dataclass(frozen=True)
class KinematicForestIndices:
    parent: np.ndarray
    frame_x: np.ndarray
    frame_y: np.ndarray
    frame_z: np.ndarray
    doftype: np.ndarray

    @property
    def n_nodes(self):
        return len(self.parent)


def prepare_kinematic_forest(parent, frame_x, frame_y, frame_z, doftype):
    """Validate and snapshot indices for the shared native kinematic kernels.

    Nonempty forests include a virtual root at index zero. Every other node
    reaches that root through an acyclic parent chain; parents need not precede
    children in storage. Frames address the same coordinate array. Coordinate
    geometry and scan ordering are separate from these index invariants.
    """
    arrays = {}
    for name, value in (
        ("parent", parent),
        ("frame_x", frame_x),
        ("frame_y", frame_y),
        ("frame_z", frame_z),
        ("doftype", doftype),
    ):
        array = np.asarray(value)
        if array.ndim != 1 or array.dtype.kind not in "iu":
            raise ValueError(f"{name} must be a one-dimensional integer array")
        if array.size > np.iinfo(np.int32).max:
            raise ValueError("kinematic forest exceeds native int32 node count")
        if array.size and (
            array.min() < np.iinfo(np.int32).min or array.max() > np.iinfo(np.int32).max
        ):
            raise ValueError(f"{name} does not fit native int32 indices")
        arrays[name] = np.array(array, dtype=np.int32, order="C", copy=True)
    n = len(arrays["parent"])
    if any(len(value) != n for value in arrays.values()):
        raise ValueError("kinematic forest arrays must have the same length")
    for name in ("parent", "frame_x", "frame_y", "frame_z"):
        array = arrays[name]
        if np.any(array < 0) or np.any(array >= n):
            raise ValueError(f"{name} contains an index outside [0, {n})")
    kinds = arrays["doftype"]
    if np.any((kinds < NodeType.root) | (kinds > NodeType.bond)):
        raise ValueError("doftype must be root (0), jump (1), or bond (2)")
    if n:
        if any(value[0] != 0 for value in arrays.values()):
            raise ValueError("node zero must be the virtual root with zero indices")
        if np.any(kinds[1:] == NodeType.root):
            raise ValueError("only node zero may have root doftype")
        # Each node is visited at most once: no dense ancestor matrix or
        # assumption that chemical order is a topological ordering.
        state = np.zeros(n, dtype=np.uint8)
        state[0] = 2
        for start in range(1, n):
            if state[start] == 2:
                continue
            path = []
            node = start
            while state[node] == 0:
                state[node] = 1
                path.append(node)
                node = int(arrays["parent"][node])
            if state[node] == 1:
                raise ValueError(f"kinematic parent cycle involving node {node}")
            state[path] = 2
    for value in arrays.values():
        value.flags.writeable = False
    return KinematicForestIndices(**arrays)
