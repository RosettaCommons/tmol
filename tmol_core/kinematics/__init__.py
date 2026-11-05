"""Framework-neutral kinematic forest metadata in kinematic coordinate order."""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class NodeType(IntEnum):
    root = 0
    jump = 1
    bond = 2


class BondDOFTypes(IntEnum):
    """Indices of the four movable bond degrees of freedom."""

    phi_p = 0
    theta = 1
    d = 2
    phi_c = 3


n_movable_bond_dof_types = 4


class JumpDOFTypes(IntEnum):
    """Jump layout: six movable DOFs, then three fixed reference angles.

    The native reverse derivative is defined for slots 0:6. Reference angles
    in slots 6:9 affect the forward transform but are not differentiable
    variables in that operation's contract.
    """

    RBx = 0
    RBy = 1
    RBz = 2
    RBdel_alpha = 3
    RBdel_beta = 4
    RBdel_gamma = 5
    RBalpha = 6
    RBbeta = 7
    RBgamma = 8


n_movable_jump_dof_types = 6


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


def _native_int32(value, name, ndim=1):
    array = np.asarray(value)
    if array.ndim != ndim or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a rank-{ndim} integer array")
    if array.size > np.iinfo(np.int32).max:
        raise ValueError("kinematic forest exceeds native int32 node count")
    needs_range_check = array.dtype.itemsize > 4 or (
        array.dtype.kind == "u" and array.dtype.itemsize == 4
    )
    if (
        array.size
        and needs_range_check
        and (
            array.min() < np.iinfo(np.int32).min or array.max() > np.iinfo(np.int32).max
        )
    ):
        raise ValueError(f"{name} does not fit native int32 indices")
    result = np.array(array, dtype=np.int32, order="C", copy=True)
    result.flags.writeable = False
    return result


def _parent_is_ordered(parent):
    return not len(parent) or (
        parent[0] == 0
        and np.all((parent[1:] >= 0) & (parent[1:] < np.arange(1, len(parent))))
    )


def _validate_parent_tree(parent):
    if _parent_is_ordered(parent):
        return True
    n = len(parent)
    if np.any(parent < 0) or np.any(parent >= n):
        raise ValueError(f"parent contains an index outside [0, {n})")
    if parent[0] != 0:
        raise ValueError("node zero must be the virtual root with zero indices")
    # Each node is visited at most once; storage order need not be topological.
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
            node = int(parent[node])
        if state[node] == 1:
            raise ValueError(f"kinematic parent cycle involving node {node}")
        state[path] = 2
    return False


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
        arrays[name] = _native_int32(value, name)
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
    _validate_parent_tree(arrays["parent"])
    return KinematicForestIndices(**arrays)
