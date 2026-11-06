"""Host movement policy using the shared chemistry and native DOF resolver."""

from typing import Any

import attrs

from tmol_core.kinematics.move_map import MoveMap as CoreMoveMap

from .pose import _prepare_palette


@attrs.define
class CartesianMoveMap:
    """Cartesian atom selection for relax minimizers; None permits every atom."""

    coord_mask: Any = None


class MoveMap(CoreMoveMap):
    """NumPy movement selections, prepared before JAX coordinate execution."""

    __slots__ = ()

    @classmethod
    def from_pose_stack(cls, pose):
        """Construct a pose-sized map with every degree of freedom disabled."""
        host, _ = _prepare_palette(pose.packed_block_types)
        return cls(
            pose.n_poses,
            pose.max_n_blocks,
            host.max_n_torsions,
            pose.max_n_atoms_per_block,
        )
