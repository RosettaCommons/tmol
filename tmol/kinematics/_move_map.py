import attrs
import torch

from tmol.types._torch import Tensor

from tmol.pose._pose_stack import PoseStack
from tmol.kinematics._datatypes import KinematicModuleData
from tmol_core.kinematics.move_map import MoveMap as CoreMoveMap


@attrs.define
class CartesianMoveMap:
    """Move map for Cartesian-space minimization.

    A lightweight wrapper around a coordinate mask, passed to
    :class:`~tmol.CartesianRelaxMinimizer` to choose the atoms that
    :func:`~tmol.fast_relax` minimizes.

    Unlike the full :class:`MoveMap`, which describes freedom in internal
    (torsion/jump) DOF space, ``CartesianMoveMap`` works directly in atomic
    Cartesian space.

    Args:
        coord_mask: Boolean tensor of shape ``[n_poses, max_n_atoms]``
            indicating which atoms are free to move.  ``None`` means all
            atoms are free to move (the default behaviour of
            ``CartesianSfxnNetwork``).
    """

    coord_mask: torch.Tensor | None = None


class MoveMap(CoreMoveMap[torch.Tensor]):
    """Hierarchical internal-coordinate degrees of freedom for a pose stack."""

    __slots__ = ()

    @classmethod
    def from_pose_stack(cls, ps: PoseStack) -> "MoveMap":
        """Construct a move map sized for ``ps`` with all movement disabled."""
        return cls(
            ps.n_poses,
            ps.max_n_blocks,
            ps.packed_block_types.max_n_torsions,
            ps.max_n_block_atoms,
            ps.device,
        )

    def __init__(
        self,
        n_poses,
        max_n_blocks,
        max_n_named_torsions,
        max_n_atoms_per_block,
        device=None,
    ):
        self._initialize(
            n_poses,
            max_n_blocks,
            max_n_named_torsions,
            max_n_atoms_per_block,
            lambda shape: torch.zeros(shape, dtype=torch.bool, device=device),
        )

    @staticmethod
    def _is_boolean_selection(value):
        return isinstance(value, torch.Tensor) and value.dtype == torch.bool

    @staticmethod
    def _is_integer_selection(value):
        return isinstance(value, torch.Tensor) and value.dtype == torch.int64


class MinimizerMap:
    """Resolved per-DOF movement mask for a pose's kinematic forest."""

    dof_mask: Tensor[torch.bool][:, :, :]

    def __init__(
        self,
        pose_stack: PoseStack,
        kmd: KinematicModuleData,
        mm: MoveMap,
    ) -> None:
        from tmol.kinematics.compiled._compiled_ops import _minimizer_map_from_movemap

        pbt = pose_stack.packed_block_types

        # fmt: off
        self.dof_mask = _minimizer_map_from_movemap(
            kmd.forest.id,
            pose_stack.max_n_pose_atoms,
            pose_stack.block_coord_offset,
            pose_stack.block_type_ind,
            pose_stack.inter_residue_connections,
            kmd.block_in_and_first_out,
            kmd.pose_stack_atom_for_jump,
            kmd.pose_stack_atom_for_root_jump,
            kmd.keep_atom_fixed,
            pbt.n_torsions,
            pbt.gen_seg_scan_path_segs.uaid_for_torsion_by_inconn,
            pbt.is_torsion_mc,
            pbt.which_mcsc_torsions,
            pbt.atom_downstream_of_conn,
            *mm.native_arguments,
        )
        # fmt: on
