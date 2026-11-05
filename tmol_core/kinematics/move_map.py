"""Hierarchical movement policy; native kinematics resolves chemical DOF priority.

The host map uses NumPy arrays. Framework adapters may supply array allocation
and index-type predicates while sharing all flag, shape, mask and setter rules.
"""

from typing import Any, Generic, TypeVar
from operator import attrgetter
import attrs
import numpy as np
from tmol_core.kinematics import n_movable_bond_dof_types, n_movable_jump_dof_types

ArrayT = TypeVar("ArrayT")
TensorSelection = Any

MOVE_MAP_FLAGS = (
    "move_all_jumps",
    "move_all_root_jumps",
    "move_all_mc",
    "move_all_sc",
    "move_all_named_torsions",
    "non_ideal",
)
MOVE_MAP_ARRAYS = (
    "move_jumps",
    "move_jumps_mask",
    "move_root_jumps",
    "move_root_jumps_mask",
    "move_mcs",
    "move_mcs_mask",
    "move_scs",
    "move_scs_mask",
    "move_named_torsions",
    "move_named_torsions_mask",
    "move_jump_dof",
    "move_jump_dof_mask",
    "move_root_jump_dof",
    "move_root_jump_dof_mask",
    "move_mc",
    "move_mc_mask",
    "move_sc",
    "move_sc_mask",
    "move_named_torsion",
    "move_named_torsion_mask",
    "move_atom_dof",
    "move_atom_dof_mask",
)


_native_arguments = attrgetter(*(MOVE_MAP_FLAGS + MOVE_MAP_ARRAYS))


@attrs.define(slots=True, auto_attribs=True)
class MoveMap(Generic[ArrayT]):
    """Hierarchical internal-coordinate degrees of freedom for a pose stack."""

    move_all_jumps: bool
    move_all_root_jumps: bool
    move_all_mc: bool
    move_all_sc: bool
    move_all_named_torsions: bool
    non_ideal: bool

    # data members on a per-residue basis; plural, representing
    # all of the torsions of a particular class for a residue
    move_jumps: ArrayT
    move_jumps_mask: ArrayT
    move_root_jumps: ArrayT
    move_root_jumps_mask: ArrayT
    move_mcs: ArrayT
    move_mcs_mask: ArrayT
    move_scs: ArrayT
    move_scs_mask: ArrayT
    move_named_torsions: ArrayT
    move_named_torsions_mask: ArrayT

    # data members on a per-torsion, per-residue basis; singular,
    # representing a single torsion of a particular class for a residue
    move_mc: ArrayT
    move_mc_mask: ArrayT
    move_sc: ArrayT
    move_sc_mask: ArrayT
    move_named_torsion: ArrayT
    move_named_torsion_mask: ArrayT

    # data members on a per-DOF basis:
    # RBx,y,z and RBdel_alpha,_beta,_gamma for jumps, and
    # phi_p, theta, d, phi_c for atoms
    move_jump_dof: ArrayT
    move_jump_dof_mask: ArrayT
    move_root_jump_dof: ArrayT
    move_root_jump_dof_mask: ArrayT
    move_atom_dof: ArrayT
    move_atom_dof_mask: ArrayT

    def set_move_all_jump_dofs_for_jump(
        self,
        pose_selection: TensorSelection,
        jump_selection: TensorSelection | None = None,
        value: bool = True,
    ) -> None:
        """Enable or disable all jump dofs for a particular set of jumps on particular poses.

        If jump_selection is None, then the two dimensional settings tensor will be indexed by
        the pose_selection tensor only; if both are not None, then the settings tensor will be indexed by
        both the pose_selection tensor and the jump_selection tensor.
        """
        self._set_val_and_mask_for_2dim(pose_selection, jump_selection, "jumps", value)

    def set_move_all_jump_dofs_for_root_jump(
        self,
        pose_selection: TensorSelection,
        root_jump_selection: TensorSelection | None = None,
        value: bool = True,
    ) -> None:
        """Enable or disable all jump dofs for a particular set of root-jumps on particular poses.

        If root_jump_selection is None, then the two dimensional settings tensor will be indexed by
        the pose_selection tensor only; if both are not None, then the settings tensor will be indexed by
        both the pose_selection tensor and the jump_selection tensor.
        """
        self._set_val_and_mask_for_2dim(
            pose_selection, root_jump_selection, "root_jumps", value
        )

    def set_move_all_mc_tors_for_blocks(
        self,
        pose_selection: TensorSelection,
        block_selection: TensorSelection | None = None,
        value: bool = True,
    ) -> None:
        """Enable or disable all DOFs for a partiular set of blocks on particular poses.

        If block_selection is None, then the two dimensional settings tensor will be indexed by
        the pose_selection tensor only; if both are not None, then the tensor will be indexed by
        the pose_selection tensor and the block_selection tensor.

        Valid combinations of pose_selection and block_selection are, e.g.:
          - pose_selection: int, block_selection: int == a particular block on a particular pose
          - pose_selection: int, block_selection: None == all blocks on a particular pose
          - pose_selection: Tensor[bool][n_poses, max_n_blocks], block_selection: None == pose/block pairs encoded in "pose_selection" tensor
          - pose_selection: Tensor[int][N], block_selection: Tensor[int][N] == different blocks on different poses, selected by index
        """
        self._assert_valid_pose_block_selection(pose_selection, block_selection)
        self._set_val_and_mask_for_2dim(pose_selection, block_selection, "mcs", value)

    def set_move_all_sc_tors_for_blocks(
        self,
        pose_selection: TensorSelection,
        block_selection: TensorSelection | None = None,
        value: bool = True,
    ) -> None:
        self._assert_valid_pose_block_selection(pose_selection, block_selection)
        self._set_val_and_mask_for_2dim(pose_selection, block_selection, "scs", value)

    def set_move_all_named_torsions_for_blocks(
        self,
        pose_selection: TensorSelection,
        block_selection: TensorSelection | None = None,
        value: bool = True,
    ) -> None:
        self._assert_valid_pose_block_selection(pose_selection, block_selection)
        self._set_val_and_mask_for_2dim(
            pose_selection, block_selection, "named_torsions", value
        )

    def set_move_mc_tor_for_blocks(
        self,
        pose_selection: TensorSelection,
        block_selection: TensorSelection | None = None,
        tor_selection: TensorSelection | None = None,
        value: bool = True,
    ) -> None:
        """Enable or disable partiular main-chain torsions for a particular set of blocks on particular poses.

        Valid combinations of block_selection and tor_selection are:
          - pose_selection: int, block_selection: int, tor_selection: int == a single DOF
          - pose_selection: Tensor[bool][n_poses, max_n_blocks, max_n_dofs], block_selection: None, tor_selection: None == pose/block/tor triples encoded in "pose_selection" tensor
          - pose_selection: Tensor[int][N], block_selection: Tensor[int][N], dof_selection: Tensor[int][N] == different torsions on different blocks on different poses, selected by index
        """
        self._assert_valid_pose_block_dof_selection(
            pose_selection, block_selection, tor_selection, self.max_n_named_torsions
        )
        self._set_val_and_mask_for_3dim(
            pose_selection, block_selection, tor_selection, "mc", value
        )

    def set_move_sc_tor_for_blocks(
        self,
        pose_selection: TensorSelection,
        block_selection: TensorSelection | None = None,
        dof_selection: TensorSelection | None = None,
        value: bool = True,
    ) -> None:
        """Enable or disable partiular side-chain torsions for a particular set of blocks on particular poses."""
        self._assert_valid_pose_block_dof_selection(
            pose_selection, block_selection, dof_selection, self.max_n_named_torsions
        )
        self._set_val_and_mask_for_3dim(
            pose_selection, block_selection, dof_selection, "sc", value
        )

    def set_move_named_torsion_for_blocks(
        self,
        pose_selection: TensorSelection,
        block_selection: TensorSelection | None = None,
        tor_selection: TensorSelection | None = None,
        value: bool = True,
    ) -> None:
        """Enable or disable partiular named-torsions for a particular set of blocks on particular poses."""
        self._assert_valid_pose_block_dof_selection(
            pose_selection, block_selection, tor_selection, self.max_n_named_torsions
        )
        self._set_val_and_mask_for_3dim(
            pose_selection, block_selection, tor_selection, "named_torsion", value
        )

    def set_move_jump_dof_for_jumps(
        self,
        pose_selection: TensorSelection,
        jump_selection: TensorSelection | None = None,
        dof_selection: TensorSelection | None = None,
        value: bool = True,
    ) -> None:
        """Enable or disable all jump dofs for a particular set of jumps on particular poses.

        If jump_selection is None, then the two dimensional settings tensor will be indexed by
        the pose_selection tensor only; if both are not None, then the settings tensor will be indexed by
        both the pose_selection tensor and the jump_selection tensor.
        """
        self._assert_valid_pose_block_dof_selection(
            pose_selection, jump_selection, dof_selection, n_movable_jump_dof_types
        )
        self._set_val_and_mask_for_3dim(
            pose_selection, jump_selection, dof_selection, "jump_dof", value
        )

    def set_move_jump_dof_for_root_jumps(
        self,
        pose_selection: TensorSelection,
        root_jump_selection: TensorSelection | None = None,
        dof_selection: TensorSelection | None = None,
        value: bool = True,
    ) -> None:
        """Enable or disable all jump dofs for a particular set of root-jumps on particular poses.

        If root_jump_selection is None, then the two dimensional settings tensor will be indexed by
        the pose_selection tensor only; if both are not None, then the settings tensor will be indexed by
        both the pose_selection tensor and the root_jump_selection tensor.
        """
        self._assert_valid_pose_block_dof_selection(
            pose_selection, root_jump_selection, dof_selection, n_movable_jump_dof_types
        )
        self._set_val_and_mask_for_3dim(
            pose_selection, root_jump_selection, dof_selection, "root_jump_dof", value
        )

    def set_move_atom_dof_for_blocks(
        self,
        pose_selection: TensorSelection,
        block_selection: TensorSelection | None = None,
        atom_selection: TensorSelection | None = None,
        dof_selection: TensorSelection | None = None,
        value: bool = True,
    ) -> None:
        """Enable or disable partiular atom dofs for a particular set of blocks on particular poses.

        Either only "pose_selection" should be not None or all four "selection" variables should be
        not None; in the former case, the settings tensor, self.move_atom_dof, will be indexed
        solely by the pose_selection tensor; in the latter case, the settings tensor will be indexed
        by all four selection tensors.

        This function offers the finest grain control over which dofs should be minimized and
        settings made using this function will override any settings made using the other
        settings tensors.
        """
        if block_selection is None:
            assert atom_selection is None
            assert dof_selection is None
            self.move_atom_dof[pose_selection] = value
            self.move_atom_dof_mask[pose_selection] = True
        else:
            self.move_atom_dof[
                pose_selection, block_selection, atom_selection, dof_selection
            ] = value
            self.move_atom_dof_mask[
                pose_selection, block_selection, atom_selection, dof_selection
            ] = True

    @property
    def n_poses(self) -> int:
        return self.move_jumps.shape[0]

    @property
    def max_n_jumps(self) -> int:
        return self.move_jumps.shape[1]

    @property
    def max_n_blocks(self) -> int:
        return self.move_mcs.shape[1]

    @property
    def max_n_named_torsions(self) -> int:
        return self.move_named_torsion.shape[2]

    @property
    def max_n_atoms_per_block(self) -> int:
        return self.move_atom_dof.shape[2]

    def __init__(
        self, n_poses, max_n_blocks, max_n_named_torsions, max_n_atoms_per_block
    ):
        self._initialize(
            n_poses,
            max_n_blocks,
            max_n_named_torsions,
            max_n_atoms_per_block,
            lambda shape: np.zeros(shape, dtype=bool),
        )

    def _initialize(
        self,
        n_poses,
        max_n_blocks,
        max_n_named_torsions,
        max_n_atoms_per_block,
        allocate,
    ):
        self.move_all_jumps = False
        self.move_all_root_jumps = False
        self.move_all_mc = False
        self.move_all_sc = False
        self.move_all_named_torsions = False
        self.non_ideal = False

        def z_bool_pb(remaining_shape=()):
            return allocate((n_poses, max_n_blocks) + remaining_shape)

        # data members on a per-residue basis; plural, representing
        # all of the torsions of a particular class for a residue
        self.move_jumps = z_bool_pb()
        self.move_jumps_mask = z_bool_pb()
        self.move_root_jumps = z_bool_pb()
        self.move_root_jumps_mask = z_bool_pb()
        self.move_mcs = z_bool_pb()
        self.move_mcs_mask = z_bool_pb()
        self.move_scs = z_bool_pb()
        self.move_scs_mask = z_bool_pb()
        self.move_named_torsions = z_bool_pb()
        self.move_named_torsions_mask = z_bool_pb()

        # data members on a per-torsion, per-residue basis; singular,
        # representing a single torsion of a particular class for a residue
        self.move_mc = z_bool_pb((max_n_named_torsions,))
        self.move_mc_mask = z_bool_pb((max_n_named_torsions,))
        self.move_sc = z_bool_pb((max_n_named_torsions,))
        self.move_sc_mask = z_bool_pb((max_n_named_torsions,))
        self.move_named_torsion = z_bool_pb((max_n_named_torsions,))
        self.move_named_torsion_mask = z_bool_pb((max_n_named_torsions,))

        # data members on a per-DOF basis:
        # RBx,y,z and RBdel_alpha,_beta,_gamma for jumps, and
        # phi_p, theta, d, phi_c for atoms
        self.move_jump_dof = z_bool_pb((n_movable_jump_dof_types,))
        self.move_jump_dof_mask = z_bool_pb((n_movable_jump_dof_types,))
        self.move_root_jump_dof = z_bool_pb((n_movable_jump_dof_types,))
        self.move_root_jump_dof_mask = z_bool_pb((n_movable_jump_dof_types,))
        self.move_atom_dof = z_bool_pb(
            (max_n_atoms_per_block, n_movable_bond_dof_types)
        )
        self.move_atom_dof_mask = z_bool_pb(
            (max_n_atoms_per_block, n_movable_bond_dof_types)
        )

    @property
    def native_arguments(self):
        """Flags and masks in the canonical native minimizer-map ABI order."""
        return _native_arguments(self)

    @staticmethod
    def _is_boolean_selection(value):
        return isinstance(value, np.ndarray) and value.dtype == np.bool_

    @staticmethod
    def _is_integer_selection(value):
        return isinstance(value, np.ndarray) and value.dtype == np.int64

    def _assert_valid_pose_block_selection(self, pose_sel, block_sel):
        if block_sel is None:
            if isinstance(pose_sel, int):
                return
            if self._is_boolean_selection(pose_sel) and pose_sel.ndim == 2:
                assert pose_sel.shape == (self.n_poses, self.max_n_blocks)
        else:
            if self._is_integer_selection(pose_sel) and self._is_integer_selection(
                block_sel
            ):
                assert pose_sel.shape == block_sel.shape

    def _assert_valid_pose_block_dof_selection(
        self, pose_sel, block_sel, dof_sel, n_dofs
    ):
        if block_sel is None:
            assert dof_sel is None
            if isinstance(pose_sel, int):
                return
            if self._is_boolean_selection(pose_sel) and pose_sel.ndim == 3:
                assert pose_sel.shape == (
                    self.n_poses,
                    self.max_n_blocks,
                    n_dofs,
                )
        else:
            assert dof_sel is not None
            if self._is_integer_selection(pose_sel) and self._is_integer_selection(
                block_sel
            ):
                assert pose_sel.shape == block_sel.shape
            if self._is_integer_selection(pose_sel) and self._is_integer_selection(
                dof_sel
            ):
                assert pose_sel.shape == dof_sel.shape
            if self._is_integer_selection(block_sel) and self._is_integer_selection(
                dof_sel
            ):
                assert block_sel.shape == dof_sel.shape

    def _set_val_and_mask_for_2dim(self, sel1, sel2, varname, value):
        var_to_set = getattr(self, "move_" + varname)
        mask_to_set = getattr(self, "move_" + varname + "_mask")
        if sel2 is None:
            var_to_set[sel1] = value
            mask_to_set[sel1] = True
        else:
            var_to_set[sel1, sel2] = value
            mask_to_set[sel1, sel2] = True

    def _set_val_and_mask_for_3dim(self, sel1, sel2, sel3, varname, value):
        var_to_set = getattr(self, "move_" + varname)
        mask_to_set = getattr(self, "move_" + varname + "_mask")
        if sel2 is None and sel3 is None:
            var_to_set[sel1] = value
            mask_to_set[sel1] = True
        else:
            var_to_set[sel1, sel2, sel3] = value
            mask_to_set[sel1, sel2, sel3] = True
