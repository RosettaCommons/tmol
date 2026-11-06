import torch
import attr
from tmol_core.pack.conformer_policy import include_current_mask

from typing import Tuple

from tmol.types._torch import Tensor
from tmol.chemical import RefinedResidueType
from tmol.pose._packed_block_types import PackedBlockTypes
from tmol.pose._pose_stack import PoseStack
from tmol.kinematics._datatypes import KinForest
from tmol.pack.rotamer._conformer_sampler import ConformerSampler


@attr.s(auto_attribs=True, frozen=True)
class IncludeCurrentSampler(ConformerSampler):
    """Add each packable residue's current conformation to its rotamer set."""

    @classmethod
    def sampler_name(cls):
        return "IncludeCurrentSampler"

    def annotate_residue_type(self, rt: RefinedResidueType):
        pass

    def annotate_packed_block_types(self, packed_block_types: PackedBlockTypes):
        pass

    def defines_rotamers_for_rt(self, rt: RefinedResidueType):
        return True

    def defines_rotamers_for_bts(
        self, pbt: PackedBlockTypes, bt_inds: Tensor[torch.int64]
    ) -> Tensor[torch.bool]:
        return torch.ones_like(bt_inds, dtype=torch.bool)

    def first_sc_atoms_for_rt(self, rt: RefinedResidueType) -> Tuple[str, ...]:
        return (rt.default_jump_connection_atom,)

    def create_samples_for_poses(
        self,
        pose_stack: PoseStack,
        task: "SetPackerTask",  # noqa: F821
    ) -> Tuple[
        Tensor[torch.int32][:],  # n_rots_for_gbt
        Tensor[torch.int32][:],  # gbt_for_rotamer
        dict,  # anything else the sampler wants to save for later
    ]:
        assert (
            id(self) in task.conformer_sampler_index
        ), "This sampler is not in the PackerTask's conformer samplers"
        self_ind_in_packer_task = task.conformer_sampler_index[id(self)]

        rows = (task.cons_bt_pose, task.cons_bt_block, task.cons_bt_which_block_type)
        n_rots_for_gbt = include_current_mask(
            task.per_block_considered_block_types_is_orig[rows],
            task.per_block_is_block_type_allowed[rows],
            task.per_block_conformer_sampler_allowed[
                task.cons_bt_pose, task.cons_bt_block, self_ind_in_packer_task
            ],
        ).to(torch.int32)

        gbt_for_rotamer = n_rots_for_gbt.nonzero(as_tuple=True)[0].to(torch.int32)
        return (n_rots_for_gbt, gbt_for_rotamer, {"copy_input_coordinates": True})

    def fill_dofs_for_samples(
        self,
        pose_stack: PoseStack,
        task: "PackerTask",  # noqa: F821
        orig_kinforest: KinForest,
        orig_dofs_kto: Tensor[torch.float32][:, 9],
        gbt_for_conformer: Tensor[torch.int64][:],
        block_type_ind_for_conformer: Tensor[torch.int64][:],
        n_dof_atoms_offset_for_conformer: Tensor[torch.int64][:],
        # which of all conformers are built by this sampler
        conformer_built_by_sampler: Tensor[torch.bool][:],
        # mapping orig conformer samples to merged conformer samples for this sampler
        conf_inds_for_sampler: Tensor[torch.int64][:],
        sampler_n_rots_for_gbt: Tensor[torch.int32][:],
        sampler_gbt_for_rotamer: Tensor[torch.int32][:],
        sample_dict: dict,
        conf_dofs_kto: Tensor[torch.float32][:, 9],
    ):
        n_rots = sampler_gbt_for_rotamer.shape[0]
        if n_rots == 0:
            return

        from ._conformer_execution import copy_current_conformer_dofs

        copy_current_conformer_dofs(
            pose_stack,
            task,
            gbt_for_conformer,
            block_type_ind_for_conformer,
            n_dof_atoms_offset_for_conformer,
            conf_inds_for_sampler,
            orig_dofs_kto,
            conf_dofs_kto,
        )


def _create_full_dof_inds_to_copy_from_orig_to_rotamers_for_include_current_sampler(
    poses: PoseStack,
    task: "SetPackerTask",  # noqa F821
    gbt_for_rot: Tensor[torch.int64][:],  # max-n-rots
    block_type_ind_for_rot: Tensor[torch.int64][:],
    conf_inds_for_sampler: Tensor[torch.int64][:],
    sampler_n_rots_for_gbt: Tensor[torch.int32][:],
    sampler_gbt_for_rotamer: Tensor[torch.int32][:],
    n_dof_atoms_offset_for_rot: Tensor[torch.int64][:],
) -> Tuple[Tensor[torch.int64][:], Tensor[torch.int64][:]]:
    if conf_inds_for_sampler.numel() == 0:
        return conf_inds_for_sampler, conf_inds_for_sampler
    from ._conformer_execution import current_conformer_copy_segments
    from ._conformer_compiled import conformer_row_indices

    sizes, source, target = current_conformer_copy_segments(
        poses,
        task,
        gbt_for_rot,
        block_type_ind_for_rot,
        n_dof_atoms_offset_for_rot,
        conf_inds_for_sampler,
    )
    ends = sizes.cumsum(0)
    packed = torch.cat((ends.new_zeros(1), ends))
    indices = conformer_row_indices(source, target, packed, int(ends[-1]), 0).to(
        poses.device
    )
    return indices[1], indices[0]
