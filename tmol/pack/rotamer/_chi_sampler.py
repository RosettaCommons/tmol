import torch
import attr

from typing import TYPE_CHECKING, Any, Union

from tmol.types._torch import Tensor
from tmol.types._functional import _validate_args
from tmol.chemical import RefinedResidueType
from tmol.pose._packed_block_types import PackedBlockTypes
from tmol.pose._pose_stack import PoseStack
from tmol.kinematics._datatypes import KinForest
from tmol.pack.rotamer._conformer_sampler import ConformerSample, ConformerSampler

if TYPE_CHECKING:
    from tmol.pack._packer_task import PackerTask


@attr.s(auto_attribs=True)
class ChiSampler(ConformerSampler):
    """Base class for samplers that define conformers through chi angles."""

    @classmethod
    def sampler_name(cls) -> str:
        """Return the stable name used for chi-sampler annotations."""
        raise NotImplementedError()

    @_validate_args
    def annotate_residue_type(self, rt: RefinedResidueType) -> None:
        """Attach optional sampler metadata to one residue type."""
        pass

    @_validate_args
    def annotate_packed_block_types(self, packed_block_types: PackedBlockTypes) -> None:
        """Attach optional sampler metadata to packed block types."""
        pass

    @_validate_args
    def defines_rotamers_for_rt(self, rt: RefinedResidueType) -> bool:
        """Return whether this sampler supports a residue type."""
        raise NotImplementedError()

    @_validate_args
    def first_sc_atoms_for_rt(self, rt_name: str) -> tuple[str, ...]:
        """Return side-chain roots used to transfer main-chain geometry."""
        raise NotImplementedError()

    def create_samples_for_poses(
        self,
        pose_stack: PoseStack,
        task: "PackerTask",
    ) -> ConformerSample:
        """Create chi samples and preserve their defining atoms and angles."""
        (
            n_rots_for_gbt,
            gbt_for_rotamer,
            chi_defining_atom_for_rotamer,
            chi_for_rotamers,
        ) = self.sample_chi_for_poses(pose_stack, task)
        return (
            n_rots_for_gbt,
            gbt_for_rotamer,
            dict(
                chi_defining_atom_for_rotamer=chi_defining_atom_for_rotamer,
                chi_for_rotamers=chi_for_rotamers,
            ),
        )

    def sample_chi_for_poses(
        self, systems: PoseStack, task: "PackerTask"
    ) -> tuple[
        Tensor[torch.int32][:, :, :],  # n_rots_for_rt
        Tensor[torch.int32][:],  # rt_for_rotamer
        Tensor[torch.int32][:, :],  # chi_defining_atom_for_rotamer
        Tensor[torch.float32][:, :],  # chi_for_rotamers, in radians
    ]:
        raise NotImplementedError()

    def fill_dofs_for_samples(
        self,
        pose_stack: PoseStack,
        task: "PackerTask",
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
        sample_dict: dict[str, Any],
        conf_dofs_kto: Tensor[torch.float32][:, 9],
    ) -> None:
        _copy_dofs_from_orig_to_rotamers_for_sampler(
            pose_stack,
            task,
            id(self),
            gbt_for_conformer,
            block_type_ind_for_conformer,
            conf_inds_for_sampler,
            sampler_n_rots_for_gbt,
            sampler_gbt_for_rotamer,
            n_dof_atoms_offset_for_conformer,
            orig_dofs_kto,
            conf_dofs_kto,
        )

        chi_atoms = sample_dict["chi_defining_atom_for_rotamer"]
        chi = sample_dict["chi_for_rotamers"]
        if chi.shape[0] == 0:
            return

        _assign_chi_dofs_from_samples(
            pose_stack.packed_block_types,
            block_type_ind_for_conformer,
            conf_inds_for_sampler,
            sampler_n_rots_for_gbt,
            sampler_gbt_for_rotamer,
            n_dof_atoms_offset_for_conformer,
            chi_atoms,
            chi,
            conf_dofs_kto,
        )


@_validate_args
def _copy_dofs_from_orig_to_rotamers_for_sampler(
    poses: PoseStack,
    task,
    sampler_name: Union[str, int],
    gbt_for_rot: Tensor[torch.int64][:],
    block_type_ind_for_rot: Tensor[torch.int64][:],
    conf_inds_for_sampler: Tensor[torch.int64][:],
    sampler_n_rots_for_gbt: Tensor[torch.int32][:],
    sampler_gbt_for_rotamer: Tensor[torch.int32][:],
    n_dof_atoms_offset_for_rot: Tensor[torch.int64][:],
    orig_dofs_kto: Tensor[torch.float32][:, 9],
    rot_dofs_kto: Tensor[torch.float32][:, 9],
) -> None:
    if conf_inds_for_sampler.numel() == 0:
        return
    if poses.packed_block_types.mc_fingerprints.atom_mapping.shape[1] == 0:
        return
    from ._conformer_execution import mainchain_copy_metadata, copy_mainchain_dofs

    metadata, sampler = mainchain_copy_metadata(
        poses,
        task,
        sampler_name,
        gbt_for_rot,
        block_type_ind_for_rot,
        conf_inds_for_sampler,
        n_dof_atoms_offset_for_rot,
    )
    copy_mainchain_dofs(*metadata, orig_dofs_kto, rot_dofs_kto, sampler)


def _create_dof_inds_to_copy_from_orig_to_rotamers_for_sampler(
    poses: PoseStack,
    task: "PackerTask",  # noqa F821
    sampler_name: Union[str, int],
    gbt_for_rot: Tensor[torch.int64][:],  # max-n-rots
    block_type_ind_for_rot: Tensor[torch.int64][:],
    conf_inds_for_sampler: Tensor[torch.int64][:],
    sampler_n_rots_for_gbt: Tensor[torch.int32][:],
    sampler_gbt_for_rotamer: Tensor[torch.int32][:],
    n_dof_atoms_offset_for_rot: Tensor[torch.int64][:],
) -> tuple[Tensor[torch.int64][:], Tensor[torch.int64][:]]:
    """Map mainchain DOFs from original residues to sampled rotamers.

    Returns:
        Destination and source indices, each shaped ``[n_copied_dofs]``.
    """

    # A sampler with no states need not have an ownership fingerprint at all.
    if conf_inds_for_sampler.numel() == 0:
        return conf_inds_for_sampler, conf_inds_for_sampler
    pbt = poses.packed_block_types
    fingerprints = pbt.mc_fingerprints
    if fingerprints.atom_mapping.shape[1] == 0:
        empty = conf_inds_for_sampler[:0]
        return empty, empty
    from ._conformer_execution import mainchain_copy_metadata, mainchain_row_indices

    metadata, sampler = mainchain_copy_metadata(
        poses,
        task,
        sampler_name,
        gbt_for_rot,
        block_type_ind_for_rot,
        conf_inds_for_sampler,
        n_dof_atoms_offset_for_rot,
    )
    src, dst = mainchain_row_indices(metadata, sampler, poses.device)
    return dst, src


@_validate_args
def _assign_chi_dofs_from_samples(
    pbt: PackedBlockTypes,
    block_type_ind_for_rot: Tensor[torch.int64][:],
    conf_inds_for_sampler: Tensor[torch.int64][:],
    sampler_n_rots_for_bt: Tensor[torch.int32][:],
    sampler_gbt_for_rotamer: Tensor[torch.int32][:],
    n_dof_atoms_offset_for_rot: Tensor[torch.int64][:],
    chi_atoms: Tensor[torch.int32][:, :],
    chi: Tensor[torch.float32][:, :],
    rot_dofs_kto: Tensor[torch.float32][:, 9],
) -> None:
    """Write sampled chis into packed ``[n_rotamer_atoms + 1, 9]`` DOFs.

    ``chi_atoms`` and ``chi`` share shape ``[n_rotamers, max_n_chi]``;
    negative atom indices mark unused chi columns.
    """
    assert chi_atoms.shape == chi.shape

    if chi_atoms.numel() == 0:
        return
    from tmol.pack.rotamer._build_rotamers import (
        _build_ring_chi_phi_c_corrections,
        _kinforest_device_indices,
    )

    from tmol.pack.rotamer._conformer_execution import assign_chi_dofs

    assign_chi_dofs(
        block_type_ind_for_rot,
        n_dof_atoms_offset_for_rot,
        conf_inds_for_sampler,
        chi_atoms,
        chi,
        _kinforest_device_indices(pbt, pbt.device),
        _build_ring_chi_phi_c_corrections(pbt).to(dtype=chi.dtype),
        rot_dofs_kto,
    )
