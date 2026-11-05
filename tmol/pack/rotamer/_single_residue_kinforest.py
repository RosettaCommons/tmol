import attr
import torch

from tmol.types._functional import _validate_args
from tmol.chemical import RefinedResidueType
from tmol.pose._packed_block_types import PackedBlockTypes
from tmol_core.pack.kinematics import (  # noqa: F401 (legacy schema reexports)
    RotamerKintree,
    PackedRotamerKintree,
    prepare_rotamer_kinematics,
    pack_rotamer_kinematics,
    annotate_residue_rotamer_kinematics,
    prepare_block_group_rotamer_kinematics,
)


def _inverse_cpu(*arrays):
    from tmol.kinematics.compiled._compiled_ops import _inverse_kin

    return _inverse_kin(*(torch.from_numpy(array) for array in arrays)).numpy()


def _rotamer_kinforest_from_data(rkd, n_atoms, atom_names, mainchain_atoms, ideal):
    return prepare_rotamer_kinematics(
        rkd, n_atoms, atom_names, mainchain_atoms, ideal, _inverse_cpu
    )


@_validate_args
def _construct_single_residue_kinforest(restype: RefinedResidueType):
    """Annotate residue-local topology and ideal DOFs through the shared core."""
    annotate_residue_rotamer_kinematics(restype, _inverse_cpu)


def _construct_block_group_kinforest(
    block_types, links, anchor: int = 0, kinforest_data=None
):
    return prepare_block_group_rotamer_kinematics(
        block_types,
        links,
        inverse_kinematics=_inverse_cpu,
        anchor=anchor,
        kinforest_data=kinforest_data,
    )


@_validate_args
def _coalesce_single_residue_kinforests(pbt: PackedBlockTypes):
    if hasattr(pbt, "rotamer_kinforest"):
        return
    packed = pack_rotamer_kinematics(pbt.active_block_types)
    ideal = packed.dofs_ideal
    dofs = (
        torch.from_numpy(ideal).to(pbt.device)
        if ideal.size
        else torch.empty(ideal.shape, dtype=torch.float32, device=pbt.device)
    )
    pbt.rotamer_kinforest = attr.evolve(packed, dofs_ideal=dofs)
