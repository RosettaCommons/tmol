"""Select a subset of blocks of a PoseStack into a compact PoseStack."""

from typing import Sequence

import attr
import torch

from tmol_core.pose.block_selection import (
    selection_plan,
    select_pdb_info,
    repeat_pdb_info,
    validate_repeat,
)

from tmol.chemical._constants import MAX_SIG_BOND_SEPARATION

from ._inter_block_bondsep import InterBlockBondsep
from ._pose_stack import PoseStack
from ._constraint_set import ConstraintSet


def _select_inter_block_bondsep(
    pose_stack: PoseStack, blocks: torch.Tensor
) -> InterBlockBondsep:
    """Restrict the sparse block-pair bond separations to ``blocks``."""
    dense = pose_stack.inter_block_bondsep.to_dense()[
        :, blocks[:, None], blocks[None, :]
    ]
    n_poses, n_blocks, _, n_conn, _ = dense.shape
    if dense.numel() == 0:
        return InterBlockBondsep.empty(n_poses, n_blocks, n_conn, pose_stack.device, 1)
    minimum = dense.amin(dim=(-2, -1))
    nearby = minimum < MAX_SIG_BOND_SEPARATION
    slots = nearby.cumsum(dim=2) - 1
    result = InterBlockBondsep.empty(
        n_poses,
        n_blocks,
        n_conn,
        pose_stack.device,
        int(nearby.sum(dim=2).max()) + 1,
    )
    pose, block1, block2 = torch.nonzero(nearby, as_tuple=True)
    slot = slots[pose, block1, block2]
    result.near_blocks[pose, block1, slot, 0] = block2.to(torch.int32)
    result.near_blocks[pose, block1, slot, 1] = minimum[pose, block1, block2].to(
        torch.int32
    )
    result.bondsep[pose, block1, slot] = dense[pose, block1, block2]
    return result


def select_blocks(pose_stack: PoseStack, blocks: Sequence[int]) -> PoseStack:
    """Compact PoseStack holding ``blocks`` (in the given order) of every pose.

    All poses must share one block layout. Connections to omitted blocks are
    cleared, and connections between retained blocks are remapped. Chain ids,
    PDB metadata and bond separations are carried over; constraints and split
    block mappings are dropped.
    """

    device = pose_stack.device
    pbt = pose_stack.packed_block_types
    plan = selection_plan(
        blocks,
        pose_stack.block_type_ind64.cpu().numpy(),
        pose_stack.block_coord_offset64.cpu().numpy(),
        [bt.n_atoms for bt in pbt.active_block_types],
        pose_stack.inter_residue_connections64.cpu().numpy(),
        split_block_mapping=pose_stack.split_block_mapping,
    )
    keep, source_atoms, new_types, offsets, connections = (
        torch.as_tensor(value, dtype=torch.int64, device=device)
        for value in (
            plan.blocks,
            plan.source_atoms,
            plan.block_types,
            plan.offsets,
            plan.connections,
        )
    )
    pdb_info = select_pdb_info(pose_stack.pdb_info, plan)
    chains = pose_stack.chain_id64[:, keep].clone()
    return PoseStack(
        packed_block_types=pbt,
        coords=pose_stack.coords[:, source_atoms].clone(),
        block_coord_offset=offsets.to(torch.int32),
        block_coord_offset64=offsets,
        inter_residue_connections=connections.to(torch.int32),
        inter_residue_connections64=connections,
        inter_block_bondsep=_select_inter_block_bondsep(pose_stack, keep),
        block_type_ind=new_types.to(torch.int32),
        block_type_ind64=new_types,
        chain_id=chains.to(torch.int32),
        chain_id64=chains,
        pdb_info=pdb_info,
        constraint_set=None,
        device=device,
    )


def repeat_pose_stack(pose_stack: PoseStack, n_poses: int) -> PoseStack:
    """A PoseStack of ``n_poses`` independent copies of a single pose.

    Unlike :meth:`PoseStackBuilder.from_poses` this keeps the coordinate dtype
    and the constraint set.
    """
    validate_repeat(pose_stack.n_poses, pose_stack.split_block_mapping)

    def tile(value: torch.Tensor) -> torch.Tensor:
        return value.repeat((n_poses,) + (1,) * (value.ndim - 1))

    constraints = ConstraintSet.concatenate(
        [pose_stack.constraint_set] * n_poses,
        from_multiple_pose_stacks=True,
        n_poses=n_poses,
        ps_offset=torch.arange(n_poses, dtype=torch.int64),
    )
    pdb_info = repeat_pdb_info(pose_stack.pdb_info, n_poses)
    tensors = (
        "coords",
        "block_coord_offset",
        "block_coord_offset64",
        "inter_residue_connections",
        "inter_residue_connections64",
        "block_type_ind",
        "block_type_ind64",
        "chain_id",
        "chain_id64",
    )
    bondsep = pose_stack.inter_block_bondsep
    return attr.evolve(
        pose_stack,
        **{name: tile(getattr(pose_stack, name)) for name in tensors},
        inter_block_bondsep=InterBlockBondsep.concatenate(
            [bondsep] * n_poses, bondsep.max_n_blocks, bondsep.max_n_conn
        ),
        pdb_info=pdb_info,
        constraint_set=constraints,
    )
