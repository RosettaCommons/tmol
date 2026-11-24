"""Select a subset of blocks of a PoseStack into a compact PoseStack."""

from typing import Sequence

import attr
import numpy
import torch

from tmol.chemical._constants import MAX_SIG_BOND_SEPARATION

from ._inter_block_bondsep import InterBlockBondsep
from ._pdb_info import PDBInfo
from ._pose_stack import PoseStack
from ._constraint_set import ConstraintSet


def _select_inter_block_bondsep(
    pose_stack: PoseStack, blocks: torch.Tensor
) -> InterBlockBondsep:
    """Restrict the sparse block-pair bond separations to ``blocks``.

    Kept slot rows are renumbered in place; the dense table is never built, as
    it is quadratic in the source stack's block count.
    """
    source = pose_stack.inter_block_bondsep
    device = pose_stack.device
    n_poses = source.n_poses
    n_blocks = int(blocks.numel())
    n_conn = source.max_n_conn
    if n_blocks == 0 or source.n_slots == 0:
        return InterBlockBondsep.empty(n_poses, n_blocks, n_conn, device, 1)
    new_of_old = torch.full(
        (source.max_n_blocks,), -1, dtype=torch.int64, device=device
    )
    new_of_old[blocks] = torch.arange(n_blocks, dtype=torch.int64, device=device)
    if int((new_of_old >= 0).sum()) != n_blocks:
        raise ValueError("selecting blocks requires distinct block indices")

    near = source.near_blocks[:, blocks]
    bondsep = source.bondsep[:, blocks]
    block2 = new_of_old[near[..., 0].to(torch.int64).clamp(min=0)]
    kept = (
        (near[..., 0] >= 0) & (block2 >= 0) & (near[..., 1] < MAX_SIG_BOND_SEPARATION)
    )

    # Rows list their block2 slots ascending, so sort the kept slots to the
    # front in their new numbering.
    order = torch.where(kept, block2, block2.new_full((), n_blocks)).argsort(dim=2)
    kept = kept.gather(2, order)
    pose, block1, slot = torch.nonzero(kept, as_tuple=True)
    result = InterBlockBondsep.empty(
        n_poses, n_blocks, n_conn, device, int(kept.sum(dim=2).max()) + 1
    )
    source_slot = order[pose, block1, slot]
    result.near_blocks[pose, block1, slot, 0] = block2[pose, block1, source_slot].to(
        torch.int32
    )
    result.near_blocks[pose, block1, slot, 1] = near[pose, block1, source_slot, 1]
    result.bondsep[pose, block1, slot] = bondsep[pose, block1, source_slot]
    return result


def select_blocks(pose_stack: PoseStack, blocks: Sequence[int]) -> PoseStack:
    """Compact PoseStack holding ``blocks`` (in the given order) of every pose.

    All poses must share one block layout. Connections to omitted blocks are
    cleared, and connections between retained blocks are remapped. Chain ids,
    PDB metadata and bond separations are carried over; constraints and split
    block mappings are dropped.
    """

    if pose_stack.split_block_mapping:
        raise ValueError("selecting blocks does not support split-block metadata")
    device = pose_stack.device
    keep = torch.as_tensor(list(blocks), dtype=torch.int64, device=device)
    block_types = pose_stack.block_type_ind64
    if keep.numel() == 0 or bool(torch.any(block_types[:, keep] < 0)):
        raise ValueError("blocks must be a non-empty set present in every pose")
    selected_types = block_types[:, keep]
    if not bool(torch.all(selected_types == selected_types[0:1])):
        raise ValueError("selecting blocks requires a homogeneous block layout")

    old_to_new = torch.full(
        (pose_stack.max_n_blocks,), -1, dtype=torch.int64, device=device
    )
    old_to_new[keep] = torch.arange(keep.numel(), dtype=torch.int64, device=device)
    connections = pose_stack.inter_residue_connections64[:, keep].clone()
    partners = connections[..., 0]
    real = partners >= 0
    mapped = torch.full_like(partners, -1)
    mapped[real] = old_to_new[partners[real]]
    connections[real & (mapped < 0)] = -1
    partners[mapped >= 0] = mapped[mapped >= 0]

    new_types = block_types[:, keep].clone()
    pbt = pose_stack.packed_block_types
    n_atoms = pbt.n_atoms[new_types[0]].to(torch.int64)
    offsets = torch.zeros_like(new_types)
    offsets[:, 1:] = torch.cumsum(n_atoms[:-1], dim=0)[None, :]
    source_atoms = torch.cat(
        [
            pose_stack.block_coord_offset64[0, b]
            + torch.arange(int(n), dtype=torch.int64, device=device)
            for b, n in zip(keep, n_atoms)
        ]
    )

    keep_np, atoms_np = keep.cpu().numpy(), source_atoms.cpu().numpy()
    pdb = pose_stack.pdb_info
    by_residue = {
        name: getattr(pdb, name)[:, keep_np].copy()
        for name in (
            "residue_labels",
            "residue_insertion_codes",
            "chain_labels",
            "metal_origins",
            "residue_annotations",
        )
        if getattr(pdb, name) is not None
    }
    pdb_info = PDBInfo(
        atom_occupancy=pdb.atom_occupancy[:, atoms_np].copy(),
        atom_b_factor=pdb.atom_b_factor[:, atoms_np].copy(),
        **by_residue,
    )
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
    if pose_stack.n_poses != 1:
        raise ValueError("only a single reference pose can be replicated")
    if pose_stack.split_block_mapping:
        raise ValueError("replicating split-block metadata is not supported")

    def tile(value: torch.Tensor) -> torch.Tensor:
        return value.repeat((n_poses,) + (1,) * (value.ndim - 1))

    constraints = ConstraintSet.concatenate(
        [pose_stack.constraint_set] * n_poses,
        from_multiple_pose_stacks=True,
        n_poses=n_poses,
        ps_offset=torch.arange(n_poses, dtype=torch.int64),
    )
    pdb_info = PDBInfo(
        **{
            name: None if value is None else numpy.repeat(value, n_poses, axis=0)
            for name, value in attr.asdict(pose_stack.pdb_info, recurse=False).items()
        }
    )
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


_POSE_TENSORS = (
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

_PDB_FIELDS = (
    "residue_labels",
    "residue_insertion_codes",
    "chain_labels",
    "atom_occupancy",
    "atom_b_factor",
    "metal_origins",
    "residue_annotations",
)


def select_poses(pose_stack: PoseStack, poses: torch.Tensor) -> PoseStack:
    """The poses at ``poses``, in that order; constraints are dropped."""
    if pose_stack.split_block_mapping:
        raise ValueError("selecting poses does not support split-block metadata")
    poses = poses.to(device=pose_stack.device, dtype=torch.int64)
    host = poses.cpu().numpy()
    pdb = pose_stack.pdb_info
    bondsep = pose_stack.inter_block_bondsep
    return attr.evolve(
        pose_stack,
        **{name: getattr(pose_stack, name)[poses].clone() for name in _POSE_TENSORS},
        inter_block_bondsep=InterBlockBondsep(
            near_blocks=bondsep.near_blocks[poses].clone(),
            bondsep=bondsep.bondsep[poses].clone(),
        ),
        pdb_info=PDBInfo(
            **{
                name: None if getattr(pdb, name) is None else getattr(pdb, name)[host]
                for name in _PDB_FIELDS
            }
        ),
        constraint_set=None,
    )


def replace_poses(
    pose_stack: PoseStack, poses: torch.Tensor, replacement: PoseStack
) -> PoseStack:
    """``pose_stack`` with the poses at ``poses`` taken from ``replacement``.

    The two must share packed block types and block count; atoms and bond
    separation slots are padded to the wider of the two. Constraints and
    split-block metadata are kept from ``pose_stack``.
    """
    if replacement.packed_block_types is not pose_stack.packed_block_types:
        raise ValueError("replacement poses use other packed block types")
    if replacement.max_n_blocks != pose_stack.max_n_blocks:
        raise ValueError("replacement poses have another block count")
    if replacement.n_poses != len(poses):
        raise ValueError("one replacement pose is needed per replaced index")
    poses = poses.to(device=pose_stack.device, dtype=torch.int64)
    host = poses.cpu().numpy()
    n_atoms = max(pose_stack.max_n_pose_atoms, replacement.max_n_pose_atoms)

    def merged(name: str) -> torch.Tensor:
        value = getattr(pose_stack, name)
        part = getattr(replacement, name)
        if name == "coords":
            out = value.new_zeros((value.shape[0], n_atoms, 3))
            out[:, : value.shape[1]] = value
            out[poses] = 0
            out[poses, : part.shape[1]] = part
            return out
        out = value.clone()
        out[poses] = part
        return out

    def merged_pdb(name: str):
        value = getattr(pose_stack.pdb_info, name)
        part = getattr(replacement.pdb_info, name)
        if value is None or part is None:
            return value
        if name.startswith("atom_"):
            out = numpy.zeros((value.shape[0], n_atoms), dtype=value.dtype)
            out[:, : value.shape[1]] = value
            out[host] = 0
            out[host, : part.shape[1]] = part
            return out
        out = value.copy()
        out[host] = part
        return out

    old, new = pose_stack.inter_block_bondsep, replacement.inter_block_bondsep
    n_conn = max(old.max_n_conn, new.max_n_conn)
    # An empty part widens the slots to the replacement's.
    widen = InterBlockBondsep.empty(
        0, old.max_n_blocks, n_conn, old.device, new.n_slots
    )
    bondsep = InterBlockBondsep.concatenate([old, widen], old.max_n_blocks, n_conn)
    k, m = new.n_slots, new.max_n_conn
    bondsep.near_blocks[poses] = MAX_SIG_BOND_SEPARATION
    bondsep.near_blocks[poses, :, :, 0] = -1
    bondsep.bondsep[poses] = MAX_SIG_BOND_SEPARATION
    bondsep.near_blocks[poses, :, :k] = new.near_blocks
    bondsep.bondsep[poses, :, :k, :m, :m] = new.bondsep
    return attr.evolve(
        pose_stack,
        **{name: merged(name) for name in _POSE_TENSORS},
        inter_block_bondsep=bondsep,
        pdb_info=PDBInfo(**{name: merged_pdb(name) for name in _PDB_FIELDS}),
    )
