import numpy
import torch
import attr

from tmol.types._torch import Tensor
from tmol.types._functional import _validate_args
from tmol.pose._packed_block_types import PackedBlockTypes

from ._build_missing_context import _build_missing_by_context
from tmol_core.io import missing_atoms as _common
from tmol_core.io.missing_atoms import (  # noqa: F401
    BlockTypeLeafAtomsAnnotation,
    BlockTypeLeafAtomICoorAnnotation,
    BlockTypeHCompletionAnnotation,
    _uaid_for_at,
    _icoor_at_is_leaf,
    _icoor_at_is_h,
    _icoor_at_is_inter_res,
)


@_validate_args
def _build_missing_leaf_atoms(
    packed_block_types: PackedBlockTypes,
    block_types64: Tensor[torch.int64][:, :],
    real_block_atoms: Tensor[torch.bool][:, :, :],
    block_coords: Tensor[torch.float32][:, :, :, 3],
    block_atom_missing: Tensor[torch.bool][:, :, :],
    inter_residue_connections: Tensor[torch.int32][:, :, :, 2],
    fail_on_missing_nonleaf_atoms: bool = True,
    packer_atoms=None,
):
    """Convert the block layout into the condensed layout used by PoseStack and
    build any missing "leaf" atoms at the same time. This is a fully differentiable
    process, and so gradients accumulated for any leaf atoms that were absent
    from the input structure will be distributed across the atoms that define the
    geometry of those atoms. A leaf atom is an atom that is not a parent to any other
    atom; these include hydrogens and carbonyl/carboxyl oxygens.
    Missing heavy atoms of non-polymer, noncanonical and nucleic-acid blocks are
    grown outward from their observed atoms and connection partners, each rotation
    the observed atoms leave free chosen to avoid clashes (see
    build_missing_by_context); their hydrogens then follow from icoors. Observed
    atoms are preserved; atoms no placed reference reaches remain unresolved.
    packer_atoms, given a block type, marks the atoms rotamer packing will place;
    those are left missing for it.
    """

    (
        real_block_atoms,
        pose_at_is_real,
        block_leaf_atom_is_missing,
        pose_stack_atom_is_missing,
        block_coord_offset,
        block_types,
        inter_residue_connections,
        block_has_missing_atoms,
    ) = _setup_for_leaf_atom_coord_building(
        packed_block_types,
        block_types64,
        real_block_atoms,
        block_coords,
        block_atom_missing,
        inter_residue_connections,
    )

    # non-canonical blocks grow missing heavy atoms by context; hydrogens and
    # canonical leaves follow from icoors
    conformer_blocks = None
    targets = block_leaf_atom_is_missing
    if packed_block_types.has_conformer_frames:
        conformer_blocks = (block_types >= 0) & packed_block_types.conformer_types[
            block_types64.clamp_min(0)
        ]
        rebuildable = packed_block_types.conformer_atoms[block_types64.clamp_min(0)]
        if packer_atoms is not None:
            rebuildable = rebuildable & ~_packer_atom_mask(
                packed_block_types, block_types64, packer_atoms
            )
        targets = targets | (
            conformer_blocks.unsqueeze(-1) & block_atom_missing & rebuildable
        )
    pose_coords = block_coords.new_zeros((*pose_stack_atom_is_missing.shape, 3))
    pose_coords[pose_at_is_real] = block_coords[real_block_atoms]
    if conformer_blocks is not None:
        heavy = packed_block_types.atom_is_hydrogen[block_types64.clamp_min(0)] == 0
        context = (
            conformer_blocks.unsqueeze(-1)
            & block_atom_missing
            & rebuildable
            & heavy
            & real_block_atoms
        )
        if torch.any(context):
            pose_targets = torch.zeros_like(pose_stack_atom_is_missing)
            pose_targets[pose_at_is_real] = context[real_block_atoms]
            pose_coords, pose_built = _build_missing_by_context(
                packed_block_types,
                pose_coords,
                pose_stack_atom_is_missing,
                pose_targets,
                block_coord_offset,
                block_types,
                inter_residue_connections,
            )
            block_built = torch.zeros_like(block_atom_missing)
            block_built[real_block_atoms] = pose_built[pose_at_is_real]
            pose_stack_atom_is_missing = pose_stack_atom_is_missing & ~pose_built
            block_atom_missing = block_atom_missing & ~block_built
            targets = targets & ~block_built
    new_pose_coords = _build_coords_from_icoors(
        packed_block_types,
        pose_coords,
        targets,
        pose_stack_atom_is_missing,
        block_coord_offset,
        block_types,  # int32s
        inter_residue_connections,
    )

    # Ligand heavy atoms can use the same differentiable construction frames.
    # Iterate only when an eligible block needs coordinate completion; observed
    # coordinates are copied unchanged and an unanchored component stays NaN.
    if conformer_blocks is not None and torch.any(
        conformer_blocks.unsqueeze(-1) & block_atom_missing
    ):
        remaining = block_atom_missing.clone()
        while True:
            missing_pose = torch.isnan(new_pose_coords).any(dim=-1)
            remaining[real_block_atoms] = missing_pose[pose_at_is_real]
            targets = (
                remaining
                & conformer_blocks.unsqueeze(-1)
                & real_block_atoms
                & rebuildable
            )
            if not torch.any(targets):
                break
            built = _build_coords_from_icoors(
                packed_block_types,
                new_pose_coords,
                targets,
                missing_pose,
                block_coord_offset,
                block_types,
                inter_residue_connections,
            )
            if torch.equal(torch.isnan(built).any(dim=-1), missing_pose):
                # no icoor frame reaches them (hydrogens on a lone heavy atom):
                #    grow them from what is placed
                pose_targets = torch.zeros_like(missing_pose)
                pose_targets[pose_at_is_real] = targets[real_block_atoms]
                built, _ = _build_missing_by_context(
                    packed_block_types,
                    built,
                    missing_pose,
                    pose_targets,
                    block_coord_offset,
                    block_types,
                    inter_residue_connections,
                    heavy_only=False,
                )
                if torch.equal(torch.isnan(built).any(dim=-1), missing_pose):
                    break
            new_pose_coords = built
        block_atom_missing = remaining
    if conformer_blocks is not None:
        block_has_missing_atoms = torch.any(
            block_atom_missing
            & ~packed_block_types.is_leaf_atom[block_types.clamp_min(0).long()]
            & real_block_atoms,
            dim=-1,
        )
    if fail_on_missing_nonleaf_atoms and torch.any(block_has_missing_atoms):
        missing = (
            block_atom_missing
            & ~packed_block_types.is_leaf_atom[block_types.clamp_min(0).long()]
            & real_block_atoms
        )
        errors = []
        for pi, bi, ai in torch.nonzero(missing).cpu().tolist():
            bt = packed_block_types.active_block_types[int(block_types[pi, bi])]
            errors.append(
                f"Error: missing non-leaf atom {bt.atoms[ai].name} on residue {bi} "
                f"{bt.name} on pose {pi} real res? True"
            )
        raise ValueError("\n".join(errors))

    # For autogen ligands, added H whose dihedral came from a different
    # conformer than the placed heavy atoms can collide with a neighbor; snap
    # such (geometrically determined) H to their parent's open vertex.
    new_pose_coords = _apply_h_geometric_completion(
        packed_block_types,
        new_pose_coords,
        block_leaf_atom_is_missing,
        block_coord_offset,
        block_types,
        inter_residue_connections,
    )

    return (
        new_pose_coords,
        block_coord_offset,
        real_block_atoms,
        pose_at_is_real,
        block_has_missing_atoms,
    )


def _packer_atom_mask(pbt, block_types64, packer_atoms):
    """[n_poses, max_n_blocks, max_n_atoms]: the atoms packer_atoms marks."""
    table = torch.zeros((pbt.n_types, pbt.max_n_atoms), dtype=torch.bool)
    for t in torch.unique(block_types64[block_types64 >= 0]).tolist():
        bt = pbt.active_block_types[t]
        table[t, : bt.n_atoms] = torch.from_numpy(
            numpy.asarray(packer_atoms(bt), dtype=bool)
        )
    return table.to(block_types64.device)[block_types64.clamp_min(0)]


def _setup_for_leaf_atom_coord_building(
    packed_block_types: PackedBlockTypes,
    block_types64: Tensor[torch.int64][:, :],
    real_block_atoms: Tensor[torch.bool][:, :, :],
    block_coords: Tensor[torch.float32][:, :, :, 3],
    block_atom_missing: Tensor[torch.bool][:, :, :],
    inter_residue_connections: Tensor[torch.int32][:, :, :, 2],
):
    # ok,
    # we're going to call gen_pose_leaf_atoms,
    # but first we need to prepare the input tensors
    # that are going to use
    pbt = packed_block_types
    device = pbt.device
    n_poses = block_coords.shape[0]
    max_n_blocks = block_coords.shape[1]

    # make sure we have all the data we need
    _annotate_packed_block_types_atom_is_leaf_atom(pbt)
    _annotate_packed_block_types_w_leaf_atom_icoors(pbt)
    _annotate_packed_block_types_w_h_completion(pbt)

    n_atoms = torch.zeros((n_poses, max_n_blocks), dtype=torch.int32, device=device)
    real_blocks = block_types64 != -1
    n_atoms[real_blocks] = packed_block_types.n_atoms[block_types64[real_blocks]]

    n_ats_inccumsum = torch.cumsum(n_atoms, dim=1, dtype=torch.int32)
    max_n_ats = torch.max(n_ats_inccumsum[:, -1])
    block_coord_offset = torch.cat(
        (
            torch.zeros((n_poses, 1), dtype=torch.int32, device=device),
            n_ats_inccumsum[:, :-1],
        ),
        dim=1,
    )
    pose_at_is_real = (
        torch.arange(max_n_ats, dtype=torch.int64, device=device).repeat(n_poses, 1)
        < n_ats_inccumsum[:, -1:]
    )

    block_at_is_leaf = torch.zeros(
        (n_poses, max_n_blocks, pbt.max_n_atoms), dtype=torch.bool, device=device
    )
    block_at_is_leaf[real_blocks] = pbt.is_leaf_atom[block_types64[real_blocks]]

    # error checking: have we been asked to build an atom that cannot be built?
    non_leaf_atom_is_missing = torch.logical_and(
        block_atom_missing, torch.logical_not(block_at_is_leaf)
    )
    block_leaf_atom_is_missing = torch.logical_and(block_at_is_leaf, block_atom_missing)

    pose_stack_atom_is_missing = torch.zeros(
        (n_poses, max_n_ats), dtype=torch.bool, device=device
    )
    # Ancestor selection must see missing non-leaf atoms too. Otherwise a
    # missing CB looks available to HA's primary frame and its backup is never
    # tried. The separate leaf mask still controls which atoms this pass builds.
    pose_stack_atom_is_missing[pose_at_is_real] = block_atom_missing[real_block_atoms]

    # Create block_has_missing_atoms tensor: True for blocks that have any missing non-leaf atoms
    block_has_missing_atoms = torch.any(non_leaf_atom_is_missing, dim=2)

    return (
        real_block_atoms,
        pose_at_is_real,
        block_leaf_atom_is_missing,
        pose_stack_atom_is_missing,
        block_coord_offset,
        block_types64.to(torch.int32),
        inter_residue_connections,
        block_has_missing_atoms,
    )


def _build_coords_from_icoors(
    pbt,
    coords,
    targets,
    missing,
    offsets,
    block_types,
    connections,
):
    from tmol.io.details.compiled._compiled import _gen_pose_leaf_atoms

    return _gen_pose_leaf_atoms(
        coords,
        targets,
        missing,
        offsets,
        block_types,
        connections,
        pbt.n_atoms,
        pbt.atom_downstream_of_conn,
        pbt.build_missing_leaf_atom_icoor_ann.anc_uaids,
        pbt.build_missing_leaf_atom_icoor_ann.geom,
        pbt.build_missing_leaf_atom_icoor_ann.anc_uaids_backup,
        pbt.build_missing_leaf_atom_icoor_ann.geom_backup,
    )


@_validate_args
def _annotate_packed_block_types_atom_is_leaf_atom(pbt: PackedBlockTypes):
    if hasattr(pbt, "is_leaf_atom"):
        return
    flags = _common.prepare_leaf_atom_flags(
        pbt.active_block_types, pbt.atom_is_hydrogen.cpu().numpy(), pbt.max_n_atoms
    )
    pbt.is_leaf_atom = torch.as_tensor(flags.is_leaf_atom, device=pbt.device)
    pbt.has_conformer_frames = flags.has_conformer_frames
    pbt.conformer_types = torch.as_tensor(flags.conformer_types, device=pbt.device)
    if flags.has_conformer_frames:
        pbt.conformer_atoms = torch.as_tensor(flags.conformer_atoms, device=pbt.device)


def _annotate_block_type_atom_is_leaf_atom(block_type, is_hydrogen):
    return _common._annotate_block_type_atom_is_leaf_atom(
        block_type,
        numpy.asarray(
            is_hydrogen.cpu() if isinstance(is_hydrogen, torch.Tensor) else is_hydrogen
        ),
    )


@attr.s(auto_attribs=True, slots=True, frozen=True)
class PackedBlockTypesLeafAtomICoorAnnotation:
    geom: Tensor[torch.float32][:, :, 3]
    anc_uaids: Tensor[torch.int32][:, :, 3, 3]
    geom_backup: Tensor[torch.float32][:, :, 3]
    anc_uaids_backup: Tensor[torch.int32][:, :, 3, 3]


def _annotate_packed_block_types_w_leaf_atom_icoors(pbt: PackedBlockTypes):
    if hasattr(pbt, "build_missing_leaf_atom_icoor_ann"):
        return
    host = _common.prepare_leaf_icoor_metadata(
        pbt.active_block_types, pbt.atom_is_hydrogen.cpu().numpy(), pbt.max_n_atoms
    )
    pbt.build_missing_leaf_atom_icoor_ann = PackedBlockTypesLeafAtomICoorAnnotation(
        **{
            field.name: torch.as_tensor(getattr(host, field.name), device=pbt.device)
            for field in attr.fields(type(host))
        }
    )


def _determine_leaf_atom_icoors_for_block_type(block_type, is_hydrogen):
    return _common._determine_leaf_atom_icoors_for_block_type(
        block_type,
        numpy.asarray(
            is_hydrogen.cpu() if isinstance(is_hydrogen, torch.Tensor) else is_hydrogen
        ),
    )


# Max heavy neighbors a parent can have while still carrying a single H
# (4-coordinate center: 3 heavy + 1 H).
_H_COMPLETION_MAX_NEIGH = _common._H_COMPLETION_MAX_NEIGH


@attr.s(auto_attribs=True, slots=True, frozen=True)
class PackedBlockTypesHCompletionAnnotation:
    eligible: Tensor[torch.bool][:, :]
    parent: Tensor[torch.int64][:, :]
    neigh: Tensor[torch.int64][:, :, 3]
    n_neigh: Tensor[torch.int64][:, :]
    dist: Tensor[torch.float32][:, :]
    par_conn: Tensor[torch.int64][:, :, 3]
    n_par_conn: Tensor[torch.int64][:, :]


def _determine_h_completion_for_block_type(block_type, is_hydrogen):
    return _common._determine_h_completion_for_block_type(
        block_type,
        numpy.asarray(
            is_hydrogen.cpu() if isinstance(is_hydrogen, torch.Tensor) else is_hydrogen
        ),
    )


def _annotate_packed_block_types_w_h_completion(pbt: PackedBlockTypes):
    if hasattr(pbt, "h_completion_ann"):
        return
    host = _common.prepare_h_completion_metadata(
        pbt.active_block_types, pbt.atom_is_hydrogen.cpu().numpy(), pbt.max_n_atoms
    )
    pbt.has_h_geometric_completion = bool(host.eligible.any())
    pbt.h_completion_ann = PackedBlockTypesHCompletionAnnotation(
        **{
            field.name: torch.as_tensor(getattr(host, field.name), device=pbt.device)
            for field in attr.fields(type(host))
        }
    )


def _apply_h_geometric_completion(
    pbt,
    pose_coords,
    block_leaf_atom_is_missing,
    block_coord_offset,
    block_types,
    inter_residue_connections,
):
    """Place eligible rebuilt H opposite the mean direction of the parent's heavy
    neighbors, replacing the conformer-specific dihedral placement.

    Only touches H flagged eligible by _determine_h_completion_for_block_type
    that were actually rebuilt.
    """
    from ._hydrogen_completion import complete_hydrogens

    if not getattr(pbt, "has_h_geometric_completion", True):
        return pose_coords
    if pose_coords.device.type == "cpu" and not block_leaf_atom_is_missing.any():
        return pose_coords
    ann = pbt.h_completion_ann
    inputs = [
        pose_coords,
        block_types,
        block_coord_offset,
        inter_residue_connections,
        pbt.conn_atom,
        block_leaf_atom_is_missing,
        ann.eligible,
        ann.parent,
        ann.neigh,
        ann.n_neigh,
        ann.dist.to(pose_coords.dtype),
        ann.par_conn,
        ann.n_par_conn,
    ]
    if pose_coords.device.type == "mps":
        inputs = [x.cpu() for x in inputs]
    return complete_hydrogens(*inputs).to(pose_coords.device)
