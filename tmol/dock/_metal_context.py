"""Receptor metals a docked ligand could reach without leaving the search box."""

import torch

from tmol.database.chemical import _metal_table
from tmol.pose._pose_stack import PoseStack
from tmol.score.gridscore.grid import compute_grid_geometry


def _max_metal_donor_distance() -> float:
    """The longest ideal metal-donor bond any ion in the table forms."""
    return max(
        distance
        for ion in _metal_table()["ions"]
        for distance in ion.get("distances", {}).values()
    )


def _metal_blocks(pose_stack: PoseStack) -> list[int]:
    """Blocks whose type declares metal coordination sites."""
    pbt = pose_stack.packed_block_types
    blocks = []
    for block in range(pose_stack.max_n_blocks):
        index = int(pose_stack.block_type_ind64[0, block])
        if index >= 0 and pbt.active_block_types[index].metal_sites:
            blocks.append(block)
    return blocks


def _has_open_site(pose_stack: PoseStack, block: int, ligand_block: int) -> bool:
    """True if any of this metal's site connections is open to the ligand.

    A site the ligand already fills counts as open: the search is free to let
    go of it, and the reference may or may not have been built coordinated.
    """
    index = int(pose_stack.block_type_ind64[0, block])
    block_type = pose_stack.packed_block_types.active_block_types[index]
    n_conn = len(block_type.connections)
    if n_conn == 0:
        return False
    partners = pose_stack.inter_residue_connections64[:, block, :n_conn, 0]
    filled = (partners >= 0) & (partners != ligand_block)
    return bool((~filled).any())


def _bridging_metal_pairs(pose_stack: PoseStack, metal_blocks) -> bool:
    """Whether two of these metals sit close enough to share one donor.

    Twice the longest ideal metal-donor bond bounds the metal-metal separation
    a bridging donor can span.
    """
    reach = 2.0 * _max_metal_donor_distance()
    centers = []
    for block in metal_blocks:
        index = int(pose_stack.block_type_ind64[0, block])
        block_type = pose_stack.packed_block_types.active_block_types[index]
        start = int(pose_stack.block_coord_offset64[0, block])
        for site in block_type.metal_sites:
            centers.append(
                pose_stack.coords[0, start + block_type.atom_to_idx[site.metal_atom]]
            )
    for i, first in enumerate(centers):
        for second in centers[i + 1 :]:
            if float((first - second).norm()) <= reach:
                return True
    return False


def ligand_donor_patches(pose_stack: PoseStack, ligand_block: int, chemical_db):
    """Every donor patch the search can give this ligand, enumerated up front.

    A donor is any ligand atom whose type is a metal donor, the same predicate
    detection applies. Each takes one connection per metal it binds; two only
    where the pocket holds a metal pair close enough to share it, since the
    receptor's metals do not move. Applying these to the packed block types
    builds every reachable bound form at once, so detection never grows them
    mid-search and a captured graph stays valid.
    """
    from tmol.io.details._metal_detection import _metal_donor_patch
    from tmol.ligand._metal_deprotonation import (
        _deprotonating_donors,
        _metal_bound_type,
    )

    atom_types = chemical_db.atom_types
    by_name = {t.name: t for t in atom_types}
    index = int(pose_stack.block_type_ind64[0, ligand_block])
    block_type = pose_stack.packed_block_types.active_block_types[index]
    leaving = _deprotonating_donors(block_type, atom_types)
    metal_blocks = [b for b in _metal_blocks(pose_stack) if b != ligand_block]
    max_metals = 2 if _bridging_metal_pairs(pose_stack, metal_blocks) else 1
    patches = []
    for atom in block_type.atoms[: block_type.n_atoms]:
        donor = by_name.get(atom.atom_type)
        if donor is None or not donor.is_metal_donor:
            continue
        hydrogen = leaving.get(atom.name)
        bound_type = (
            _metal_bound_type(block_type, atom_types, atom.name, hydrogen)
            if hydrogen is not None
            else None
        )
        for n_metals in range(1, max_metals + 1):
            patches.append(
                _metal_donor_patch(
                    block_type.base_name,
                    atom.name,
                    n_metals,
                    leaving=hydrogen,
                    bound_type=bound_type,
                )
            )
    return tuple(patches)


def ligand_variant_atom_maps(pose_stack: PoseStack, ligand_block: int) -> dict:
    """``block type index -> [slot] -> that form's local atom``, -1 if absent.

    Slots are the unbound ligand's atoms, which every bound form is a subset of:
    binding only takes a hydrogen away. Forms are matched by atom name.
    """
    pbt = pose_stack.packed_block_types
    index = int(pose_stack.block_type_ind64[0, ligand_block])
    reference = pbt.active_block_types[index]
    slots = [atom.name for atom in reference.atoms[: reference.n_atoms]]
    base = reference.name.split(":")[0]
    maps = {}
    for other, block_type in enumerate(pbt.active_block_types):
        if block_type.name.split(":")[0] != base:
            continue
        local = {
            atom.name: position
            for position, atom in enumerate(block_type.atoms[: block_type.n_atoms])
        }
        if set(local) - set(slots):
            continue
        maps[other] = torch.tensor(
            [local.get(name, -1) for name in slots],
            dtype=torch.int64,
            device=pose_stack.device,
        )
    return maps


def ligand_metal_atom_types(block_type, chemical_db) -> tuple[str, ...]:
    """Atom types the ligand's donors take once they bind a metal.

    Binding deprotonates some donors, which retypes them, and a grid built
    without those types cannot score the bound form.
    """
    from tmol.ligand._metal_deprotonation import (
        _deprotonating_donors,
        _metal_bound_type,
    )

    atom_types = chemical_db.atom_types
    return tuple(
        sorted(
            {
                _metal_bound_type(block_type, atom_types, atom, hydrogen)
                for atom, hydrogen in _deprotonating_donors(
                    block_type, atom_types
                ).items()
            }
        )
    )


def reachable_metal_blocks(
    pose_stack: PoseStack,
    ligand_block: int,
    *,
    voxel_spacing: float,
    bbox_padding: float,
    tolerance: float = 0.55,
) -> tuple[int, ...]:
    """Metals with an open site close enough to the search box to be bound.

    The grid confines the ligand to its box, so a metal further from that box
    than one metal-donor bond can never be coordinated however the search moves
    the ligand; keeping it would only widen the compacted pose.
    """

    offset = int(pose_stack.block_coord_offset64[0, ligand_block])
    n_atoms = int(pose_stack.n_ats_per_block[0, ligand_block])
    geometry = compute_grid_geometry(
        pose_stack.coords[0, offset : offset + n_atoms].to(torch.float64),
        voxel_spacing,
        bbox_padding,
    )
    reach = _max_metal_donor_distance() + tolerance
    lower = geometry.origin - reach
    upper = (
        geometry.origin
        + geometry.origin.new_tensor(geometry.dims).sub(1) * geometry.voxel_spacing
        + reach
    )

    reachable = []
    for block in _metal_blocks(pose_stack):
        if not _has_open_site(pose_stack, block, ligand_block):
            continue
        index = int(pose_stack.block_type_ind64[0, block])
        block_type = pose_stack.packed_block_types.active_block_types[index]
        start = int(pose_stack.block_coord_offset64[0, block])
        metal_atoms = [
            start + block_type.atom_to_idx[site.metal_atom]
            for site in block_type.metal_sites
        ]
        xyz = pose_stack.coords[0, metal_atoms].to(torch.float64)
        if bool(((xyz >= lower) & (xyz <= upper)).all(dim=1).any()):
            reachable.append(block)
    return tuple(reachable)
