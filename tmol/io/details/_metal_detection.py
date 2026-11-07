"""Torch buffers and packed-type growth around shared metal preparation."""

import torch

from tmol_core.io import metal_detection as _common
from tmol_core.chemical import _metal_table
from tmol.pose._packed_block_types import PackedBlockTypes


def __getattr__(name):
    return getattr(_common, name)


def _find_metal_geometries(
    canonical_ordering,
    chemical_db,
    res_types,
    coords,
    table=None,
    geometries=None,
    tolerance=0.55,
    excluded_donor_residues=None,
    declared_sites=None,
    find_additional=True,
    required_donors=None,
    require_all_donors=False,
):
    table = _metal_table() if table is None else table
    tables = _common._build_canonical_metal_tables(
        canonical_ordering, chemical_db, table
    )
    if (
        not tables.ion_for_class
        and not tables.cluster_for_class
        and not (declared_sites or required_donors or geometries)
    ):
        return torch.zeros_like(res_types, dtype=torch.int32), []
    variants, assignments = _common.find_metal_geometries(
        canonical_ordering,
        chemical_db,
        res_types.cpu().numpy(),
        lambda: coords.detach().cpu().numpy(),
        table,
        geometries,
        tolerance,
        (
            None
            if excluded_donor_residues is None
            else excluded_donor_residues.cpu().numpy()
        ),
        declared_sites,
        find_additional,
        required_donors,
        require_all_donors,
    )
    return torch.from_numpy(variants).to(res_types.device), assignments


def _donor_patches(canonical_ordering, chemical_db, res_types, assignments):
    if not assignments:
        return ()
    return _common._donor_patches(
        canonical_ordering, chemical_db, res_types.cpu().numpy(), assignments
    )


def _with_donor_patches(pbt: PackedBlockTypes, patches) -> PackedBlockTypes:
    """The newest packed block types grown from ``pbt`` carrying every patch; each
    generation's residues are a leading run of the next's.
    """
    if not patches:
        return pbt
    root = getattr(pbt, "_donor_patch_root", pbt)
    newest = getattr(root, "_donor_patch_newest", root)
    known = {v.name for v in newest.chem_db.variants}
    extra = tuple(p for p in patches if p.name not in known)
    if extra:
        chem_db = newest.chem_db.with_variants_applied(extra)
        rts = newest.restype_set.extended(chem_db)
        newest = PackedBlockTypes.from_restype_list(
            chem_db, rts, rts.residue_types, pbt.device
        )
        setattr(newest, "_donor_patch_root", root)
        setattr(root, "_donor_patch_newest", newest)
    return newest


def _translated(metal, offset, wide):
    if wide:
        # MPS does not support float64; the fixed-fit translation remains
        # differentiable through the CPU conversion and return to its device.
        source = metal.cpu() if metal.device.type == "mps" else metal
        return (
            source.double()
            + torch.as_tensor(offset, dtype=torch.float64, device=source.device)
        ).to(device=metal.device, dtype=metal.dtype)
    return metal + metal.new_tensor(offset)


def _place_cluster_virtuals(bt, coords, missing, rotations):
    for atom, metal, offset, wide in _common.cluster_virtual_placements(
        bt, missing.cpu().numpy(), rotations
    ):
        coords[atom] = _translated(coords[metal], offset, wide)
        missing[atom] = False


def _place_site_virtuals(pbt, block_types64, block_coords, missing_atoms, assignments):
    if not assignments:
        return block_coords, missing_atoms
    for pose, res, atom, metal, offset, wide in _common.metal_virtual_placements(
        pbt.active_block_types,
        block_types64.cpu().numpy(),
        missing_atoms.cpu().numpy(),
        assignments,
    ):
        block_coords[pose, res, atom] = _translated(
            block_coords[pose, res, metal], offset, wide
        )
        missing_atoms[pose, res, atom] = False
    return block_coords, missing_atoms
