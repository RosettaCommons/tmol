"""Add, remove or redetect metal-donor bonds on an existing pose stack."""

from typing import Optional

import attr
import numpy
import torch

from tmol.database.chemical import _site_connections, _site_metal
from tmol.io._canonical_ordering import CanonicalOrdering
from tmol.io._pose_stack_construction import _pose_stack_from_canonical_form
from tmol.io._pose_stack_deconstruction import _canonical_form_from_pose_stack
from tmol.io.details._metal_geometry import _unit
from tmol.pose._pose_stack import PoseStack
from tmol.types._torch import Tensor


def _add_metal_coordination(
    canonical_ordering: CanonicalOrdering,
    pose_stack: PoseStack,
    pose: int,
    metal: int,
    donor: int,
    atom: str,
    site: Optional[int] = None,
) -> PoseStack:
    """Bond ``atom`` of ``donor`` to an open site of ``metal`` (by default the one whose
    virtual faces it best); the donor atom loses its hydrogens.
    """
    # Pass the caller's chain ids through: deduced ones would split a pose whose
    # blocks are a non-contiguous selection of one chain.
    cf = _canonical_form_from_pose_stack(
        canonical_ordering, pose_stack, chain_id=pose_stack.chain_id
    )
    rows = _rows(cf.metal_coordination)
    metal_bt = _block_type(pose_stack, pose, metal)
    if not metal_bt.metal_sites:
        raise ValueError(f"residue {metal} ({metal_bt.name}) is not a metal")
    filled = {site for p, m, site, _, _ in rows if (p, m) == (pose, metal)}
    open_sites = [k for k in range(len(_site_connections(metal_bt))) if k not in filled]
    if site is None:
        if not open_sites:
            raise ValueError(f"{metal_bt.name} at residue {metal} has no open site")
        site = _closest_open_site(pose_stack, pose, metal, donor, atom, open_sites)
    elif site not in open_sites:
        raise ValueError(f"site {site} of {metal_bt.name} is not open")

    donor_bt = _block_type(pose_stack, pose, donor)
    index = canonical_ordering.restypes_atom_index_mapping[donor_bt.io_equiv_class]
    rows.append((pose, metal, site, donor, index[atom]))
    is_h = pose_stack.packed_block_types.atom_is_hydrogen[
        int(pose_stack.block_type_ind64[pose, donor])
    ]
    bonded = donor_bt.bond_indices
    donor_atom = donor_bt.atom_to_idx[atom]
    hydrogens = [
        int(b)
        for a, b in bonded.tolist()
        if a == donor_atom and bool(is_h[b]) and donor_bt.atoms[b].name in index
    ]
    if hydrogens:
        coords = cf.coords.clone()
        for h in hydrogens:
            coords[pose, donor, index[donor_bt.atoms[h].name]] = float("nan")
        cf = attr.evolve(cf, coords=coords)
    return _rebuild(canonical_ordering, pose_stack, cf, rows)


def _remove_metal_coordination(
    canonical_ordering: CanonicalOrdering,
    pose_stack: PoseStack,
    pose: int,
    metal: int,
    site: int,
) -> PoseStack:
    """Open one filled site of a metal; the donor keeps its hydrogens."""
    cf = _canonical_form_from_pose_stack(
        canonical_ordering, pose_stack, chain_id=pose_stack.chain_id
    )
    rows = _rows(cf.metal_coordination)
    kept = [r for r in rows if r[:3] != (pose, metal, site)]
    if len(kept) == len(rows):
        raise ValueError(f"site {site} of residue {metal} is not filled")
    return _rebuild(canonical_ordering, pose_stack, cf, kept)


def _rows(metal_coordination):
    if metal_coordination is None:
        return []
    return [tuple(r) for r in metal_coordination.tolist()]


def _block_type(pose_stack, pose, res):
    bt_ind = int(pose_stack.block_type_ind64[pose, res])
    if bt_ind < 0:
        raise ValueError(f"pose {pose} has no residue {res}")
    return pose_stack.packed_block_types.active_block_types[bt_ind]


def _closest_open_site(pose_stack, pose, metal, donor, atom, open_sites):
    metal_bt = _block_type(pose_stack, pose, metal)
    if not any(site.site_virts for site in metal_bt.metal_sites):
        return open_sites[0]
    xyz = pose_stack.coords[pose].detach().cpu().numpy().astype(numpy.float64)
    offset = int(pose_stack.block_coord_offset64[pose, metal])
    donor_bt = _block_type(pose_stack, pose, donor)
    donor_offset = int(pose_stack.block_coord_offset64[pose, donor])
    donor_xyz = xyz[donor_offset + donor_bt.atom_to_idx[atom]]

    def alignment(k):
        site, local = _site_metal(metal_bt, k)
        metal_xyz = xyz[offset + metal_bt.atom_to_idx[site.metal_atom]]
        ray = xyz[offset + metal_bt.atom_to_idx[site.site_virts[local]]] - metal_xyz
        return _unit(ray) @ _unit(donor_xyz - metal_xyz)

    return max(open_sites, key=alignment)


def _rebuild(canonical_ordering, pose_stack, cf, rows):
    device = pose_stack.device
    coordination = (
        torch.tensor(rows, dtype=torch.int64, device=device).view(-1, 5)
        if rows
        else None
    )
    cf = attr.evolve(cf, metal_coordination=coordination)
    return _pose_stack_from_canonical_form(
        canonical_ordering,
        pose_stack.packed_block_types,
        *cf,
        ligand_hydrogens="input",
        find_additional_disulfides=False,
        find_additional_cyclic_closures=False,
        find_additional_metal_coordination=False,
    )


def _site_virtual_indices(pose_stack, canonical_ordering, pose, metal):
    """Canonical indices of a metal's site virtuals, which its fitted geometry orients."""
    metal_bt = _block_type(pose_stack, pose, metal)
    index = canonical_ordering.restypes_atom_index_mapping[metal_bt.io_equiv_class]
    return [
        index[name]
        for site in metal_bt.metal_sites
        for name in site.site_virts
        if name in index
    ]


def _declared_geometries(pose_stack: PoseStack) -> dict:
    """{(pose, block): geometry name} for every metal block, from its own type."""
    pbt = pose_stack.packed_block_types
    geometry_of = {
        index: bt.metal_sites[0].geometry
        for index, bt in enumerate(pbt.active_block_types)
        if bt.metal_sites
    }
    if not geometry_of:
        return {}
    present = pose_stack.block_type_ind64.cpu().numpy()
    return {
        (int(pose), int(block)): geometry_of[int(present[pose, block])]
        for pose, block in zip(*numpy.nonzero(numpy.isin(present, list(geometry_of))))
    }


def _holds_a_metal(pose_stack: PoseStack) -> bool:
    """Whether any block of the stack is a metal with sites to fill."""
    pbt = pose_stack.packed_block_types
    with_sites = torch.tensor(
        [bool(bt.metal_sites) for bt in pbt.active_block_types],
        dtype=torch.bool,
        device=pose_stack.device,
    )
    present = pose_stack.block_type_ind64
    return bool(with_sites[present[present >= 0]].any()) if with_sites.any() else False


def redetect_metal_coordination(
    pose_stack: PoseStack,
    *,
    canonical_ordering: Optional[CanonicalOrdering] = None,
    protonated_donors: Optional[Tensor[torch.bool][:, :]] = None,
    tolerance: float = 0.55,
    drop_existing=False,
    only_donor_blocks=None,
) -> PoseStack:
    """A pose stack whose metal sites are redeclared from its own coordinates.

    A donor that has moved into an open site fills it, which is what lets a
    docked ligand coordinate a receptor metal it was not built against. Sites
    already filled stay filled unless ``drop_existing``, which redeclares them
    from geometry alone, so a donor that has moved away lets go. A search that
    moves a bound donor needs that: a site held open by declaration alone puts
    no bound on its harmonic. Pass a sequence of block indices to let go of
    only those donors' bonds, which leaves a rigid receptor's own coordination
    exactly as it was built. ``only_donor_blocks`` keeps every other block from
    taking a site it does not already hold, which is what lets a search move one
    ligand without re-fitting the receptor around its metals. ``protonated_donors`` marks the blocks whose
    donors may coordinate while still holding a hydrogen -- a ligand, typically,
    whose protonation the input fixed before it reached the metal.

    The metals keep the geometry they were built with: this fills open sites, it
    does not re-fit a site the structure already defines.
    """
    from tmol.database.chemical import GEOMETRY_NAMES, _metal_table
    from tmol.io.details._metal_detection import (
        _find_metal_geometries,
        _metal_connection_rows,
    )

    if not _holds_a_metal(pose_stack):
        return pose_stack
    if canonical_ordering is None:
        canonical_ordering = CanonicalOrdering.from_chemdb(
            pose_stack.packed_block_types.chem_db
        )
    dtype = pose_stack.coords.dtype
    if dtype != torch.float32:
        pose_stack = attr.evolve(pose_stack, coords=pose_stack.coords.to(torch.float32))
    cf = _canonical_form_from_pose_stack(
        canonical_ordering, pose_stack, chain_id=pose_stack.chain_id
    )
    held = _rows(cf.metal_coordination)
    required = {}
    droppable = (
        None
        if drop_existing is True
        else (
            set() if drop_existing is False else {int(block) for block in drop_existing}
        )
    )
    for pose, metal, _, donor, atom in held:
        if droppable is None or donor in droppable:
            continue
        required.setdefault((pose, metal), set()).add((donor, atom))
    # A metal keeps the geometry it was built with: redetection fills its open
    # sites, it does not re-fit the site the structure already defines. The
    # block type is what says which geometry that is, and a free fit choosing a
    # larger one would hand back site indices the type cannot hold.
    geometries = _declared_geometries(pose_stack)
    # The rebuild reads the geometries back out of the canonical form, so state
    # them there too: one source, or a site index valid for one geometry lands
    # in another that cannot hold it.
    if geometries:
        cf = attr.evolve(
            cf,
            metal_sites=torch.tensor(
                [
                    [pose, metal, GEOMETRY_NAMES.index(name)]
                    for (pose, metal), name in sorted(geometries.items())
                ],
                dtype=torch.int64,
                device=cf.coords.device,
            ),
        )

    excluded = None
    if only_donor_blocks is not None:
        excluded = torch.ones(
            cf.res_types.shape, dtype=torch.bool, device=cf.res_types.device
        )
        for block in only_donor_blocks:
            excluded[:, int(block)] = False
    _, assignments = _find_metal_geometries(
        canonical_ordering,
        pose_stack.packed_block_types.chem_db,
        cf.res_types,
        cf.coords,
        geometries=geometries,
        tolerance=tolerance,
        excluded_donor_residues=excluded,
        required_donors=required,
        protonated_donors=(
            None if protonated_donors is None else protonated_donors.cpu().numpy()
        ),
    )
    rows = _metal_connection_rows(assignments)
    # A metal takes no more donors than it has sites: where the donors already
    # declared and the ones newly in reach exceed them, the fit numbers the
    # excess past the end of the geometry, and those do not coordinate.
    sites_in = {
        name: len(geometry["vertices"])
        for geometry in _metal_table()["geometries"]
        for name in (geometry["name"],)
    }
    rows = [
        row
        for row in rows
        if row[2] < sites_in.get(geometries.get((row[0], row[1]), ""), 0)
    ]
    if sorted(rows) == sorted(held):
        return (
            pose_stack
            if dtype == torch.float32
            else attr.evolve(pose_stack, coords=pose_stack.coords.to(dtype))
        )

    coords = cf.coords.clone()
    # Site virtuals carry the orientation of the geometry they were built under;
    # drop them so the rebuild re-places them against the donors found here.
    for pose, metal in {(pose, metal) for pose, metal, _, _, _ in rows}:
        for virtual in _site_virtual_indices(
            pose_stack, canonical_ordering, pose, metal
        ):
            coords[pose, metal, virtual] = float("nan")
    cf = attr.evolve(cf, coords=coords)
    out = _rebuild(canonical_ordering, pose_stack, cf, rows)
    if dtype != torch.float32:
        out = attr.evolve(out, coords=out.coords.to(dtype))
    return out


def _without_metal_coordination(
    pose_stack: PoseStack,
    blocks,
    canonical_ordering: Optional[CanonicalOrdering] = None,
) -> PoseStack:
    """A pose stack with every metal bond to ``blocks`` opened, in one rebuild.

    The unbound state of a ligand that coordinates a receptor metal, and the
    free ligand that grid docking starts from, are the same structure: one whose
    donor has let go.
    """
    if not _holds_a_metal(pose_stack):
        return pose_stack
    if canonical_ordering is None:
        canonical_ordering = CanonicalOrdering.from_chemdb(
            pose_stack.packed_block_types.chem_db
        )
    dtype = pose_stack.coords.dtype
    if dtype != torch.float32:
        pose_stack = attr.evolve(pose_stack, coords=pose_stack.coords.to(torch.float32))
    cf = _canonical_form_from_pose_stack(
        canonical_ordering, pose_stack, chain_id=pose_stack.chain_id
    )
    rows = _rows(cf.metal_coordination)
    kept = [row for row in rows if row[3] not in set(blocks)]
    if len(kept) == len(rows):
        out = pose_stack
    else:
        out = _rebuild(canonical_ordering, pose_stack, cf, kept)
    if dtype != torch.float32:
        out = attr.evolve(out, coords=out.coords.to(dtype))
    return out
