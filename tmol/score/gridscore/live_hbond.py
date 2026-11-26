"""Live ligand--receptor hydrogen bonds for rigid-receptor grid scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Sequence

import attr
import torch

from tmol.database import ParameterDatabase
from tmol.pose._pose_stack import PoseStack
from tmol.score.hbond._hbond_energy_term import HBondEnergyTerm

from ._common import (
    LiveData,
    build_cell_list,
    cell_grid,
    derived_atom_indices,
    validate_ligand_coords,
)
from ._movable import movable_atoms as movable_atoms_of


@dataclass(frozen=True)
class LiveHBondData(LiveData):
    """One shared rigid-receptor hbond structure and one ligand topology.

    Receptor coordinates and cell lists occur once, independently of the
    number of ligand poses. Ligand tensors contain only block-local atom
    indices, so every pose supplies its own current base geometry.
    """

    _float64_fields: ClassVar[frozenset] = frozenset({"pair_polynomials"})

    n_ligand_atoms: int
    receptor_donors: torch.Tensor
    receptor_donor_types: torch.Tensor
    receptor_acceptors: torch.Tensor
    receptor_acceptor_types: torch.Tensor
    ligand_donor_atoms: torch.Tensor
    ligand_donor_types: torch.Tensor
    ligand_acceptor_atoms: torch.Tensor
    ligand_acceptor_types: torch.Tensor
    cell_origin: torch.Tensor
    cell_dims: torch.Tensor
    receptor_donor_cell_offsets: torch.Tensor
    receptor_donor_cell_indices: torch.Tensor
    receptor_acceptor_cell_offsets: torch.Tensor
    receptor_acceptor_cell_indices: torch.Tensor
    pair_params: torch.Tensor
    pair_polynomials: torch.Tensor
    global_params: torch.Tensor
    #: Ligand site rows per block type the search can give the ligand, in
    #: movable-slot indices and padded to one width; empty for one molecule.
    variant_sites: dict = field(default_factory=dict, compare=False, repr=False)


def _padded_variant_tables(base, variant_sites, width):
    """Full ligand+sidechain site rows per block type, padded to one width.

    ``base`` holds the reference rows; a site whose heavy atom lies in the
    ligand's slots is replaced by the form's own, the rest (flexible sidechain
    sites) are shared. Padding rows carry type -1, which the kernel skips.
    """
    donor_atoms, donor_types, acceptor_atoms, acceptor_types = base
    keep_d = donor_atoms[:, 0] >= width
    keep_a = acceptor_atoms[:, 0] >= width
    tables = {}
    for index, (v_da, v_dt, v_aa, v_at) in variant_sites.items():
        tables[index] = (
            torch.cat((v_da, donor_atoms[keep_d])),
            torch.cat((v_dt, donor_types[keep_d])),
            torch.cat((v_aa, acceptor_atoms[keep_a])),
            torch.cat((v_at, acceptor_types[keep_a])),
        )
    n_donor = max(t[0].shape[0] for t in tables.values())
    n_acceptor = max(t[2].shape[0] for t in tables.values())

    def pad(atoms, types, rows, arity):
        if atoms.shape[0] == rows:
            return atoms, types
        extra = rows - atoms.shape[0]
        return (
            torch.cat((atoms, atoms.new_zeros((extra, arity)))),
            torch.cat((types, types.new_full((extra,), -1))),
        )

    for index, (da, dt, aa, at) in tables.items():
        da, dt = pad(da, dt, n_donor, 2)
        aa, at = pad(aa, at, n_acceptor, 3)
        tables[index] = (da, dt, aa, at)
    return tables


def _hbond_sites(pose_stack: PoseStack, derived: torch.Tensor):
    """Every donor ``(heavy, H, type)`` and acceptor ``(A, base, base0, type)``.

    Rows hold global atom indices in block, tile, site order.
    """

    params = pose_stack.packed_block_types.hbpbt_params
    block_types = pose_stack.block_type_ind64[0]
    blocks = torch.nonzero(block_types >= 0).squeeze(1)
    types = block_types[blocks]
    offsets = pose_stack.block_coord_offset64[0, blocks]
    device = block_types.device

    def sites(n_per_tile, atom_inds, site_types):
        n_tiles, width = atom_inds.shape[1:]
        tiles = torch.arange(n_tiles, device=device) * HBondEnergyTerm.tile_size
        chosen = torch.arange(width, device=device) < n_per_tile[types].unsqueeze(-1)
        atoms = offsets[:, None, None] + tiles[None, :, None] + atom_inds[types]
        return atoms[chosen].to(torch.int64), site_types[types][chosen].to(torch.int64)

    hydrogen, donor_type = sites(
        params.tile_n_donH, params.tile_donH_inds, params.tile_donorH_type
    )
    acceptor, acceptor_type = sites(
        params.tile_n_acc, params.tile_acc_inds, params.tile_acceptor_type
    )
    donor = derived[hydrogen, 0]
    base, base0 = derived[acceptor, 1], derived[acceptor, 2]
    if bool((donor < 0).any() | (base < 0).any() | (base0 < 0).any()):
        raise RuntimeError("hbond base generation returned no base atom")
    return (
        (donor, hydrogen, donor_type),
        (acceptor, base, base0, acceptor_type),
    )


def ligand_variant_sites(
    pose_stack: PoseStack,
    ligand_block: int,
    param_db: ParameterDatabase,
    atom_maps: dict,
) -> dict:
    """Each bound form's own ligand hbond sites, in movable-slot indices.

    Binding as a conjugate base takes a donor's hydrogen away: that donor site
    goes, and the acceptor left behind has different parameters and a different
    second base atom. Those follow from the block type, so each form's sites are
    read off a one-block pose of its own and mapped back onto the shared slots.

    ``atom_maps`` is block type index -> ``[slot] -> that form's local atom``,
    -1 where the form has no atom for a slot.
    """
    from tmol.pose import select_blocks

    one = select_blocks(pose_stack, [ligand_block])
    width = one.max_n_pose_atoms
    sites = {}
    for index, atom_map in atom_maps.items():
        local = atom_map.to(device=one.device, dtype=torch.int64)
        inverse = torch.full((width,), -1, dtype=torch.int64, device=one.device)
        present = local >= 0
        inverse[local[present]] = torch.nonzero(present).squeeze(1)
        coords = one.coords.clone()
        coords[0, local[present]] = one.coords[0, torch.nonzero(present).squeeze(1)]
        types = torch.full_like(one.block_type_ind64, index)
        variant = attr.evolve(
            one,
            coords=coords,
            block_type_ind=types.to(torch.int32),
            block_type_ind64=types,
        )
        term = HBondEnergyTerm(param_db, variant.device)
        for block_type in variant.packed_block_types.active_block_types:
            term.setup_block_type(block_type)
        term.setup_packed_block_types(variant.packed_block_types)
        donors, acceptors = _hbond_sites(variant, derived_atom_indices(variant))
        donor, hydrogen, donor_type = donors
        acceptor, base, base0, acceptor_type = acceptors
        sites[index] = (
            torch.stack((inverse[donor], inverse[hydrogen]), 1),
            donor_type.to(torch.int64),
            torch.stack((inverse[acceptor], inverse[base], inverse[base0]), 1),
            acceptor_type.to(torch.int64),
        )
    return sites


def build_live_hbond_data(
    pose_stack: PoseStack,
    ligand_block: int,
    param_db: ParameterDatabase,
    *,
    sidechain_blocks: tuple[int, ...] = (),
    movable_atoms: Sequence[int] | None = None,
    extra_blocks: tuple[int, ...] = (),
    ligand_atom_maps: dict | None = None,
) -> LiveHBondData:
    """Build movable-atom topology and shared fixed-receptor hbond cells.

    ``movable_atoms`` are the compact-pose atoms that move (all ligand atoms by
    default; required with ``sidechain_blocks``).
    """

    if pose_stack.n_poses != 1:
        raise ValueError("live hbond construction requires one pose")
    if sidechain_blocks and movable_atoms is None:
        raise ValueError("sidechain_blocks require the compact movable_atoms")
    device = pose_stack.device
    moving = movable_atoms_of(pose_stack, ligand_block, sidechain_blocks, extra_blocks)
    atom_map = moving.atom_map
    is_query = torch.ones(moving.n_atoms, dtype=torch.bool, device=device)
    if movable_atoms is not None:
        is_query = torch.zeros(moving.n_atoms, dtype=torch.bool, device=device)
        chosen = torch.as_tensor(list(movable_atoms), dtype=torch.int64, device=device)
        is_query[chosen] = True

    term = HBondEnergyTerm(param_db, device)
    for block_type in pose_stack.packed_block_types.active_block_types:
        term.setup_block_type(block_type)
    term.setup_packed_block_types(pose_stack.packed_block_types)
    donors, acceptors = _hbond_sites(pose_stack, derived_atom_indices(pose_stack))
    donor, hydrogen, donor_type = donors
    acceptor, base, base0, acceptor_type = acceptors

    def compact(atoms):
        return atom_map[atoms]

    # A site of a moving block is a ligand site when its heavy atom moves;
    # every other site belongs to the fixed receptor.
    donor_fixed = compact(hydrogen) < 0
    donor_mine = (compact(donor) >= 0) & is_query[compact(donor).clamp(min=0)]
    acceptor_fixed = compact(acceptor) < 0
    acceptor_mine = (compact(acceptor) >= 0) & is_query[compact(acceptor).clamp(min=0)]
    if bool((compact(donor)[donor_fixed] >= 0).any()) or bool(
        (
            (compact(base)[acceptor_fixed] >= 0) | (compact(base0)[acceptor_fixed] >= 0)
        ).any()
    ):
        raise ValueError("receptor hbond base points into the moving blocks")
    if bool((compact(hydrogen)[donor_mine] < 0).any()) or bool(
        ((compact(base)[acceptor_mine] < 0) | (compact(base0)[acceptor_mine] < 0)).any()
    ):
        raise ValueError("movable hbond geometry crosses a block boundary")

    coords = pose_stack.coords.reshape(-1, 3).detach().to(torch.float64)
    receptor_donors = coords[torch.stack((donor, hydrogen), 1)[donor_fixed]]
    receptor_acceptors = coords[torch.stack((acceptor, base, base0), 1)[acceptor_fixed]]

    pair_params, pair_polynomials, global_params = term._param_tables(torch.float64)
    max_ha_dis = float(global_params[0, 5])
    if not (max_ha_dis > 0):
        raise RuntimeError("hbond max H--A distance must be positive")

    anchors = torch.cat((receptor_donors[:, 1], receptor_acceptors[:, 0]))
    cell_origin, cell_dims = cell_grid(anchors, max_ha_dis)
    donor_offsets, donor_cell_indices = build_cell_list(
        receptor_donors[:, 1], cell_origin, cell_dims, max_ha_dis
    )
    acceptor_offsets, acceptor_cell_indices = build_cell_list(
        receptor_acceptors[:, 0], cell_origin, cell_dims, max_ha_dis
    )
    base_tables = (
        torch.stack((compact(donor), compact(hydrogen)), 1)[donor_mine],
        donor_type[donor_mine].to(torch.int64),
        torch.stack((compact(acceptor), compact(base), compact(base0)), 1)[
            acceptor_mine
        ],
        acceptor_type[acceptor_mine].to(torch.int64),
    )
    variant_tables = {}
    if ligand_atom_maps:
        variant_tables = _padded_variant_tables(
            base_tables,
            ligand_variant_sites(pose_stack, ligand_block, param_db, ligand_atom_maps),
            pose_stack.block_type(0, ligand_block).n_atoms,
        )
    return LiveHBondData(
        n_ligand_atoms=moving.n_atoms,
        receptor_donors=receptor_donors,
        receptor_donor_types=donor_type[donor_fixed],
        receptor_acceptors=receptor_acceptors,
        receptor_acceptor_types=acceptor_type[acceptor_fixed],
        # One table row while every pose holds the same ligand; detection
        # swaps in a row per pose once the search can bind a metal.
        ligand_donor_atoms=base_tables[0][None].contiguous(),
        ligand_donor_types=base_tables[1][None].contiguous(),
        ligand_acceptor_atoms=base_tables[2][None].contiguous(),
        ligand_acceptor_types=base_tables[3][None].contiguous(),
        variant_sites=variant_tables,
        cell_origin=cell_origin.to(torch.float64).contiguous(),
        cell_dims=cell_dims.contiguous(),
        receptor_donor_cell_offsets=donor_offsets,
        receptor_donor_cell_indices=donor_cell_indices,
        receptor_acceptor_cell_offsets=acceptor_offsets,
        receptor_acceptor_cell_indices=acceptor_cell_indices,
        pair_params=pair_params.detach().clone().contiguous(),
        pair_polynomials=pair_polynomials.detach().clone().contiguous(),
        global_params=global_params.detach().clone().contiguous(),
    )


def set_hbond_pose_variants(data: LiveHBondData, block_types) -> None:
    """Point the kernel's ligand tables at each pose's own bound form."""
    if not data.variant_sites:
        return
    unknown = sorted(set(int(b) for b in block_types) - set(data.variant_sites))
    if unknown:
        raise ValueError(f"ligand block types {unknown} have no hbond sites")
    rows = [data.variant_sites[int(b)] for b in block_types]
    for position, name in enumerate(
        (
            "ligand_donor_atoms",
            "ligand_donor_types",
            "ligand_acceptor_atoms",
            "ligand_acceptor_types",
        )
    ):
        current = getattr(data, name)
        # Rows are built once on the reference data; each scorer holds its own
        # device and dtype copy of the live tables.
        stacked = (
            torch.stack([row[position] for row in rows])
            .to(device=current.device, dtype=current.dtype)
            .contiguous()
        )
        if current.shape == stacked.shape:
            current.copy_(stacked)
        else:
            object.__setattr__(data, name, stacked)


def _validate_ligand_coords(data: LiveHBondData, ligand_coords: torch.Tensor) -> None:
    validate_ligand_coords(
        ligand_coords, data.n_ligand_atoms, data.receptor_donors, "hbond"
    )


def _live_hbond_op(
    data: LiveHBondData, ligand_coords: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    from .potentials.compiled import live_hbond

    return live_hbond(
        ligand_coords.contiguous(),
        data.ligand_donor_atoms,
        data.ligand_donor_types,
        data.ligand_acceptor_atoms,
        data.ligand_acceptor_types,
        data.receptor_donors,
        data.receptor_donor_types,
        data.receptor_acceptors,
        data.receptor_acceptor_types,
        data.cell_origin,
        data.cell_dims,
        data.receptor_donor_cell_offsets,
        data.receptor_donor_cell_indices,
        data.receptor_acceptor_cell_offsets,
        data.receptor_acceptor_cell_indices,
        data.pair_params,
        data.pair_polynomials,
        data.global_params,
    )


class _LiveHBondAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ligand_coords: torch.Tensor, data: LiveHBondData):
        energies, gradients = _live_hbond_op(data, ligand_coords)
        ctx.save_for_backward(gradients)
        return energies

    @staticmethod
    def backward(ctx, grad_energies):
        (gradients,) = ctx.saved_tensors
        return gradients * grad_energies[:, None, None], None


def score_ligand_hbond(
    data: LiveHBondData, ligand_coords: torch.Tensor
) -> torch.Tensor:
    """Return unweighted live interface hbond energy for every ligand pose."""

    _validate_ligand_coords(data, ligand_coords)
    if ligand_coords.requires_grad:
        return _LiveHBondAutograd.apply(ligand_coords, data)
    return _live_hbond_op(data, ligand_coords)[0]


def score_ligand_hbond_with_gradients(
    data: LiveHBondData, ligand_coords: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unweighted live hbond energies and analytic ligand gradients."""

    _validate_ligand_coords(data, ligand_coords)
    with torch.no_grad():
        return _live_hbond_op(data, ligand_coords.detach())
