import numpy
import torch
import attr
import logging

from typing import Optional, Tuple
from tmol.types._torch import Tensor
from tmol.types._array import NDArray
from tmol.types._functional import _validate_args
from tmol.database.chemical import (
    NEUTRAL_TERMINUS_VAR_BASE,
)
from tmol.io._canonical_ordering import CanonicalOrdering
from tmol.pose._inter_block_bondsep import InterBlockBondsep
from tmol.pose._packed_block_types import (
    PackedBlockTypes,
    _annotate_packed_block_types_w_dslf_conn_inds,
)
from tmol.pose._pose_stack_builder import PoseStackBuilder

from tmol_core.io import canonical as _canonical
from tmol_core.io import connections as _connections
from tmol_core.io.canonical import (  # noqa: F401
    _map_term_to_int,
    _term_and_spcase_var_candidate_lists,
    _assign_var_inds_for_bt,
)

logger = logging.getLogger(__name__)

# Treatment of a generated ligand's input hydrogens: keep names and coordinates,
# keep names only, or keep neither.
LIGAND_HYDROGEN_MODES = ("input", "names", "rebuild")

# An input hydrogen is on its parent when within this distance of its ideal
# bond length (A).
_BONDED_H_TOLERANCE = 0.4


def _validate_ligand_hydrogens(mode: str):
    """Raise unless mode is one of LIGAND_HYDROGEN_MODES."""
    if mode not in LIGAND_HYDROGEN_MODES:
        raise ValueError(
            f"ligand_hydrogens must be one of {LIGAND_HYDROGEN_MODES}, not {mode!r}"
        )


@_validate_args
def _assign_block_types(
    canonical_ordering: CanonicalOrdering,
    packed_block_types: PackedBlockTypes,
    atom_is_present: Tensor[torch.bool][:, :, :],
    chain_id: Tensor[torch.int32][:, :],
    res_types: Tensor[torch.int32][:, :],
    res_type_variants: Tensor[torch.int32][:, :],
    found_disulfides64: Tensor[torch.int64][:, 3],
    res_not_connected: Optional[Tensor[torch.bool][:, :, 2]] = None,
    cyclic_closures64: Optional[Tensor[torch.int64][:, 3]] = None,
    covalent_bonds64: Optional[Tensor[torch.int64][:, 5]] = None,
    metal_connections: Optional[list] = None,
) -> Tuple[
    Tensor[torch.int64][:, :],
    Tensor[torch.int64][:, :, :, 2],
    InterBlockBondsep,
]:
    """Choose each residue's block type and wire every inter-residue connection.

    ``metal_connections`` are (pose, metal, site, donor, donor canonical atom):
    site is the index into the metal's site connections the donor fills.
    """
    pbt = packed_block_types
    _annotate_packed_block_types_w_canonical_res_order(canonical_ordering, pbt)
    _annotate_packed_block_types_w_dslf_conn_inds(pbt)

    device = pbt.device
    n_poses = chain_id.shape[0]
    max_n_res = chain_id.shape[1]

    res_types64 = res_types.to(torch.int64)
    res_type_variants64 = res_type_variants.to(torch.int64)
    is_real_res = res_types64 != -1
    nz_is_real_res = torch.nonzero(is_real_res)

    if res_not_connected is None:
        res_not_connected = torch.zeros(
            (n_poses, max_n_res, 2), dtype=torch.bool, device=device
        )

    explicit_polymer_connections = torch.zeros(
        (n_poses, max_n_res, 2), dtype=torch.bool, device=device
    )
    if covalent_bonds64 is not None and covalent_bonds64.numel():
        poses = covalent_bonds64[:, 0].repeat_interleave(2)
        residues = covalent_bonds64[:, (1, 3)].reshape(-1)
        atoms = covalent_bonds64[:, (2, 4)].reshape(-1)
        ports = pbt.canonical_ordering_annotation.co_polymer_connection_atoms[
            res_types64[poses, residues]
        ]
        endpoints, sides = torch.nonzero(atoms[:, None] == ports, as_tuple=True)
        explicit_polymer_connections[poses[endpoints], residues[endpoints], sides] = (
            True
        )

    (
        termini_variants,
        is_actual_first_chain_res,
        is_actual_last_chain_res,
        is_polymeric,
    ) = _determine_chain_ending_status(
        pbt,
        chain_id,
        res_types64,
        res_not_connected,
        is_real_res,
        cyclic_closures64,
        explicit_polymer_connections,
    )

    termini_variants = _termini_the_atoms_state(
        pbt,
        atom_is_present,
        res_types64,
        res_type_variants64,
        termini_variants,
        res_not_connected & is_polymeric[..., None],
        is_real_res,
    )
    from tmol.io.details.compiled._compiled import _canonical_variants

    variant_inputs = [
        res_types64,
        res_type_variants64,
        termini_variants,
        pbt.canonical_ordering_annotation.var_combo_is_real_candidate,
    ]
    if device.type == "mps":
        variant_inputs = [x.cpu() for x in variant_inputs]
    res_type_variants64 = _canonical_variants(
        variant_inputs, NEUTRAL_TERMINUS_VAR_BASE
    ).to(device)
    block_type_ind64 = _select_best_block_type_candidate(
        canonical_ordering,
        pbt,
        atom_is_present,
        is_real_res,
        nz_is_real_res,
        res_types64,
        termini_variants,
        res_type_variants64,
    )
    conjugation_conns = _apply_conjugated_variants(
        canonical_ordering, pbt, block_type_ind64, covalent_bonds64
    )
    conjugation_conns += _apply_metal_connections(
        canonical_ordering, pbt, block_type_ind64, metal_connections
    )

    inter_residue_connections64 = _wire_canonical_connections(
        pbt,
        block_type_ind64,
        is_polymeric,
        is_actual_first_chain_res,
        is_actual_last_chain_res,
        res_not_connected,
        found_disulfides64,
        cyclic_closures64,
        conjugation_conns,
    )

    inter_block_bondsep = PoseStackBuilder._inter_block_bondsep_from_connections(
        pbt, block_type_ind64, is_real_res, inter_residue_connections64
    )

    return (block_type_ind64, inter_residue_connections64, inter_block_bondsep)


def _wire_canonical_connections(
    pbt, types, polymeric, first, last, broken, disulfides=None, cycles=None, extra=()
):
    from tmol.io.details.compiled._compiled import _canonical_wiring

    empty = torch.empty((0, 3), dtype=types.dtype, device=types.device)
    inputs = [
        types,
        polymeric,
        first,
        last,
        broken,
        pbt.up_conn_inds.to(types.dtype),
        pbt.down_conn_inds.to(types.dtype),
        pbt.canonical_dslf_conn_ind.to(types.dtype),
        empty if disulfides is None else disulfides,
        empty if cycles is None else cycles,
        torch.tensor(extra, dtype=types.dtype, device=types.device).reshape(-1, 5),
    ]
    if types.device.type == "mps":
        inputs = [x.cpu() for x in inputs]
    connections, status = _canonical_wiring(inputs, pbt.max_n_conn)
    connections = connections.to(types.device)
    if torch.any(status):
        _assert_connections_are_well_formed(pbt, types, connections)
        raise RuntimeError(
            f"malformed inter-residue connections: pose status {status.tolist()}"
        )
    return connections


def _assert_connections_are_well_formed(pbt, block_type_ind64, connections):
    """Validate endpoint bounds, reciprocity, and self-connections in the core."""
    from tmol.io.details.compiled._compiled import _canonical_connection_errors

    types = block_type_ind64
    if types.device.type == "mps":
        types, connections = types.cpu(), connections.cpu()
    bad = _canonical_connection_errors(types, connections)
    if bad.any():
        rows = torch.nonzero(bad)[:20]
        partners = connections[rows[:, 0], rows[:, 1], rows[:, 2]]
        rows = torch.cat((rows, partners), dim=1)
        raise RuntimeError(
            "malformed inter-residue connections (pose, block, port, partner, "
            f"partner port): {rows.tolist()}"
        )


@_validate_args
def _determine_chain_ending_status(
    pbt: PackedBlockTypes,
    chain_id: Tensor[torch.int32][:, :],
    res_types64: Tensor[torch.int64][:, :],
    res_not_connected: Optional[Tensor[torch.bool][:, :, 2]],
    is_real_res: Tensor[torch.bool][:, :],
    cyclic_closures64: Optional[Tensor[torch.int64][:, 3]] = None,
    explicit_polymer_connections: Optional[Tensor[torch.bool][:, :, 2]] = None,
):
    """Infer chain boundaries and termini from prepared ports and explicit topology."""
    from tmol.io.details.compiled._compiled import _canonical_chain_endings

    annotation = pbt.canonical_ordering_annotation
    if cyclic_closures64 is None:
        cyclic_closures64 = torch.empty((0, 3), dtype=torch.int64, device=pbt.device)
    if res_not_connected is None:
        res_not_connected = torch.zeros(
            (*chain_id.shape, 2), dtype=torch.bool, device=pbt.device
        )
    if explicit_polymer_connections is None:
        explicit_polymer_connections = torch.zeros_like(res_not_connected)
    inputs = [
        chain_id,
        res_types64,
        res_not_connected,
        is_real_res,
        annotation.co_equiv_class_is_polymeric,
        annotation.co_polymer_connection_atoms,
        cyclic_closures64,
        explicit_polymer_connections,
    ]
    if pbt.device.type == "mps":
        inputs = [x.cpu() for x in inputs]
    return tuple(x.to(pbt.device) for x in _canonical_chain_endings(inputs))


def _termini_the_atoms_state(pbt, present, restypes, variants, termini, broken, real):
    """Infer terminal patches at gaps without trading away any supplied atom."""
    from tmol.io.details.compiled._compiled import _canonical_atom_termini

    annotation = pbt.canonical_ordering_annotation
    inputs = [
        present,
        restypes,
        variants,
        termini,
        broken,
        real,
        annotation.var_combo_candidate_bt_index,
        annotation.var_combo_is_real_candidate,
        annotation.bt_canonical_atom_is_absent,
    ]
    if present.device.type == "mps":
        inputs = [x.cpu() for x in inputs]
    return _canonical_atom_termini(inputs).to(present.device)


@_validate_args
def _select_best_block_type_candidate(  # noqa: C901
    canonical_ordering: CanonicalOrdering,
    pbt: PackedBlockTypes,
    atom_is_present: Tensor[torch.bool][:, :, :],
    is_real_res: Tensor[torch.bool][:, :],
    nz_is_real_res: Tensor[torch.int64][:, :],
    res_types64: Tensor[torch.int64][:, :],
    termini_variants: Tensor[torch.int64][:, :],
    res_type_variants64: Tensor[torch.int64][:, :],
):
    """Select by missing/extraneous atoms and terminal preference in the common kernel."""
    from tmol.io.details.compiled._compiled import _select_canonical_candidates

    device = pbt.device
    n_poses, max_n_res = is_real_res.shape
    can_ann = pbt.canonical_ordering_annotation
    inputs = [
        atom_is_present,
        is_real_res,
        res_types64,
        termini_variants,
        res_type_variants64,
        can_ann.var_combo_candidate_bt_index,
        can_ann.var_combo_is_real_candidate,
        can_ann.bt_canonical_atom_is_absent,
        can_ann.bt_non_term_patch_added_canonical_atom_is_present,
        can_ann.bt_is_non_default_terminus,
    ]
    # Metal pose construction uses the same native CPU metadata kernel.
    if device.type == "mps":
        inputs = [x.cpu() for x in inputs]
    selected, best_candidate_score, _ = _select_canonical_candidates(inputs)
    selected, best_candidate_score = selected.to(device), best_candidate_score.to(
        device
    )
    warning_threshold, failure_threshold = _canonical.candidate_score_thresholds(
        canonical_ordering.max_n_canonical_atoms
    )
    if not torch.any(best_candidate_score >= warning_threshold):
        return selected

    max_n_candidates = can_ann.max_n_candidates_for_var_combo
    block_type_candidates = torch.full(
        (n_poses, max_n_res, max_n_candidates),
        -1,
        dtype=torch.int64,
        device=device,
    )
    is_real_candidate = torch.zeros(
        (n_poses, max_n_res, max_n_candidates),
        dtype=torch.bool,
        device=device,
    )
    real_res_res_types64 = res_types64[is_real_res]
    real_res_termini_variants = termini_variants[is_real_res]
    real_res_res_type_variants64 = res_type_variants64[is_real_res]

    real_res_block_type_candidates = can_ann.var_combo_candidate_bt_index[
        real_res_res_types64, real_res_termini_variants, real_res_res_type_variants64
    ]
    is_real_cand_for_real_res = can_ann.var_combo_is_real_candidate[
        real_res_res_types64, real_res_termini_variants, real_res_res_type_variants64
    ]
    block_type_candidates[is_real_res] = real_res_block_type_candidates
    is_real_candidate[is_real_res] = is_real_cand_for_real_res
    real_candidate_block_type = real_res_block_type_candidates[
        is_real_cand_for_real_res
    ]
    atom_is_present = atom_is_present.unsqueeze(2).expand(-1, -1, max_n_candidates, -1)
    real_candidate_atom_is_absent = can_ann.bt_canonical_atom_is_absent[
        real_candidate_block_type
    ]

    real_candidate_provided_atoms_absent = torch.logical_and(
        atom_is_present[is_real_candidate], real_candidate_atom_is_absent
    )

    atom_is_absent = torch.logical_not(atom_is_present)
    real_candidate_non_term_patch_atom_is_present = (
        can_ann.bt_non_term_patch_added_canonical_atom_is_present[
            real_candidate_block_type
        ]
    )

    real_candidate_canonical_atom_was_not_provided = torch.logical_and(
        atom_is_absent[is_real_candidate], real_candidate_non_term_patch_atom_is_present
    )
    if torch.any(best_candidate_score >= warning_threshold):
        nz_is_real_candidate = torch.nonzero(is_real_candidate)
        err_msg = []
        for i, j in torch.nonzero(is_real_res & ~is_real_candidate.any(dim=2)).tolist():
            name = canonical_ordering.restype_io_equiv_classes[res_types64[i, j]]
            termini = ("down", "neither", "up", "both")[termini_variants[i, j]]
            err_msg.append(
                f"No block type candidates for pose={i} residue={j} {name}: "
                f"termini={termini}, variant={res_type_variants64[i, j].item()}. "
                "Check prepared polymer connections and input chain topology.\n"
            )
        for cand_ind in range(nz_is_real_candidate.shape[0]):
            i = nz_is_real_candidate[cand_ind, 0]
            j = nz_is_real_candidate[cand_ind, 1]
            k = nz_is_real_candidate[cand_ind, 2]

            if best_candidate_score[i, j] < warning_threshold:
                continue
            ij_equiv_class = canonical_ordering.restype_io_equiv_classes[
                res_types64[i, j]
            ]
            err_msg.append("Failed to resolve block type for")
            err_msg.extend([str(x) for x in [i.item(), j.item()]])
            err_msg.append(str(ij_equiv_class) + "\n")

            which_bt = real_candidate_block_type[cand_ind]
            cand_bt = pbt.active_block_types[which_bt]
            err_msg.extend(
                [
                    str(x)
                    for x in (
                        i.item(),
                        j.item(),
                        k.item(),
                        which_bt.item(),
                        cand_bt.name,
                        "restype",
                    )
                ]
            )
            err_msg.extend([str(x) for x in (res_types64[i, j].item(), "equiv class")])
            err_msg.append(
                canonical_ordering.restype_io_equiv_classes[res_types64[i, j]] + "\n"
            )

            atom_names = canonical_ordering.restypes_ordered_atom_names[
                cand_bt.io_equiv_class
            ]
            for mask, message in (
                (
                    real_candidate_provided_atoms_absent[cand_ind],
                    "provided but absent from candidate",
                ),
                (
                    real_candidate_canonical_atom_was_not_provided[cand_ind],
                    "missing but present in candidate",
                ),
            ):
                for atom_index in torch.nonzero(mask).flatten().tolist():
                    err_msg.append(
                        f" atom {atom_names[atom_index]} {message} {cand_bt.name}\n"
                    )

        err_msg = " ".join(
            [
                "failed to resolve a block type from the candidates available\n",
                *err_msg,
            ]
        )

        if torch.any(best_candidate_score >= failure_threshold):
            raise RuntimeError(err_msg + "Best candidate exceeds failure threshold")
        logger.warning(err_msg)

    return selected


@_validate_args
def _take_block_type_atoms_from_canonical(
    packed_block_types: PackedBlockTypes,
    block_types64: Tensor[torch.int64][:, :],
    coords: Tensor[torch.float32][:, :, :, 3],
    atom_is_present: Tensor[torch.bool][:, :, :],
    atom_occupancy: Optional[NDArray[numpy.float32][:, :, :]] = None,
    atom_b_factor: Optional[NDArray[numpy.float32][:, :, :]] = None,
    *,
    ligand_hydrogens: str = "names",
):
    """Now that we have decided which block type each canonical residue
    is, we want to select only those atoms from the canonically-ordered
    coords and atom_is_present tensors
    In the case of the atom_is_present tensor, we will here forward only
    concern ourselves with the atoms that are missing and not with
    the (perhaps more than one) ways in which an atom can be (tentatively)
    provided; thus we will invert atom_is_present and return the new
    tensor as missing_atoms
    """
    pbt = packed_block_types
    assert hasattr(pbt, "canonical_ordering_annotation")
    can_ann = pbt.canonical_ordering_annotation
    device = pbt.device

    n_poses = block_types64.shape[0]
    max_n_blocks = block_types64.shape[1]
    real_block_types = block_types64 != -1
    real_atoms = torch.zeros(
        (n_poses, max_n_blocks, pbt.max_n_atoms), dtype=torch.bool, device=pbt.device
    )
    real_atoms[real_block_types] = pbt.atom_is_real[block_types64[real_block_types]]

    canonical_atom_inds = torch.full(
        (n_poses, max_n_blocks, pbt.max_n_atoms),
        -1,
        dtype=torch.int64,
        device=pbt.device,
    )

    canonical_atom_inds[real_block_types] = can_ann.bt_canonical_atom_ind_map[
        block_types64[real_block_types]
    ]
    nz_real_pose_ind, nz_real_block_ind, _ = torch.nonzero(real_atoms, as_tuple=True)

    block_coords = torch.zeros(
        (n_poses, max_n_blocks, pbt.max_n_atoms, 3), dtype=torch.float32, device=device
    )
    missing_atoms = torch.zeros(
        (n_poses, max_n_blocks, pbt.max_n_atoms), dtype=torch.bool, device=pbt.device
    )
    real_canonical_atom_inds = canonical_atom_inds[real_atoms]
    block_coords[real_atoms] = coords[
        nz_real_pose_ind, nz_real_block_ind, real_canonical_atom_inds
    ]
    missing_atoms[real_atoms] = torch.logical_not(
        atom_is_present[nz_real_pose_ind, nz_real_block_ind, real_canonical_atom_inds]
    )

    # A generated ligand keeps an input hydrogen only in "input" mode, and only
    # where it sits on the parent its name gives it.
    _validate_ligand_hydrogens(ligand_hydrogens)
    regenerate = torch.tensor(
        [bool(bt.hydrogens_regenerated) for bt in pbt.active_block_types],
        dtype=torch.bool,
        device=device,
    )
    indices = block_types64.clamp_min(0)
    rebuild = (
        regenerate[indices].unsqueeze(-1)
        & pbt.atom_is_hydrogen[indices].bool()
        & real_atoms
    )
    if ligand_hydrogens == "input":
        rebuild &= ~_hydrogens_on_their_parents(
            pbt, indices, block_coords.detach(), missing_atoms
        )
    missing_atoms |= rebuild

    canonical_atom_occupancy = canonical_atom_b_factor = None
    if atom_occupancy is not None or atom_b_factor is not None:
        real_atoms_n = real_atoms.cpu().numpy()
        source = (
            nz_real_pose_ind.cpu().numpy(),
            nz_real_block_ind.cpu().numpy(),
            real_canonical_atom_inds.cpu().numpy(),
        )

        def block_layout(values):
            if values is None:
                return None
            laid_out = numpy.zeros(real_atoms_n.shape, dtype=numpy.float32)
            laid_out[real_atoms_n] = values[source]
            return laid_out

        canonical_atom_occupancy = block_layout(atom_occupancy)
        canonical_atom_b_factor = block_layout(atom_b_factor)

    return (
        block_coords,
        missing_atoms,
        real_atoms,
        real_canonical_atom_inds,
        canonical_atom_occupancy,
        canonical_atom_b_factor,
    )


def _annotate_pbt_w_hydrogen_parents(pbt: PackedBlockTypes):
    """Per block type and atom, a hydrogen's bonded heavy atom and its ideal distance."""
    if hasattr(pbt, "hydrogen_parent"):
        return
    is_h = pbt.atom_is_hydrogen.cpu().numpy().astype(bool)
    parent = numpy.full((pbt.n_types, pbt.max_n_atoms), -1, dtype=numpy.int64)
    length = numpy.zeros((pbt.n_types, pbt.max_n_atoms), dtype=numpy.float32)
    for i, bt in enumerate(pbt.active_block_types):
        for a0, a1 in bt.bond_indices:
            h, heavy = int(a1), int(a0)
            if is_h[i, h] and not is_h[i, heavy]:
                parent[i, h] = heavy
                length[i, h] = bt.icoors[bt.icoors_index[bt.atoms[h].name]].d
    setattr(pbt, "hydrogen_parent", torch.tensor(parent, device=pbt.device))
    setattr(pbt, "hydrogen_parent_distance", torch.tensor(length, device=pbt.device))


def _hydrogens_on_their_parents(pbt, block_types64, block_coords, missing_atoms):
    """[pose, block, atom] mask of present hydrogens at bond length from their parent."""
    _annotate_pbt_w_hydrogen_parents(pbt)
    parent = pbt.hydrogen_parent[block_types64]
    clamped = parent.clamp_min(0)
    parent_xyz = torch.gather(
        block_coords, 2, clamped.unsqueeze(-1).expand(-1, -1, -1, 3)
    )
    distance = torch.linalg.vector_norm(block_coords - parent_xyz, dim=-1)
    ideal = pbt.hydrogen_parent_distance[block_types64]
    return (
        (parent >= 0)
        & ~missing_atoms
        & ~torch.gather(missing_atoms, 2, clamped)
        & ((distance - ideal).abs() < _BONDED_H_TOLERANCE)
    )


@attr.s(auto_attribs=True, frozen=True)
class CanonicalOrderingAnnotation:
    max_n_candidates_for_var_combo: int
    # n-co-equiv-class
    co_equiv_class_is_polymeric: Tensor[torch.bool][:]
    co_polymer_connection_atoms: Tensor[torch.int64][:, 2]
    # n-co-equiv-class x n-term-opts x n-spcase-var x max-n-candidates
    var_combo_is_real_candidate: Tensor[torch.bool][:, :, :, :]
    # n-co-equiv-class x n-term-opts x n-spcase-var x max-n-candidates
    var_combo_candidate_bt_index: Tensor[torch.int64][:, :, :, :]
    # n-pbt-block-types x max-n-canonical-atoms
    bt_canonical_atom_is_absent: Tensor[torch.bool][:, :]
    # n-pbt-block-types x max-n-canonical-atoms
    bt_non_term_patch_added_canonical_atom_is_present: Tensor[torch.bool][:, :]
    # n-pbt-block-types
    bt_is_non_default_terminus: Tensor[torch.bool][:]

    # n-pbt-block-types
    # needed for output?
    bt_ind_to_canonical_io_equiv_class_ind: Tensor[torch.int64][:]

    # n-pbt-block-types x max-n-atoms
    bt_canonical_atom_ind_map: Tensor[torch.int64][:, :]


def _get_co_equiv_class_is_polymeric(*args):
    host = _canonical._get_co_equiv_class_is_polymeric(*args)
    return torch.as_tensor(host)


def _collect_var_combo_candidates(*args):
    host = _canonical._collect_var_combo_candidates(*args)
    return tuple(torch.as_tensor(value) for value in host)


def _note_atoms_present_and_absent_from_variants(*args):
    host = _canonical._note_atoms_present_and_absent_from_variants(*args)
    return tuple(torch.as_tensor(value) for value in host)


def _create_bt_to_canonical_atom_mapping(*args):
    host = _canonical._create_bt_to_canonical_atom_mapping(*args)
    return tuple(torch.as_tensor(value) for value in host)


@_validate_args
def _annotate_packed_block_types_w_canonical_res_order(
    canonical_ordering, pbt: PackedBlockTypes
):
    if getattr(pbt, "_canonical_ordering_source", None) is canonical_ordering:
        return
    _annotate_packed_block_types_w_conjugations(pbt)
    host = _canonical.prepare_canonical_selection(
        canonical_ordering, pbt.active_block_types, pbt.max_n_atoms
    )
    pbt.canonical_ordering_annotation = CanonicalOrderingAnnotation(
        **{
            field.name: (
                torch.as_tensor(value, device=pbt.device)
                if isinstance(value, numpy.ndarray)
                else value
            )
            for field in attr.fields(type(host))
            for value in (getattr(host, field.name),)
        }
    )
    pbt._canonical_ordering_source = canonical_ordering


def _apply_connection_plan(pbt, block_type_ind64, plan):
    block_type_ind64.copy_(
        torch.tensor(plan.block_type_rows, dtype=torch.int64, device=pbt.device)
    )
    for message in plan.warnings:
        logger.warning(message)
    return plan.connections


@_validate_args
def _apply_conjugated_variants(
    canonical_ordering, pbt, block_type_ind64, covalent_bonds64
):
    if covalent_bonds64 is None or covalent_bonds64.shape[0] == 0:
        return []
    _annotate_packed_block_types_w_conjugations(pbt)
    return _apply_connection_plan(
        pbt,
        block_type_ind64,
        _connections.prepare_covalent_connections(
            canonical_ordering,
            pbt.active_block_types,
            pbt._conjugation_metadata,
            block_type_ind64.cpu().tolist(),
            covalent_bonds64.cpu().tolist(),
        ),
    )


def _apply_metal_connections(
    canonical_ordering, pbt, block_type_ind64, metal_connections
):
    if not metal_connections:
        return []
    _annotate_packed_block_types_w_conjugations(pbt)
    return _apply_connection_plan(
        pbt,
        block_type_ind64,
        _connections.prepare_metal_connections(
            canonical_ordering,
            pbt.active_block_types,
            pbt._conjugation_metadata,
            block_type_ind64.cpu().tolist(),
            metal_connections,
        ),
    )


def _annotate_packed_block_types_w_conjugations(pbt: PackedBlockTypes):
    if hasattr(pbt, "_conjugation_metadata"):
        return
    _annotate_packed_block_types_w_dslf_conn_inds(pbt)
    pbt._conjugation_metadata = _canonical.prepare_conjugations(pbt.active_block_types)
    for name, value in vars(pbt._conjugation_metadata).items():
        setattr(pbt, name, value)
