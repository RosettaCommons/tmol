from typing import Optional

import attr

import numpy
import torch

from tmol.chemical import ResidueTypeSet
from tmol.database import ParameterDatabase
from tmol.io._build_context import PoseBuildContext
from tmol.io._canonical_ordering import CanonicalOrdering, _default_packed_block_types
from tmol.pose._packed_block_types import PackedBlockTypes
from tmol.pose._pose_stack import PoseStack
from tmol.pose._pose_stack_builder import PoseStackBuilder
from tmol.pose._sequence import (
    _resolve_block_type_names,
    _smiles_in_tokens,
    _tokenize_sequences,
)
from tmol.utility import resolve_device
from tmol_core.io import extended as _extended
from tmol_core.io.extended import EXTENDED_BACKBONE_TORSIONS  # noqa: F401


def _create_pose_stack_from_sequences(
    seqs,  # str | Sequence[str]
    packed_block_types: Optional[PackedBlockTypes] = None,
    device: Optional[torch.device] = None,
    param_db=None,
    termini: bool = True,
    context: Optional[PoseBuildContext] = None,
    return_context: bool = False,
):
    """Construct a PoseStack with zero coordinates from sequence strings.

    See tmol.pose._sequence for the grammar. Returns
    (PoseStack, PoseBuildContext) when return_context is set; the context
    carries the database extended with any ligands the sequence names.
    """
    device = resolve_device(device if device is not None else torch.device("cpu"))
    if context is not None and param_db is not None:
        raise ValueError("pass either context= or param_db=, not both")

    tokens, chain_lengths = _tokenize_sequences(seqs)

    if context is not None:
        context_device = context.packed_block_types.device
        if context_device.type != device.type or (
            context_device.type == "cuda" and context_device.index != device.index
        ):
            raise ValueError(
                f"context was built for device "
                f"{context.packed_block_types.device} but device is {device}"
            )
        param_db = context.parameter_database
        restype_set = context.restype_set
        ligand_names = context.ligand_names
        canonical_ordering = context.canonical_ordering
    else:
        smiles = _smiles_in_tokens(tokens)
        if smiles:
            from tmol.ligand._preparation import _prepare_ligands_from_smiles

            param_db, ligand_names = _prepare_ligands_from_smiles(
                smiles, param_db=param_db
            )
        else:
            param_db = param_db or ParameterDatabase.get_default()
            ligand_names = {}
        if param_db is ParameterDatabase.get_default():
            restype_set = ResidueTypeSet.get_default()
        else:
            restype_set = ResidueTypeSet.from_database(param_db.chemical)
        canonical_ordering = None

    names, chain_lengths = _resolve_block_type_names(
        tokens, chain_lengths, restype_set, ligand_names, termini
    )

    if packed_block_types is None:
        if restype_set.chem_db is ParameterDatabase.get_default().chemical:
            packed_block_types = _default_packed_block_types(device)
        else:
            packed_block_types = PackedBlockTypes.from_restype_list(
                restype_set.chem_db, restype_set, restype_set.residue_types, device
            )

    pose_stack = PoseStackBuilder.from_block_type_names(
        packed_block_types, names, chain_lengths
    )
    if not return_context:
        return pose_stack

    if canonical_ordering is None:
        canonical_ordering = CanonicalOrdering.from_chemdb(param_db.chemical)
    return pose_stack, PoseBuildContext(
        canonical_ordering=canonical_ordering,
        packed_block_types=packed_block_types,
        parameter_database=param_db,
        restype_set=restype_set,
        ligand_names=ligand_names,
    )


# ---------------------------------------------------------------------------
# Backbone torsion values to 'build extended'
#    beta strand for protein / B-form DNA / A-form RNA
# ---------------------------------------------------------------------------


def extended_pose_stack_from_sequences(
    seqs,  # str | Sequence[str]
    device: Optional[torch.device] = None,
    param_db=None,
    termini: bool = True,
    context=None,
    return_context: bool = False,
):
    """Build a PoseStack from sequences with ideal geometry and extended
    backbone torsions.

    See tmol.pose._sequence for the grammar. Returns
    (PoseStack, PoseBuildContext) when return_context is set.
    """
    pose_stack, build_context = _create_pose_stack_from_sequences(
        seqs,
        device=device,
        param_db=param_db,
        termini=termini,
        context=context,
        return_context=True,
    )
    pose_stack = attr.evolve(pose_stack, coords=_ideal_chained_coords(pose_stack))
    pose_stack = _set_ideal_backbone_torsions(pose_stack)
    pose_stack = _separate_pose_stack_chains(pose_stack)
    return (pose_stack, build_context) if return_context else pose_stack


# ---------------------------------------------------------------------------
# Helpers used only by extended_pose_stack_from_sequences
# ---------------------------------------------------------------------------


def _set_ideal_backbone_torsions(pose_stack: PoseStack) -> PoseStack:
    from tmol.pose._util import _measure_torsions
    from tmol.kinematics._pose_stack_kinematics import _apply_torsion_deltas

    requests, values = _extended.extended_torsion_requests(
        pose_stack.packed_block_types.active_block_types,
        pose_stack.block_type_ind64.cpu().numpy(),
    )
    if not requests:
        return pose_stack

    current = _measure_torsions(pose_stack, requests, degrees=False)
    deltas = numpy.radians(numpy.array(values)) - current
    defined = ~numpy.isnan(deltas)
    requests = [r for r, keep in zip(requests, defined) if keep]
    return _apply_torsion_deltas(pose_stack, requests, deltas[defined])


def _ideal_chained_coords(pose_stack: PoseStack) -> torch.Tensor:
    coords = _extended.ideal_chained_coords(
        pose_stack.packed_block_types.active_block_types,
        pose_stack.block_type_ind64.cpu().numpy(),
        pose_stack.block_coord_offset64.cpu().numpy(),
        pose_stack.inter_residue_connections64.cpu().numpy(),
        pose_stack.coords.shape,
    )
    return torch.tensor(coords, dtype=torch.float32, device=pose_stack.device)


def _separate_pose_stack_chains(pose_stack: PoseStack) -> PoseStack:
    """Slide each chain along x so no two chains overlap."""
    shifts = _extended.chain_separation_shifts(
        pose_stack.packed_block_types.active_block_types,
        pose_stack.block_type_ind64.cpu().numpy(),
        pose_stack.block_coord_offset64.cpu().numpy(),
        pose_stack.chain_id.cpu().numpy(),
        pose_stack.coords.detach().cpu().numpy(),
    )
    coords = pose_stack.coords.clone()
    for pose, indices, shift in shifts:
        inds = torch.from_numpy(indices).to(pose_stack.device)
        coords[pose, inds, 0] += shift
    return attr.evolve(pose_stack, coords=coords)
