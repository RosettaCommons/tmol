"""Torch buffers and diagnostics for shared protonation evidence/selection."""

import logging
import torch
from tmol_core.chemical import DEPROTONATED_VAR_IND
from tmol_core.io.protonation import (
    prepare_protonation_plan,
    protonation_is_identity,
    required_donor_rows,
    neutral_terminus_patches,
    variant_atoms,
)

logger = logging.getLogger(__name__)
_LAST_BUFFERS = None


def _plan_buffers(plan, device):
    global _LAST_BUFFERS
    if _LAST_BUFFERS is not None:
        old, old_device, buffers = _LAST_BUFFERS
        if old is plan and old_device == device:
            return buffers
    buffers = tuple(torch.from_numpy(a).to(device) for a in plan.arrays)
    _LAST_BUFFERS = plan, device, buffers
    return buffers


def _select_protonation_variants(
    canonical_ordering,
    chemical_db,
    res_types,
    res_type_variants,
    atom_is_present,
    metal_assignments,
    covalent_bonds=None,
):
    from .compiled._compiled import _ops

    plan = prepare_protonation_plan(
        canonical_ordering, chemical_db, res_types.cpu().numpy()
    )
    device = atom_is_present.device
    required_rows = required_donor_rows(metal_assignments)
    if not len(required_rows) and (
        not plan.expecting.any()
        or (
            device.type == "cpu"
            and protonation_is_identity(
                plan,
                res_types.numpy(),
                res_type_variants.numpy(),
                atom_is_present.numpy(),
                required_rows,
            )
        )
    ):
        return res_type_variants.clone()
    execution = torch.device("cpu") if device.type == "mps" else device
    bonds = (
        torch.empty((0, 5), dtype=torch.int64, device=execution)
        if covalent_bonds is None
        else covalent_bonds.to(device=execution, dtype=torch.int64)
    )
    required = torch.from_numpy(required_rows).to(execution)
    inputs = [
        res_types.to(execution),
        res_type_variants.to(execution),
        atom_is_present.to(execution),
        *_plan_buffers(plan, execution),
        bonds,
        required,
    ]
    variants, unresolved = _ops.protonation(inputs, DEPROTONATED_VAR_IND)
    for pose, res in torch.nonzero(unresolved).cpu().tolist():
        cls = int(res_types[pose, res])
        logger.warning(
            "%s %d coordinates through atoms no form of it can donate",
            canonical_ordering.restype_io_equiv_classes[cls],
            res,
        )
    return variants.to(device)


def _variant_atoms(canonical_ordering, chemical_db, res_types):
    return variant_atoms(canonical_ordering, chemical_db, res_types.cpu().numpy())


def _neutral_terminus_patches(canonical_ordering, chemical_db, res_types, variants):
    return neutral_terminus_patches(
        canonical_ordering, chemical_db, res_types.cpu().numpy(), variants.cpu().numpy()
    )
