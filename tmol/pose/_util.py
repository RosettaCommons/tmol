"""Read-only queries on a PoseStack: sequence and named torsions.

Setting torsions (which requires inverse kinematics) lives in
``tmol.kinematics._pose_stack_kinematics`` so that this module carries no
kinematics dependency.
"""

from typing import List

import numpy
import torch

from tmol_core.pose import query as _query
from tmol.numeric import coord_dihedrals
from tmol.pose._pose_stack import PoseStack
from tmol_core.chemical.atom_resolution import torsion_atom_indices, resolve_atom_id
from tmol_core.pose.query import (
    torsion_requests,
    format_named_torsions,
)
from tmol_core.pose.query import _is_scalar, _as_list  # noqa: F401


def get_sequence(pose_stack: PoseStack, pose: int = 0) -> str:
    """One-letter sequence of one pose's real (non-padding) blocks.

    Blocks whose three-letter code has no one-letter equivalent (ligands,
    ions, and other non-canonical residues) appear as ``"X"``.
    """
    return _query.sequence_from_types(
        pose_stack.packed_block_types.active_block_types,
        pose_stack.block_type_ind64[pose].cpu().numpy(),
    )


def get_ligand_mask(pose_stack: PoseStack) -> torch.Tensor:
    """Boolean mask shaped ``[n_poses, max_n_blocks]``, true where a block is a
    ligand (or any other non-polymer residue, such as an ion). Padding blocks
    are false.
    """
    flags = torch.tensor(
        _query.ligand_flags(pose_stack.packed_block_types.active_block_types),
        dtype=torch.bool,
        device=pose_stack.device,
    )
    bti = pose_stack.block_type_ind64
    real = bti >= 0
    safe_ind = torch.where(real, bti, torch.zeros_like(bti))
    return real & flags[safe_ind]


def get_polymer_type_mask(pose_stack: PoseStack, polymer_type) -> torch.Tensor:
    """Boolean mask shaped ``[n_poses, max_n_blocks]``, true where a block's
    ``properties.polymer.polymer_type`` equals ``polymer_type`` (one of
    ``"amino_acid"``, ``"nucleic_acid"``, or ``None`` for a non-polymer like a
    ligand or ion). Padding blocks are false.
    """
    flags = torch.tensor(
        _query.polymer_type_flags(
            pose_stack.packed_block_types.active_block_types, polymer_type
        ),
        dtype=torch.bool,
        device=pose_stack.device,
    )
    bti = pose_stack.block_type_ind64
    real = bti >= 0
    safe_ind = torch.where(real, bti, torch.zeros_like(bti))
    return real & flags[safe_ind]


def metal_blocks(pose_stack: PoseStack, pose: int) -> List[int]:
    """Indices of the blocks in one pose that carry metal sites."""
    return _query.metal_blocks(
        pose_stack.packed_block_types.active_block_types,
        pose_stack.block_type_ind64[pose].cpu().numpy(),
    )


def residue_label(pose_stack: PoseStack, pose: int, block: int) -> str:
    """ "chain resnum name3" for one block."""
    return _query.residue_label(
        pose_stack.block_type(pose, block), pose_stack.pdb_info, pose, block
    )


def atom_label(pose_stack: PoseStack, pose: int, atom: int) -> str:
    """ "chain resnum name3 atom" for a per-pose atom index."""
    return _query.atom_label(
        pose_stack.packed_block_types.active_block_types,
        pose_stack.block_type_ind64[pose].cpu().numpy(),
        pose_stack.block_coord_offset64[pose].cpu().numpy(),
        pose_stack.pdb_info,
        pose,
        atom,
    )


_RESIDUE_ID = _query._RESIDUE_ID


def block_for_residue_label(
    pose_stack: PoseStack, pose_index: int, chain: str, residue_id: str
) -> int:
    """Block index of one (chain, author residue label).

    ``residue_id`` is a residue number as a string, with an optional PDB
    insertion-code suffix (e.g. ``"12"`` or ``"12A"``).

    Raises ValueError if no block in the pose matches.
    """
    return _query.block_for_residue_label(
        pose_stack.pdb_info, pose_index, chain, residue_id
    )


def blocks_for_residue_labels(
    pose_stack: PoseStack,
    pose_index: int,
    chain: str,
    residue_numbers: torch.Tensor,
) -> torch.Tensor:
    """Block index of each (chain, author residue number) label in a batch.

    ``residue_numbers`` may have any shape; the result has the same shape.
    Entries with no matching block resolve to ``-1``.
    """
    block = _query.blocks_for_residue_labels(
        pose_stack.pdb_info, pose_index, chain, residue_numbers.cpu().numpy()
    )
    return torch.as_tensor(block, dtype=torch.int64, device=pose_stack.device)


def atom_for_residue_label(
    pose_stack: PoseStack, pose_ind: int, chain_id: str, res_id: str, atom_id: str
) -> torch.Tensor:
    """``[pose, block, atom]`` triple for one atom, addressed entirely in PDB space.

    ``res_id`` is a residue number as a string, with an optional PDB
    insertion-code suffix (e.g. ``"12"`` or ``"12A"``); ``atom_id`` is that
    residue's atom name (e.g. ``"CA"``). The result is an ``int32`` tensor on
    ``pose_stack``'s device, suitable directly as one row of the ``atoms``
    tensor :meth:`ConstraintSet.add_constraints` expects.

    Raises ValueError if no block matches the chain/residue, or if the
    matching block's residue type has no atom by that name.
    """
    atoms = _query.atom_for_residue_label(
        pose_stack.packed_block_types.active_block_types,
        pose_stack.block_type_ind64.cpu().numpy(),
        pose_stack.pdb_info,
        pose_ind,
        chain_id,
        res_id,
        atom_id,
    )
    return torch.as_tensor(atoms, dtype=torch.int32, device=pose_stack.device)


def constraint_atoms_for_labels(
    pose_stack: PoseStack,
    atoms,
) -> torch.Tensor:
    """The atoms of one constraint, addressed entirely in PDB space.

    ``atoms`` is a sequence of 1 to 4 ``(pose_ind, chain_id, res_id, atom_id)``
    labels (see :func:`atom_for_residue_label`) -- one atom per constraint
    slot. The slot count matches whichever ``ConstraintEnergyTerm`` function
    the result is eventually passed to: 1 for a coordinate constraint, 2 for
    a distance constraint (``harmonic``/``bounded``), 4 for a torsion-angle
    constraint (``circularharmonic``).

    Returns an ``int32`` tensor shaped ``[len(atoms), 3]`` on ``pose_stack``'s
    device. Stack the result for each constraint (e.g. ``torch.stack(...)``)
    to build the ``[n_constraints, n_atoms, 3]`` tensor
    :meth:`ConstraintSet.add_constraints` expects.
    """
    indices = _query.constraint_atoms_for_labels(
        pose_stack.packed_block_types.active_block_types,
        pose_stack.block_type_ind64.cpu().numpy(),
        pose_stack.pdb_info,
        atoms,
    )
    return torch.as_tensor(indices, dtype=torch.int32, device=pose_stack.device)


def get_torsion_names(pose_stack: PoseStack, pose: int, block: int) -> List[str]:
    """Names of the torsions defined on a block's type, in database order."""
    bt_ind = int(pose_stack.block_type_ind64[pose, block])
    if bt_ind == -1:
        raise ValueError(f"block {block} of pose {pose} is not a real block")
    bt = pose_stack.packed_block_types.active_block_types[bt_ind]
    return [tor.name for tor in bt.torsions]


def get_named_torsions(
    pose_stack: PoseStack,
    poses=None,
    blocks=None,
    names=None,
    degrees: bool = True,
):
    """Measure named torsions.

    Returns a single float when ``poses``, ``blocks`` and ``names`` are all
    scalars, a {name: value} dict when only ``names`` is left open, and
    otherwise a list-of-lists of such dicts indexed [pose][block]. Torsions that
    reach an absent neighboring residue measure as nan.
    """
    requests = _torsion_requests(pose_stack, poses, blocks, names)
    values = _measure_torsions(pose_stack, requests, degrees)

    return format_named_torsions(
        requests,
        values,
        (pose_stack.n_poses, pose_stack.max_n_blocks),
        poses,
        blocks,
        names,
    )


def _torsion_requests(pose_stack, poses, blocks, names):
    return torsion_requests(
        pose_stack.packed_block_types.active_block_types,
        pose_stack.block_type_ind64.cpu().numpy(),
        poses,
        blocks,
        names,
    )


def _resolve_uaid(pose_stack, pose, block, uaid):
    """Flat (pose, atom) coordinate index for a uaid, or -1 if unresolvable."""
    return resolve_atom_id(
        pose,
        block,
        uaid,
        pose_stack.block_type_ind64,
        pose_stack.block_coord_offset64,
        pose_stack.inter_residue_connections64,
        pose_stack.packed_block_types.atom_downstream_of_conn,
        pose_stack.max_n_pose_atoms,
    )


def _measure_torsions(pose_stack, requests, degrees):
    if not requests:
        return numpy.empty(0, dtype=numpy.float64)
    pbt = pose_stack.packed_block_types
    inds = torsion_atom_indices(
        pbt.active_block_types,
        requests,
        pose_stack.block_type_ind64.cpu().numpy(),
        pose_stack.block_coord_offset64.cpu().numpy(),
        pose_stack.inter_residue_connections64.cpu().numpy(),
        pbt.atom_downstream_of_conn.cpu().numpy(),
        pose_stack.max_n_pose_atoms,
    )
    return _measure_torsion_indices(pose_stack, inds, degrees)


def _measure_torsion_indices(pose_stack, inds, degrees):
    values = numpy.full(len(inds), numpy.nan)
    ok = (inds != -1).all(axis=1)
    if ok.any():
        flat = pose_stack.coords.reshape(-1, 3).double()
        sel = torch.from_numpy(inds[ok]).to(pose_stack.device)
        measured = coord_dihedrals(
            flat[sel[:, 0]],
            flat[sel[:, 1]],
            flat[sel[:, 2]],
            flat[sel[:, 3]],
            output_dtype=pose_stack.coords.dtype,
        )
        values[ok] = measured.cpu().numpy().astype(numpy.float64)
    return numpy.degrees(values) if degrees else values
