"""Host plans for selecting molecular blocks without copying coordinates.

Frontends gather coordinates (preserving derivatives), logical chains and
sparse bond separations on their execution device. This module owns layout,
connection remapping and author metadata, including custom residue types.
"""

from dataclasses import dataclass

import attr
import numpy

from tmol_core.io.pdb_info import PDBInfo


@dataclass(frozen=True)
class BlockSelectionPlan:
    blocks: numpy.ndarray
    source_atoms: numpy.ndarray
    block_types: numpy.ndarray
    offsets: numpy.ndarray
    connections: numpy.ndarray


def selection_plan(
    blocks,
    block_types,
    block_offsets,
    n_atoms_by_type,
    connections,
    *,
    split_block_mapping=None,
):
    """Plan the mature homogeneous-layout selection using host integer arrays.

    Block order and repeated/negative indices follow array indexing. Bond
    separations are restricted by the frontend, not recomputed after deletion.
    Constraints and split mappings are not transferred to the selected pose.
    """
    if split_block_mapping:
        raise ValueError("selecting blocks does not support split-block metadata")
    keep = numpy.asarray(list(blocks), dtype=numpy.int64)
    block_types = numpy.asarray(block_types, dtype=numpy.int64)
    if keep.size == 0 or numpy.any(block_types[:, keep] < 0):
        raise ValueError("blocks must be a non-empty set present in every pose")
    block_offsets = numpy.asarray(block_offsets, dtype=numpy.int64)
    selected_types = block_types[:, keep]
    if not numpy.all(selected_types == selected_types[0:1]):
        raise ValueError("selecting blocks requires a homogeneous block layout")

    old_to_new = numpy.full(block_types.shape[1], -1, dtype=numpy.int64)
    old_to_new[keep] = numpy.arange(keep.size, dtype=numpy.int64)
    connections = numpy.asarray(connections, dtype=numpy.int64)[:, keep].copy()
    partners = connections[..., 0]
    real = partners >= 0
    mapped = numpy.full_like(partners, -1)
    mapped[real] = old_to_new[partners[real]]
    connections[real & (mapped < 0)] = -1
    partners[mapped >= 0] = mapped[mapped >= 0]

    new_types = selected_types.copy()
    n_atoms = numpy.asarray(n_atoms_by_type, dtype=numpy.int64)[new_types[0]]
    offsets = numpy.zeros_like(new_types)
    offsets[:, 1:] = numpy.cumsum(n_atoms[:-1])[None, :]
    source_atoms = numpy.concatenate(
        [
            block_offsets[0, b] + numpy.arange(n, dtype=numpy.int64)
            for b, n in zip(keep, n_atoms)
        ]
    )
    return BlockSelectionPlan(keep, source_atoms, new_types, offsets, connections)


def select_pdb_info(pdb, plan):
    """Copy atom and residue annotations using the same selection plan."""
    by_residue = {
        name: getattr(pdb, name)[:, plan.blocks].copy()
        for name in (
            "residue_labels",
            "residue_insertion_codes",
            "chain_labels",
            "metal_origins",
            "residue_annotations",
        )
        if getattr(pdb, name) is not None
    }
    return PDBInfo(
        atom_occupancy=pdb.atom_occupancy[:, plan.source_atoms].copy(),
        atom_b_factor=pdb.atom_b_factor[:, plan.source_atoms].copy(),
        **by_residue,
    )


def validate_repeat(n_source_poses, split_block_mapping=None):
    if n_source_poses != 1:
        raise ValueError("only a single reference pose can be replicated")
    if split_block_mapping:
        raise ValueError("replicating split-block metadata is not supported")


def repeat_pdb_info(pdb, n_poses):
    return PDBInfo(
        **{
            name: None if value is None else numpy.repeat(value, n_poses, axis=0)
            for name, value in attr.asdict(pdb, recurse=False).items()
        }
    )
