"""Host chemistry and identity plans for heterogeneous pose concatenation."""

from dataclasses import dataclass, replace

import numpy

from tmol_core.io.split_block_mapping import SplitBlockMapping


def is_leading_run(items, of) -> bool:
    """Whether ``items`` begin ``of``, as the same objects."""
    return len(items) <= len(of) and all(a is b for a, b in zip(items, of))


@dataclass(frozen=True)
class ConcatenationLayout:
    n_poses: int
    max_n_blocks: int
    max_n_atoms: int
    pose_offsets: numpy.ndarray


def concatenation_layout(pose_stacks):
    """Describe padding and original batch offsets without reading coordinates."""
    counts = numpy.asarray([ps.n_poses for ps in pose_stacks], dtype=numpy.int64)
    return ConcatenationLayout(
        int(counts.sum()),
        max(ps.max_n_blocks for ps in pose_stacks),
        max(ps.coords.shape[1] for ps in pose_stacks),
        numpy.cumsum(counts) - counts,
    )


def widest_packed_block_types(pose_stacks):
    """The packed block types over the largest chemical database.

    Databases grow only by appending residues; each must be a leading run of it.
    """
    widest = max(
        (ps.packed_block_types for ps in pose_stacks),
        key=lambda pbt: (len(pbt.chem_db.residues), pbt.n_types),
    )
    for ps in pose_stacks:
        residues = ps.packed_block_types.chem_db.residues
        if not is_leading_run(residues, widest.chem_db.residues):
            raise ValueError(
                "pose stacks were built from chemical databases neither of "
                "which extends the other; build them from one context"
            )
    return widest


def merged_block_types(pose_stacks):
    """First-seen residue identities in stable name order."""
    by_name = {}
    for pose in pose_stacks:
        for block_type in pose.packed_block_types.active_block_types:
            by_name.setdefault(block_type.name, block_type)
    return list(by_name.values())


def concatenate_split_mapping(packed_block_types, pose_stacks):
    """Concatenate split-block metadata and remap original block types."""
    if not any(
        pose_stack.split_block_mapping is not None
        and pose_stack.split_block_mapping.entries
        for pose_stack in pose_stacks
    ):
        return None

    block_type_index_by_name = {
        block_type.name: index
        for index, block_type in enumerate(packed_block_types.active_block_types)
    }
    combined_entries = []
    pose_offset = 0
    for pose_stack in pose_stacks:
        mapping = pose_stack.split_block_mapping
        if mapping is not None:
            source_block_types = pose_stack.packed_block_types.active_block_types
            combined_entries.extend(
                replace(
                    entry,
                    pose_ind=pose_offset + entry.pose_ind,
                    orig_block_type_ind=block_type_index_by_name[
                        source_block_types[entry.orig_block_type_ind].name
                    ],
                )
                for entry in mapping.entries
            )
        pose_offset += pose_stack.n_poses

    return SplitBlockMapping(entries=tuple(combined_entries))


def concatenate_pdb_info(pose_stacks, n_poses, ps_offsets, max_n_blocks):
    """Return common metadata fields, retaining mature padding and optional fields."""
    ps_offsets = numpy.asarray(ps_offsets, dtype=numpy.int64)
    max_n_atoms = max(ps.coords.shape[1] for ps in pose_stacks)
    block_shape, atom_shape = (n_poses, max_n_blocks), (n_poses, max_n_atoms)
    fields = {
        "residue_labels": numpy.full(block_shape, 0, dtype=int),
        "residue_insertion_codes": numpy.full(block_shape, "", dtype=object),
        "chain_labels": numpy.full(block_shape, "", dtype=object),
        "atom_occupancy": numpy.full(atom_shape, 1.0, dtype=numpy.float32),
        # Concatenation historically pads B factors with zero, independently
        # of the default applied when reading atoms with absent B factors.
        "atom_b_factor": numpy.full(atom_shape, 0.0, dtype=numpy.float32),
        "metal_origins": (
            numpy.full(block_shape, None, dtype=object)
            if any(ps.pdb_info.metal_origins is not None for ps in pose_stacks)
            else None
        ),
    }
    annotations = [ps.pdb_info.residue_annotations for ps in pose_stacks]
    fields["residue_annotations"] = (
        numpy.zeros(block_shape, dtype=annotations[0].dtype)
        if all(a is not None for a in annotations)
        and len({a.dtype for a in annotations}) == 1
        else None
    )
    for pose, offset in zip(pose_stacks, ps_offsets):
        for name, destination in fields.items():
            value = getattr(pose.pdb_info, name)
            if destination is None or (name == "metal_origins" and value is None):
                continue
            width = (
                pose.coords.shape[1]
                if name in ("atom_occupancy", "atom_b_factor")
                else pose.pdb_info.residue_labels.shape[1]
            )
            destination[offset : offset + pose.n_poses, :width] = value
    return fields
