"""Host plans for fragment identity, named bonds and original block recovery."""

from dataclasses import replace
import numpy as np

from tmol_core.io.split_block_mapping import SplitBlockEntry, SplitBlockMapping
from ._fragmentation import FragmentedLigandPoseMapping


def resolve_fragment_mapping(mapping, residue_labels, block_types, active_block_types):
    """Resolve temporary author labels and require consistent model topology."""

    resolved_blocks = []
    temporary_to_actual: dict[int, int] = {}
    for record in mapping.blocks:
        candidates = np.flatnonzero(
            residue_labels[record.pose_index] == record.pose_residue_label
        ).tolist()
        if len(candidates) != 1:
            raise ValueError(
                f"Could not uniquely map {record.fragment_name} "
                f"{record.chain_label}:{record.residue_label}{record.insertion_code}; "
                f"found blocks {candidates}"
            )
        actual_index = candidates[0]
        block_type_index = int(block_types[record.pose_index, actual_index])
        block_type = active_block_types[block_type_index]
        if block_type.name != record.fragment_name:
            raise ValueError(
                f"Mapped residue label {record.pose_residue_label} to "
                f"{block_type.name}, expected {record.fragment_name}"
            )
        previous_index = temporary_to_actual.setdefault(
            record.block_index, actual_index
        )
        if previous_index != actual_index:
            raise ValueError(
                "Fragmented AtomArrayStack models must have identical block "
                "topology; fragment block indices differ between models"
            )
        resolved_blocks.append(replace(record, block_index=actual_index))

    mapping = FragmentedLigandPoseMapping(
        blocks=tuple(resolved_blocks),
        connection_pairs=tuple(
            (
                temporary_to_actual[block_a],
                name_a,
                temporary_to_actual[block_b],
                name_b,
            )
            for block_a, name_a, block_b, name_b in mapping.connection_pairs
        ),
    )
    return mapping


def connect_fragment_ports(mapping, block_types, active_block_types, connections):
    """Install named cut bonds while rejecting already occupied ports."""
    out = np.array(connections, copy=True)
    for pose in range(len(block_types)):
        for left, name_left, right, name_right in mapping.connection_pairs:
            left_type = active_block_types[block_types[pose, left]]
            right_type = active_block_types[block_types[pose, right]]
            lc = left_type.connection_to_cidx[name_left]
            rc = right_type.connection_to_cidx[name_right]
            for block, port, name in ((left, lc, name_left), (right, rc, name_right)):
                if np.any(out[pose, block, port] != -1):
                    raise ValueError(
                        f"fragment connection {block}:{name} is already occupied"
                    )
            out[pose, left, lc] = right, rc
            out[pose, right, rc] = left, lc
    return out


def missing_original_type_names(mapping, block_types, active_block_types):
    """List absent original types in deterministic first occurrence order."""
    active = {block.name for block in active_block_types}
    names = {}
    for record in mapping.blocks:
        fragment = active_block_types[
            block_types[record.pose_index, record.block_index]
        ]
        original = fragment.name.rsplit(".", 1)[0]
        if original not in active:
            names.setdefault(original, None)
    return tuple(names)


def build_split_block_mapping(mapping, block_types, active_block_types):
    """Map fragment atoms and distinct physical ligands to their original identity."""
    by_name = {block.name: i for i, block in enumerate(active_block_types)}
    groups, counts, entries = {}, {}, []
    for record in mapping.blocks:
        pose, block = record.pose_index, record.block_index
        fragment = active_block_types[block_types[pose, block]]
        original_name = fragment.name.rsplit(".", 1)[0]
        # Repeated instances of the same ligand type remain separate molecules.
        key = (
            pose,
            original_name,
            record.chain_label,
            record.residue_label,
            record.insertion_code,
        )
        if key not in groups:
            groups[key] = counts.get(pose, 0)
            counts[pose] = groups[key] + 1
        original_index = by_name[original_name]
        original = active_block_types[original_index]
        entries.append(
            SplitBlockEntry(
                pose_ind=pose,
                block_ind=block,
                group_ind=groups[key],
                orig_block_type_ind=original_index,
                split_to_orig_atom_inds=np.array(
                    [original.atom_to_idx[a.name] for a in fragment.atoms], np.int32
                ),
                orig_residue_label=record.residue_label,
                orig_chain_label=record.chain_label,
                orig_ins_code=record.insertion_code,
            )
        )
    return SplitBlockMapping(tuple(entries))


def recombine_fragmented_ligands(
    structure, sbm, residue_labels, block_types, active_block_types
):
    """Restore original author identities on exported fragment atoms."""
    if sbm is None or not sbm.entries:
        return structure

    result = structure.copy()

    for entry in sbm.entries:
        if entry.pose_ind != 0:
            continue
        pose_res_label = int(residue_labels[entry.pose_ind, entry.block_ind])
        frag_name = active_block_types[
            int(block_types[entry.pose_ind, entry.block_ind])
        ].name
        orig_name = active_block_types[entry.orig_block_type_ind].name

        fragment_atoms = (result.res_id == pose_res_label) & (
            result.res_name == frag_name
        )
        result.res_name[fragment_atoms] = orig_name
        result.res_id[fragment_atoms] = entry.orig_residue_label
        result.chain_id[fragment_atoms] = entry.orig_chain_label
        result.ins_code[fragment_atoms] = entry.orig_ins_code
    return result
