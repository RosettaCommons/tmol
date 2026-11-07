"""Residue mutation selection and task restrictions shared by execution adapters."""

import numpy

from tmol_core.chemical.restypes import _one2three


def sequence_mutations(active_types, block_type_ind_row, sequence, pose_index):
    """Return only changed real blocks, preserving the public one-letter contract."""
    real_blocks = numpy.flatnonzero(numpy.asarray(block_type_ind_row) >= 0)
    if len(sequence) != len(real_blocks):
        raise ValueError(
            f"sequence has {len(sequence)} residues but pose {pose_index} has "
            f"{len(real_blocks)} real blocks"
        )
    name3_by_block = {}
    for block, letter in zip(real_blocks, sequence):
        new_name3 = _one2three(letter)
        if new_name3 is None:
            raise ValueError(f"{letter!r} is not a valid one-letter amino acid code")
        current_name3 = active_types[int(block_type_ind_row[block])].name3
        if new_name3 != current_name3:
            name3_by_block[int(block)] = new_name3
    return name3_by_block


def configure_mutation_task(task, shape, pose_index, name3_by_block, mask_to_device):
    """Freeze other blocks/poses and restrict requested identities on an existing task.

    The adapter supplies only boolean-array placement; task methods implement
    the common packing policy. Samplers and packing execution stay in frontends.
    """
    target_mask = numpy.zeros(shape, dtype=bool)
    for block in name3_by_block:
        target_mask[pose_index, block] = True
    task.disable_packing_by_block_mask(mask_to_device(~target_mask))
    for block, name3 in name3_by_block.items():
        single_mask = numpy.zeros_like(target_mask)
        single_mask[pose_index, block] = True
        task.restrict_absent_name3s({name3}, mask_to_device(single_mask))
