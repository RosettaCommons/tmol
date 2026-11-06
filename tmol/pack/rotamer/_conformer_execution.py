"""Torch buffer preparation for the common native conformer row-copy kernel."""

import torch


def copy_conformer_rows(
    sizes, source_offsets, target_offsets, source_values, target_values, root_shift=0
):
    if sizes.numel() == 0:
        return
    from ._conformer_compiled import copy_conformer_rows as native_copy

    ends = sizes.to(torch.int64).cumsum(0)
    packed_offsets = torch.cat((ends.new_zeros(1), ends))
    # Eager sampling already has a dynamic selected-atom count. Resolve it here;
    # the common launch and a JAX FFI call require no allocation or host sync.
    native_copy(
        source_offsets,
        target_offsets,
        packed_offsets,
        source_values,
        target_values,
        int(ends[-1]),
        root_shift,
    )
