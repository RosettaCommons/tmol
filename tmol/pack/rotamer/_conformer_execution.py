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
    operation = (
        _CopyConformerRows.apply
        if torch.is_grad_enabled()
        and (source_values.requires_grad or target_values.requires_grad)
        else native_copy
    )
    operation(
        source_offsets,
        target_offsets,
        packed_offsets,
        source_values,
        target_values,
        int(ends[-1]),
        root_shift,
    )


class _CopyConformerRows(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        source_offsets,
        target_offsets,
        packed_offsets,
        source,
        target,
        n_rows,
        shift,
    ):
        from ._conformer_compiled import copy_conformer_rows, conformer_row_indices

        indices = conformer_row_indices(
            source_offsets, target_offsets, packed_offsets, n_rows, shift
        )
        ctx.save_for_backward(indices)
        ctx.source_shape = source.shape
        ctx.mark_dirty(target)
        copy_conformer_rows(
            source_offsets,
            target_offsets,
            packed_offsets,
            source,
            target,
            n_rows,
            shift,
        )
        return target

    @staticmethod
    def backward(ctx, gradient):
        (indices,) = ctx.saved_tensors
        source, target = indices
        grad_source = gradient.new_zeros(ctx.source_shape).index_add(
            0, source, gradient[target]
        )
        grad_target = gradient.index_fill(0, target, 0)
        return None, None, None, grad_source, grad_target, None, None
