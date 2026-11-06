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


def assign_chi_dofs(
    types, offsets, conformers, atoms, chi, kinematic_index, corrections, dofs
):
    from ._conformer_compiled import assign_chi_dofs as native_assign

    operation = (
        _AssignChiDofs.apply
        if torch.is_grad_enabled()
        and any(value.requires_grad for value in (chi, corrections, dofs))
        else native_assign
    )
    operation(
        types, offsets, conformers, atoms, chi, kinematic_index, corrections, dofs
    )


class _AssignChiDofs(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, types, offsets, conformers, atoms, chi, kinematic_index, corrections, dofs
    ):
        from ._conformer_compiled import assign_chi_dofs, chi_dof_indices

        indices = chi_dof_indices(
            types, offsets, conformers, atoms, kinematic_index, corrections.shape[1]
        )
        ctx.save_for_backward(indices)
        ctx.chi_shape, ctx.correction_shape = chi.shape, corrections.shape
        ctx.chi_size, ctx.correction_size = chi.numel(), corrections.numel()
        ctx.mark_dirty(dofs)
        assign_chi_dofs(
            types, offsets, conformers, atoms, chi, kinematic_index, corrections, dofs
        )
        return dofs

    @staticmethod
    def backward(ctx, gradient):
        (indices,) = ctx.saved_tensors
        active = indices[0] >= 0
        destination, parameter = indices[:, active]
        grad_values = gradient.flatten()[destination]
        grad_chi = (
            gradient.new_zeros(ctx.chi_size)
            .masked_scatter(active, grad_values)
            .reshape(ctx.chi_shape)
        )
        grad_parameters = (
            gradient.new_zeros(ctx.correction_size)
            .index_add(0, parameter, -grad_values)
            .reshape(ctx.correction_shape)
        )
        grad_dofs = (
            gradient.flatten().index_fill(0, destination, 0).reshape_as(gradient)
        )
        return None, None, None, None, grad_chi, None, grad_parameters, grad_dofs
