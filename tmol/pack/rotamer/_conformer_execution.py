"""Torch execution and derivatives for common native conformer plans."""

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
    if target_values.device.type not in ("cpu", "cuda"):
        from ._conformer_compiled import conformer_row_indices

        indices = conformer_row_indices(
            source_offsets.cpu(),
            target_offsets.cpu(),
            packed_offsets.cpu(),
            int(ends[-1]),
            root_shift,
        ).to(target_values.device)
        target_values[indices[1]] = source_values[indices[0]]
        return
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

    if dofs.device.type not in ("cpu", "cuda"):
        from ._conformer_compiled import chi_dof_indices

        indices = chi_dof_indices(
            types.cpu(),
            offsets.cpu(),
            conformers.cpu(),
            atoms.cpu(),
            kinematic_index.cpu(),
            corrections.shape[1],
        ).to(dofs.device)
        active = indices[0] >= 0
        dofs.view(-1)[indices[0, active]] = (
            chi.flatten()[active] - corrections.flatten()[indices[1, active]]
        )
        return
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


def current_conformer_copy_segments(poses, task, considered, types, offsets, selected):
    from ._conformer_compiled import conformer_copy_plan

    device = (
        poses.device if poses.device.type in ("cpu", "cuda") else torch.device("cpu")
    )
    inputs = [
        poses.block_type_ind.long(),
        poses.packed_block_types.n_atoms,
        considered,
        task.global_block_ind_for_considered_block_types,
        types,
        offsets,
        selected,
    ]
    return conformer_copy_plan([v.to(device).contiguous() for v in inputs])


def copy_current_conformer_dofs(
    poses, task, considered, types, offsets, selected, source, target
):
    if selected.numel() == 0:
        return
    sizes, origins, destinations = current_conformer_copy_segments(
        poses, task, considered, types, offsets, selected
    )
    copy_conformer_rows(sizes, origins, destinations, source, target, root_shift=1)


def reset_opth_dofs(types, offsets, selected, chi_atoms, cache, ideal, target, flip):
    if not flip or selected.numel() == 0:
        return
    from ._conformer_compiled import opth_reset_plan

    device = (
        target.device if target.device.type in ("cpu", "cuda") else torch.device("cpu")
    )
    inputs = [
        types,
        offsets,
        selected,
        chi_atoms,
        cache.nhq_downstream_kfo,
        cache.nhq_downstream_count,
    ]
    indices = opth_reset_plan(
        [v.to(device).contiguous() for v in inputs], ideal.shape[1], flip
    ).to(target.device)
    active = indices[0] >= 0
    target[indices[1, active]] = ideal.reshape(-1, 9)[indices[0, active]]
