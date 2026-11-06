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


def mainchain_copy_metadata(poses, task, sampler, considered, types, selected, offsets):
    from ._build_rotamers import _kinforest_device_indices

    pbt = poses.packed_block_types
    fingerprints = pbt.mc_fingerprints
    device = (
        poses.device if poses.device.type in ("cpu", "cuda") else torch.device("cpu")
    )
    # Current-copy and mainchain-copy share the source-residue prefix plan.
    _, source_offsets, target_offsets = current_conformer_copy_segments(
        poses, task, considered, types, offsets, selected
    )
    residues = task.global_block_ind_for_considered_block_types[considered[selected]]
    metadata = (
        poses.block_type_ind.reshape(-1)[residues].long(),
        source_offsets,
        types[selected],
        target_offsets,
        fingerprints.source_fingerprint,
        fingerprints.source_atom_mapping,
        fingerprints.atom_mapping,
        _kinforest_device_indices(pbt, pbt.device),
    )
    return [v.to(device).contiguous() for v in metadata], fingerprints.sampler_mapping[
        sampler
    ]


def mainchain_row_indices(metadata, sampler, device):
    from ._conformer_compiled import mainchain_row_indices as native_indices

    indices = native_indices(metadata, sampler).to(device)
    active = indices[0] >= 0
    return indices[:, active]


def copy_mainchain_dofs(
    source_types,
    source_offsets,
    target_types,
    target_offsets,
    fingerprints,
    source_atoms,
    target_atoms,
    kinematic_index,
    source,
    target,
    sampler,
):
    from ._conformer_compiled import copy_mainchain_dofs as native_copy

    if target.device.type not in ("cpu", "cuda"):
        src, dst = mainchain_row_indices(
            [
                source_types,
                source_offsets,
                target_types,
                target_offsets,
                fingerprints,
                source_atoms,
                target_atoms,
                kinematic_index,
            ],
            sampler,
            target.device,
        )
        measured = source[src]
        target[dst] = torch.where(torch.isfinite(measured), measured, target[dst])
        return
    operation = (
        _CopyMainchainDofs.apply
        if torch.is_grad_enabled() and (source.requires_grad or target.requires_grad)
        else native_copy
    )
    operation(
        source_types,
        source_offsets,
        target_types,
        target_offsets,
        fingerprints,
        source_atoms,
        target_atoms,
        kinematic_index,
        source,
        target,
        sampler,
    )


class _CopyMainchainDofs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        from ._conformer_compiled import copy_mainchain_dofs, mainchain_dof_indices

        indices = mainchain_dof_indices(list(args[:10]), args[10])
        ctx.save_for_backward(indices)
        ctx.source_shape = args[8].shape
        ctx.source_size = args[8].numel()
        ctx.mark_dirty(args[9])
        copy_mainchain_dofs(*args)
        return args[9]

    @staticmethod
    def backward(ctx, gradient):
        (indices,) = ctx.saved_tensors
        active = indices[0] >= 0
        source, target = indices[:, active]
        result = [None] * 11
        if ctx.needs_input_grad[8]:
            result[8] = (
                gradient.new_zeros(ctx.source_size)
                .index_add(0, source, gradient.flatten()[target])
                .reshape(ctx.source_shape)
            )
        if ctx.needs_input_grad[9]:
            result[9] = gradient.flatten().index_fill(0, target, 0).reshape_as(gradient)
        return tuple(result)


def correct_chi_dofs(
    types, offsets, conformers, atoms, chi, coords, chi_atoms, kinematic_index, dofs
):
    from ._conformer_compiled import correct_chi_dofs as native_correct

    if dofs.device.type not in ("cpu", "cuda"):
        from tmol.numeric import coord_dihedrals
        from ._conformer_compiled import chi_correction_indices

        metadata = [types, offsets, conformers, atoms, chi_atoms, kinematic_index]
        indices = chi_correction_indices([v.cpu() for v in metadata]).to(dofs.device)
        active = indices[0] >= 0
        destination = indices[0, active]
        xyz = coords[indices[1:, active]].double()
        measured = coord_dihedrals(*xyz, output_dtype=chi.dtype)
        delta = chi.flatten()[active].double() - measured.double()
        delta = (delta + torch.pi) % (2 * torch.pi) - torch.pi
        dofs.view(-1).index_add_(0, destination, delta.to(dofs.dtype))
        return
    operation = (
        _CorrectChiDofs.apply
        if torch.is_grad_enabled() and any(v.requires_grad for v in (chi, coords, dofs))
        else native_correct
    )
    operation(
        types, offsets, conformers, atoms, chi, coords, chi_atoms, kinematic_index, dofs
    )


class _CorrectChiDofs(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        types,
        offsets,
        conformers,
        atoms,
        chi,
        coords,
        chi_atoms,
        kinematic_index,
        dofs,
    ):
        from ._conformer_compiled import correct_chi_dofs, chi_correction_indices

        indices = chi_correction_indices(
            [types, offsets, conformers, atoms, chi_atoms, kinematic_index]
        )
        ctx.save_for_backward(coords, indices)
        ctx.chi_shape = chi.shape
        ctx.chi_size = chi.numel()
        ctx.mark_dirty(dofs)
        correct_chi_dofs(
            types,
            offsets,
            conformers,
            atoms,
            chi,
            coords,
            chi_atoms,
            kinematic_index,
            dofs,
        )
        return dofs

    @staticmethod
    def backward(ctx, gradient):
        coords, indices = ctx.saved_tensors
        result = [None] * 9
        if ctx.needs_input_grad[4]:
            active = indices[0] >= 0
            result[4] = (
                gradient.new_zeros(ctx.chi_size)
                .masked_scatter(active, gradient.flatten()[indices[0, active]])
                .reshape(ctx.chi_shape)
            )
        if ctx.needs_input_grad[5]:
            result[5] = _ChiCorrectionCoordinateGradient.apply(
                coords, indices, gradient.contiguous()
            )
        if ctx.needs_input_grad[8]:
            result[8] = gradient
        return tuple(result)


class _ChiCorrectionCoordinateGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coords, indices, weights):
        from ._conformer_compiled import chi_correction_derivatives

        ctx.save_for_backward(coords, indices, weights)
        return chi_correction_derivatives(indices, coords, weights, coords, False)[0]

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, direction):
        from ._conformer_compiled import chi_correction_derivatives

        coords, indices, weights = ctx.saved_tensors
        grad_coords, grad_weights = chi_correction_derivatives(
            indices, coords, weights, direction.contiguous(), True
        )
        return grad_coords, None, grad_weights
