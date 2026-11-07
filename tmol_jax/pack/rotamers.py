"""Eager conformer assembly using common topology plans and native kernels."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from tmol_core.pack.conformer_merge import (
    check_task_sample_budget,
    prepare_conformer_origins,
)
from tmol_core.pack.correlation import (
    correlated_groups_for_samples,
    correlation_indices,
)
from .task import PackerTask, SetPackerTask
from .kinematics import prepare_rotamer_kinematics
from .merge import merge_conformer_samples
from .conformer_geometry import geometry_for_task
from .rotamer_set import RotamerSet, ConformerContext
from tmol_core.pack.chi import chi4_by_defining_atom, ring_chi_phi_c_corrections
from .fingerprint import prepare_atom_copy_plan
from ._native import (
    copy_conformer_rows,
    copy_conformer_dofs,
    assign_conformer_chi,
    correct_conformer_chi,
    current_conformer_copy_plan,
    reset_opth_dofs,
)


def _sample_conformers(poses, task):
    samplers = tuple(task.conformer_samplers)
    if not samplers:
        raise ValueError("At least one conformer sampler is required")
    samples = [None] * len(samplers)
    deferred = [
        i for i, sampler in enumerate(samplers) if sampler.samples_after_other_samplers
    ]
    for i, sampler in enumerate(samplers):
        if i not in deferred:
            samples[i] = sampler.create_samples_for_poses(poses, task)
    built = None
    for sample in samples:
        if sample is not None:
            built = sample[0] if built is None else built + sample[0]
    for i in deferred:
        samples[i] = samplers[i].create_samples_for_poses(
            poses, task, built_rotamer_counts=built
        )
    _validate_payloads(samples, poses.coords)
    host_counts = np.stack([np.asarray(sample[0]) for sample in samples])
    pose_of, block_of = np.asarray(task.cons_bt_pose), np.asarray(task.cons_bt_block)
    shape = poses.block_type_ind.shape
    check_task_sample_budget(
        shape, pose_of, block_of, host_counts, task.chi_sample_budget
    )
    declarations = [
        (i, rows)
        for i, sample in enumerate(samples)
        for rows in sample[2].get("correlated_gbts", ())
        if rows
    ]
    groups = (
        correlated_groups_for_samples(pose_of, block_of, host_counts, declarations)
        if declarations
        else ()
    )
    _, _, choices, _, destinations = merge_conformer_samples(samples)
    return samplers, samples, choices, destinations, groups


def _indices(value, *, device):
    if (
        isinstance(value, np.ndarray)
        and value.size
        and (value.min() < -(1 << 31) or value.max() >= 1 << 31)
    ):
        raise OverflowError("conformer indices exceed the native int32 capacity")
    return jax.device_put(np.asarray(value, dtype=np.int32), device)


def _validate_payloads(samples, coordinates):
    for counts, choices, payload in samples:
        fields = set(payload) - {"correlated_gbts"}
        if "opth_dof_initialization" in fields:
            if payload["opth_dof_initialization"] is not True:
                raise ValueError("OptH DOF initialization requires a true marker")
            fields.remove("opth_dof_initialization")
        if fields == {"copy_input_coordinates"}:
            if payload["copy_input_coordinates"] is not True:
                raise ValueError("input-coordinate payload requires a true copy marker")
        elif fields == {"conformer_coordinates"}:
            value = payload["conformer_coordinates"]
            if (
                value.ndim != 2
                or value.shape[1] != 3
                or value.dtype != coordinates.dtype
                or value.device != coordinates.device
            ):
                raise ValueError(
                    "conformer coordinates must match pose dtype/device and xyz shape"
                )
        elif fields == {"chi_for_rotamers", "chi_defining_atom_for_rotamer"}:
            chi, atoms = (
                payload["chi_for_rotamers"],
                payload["chi_defining_atom_for_rotamer"],
            )
            if (
                chi.ndim != 2
                or chi.shape != atoms.shape
                or chi.shape[0] != choices.size
                or chi.dtype != coordinates.dtype
                or atoms.dtype != jnp.int32
                or chi.device != coordinates.device
                or atoms.device != coordinates.device
            ):
                raise ValueError(
                    "chi samples require matching pose dtype/device and int32 atom identities"
                )
        else:
            raise NotImplementedError(
                f"unsupported conformer payload fields: {sorted(fields)}"
            )


def build_rotamers(poses, task, chem_db=None):
    """Build task-selected coordinates using a bounded reusable execution plan.

    Sampling remains eager because it determines output cardinality. The plan
    contains topology and array operations; coordinates and sampled chi are
    fresh inputs on every call, including when conformer identities are unchanged.
    """
    if isinstance(poses.coords, jax.core.Tracer):
        raise TypeError("conformer assembly requires eager coordinates")
    if isinstance(task, PackerTask):
        task = SetPackerTask.from_packer_task(task)
    task._validate_pose(poses)
    if chem_db is not None and chem_db is not poses.packed_block_types.chem_db:
        raise ValueError("conformer chemistry must match the pose parameter context")
    samplers, samples, choices, destinations, groups = _sample_conformers(poses, task)
    copy_input = tuple(s[2].get("copy_input_coordinates", False) for s in samples)
    direct = tuple("conformer_coordinates" in s[2] for s in samples)
    atom_counts = np.asarray(poses.packed_block_types.n_atoms)
    for sample, enabled in zip(samples, direct):
        if enabled:
            kinds = np.asarray(task.cons_bt_block_type)[np.asarray(sample[1])]
            if sample[2]["conformer_coordinates"].shape[0] != int(
                atom_counts[kinds].sum()
            ):
                raise ValueError(
                    "conformer coordinate rows must cover every selected atom"
                )
    opth = tuple(s[2].get("opth_dof_initialization", False) for s in samples)
    opth_options = tuple(
        (s.flip_NHQ, task.chi_sample_budget or s.budget) if enabled else None
        for s, enabled in zip(samplers, opth)
    )
    key = (
        poses.coords.dtype,
        tuple(map(id, samplers)),
        copy_input,
        direct,
        opth_options,
        groups,
    )
    identities = tuple(
        np.asarray(value)
        for value in (
            choices,
            task.cons_bt_pose,
            task.cons_bt_block,
            task.cons_bt_block_type,
            task.global_block_ind_for_considered_block_types,
            *destinations,
        )
    )
    previous = getattr(task, "_assembly_execution", None)
    if (
        previous is None
        or previous[0] != key
        or any(
            not np.array_equal(old, new)
            for old, new in zip(previous[1], identities, strict=True)
        )
    ):
        execute, fields = _assembly_plan(
            poses, task, samplers, copy_input, direct, opth_options, identities, groups
        )
        previous = (
            key,
            tuple(np.array(a, copy=True) for a in identities),
            execute,
            fields,
        )
        task._assembly_execution = previous
    payloads = tuple(
        (
            ()
            if copy
            else (
                sample[2]["conformer_coordinates"]
                if coordinates
                else (
                    sample[2]["chi_for_rotamers"],
                    sample[2]["chi_defining_atom_for_rotamer"],
                )
            )
        )
        for copy, coordinates, sample in zip(copy_input, direct, samples)
    )
    # Empty assembly can be constant-folded, removing all device-bearing inputs.
    # Keep its output on the pose device even when another backend is default.
    with jax.default_device(poses.device):
        coordinates = previous[2](poses.coords, payloads)
    return poses, RotamerSet(coords=coordinates, **previous[3])


def _copy_coordinate_payloads(coordinates, payloads, plans):
    for payload, plan in zip(payloads, plans):
        if plan is not None:
            source, target, prefix, rows = plan
            coordinates = copy_conformer_rows(
                coordinates,
                payload,
                source,
                target,
                prefix,
                n_rows=rows,
                root_shift=0,
            )
    return coordinates


def _coordinate_row_plan(enabled, layout, selected, integer):
    if not enabled:
        return None
    prefix = np.r_[0, np.cumsum(layout.n_atoms[selected], dtype=np.int64)]
    return (
        integer(prefix[:-1]),
        integer(layout.offsets[selected]),
        integer(prefix),
        int(prefix[-1]),
    )


def _chi_copy_plan(
    enabled,
    selected,
    source_residues,
    source_types,
    source_offsets,
    types,
    layout,
    integer,
):
    if not enabled:
        return None
    source = source_residues[selected]
    return (
        integer(source_types[source]),
        integer(source_offsets[source]),
        integer(types[selected]),
        integer(layout.offsets[selected]),
    )


def _assembly_plan(
    poses, task, samplers, copy_input, direct, opth_options, identities, groups
):
    """Bind immutable native layouts and compile one coordinate execution graph."""
    pbt, device = poses.packed_block_types, poses.device
    packed = prepare_rotamer_kinematics(pbt)
    choices, pose_of, block_of, considered, global_blocks, *destinations = identities
    types = considered[choices]
    shape = poses.block_type_ind.shape
    n_atoms_by_type = np.asarray(
        [bt.n_atoms for bt in pbt.active_block_types], np.int32
    )
    layout, geometry = geometry_for_task(
        task,
        "_target_conformer_geometry",
        packed,
        pbt.max_n_atoms,
        n_atoms_by_type,
        types,
        device,
    )
    integer = partial(_indices, device=device)
    source_types = np.asarray(poses.block_type_ind).ravel()
    real = source_types >= 0
    coordinate_offsets = (
        np.asarray(poses.block_coord_offset)
        + np.arange(poses.n_poses)[:, None] * poses.max_n_atoms_per_pose
    ).ravel()
    source_layout, source_geometry = geometry_for_task(
        task,
        "_source_conformer_geometry",
        packed,
        pbt.max_n_atoms,
        n_atoms_by_type,
        source_types[real],
        device,
        coordinate_offsets[real],
    )
    source_offsets = np.zeros(len(source_types), np.int32)
    source_offsets[real] = source_layout.offsets
    source_residues = global_blocks[choices]
    type_indices, target_offsets = integer(types), integer(layout.offsets)
    with jax.default_device(device):
        ideal = jnp.concatenate(
            (
                jnp.zeros((1, 9), poses.coords.dtype),
                packed.dofs_ideal.reshape(-1, 9)[integer(layout.ideal_indices)].astype(
                    poses.coords.dtype
                ),
            )
        )

    def row_plan(selected, offsets):
        # Cardinality is already known from the shared host layout. Prepare its
        # row-copy offsets once, avoiding device synchronization during execution.
        prefix = np.concatenate(
            ([0], np.cumsum(layout.n_atoms[selected], dtype=np.int64))
        )
        return (
            integer(offsets[source_residues[selected]]),
            integer(layout.offsets[selected]),
            integer(prefix),
            int(prefix[-1]),
        )

    copy_plans, chi_plans, copied, opth_metadata = [], [], [], []
    coordinate_plans = []
    for sampler, copy, coordinates, opt, selected in zip(
        samplers, copy_input, direct, opth_options, destinations
    ):
        coordinate_plans.append(
            _coordinate_row_plan(coordinates, layout, selected, integer)
        )
        chi_plans.append(
            _chi_copy_plan(
                not (copy or coordinates),
                selected,
                source_residues,
                source_types,
                source_offsets,
                types,
                layout,
                integer,
            )
        )
        if copy or opt is not None:
            plan = current_conformer_copy_plan(
                integer(source_types.reshape(shape)),
                integer(n_atoms_by_type),
                integer(choices),
                integer(global_blocks),
                type_indices,
                target_offsets,
                integer(selected),
            )
            copy_plans.append((*plan, int(layout.n_atoms[selected].sum())))
        else:
            copy_plans.append(None)
        opth_metadata.append(
            sampler._metadata(pbt, device, opt[1])[0] if opt is not None else None
        )
        if copy:
            if not np.array_equal(
                source_types[source_residues[selected]], types[selected]
            ):
                raise ValueError(
                    "input conformers must retain their original chemical identity"
                )
            copied.append(selected)
    copied_rows = (
        row_plan(np.concatenate(copied), coordinate_offsets) if copied else None
    )
    selections = tuple(integer(s) for s in destinations)
    chi_samplers = tuple(
        not (copy or coordinates) for copy, coordinates in zip(copy_input, direct)
    )
    has_chi = any(chi_samplers)
    if has_chi:
        fingerprints = prepare_atom_copy_plan(
            pbt, [s for s, chi in zip(samplers, chi_samplers) if chi]
        )
        kinematic_index = integer(packed.kinforest_idx)
        chi_atoms = integer(
            chi4_by_defining_atom(pbt.active_block_types, pbt.max_n_atoms)
        )
        ring_corrections = jax.device_put(
            ring_chi_phi_c_corrections(pbt.active_block_types, pbt.max_n_atoms).astype(
                poses.coords.dtype
            ),
            device,
        )
        sampler_indices = tuple(
            (
                fingerprints.sampler_mapping[id(s)]
                if chi and len(selected) and fingerprints.atom_mapping.shape[1]
                else -1
            )
            for s, chi, selected in zip(samplers, chi_samplers, destinations)
        )

    def arrays(plans):
        return tuple(None if p is None else p[:-1] for p in plans)

    def counts(plans):
        return tuple(None if p is None else p[-1] for p in plans)

    execute = partial(
        _execute_assembly,
        plan=(
            geometry,
            source_geometry,
            ideal,
            type_indices,
            target_offsets,
            selections,
            tuple(chi_plans),
            arrays(copy_plans),
            arrays(coordinate_plans),
            None if copied_rows is None else copied_rows[:-1],
            (
                (
                    fingerprints.source_fingerprint,
                    fingerprints.source_atom_mapping,
                    fingerprints.atom_mapping,
                )
                if has_chi
                else ()
            ),
            kinematic_index if has_chi else None,
            chi_atoms if has_chi else None,
            ring_corrections if has_chi else None,
            tuple(opth_metadata),
            packed.dofs_ideal,
        ),
        total=layout.total,
        copy_input=copy_input,
        opth_options=tuple(None if opt is None else opt[0] for opt in opth_options),
        sampler_indices=sampler_indices if has_chi else (),
        copy_counts=counts(copy_plans),
        coordinate_counts=counts(coordinate_plans),
        copied_count=None if copied_rows is None else copied_rows[-1],
    )

    origins = prepare_conformer_origins(shape, pose_of, block_of, choices)
    counts_pose, offsets_pose, counts_block, offsets_block, rot_pose, rot_block = map(
        integer, origins
    )
    with jax.default_device(device):
        first_type = (
            jnp.where(
                counts_block > 0,
                type_indices[jnp.minimum(offsets_block, len(types) - 1)],
                -1,
            )
            if len(types)
            else jnp.full(shape, -1, jnp.int32)
        )
        atom_poses = jnp.repeat(
            rot_pose, integer(layout.n_atoms), total_repeat_length=layout.total
        )
    fields = dict(
        n_rots_for_pose=counts_pose,
        rot_offset_for_pose=offsets_pose,
        n_rots_for_block=counts_block,
        rot_offset_for_block=offsets_block,
        pose_for_rot=rot_pose,
        block_type_ind_for_rot=type_indices,
        block_ind_for_rot=rot_block,
        coord_offset_for_rot=target_offsets,
        group_for_block=integer(correlation_indices(shape, groups, origins[2])),
        first_rot_block_type=first_type,
        pose_ind_for_atom=atom_poses,
        max_n_rots_per_pose=int(origins[0].max(initial=0)),
        correlated_groups=groups,
        context=ConformerContext(poses),
    )
    return execute, fields


def _copy_rows(target, source, plan, root_shift):
    source_offsets, destination_offsets, prefix, n_rows = plan
    return copy_conformer_rows(
        target,
        source,
        source_offsets,
        destination_offsets,
        prefix,
        n_rows=n_rows,
        root_shift=root_shift,
    )


@partial(
    jax.jit,
    static_argnames=(
        "total",
        "copy_input",
        "opth_options",
        "sampler_indices",
        "copy_counts",
        "coordinate_counts",
        "copied_count",
    ),
)
def _execute_assembly(
    coords,
    payloads,
    plan,
    *,
    total,
    copy_input,
    opth_options,
    sampler_indices,
    copy_counts,
    coordinate_counts,
    copied_count,
):
    """Execute shared prepared topology with explicit arrays, reusable across tasks."""
    (
        geometry,
        source_geometry,
        ideal,
        type_indices,
        target_offsets,
        selections,
        chi_plans,
        copy_arrays,
        coordinate_arrays,
        copied_arrays,
        fingerprints,
        kinematic_index,
        chi_atoms,
        ring_corrections,
        opth_metadata,
        packed_ideal,
    ) = plan
    copy_plans = tuple(
        None if n is None else (*a, n) for a, n in zip(copy_arrays, copy_counts)
    )
    coordinate_plans = tuple(
        None if n is None else (*a, n)
        for a, n in zip(coordinate_arrays, coordinate_counts)
    )
    copied_rows = None if copied_count is None else (*copied_arrays, copied_count)
    chi_samplers = tuple(
        not (copy or count is not None)
        for copy, count in zip(copy_input, coordinate_counts)
    )
    has_chi = any(chi_samplers)
    dof_selections = tuple(
        (i, selected)
        for i, selected in enumerate(selections)
        if coordinate_counts[i] is None
    )
    if not has_chi:
        coordinates = jnp.zeros((total, 3), coords.dtype)
        if copied_rows is not None:
            coordinates = _copy_rows(coordinates, coords.reshape(-1, 3), copied_rows, 0)
        return _copy_coordinate_payloads(coordinates, payloads, coordinate_plans)
    dofs = ideal
    source_dofs = source_geometry.inverse(coords)
    for i, selected in dof_selections:
        if copy_input[i]:
            dofs = _copy_rows(dofs, source_dofs, copy_plans[i], 1)
        else:
            if opth_options[i] is not None:
                dofs = _copy_rows(dofs, source_dofs, copy_plans[i], 1)
                dofs = reset_opth_dofs(
                    dofs,
                    packed_ideal.astype(coords.dtype),
                    type_indices,
                    target_offsets,
                    selected,
                    payloads[i][1],
                    opth_metadata[i]["nhq_downstream_kfo"],
                    opth_metadata[i]["nhq_downstream_count"],
                    flip=opth_options[i],
                )
            elif len(selected) and fingerprints[2].shape[1]:
                dofs = copy_conformer_dofs(
                    dofs,
                    source_dofs,
                    *chi_plans[i],
                    fingerprints[0],
                    fingerprints[1],
                    fingerprints[2],
                    kinematic_index,
                    sampler=sampler_indices[i],
                )
            angles, defining = payloads[i]
            dofs = assign_conformer_chi(
                dofs,
                angles,
                type_indices,
                target_offsets,
                selected,
                defining,
                kinematic_index,
                ring_corrections,
            )
    trial = geometry.forward(dofs)
    for i, (copy, selected) in enumerate(zip(copy_input, selections)):
        if chi_samplers[i]:
            angles, defining = payloads[i]
            dofs = correct_conformer_chi(
                dofs,
                angles,
                trial,
                type_indices,
                target_offsets,
                selected,
                defining,
                chi_atoms,
                kinematic_index,
            )
    coordinates = geometry.forward(dofs)
    if copied_rows is not None:
        coordinates = _copy_rows(coordinates, coords.reshape(-1, 3), copied_rows, 0)
    return _copy_coordinate_payloads(coordinates, payloads, coordinate_plans)
