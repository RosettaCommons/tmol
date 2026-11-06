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
from .rotamer_set import RotamerSet
from tmol_core.pack.chi import chi4_by_defining_atom, ring_chi_phi_c_corrections
from .fingerprint import prepare_atom_copy_plan
from ._native import (
    copy_conformer_rows,
    copy_conformer_dofs,
    assign_conformer_chi,
    correct_conformer_chi,
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
        if fields == {"copy_input_coordinates"}:
            if payload["copy_input_coordinates"] is not True:
                raise ValueError("input-coordinate payload requires a true copy marker")
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
    """Build task-selected coordinates and retain pose, residue and atom identity.

    Assembly is eager: discrete sampling determines coordinate array shapes.
    Samplers must be registered explicitly, including any desired fallback.
    """
    if isinstance(poses.coords, jax.core.Tracer):
        raise TypeError("conformer assembly requires eager coordinates")
    if isinstance(task, PackerTask):
        task = SetPackerTask.from_packer_task(task)
    task._validate_pose(poses)
    pbt, device = poses.packed_block_types, poses.device
    if chem_db is not None and chem_db is not pbt.chem_db:
        raise ValueError("conformer chemistry must match the pose parameter context")
    packed = prepare_rotamer_kinematics(pbt)
    samplers, samples, choices, destinations, groups = _sample_conformers(poses, task)
    pose_of, block_of = np.asarray(task.cons_bt_pose), np.asarray(task.cons_bt_block)
    shape = poses.block_type_ind.shape
    host_choices = np.asarray(choices)
    types = np.asarray(task.cons_bt_block_type)[host_choices]
    n_atoms_by_type = np.asarray(
        [block.n_atoms for block in pbt.active_block_types], np.int32
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

    with jax.default_device(device):
        dofs = jnp.concatenate(
            (
                jnp.zeros((1, 9), poses.coords.dtype),
                packed.dofs_ideal.reshape(-1, 9)[integer(layout.ideal_indices)].astype(
                    poses.coords.dtype
                ),
            )
        )
    source_types = np.asarray(poses.block_type_ind).ravel()
    real = source_types >= 0
    coordinate_offsets = (
        np.asarray(poses.block_coord_offset)
        + np.arange(poses.n_poses)[:, None] * poses.max_n_atoms_per_pose
    ).ravel()
    coordinate_offsets_device = integer(coordinate_offsets)
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
    source_dofs = source_geometry.inverse(poses.coords)
    source_offsets = np.zeros(len(source_types), np.int32)
    source_offsets[real] = source_layout.offsets
    source_offsets, coordinate_offsets = (
        integer(source_offsets),
        coordinate_offsets_device,
    )
    target_offsets, atom_counts = integer(layout.offsets), integer(layout.n_atoms)
    source_residues = task.global_block_ind_for_considered_block_types[choices]

    def copy_rows(target, source, selected, offsets, root_shift):
        sizes = atom_counts[selected]
        with jax.default_device(device):
            packed_offsets = jnp.concatenate(
                (jnp.zeros(1, jnp.int32), jnp.cumsum(sizes))
            )
        return copy_conformer_rows(
            target,
            source,
            offsets[source_residues[selected]],
            target_offsets[selected],
            packed_offsets,
            n_rows=int(packed_offsets[-1]),
            root_shift=root_shift,
        )

    type_indices = integer(types)
    chi_samples = any("chi_for_rotamers" in sample[2] for sample in samples)
    if chi_samples:
        fingerprints = prepare_atom_copy_plan(pbt, samplers)
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
    copied = []
    for sampler, sample, selected in zip(samplers, samples, destinations):
        metadata = sample[2]
        if metadata.get("copy_input_coordinates", False):
            original = poses.block_type_ind.reshape(-1)[source_residues[selected]]
            if not bool(jnp.all(original == integer(types)[selected])):
                raise ValueError(
                    "input conformers must retain their original chemical identity"
                )
            dofs = copy_rows(dofs, source_dofs, selected, source_offsets, 1)
            copied.append(selected)
        elif "chi_for_rotamers" in metadata:
            if len(selected) and fingerprints.atom_mapping.shape[1]:
                source = source_residues[selected]
                dofs = copy_conformer_dofs(
                    dofs,
                    source_dofs,
                    poses.block_type_ind.reshape(-1)[source],
                    source_offsets[source],
                    type_indices[selected],
                    target_offsets[selected],
                    fingerprints.source_fingerprint,
                    fingerprints.source_atom_mapping,
                    fingerprints.atom_mapping,
                    kinematic_index,
                    sampler=fingerprints.sampler_mapping[id(sampler)],
                )
            dofs = assign_conformer_chi(
                dofs,
                metadata["chi_for_rotamers"],
                type_indices,
                target_offsets,
                selected,
                metadata["chi_defining_atom_for_rotamer"],
                kinematic_index,
                ring_corrections,
            )
        else:
            raise NotImplementedError(
                f"unsupported conformer payload from {sampler.sampler_name()}"
            )
    if chi_samples:
        trial = geometry.forward(dofs)
        for sample, selected in zip(samples, destinations):
            metadata = sample[2]
            if "chi_for_rotamers" in metadata:
                dofs = correct_conformer_chi(
                    dofs,
                    metadata["chi_for_rotamers"],
                    trial,
                    type_indices,
                    target_offsets,
                    selected,
                    metadata["chi_defining_atom_for_rotamer"],
                    chi_atoms,
                    kinematic_index,
                )
    coordinates = geometry.forward(dofs)
    if copied:
        coordinates = copy_rows(
            coordinates,
            poses.coords.reshape(-1, 3),
            jnp.concatenate(copied),
            coordinate_offsets,
            0,
        )
    origins = prepare_conformer_origins(shape, pose_of, block_of, host_choices)
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
        atom_poses = jnp.repeat(rot_pose, atom_counts, total_repeat_length=layout.total)
    return poses, RotamerSet(
        counts_pose,
        offsets_pose,
        counts_block,
        offsets_block,
        rot_pose,
        type_indices,
        rot_block,
        target_offsets,
        coordinates,
        integer(correlation_indices(shape, groups, origins[2])),
        first_type,
        atom_poses,
        int(origins[0].max(initial=0)),
        groups,
    )
