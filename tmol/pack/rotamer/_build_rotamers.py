import numpy
import torch

from tmol_core.pack.conformer_layout import (  # noqa: F401 (legacy helper imports)
    KINFOREST_FIELDS,
    prepare_conformer_layout,
    compact_conformer_forest,
    construct_conformer_scans,
    _update_nodes,
    _update_scan_starts,
    _load_from_rotamers,
    _load_from_rotamers_w_offsets,
    _load_rotamer_parents,
)

from typing import Any

from tmol.types._array import NDArray
from tmol.types._torch import Tensor
from tmol.utility.tensor._common_operations import _exclusive_cumsum1d
from tmol.database.chemical import ChemicalDatabase
from tmol.kinematics._datatypes import KinForest
from tmol.chemical import RefinedResidueType
from tmol.pose._packed_block_types import PackedBlockTypes
from tmol.pose._pose_stack import PoseStack
from tmol.pack.rotamer._rotamer_set import RotamerSet
from tmol.pack.rotamer._chi_sampler import ChiSampler
from tmol.pack.rotamer._single_residue_kinforest import (
    _construct_single_residue_kinforest,
    _coalesce_single_residue_kinforests,
)
from tmol.pack.rotamer._mainchain_fingerprint import (
    _annotate_residue_type_with_sampler_fingerprints,
    _find_unique_fingerprints,
)
from tmol.pack._packer_task import SetPackerTask
from tmol_core.pack.chi import (  # noqa: F401
    chi_moves_fourth_atom as _turns_its_fourth_atom,
    chi4_by_defining_atom,
    ring_chi_phi_c_corrections,
)

ConformerSample = tuple[
    Tensor[torch.int32][:],
    Tensor[torch.int32][:],
    dict[str, Any],
]


def _build_chi4_by_defining_atom(pbt):
    if not hasattr(pbt, "_chi4_by_defining_atom"):
        values = chi4_by_defining_atom(pbt.active_block_types, pbt.max_n_atoms)
        object.__setattr__(pbt, "_chi4_by_defining_atom", values)
    return pbt._chi4_by_defining_atom


def _build_ring_chi_phi_c_corrections(pbt):
    if not hasattr(pbt, "_ring_chi_phi_c_corrections"):
        values = ring_chi_phi_c_corrections(pbt.active_block_types, pbt.max_n_atoms)
        values = torch.as_tensor(values, device=pbt.device)
        object.__setattr__(pbt, "_ring_chi_phi_c_corrections", values)
    return pbt._ring_chi_phi_c_corrections


def _kinforest_device_indices(pbt, device):
    """One RTO-to-KFO device table shared by DOF copying and chi correction."""
    if not hasattr(pbt, "_kinforest_device_indices"):
        object.__setattr__(
            pbt,
            "_kinforest_device_indices",
            torch.as_tensor(
                pbt.rotamer_kinforest.kinforest_idx, dtype=torch.int64, device=device
            ),
        )
    return pbt._kinforest_device_indices


def _chi4_and_kfo_device_tables(pbt, device):
    """Device copies of the per-atom chi table and the RTO -> KFO atom map."""
    if not hasattr(pbt, "_chi4_kfo_device_tables"):
        tables = (
            torch.as_tensor(_build_chi4_by_defining_atom(pbt), device=device).to(
                torch.int64
            ),
            _kinforest_device_indices(pbt, device),
        )
        object.__setattr__(pbt, "_chi4_kfo_device_tables", tables)
    return pbt._chi4_kfo_device_tables


def _correct_phi_c_from_measured_chi(
    pbt,
    conformer_samples,
    new_ind_for_sampler_rotamer,
    block_type_ind_for_conformer_torch,
    n_atoms_offset_for_conformer_torch,
    conformer_kinforest,
    nodes,
    scans,
    gens,
    conf_dofs_kto,
):
    """Make every chi dihedral come out at the value its sampler asked for.

    The phi_c written by assign_chi_dofs_from_samples is offset from the chi
    dihedral by an amount that depends on the rotamer's own geometry, so it is
    measured rather than derived:
      1. Does a trial forward pass with the current DOFs.
      2. Measures each chi's actual dihedral from the trial coords.
      3. Adds (intended - measured) to conf_dofs_kto[atom_kto, 3] so the
         final forward pass produces the intended dihedral.

    A dihedral is invariant under rotation of its own upstream frame, so one
    trial pass corrects every chi at once.
    """
    # trial forward pass to get coords in RTO
    n_at_total = conformer_kinforest.id.numel() - 1
    # The correction mutates DOFs after the trial pass. Preserve the values
    # retained by forward-kinematics autograd when construction is differentiable.
    trial_dofs = (
        conf_dofs_kto.clone()
        if torch.is_grad_enabled() and conf_dofs_kto.requires_grad
        else conf_dofs_kto
    )
    trial_coords_rto = _calculate_rotamer_coords(
        pbt, n_at_total, conformer_kinforest, nodes, scans, gens, trial_dofs
    )

    chi4_by_atom, kfidx = _chi4_and_kfo_device_tables(pbt, conf_dofs_kto.device)

    for i, sample_data in enumerate(conformer_samples):
        sample_dict = sample_data[2]
        if "chi_for_rotamers" not in sample_dict:
            continue
        chi_intended = sample_dict["chi_for_rotamers"]  # (n_samp_rots, max_n_chi)
        cda_rto = sample_dict["chi_defining_atom_for_rotamer"]
        if chi_intended.shape[0] == 0:
            continue

        from ._conformer_execution import correct_chi_dofs

        correct_chi_dofs(
            block_type_ind_for_conformer_torch,
            n_atoms_offset_for_conformer_torch,
            new_ind_for_sampler_rotamer[i],
            cda_rto,
            chi_intended,
            trial_coords_rto,
            chi4_by_atom,
            kfidx,
            conf_dofs_kto,
        )


def _exc_cumsum_from_inc_cumsum(cumsum):
    return numpy.concatenate((numpy.zeros(1, dtype=numpy.int64), cumsum[:-1]))


# from tmol.system.restype import RefinedResidueType


# residue type annotations:
#  - icoors
#  - parent index (i.e. tree)
#  - list of atoms whose icoors are controlled by dunbrack
#  - bonds that are part of named torsions
#  - mainchain atoms
#  - rottable set for this RT
#
# condensed residue type annotations:
#  - (all data in residue-type annotations)
#
#
# NOTE: sidechain samplers should be listed in the residue-level packer task
# NOTE: all samplers should be given the same starting data and then a mask
#       to indicate which residues they should operate on


def _annotate_restype(
    restype: RefinedResidueType,
    samplers: tuple[ChiSampler, ...],
    chem_db: ChemicalDatabase,
) -> None:
    _construct_single_residue_kinforest(restype)
    _annotate_residue_type_with_sampler_fingerprints(restype, samplers, chem_db)


def _annotate_packed_block_types(pbt: PackedBlockTypes) -> None:
    _coalesce_single_residue_kinforests(pbt)
    _find_unique_fingerprints(pbt)


def _annotate_everything(
    chem_db: ChemicalDatabase,
    samplers: tuple[ChiSampler, ...],
    pbt: PackedBlockTypes,
) -> None:
    """Refresh task ownership while reusing each annotation's scoped cache."""
    for sampler in samplers:
        for rt in pbt.active_block_types:
            sampler.annotate_residue_type(rt)
        sampler.annotate_packed_block_types(pbt)

    for rt in pbt.active_block_types:
        _annotate_restype(rt, samplers, chem_db)
    _annotate_packed_block_types(pbt)


def _construct_scans_for_conformers(
    pbt, block_type_ind_for_conf, n_atoms_for_conf, n_atoms_offset_for_conf
):
    return construct_conformer_scans(
        pbt.rotamer_kinforest, block_type_ind_for_conf, n_atoms_offset_for_conf
    )


def _construct_kinforest_for_conformers(
    pbt,
    conf_block_type_ind,
    n_atoms_total,
    n_atoms_for_conf,
    block_offset_for_conf,
    device,
):
    compact = compact_conformer_forest(
        pbt.rotamer_kinforest,
        pbt.max_n_atoms,
        conf_block_type_ind,
        n_atoms_total,
        n_atoms_for_conf.cpu().numpy(),
        block_offset_for_conf,
    )
    packed = torch.as_tensor(compact, device=device)
    return KinForest(**dict(zip(KINFOREST_FIELDS, packed.unbind(0))))


def _measure_dofs_from_orig_coords(
    coords: Tensor[torch.float32][:, :, :], kinforest: KinForest
) -> Tensor[torch.float32][:, 9]:
    from tmol.kinematics.compiled._compiled_ops import _inverse_kin

    kinforest_coords = coords.view(-1, 3)[kinforest.id.to(torch.int64)]
    kinforest_coords[0, :] = 0  # reset root

    dofs_orig = _inverse_kin(
        kinforest_coords,
        kinforest.parent,
        kinforest.frame_x,
        kinforest.frame_y,
        kinforest.frame_z,
        kinforest.doftype,
    )
    return dofs_orig


def _measure_pose_dofs(
    poses: PoseStack,
) -> tuple[KinForest, Tensor[torch.float32][:, 9]]:
    """Measure the internal coordinates of every real residue in a pose stack.

    Args:
        poses: Poses providing residue topology and Cartesian coordinates.

    Returns:
        A residue-local kinematic forest and its measured degrees of freedom.
    """
    # measure the DOFs for the original residues
    # -- first build kinforests for the original residues,
    # -- then call measure_dofs_from_orig_coords

    pbt = poses.packed_block_types
    pbti = poses.block_type_ind.view(-1)
    real_poses_blocks = pbti != -1
    orig_res_block_type_ind = pbti[real_poses_blocks]

    # coordinate layout: n-poses x max-n-atoms-per-pose x 3
    # offsets provided by the pose stack
    n_poses = poses.coords.shape[0]
    max_n_atoms_per_pose = poses.max_n_pose_atoms
    per_pose_offset = max_n_atoms_per_pose * torch.arange(
        n_poses, dtype=torch.int64, device=poses.device
    ).unsqueeze(1)
    orig_atom_offset_for_poses_blocks = (
        (poses.block_coord_offset.to(torch.int64) + per_pose_offset)
        .flatten()[real_poses_blocks]
        .cpu()
        .numpy()
    )

    n_atoms_for_orig = pbt.n_atoms[orig_res_block_type_ind.to(torch.int64)]
    n_orig_atoms_total = int(n_atoms_for_orig.sum())

    orig_kinforest = _construct_kinforest_for_conformers(
        poses.packed_block_types,
        orig_res_block_type_ind.cpu().numpy(),
        n_orig_atoms_total,
        n_atoms_for_orig,
        orig_atom_offset_for_poses_blocks,
        poses.device,
    )

    # orig_dofs returned in kinforest order
    return orig_kinforest, _measure_dofs_from_orig_coords(poses.coords, orig_kinforest)


def _merge_conformer_samples(
    conformer_samples: list[ConformerSample],
) -> tuple[
    Tensor[torch.int32][:],
    Tensor[torch.int64][:],
    Tensor[torch.int32][:],
    list[Tensor[torch.bool][:]],
    list[Tensor[torch.int64][:]],
]:
    """Merge sampler rows in considered-type, sampler, then source-row order.

    Each sampler supplies a count per considered type and its source rows in
    ascending considered-type order. Prefix offsets locate each row directly;
    the returned destinations map original sampler rows into the merged arrays.
    """
    if not conformer_samples:
        raise ValueError("At least one conformer sampler is required")
    device = conformer_samples[0][0].device
    count_dtype = conformer_samples[0][0].dtype
    index_dtype = conformer_samples[0][1].dtype
    for counts, indices, *_ in conformer_samples:
        if counts.device != device or indices.device != device:
            raise ValueError("Conformer samples must share a device")
        if counts.dtype not in (torch.int32, torch.int64) or indices.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("Conformer sample counts and indices must be integers")
        count_dtype = torch.promote_types(count_dtype, counts.dtype)
        index_dtype = torch.promote_types(index_dtype, indices.dtype)

    n_conformers = sum(sample[1].numel() for sample in conformer_samples)
    if (
        max(n_conformers, conformer_samples[0][0].numel())
        > torch.iinfo(torch.int32).max
    ):
        raise ValueError("Conformer sample counts exceed native 32-bit indexing")
    counts = torch.stack([sample[0].to(torch.int64) for sample in conformer_samples])
    source_ends = counts.cumsum(dim=1)
    expected_types = torch.arange(counts.shape[1], device=device)
    valid = []
    for i, sample in enumerate(conformer_samples):
        indices = sample[1]
        if indices.numel() == 0:
            valid.append(counts.new_ones((), dtype=torch.bool))
            continue
        # Clamp both ends while counts are still unvalidated. Invalid source
        # counts must produce a diagnostic, not an out-of-bounds device read.
        starts = (source_ends[i] - counts[i]).clamp(0, indices.numel() - 1)
        ends = (source_ends[i] - 1).clamp(0, indices.numel() - 1)
        boundaries_match = (indices[starts] == expected_types) & (
            indices[ends] == expected_types
        )
        # Monotonic rows with matching interval endpoints must match throughout.
        valid.append(
            ((counts[i] == 0) | boundaries_match).all()
            & (indices[1:] >= indices[:-1]).all()
        )
    totals = counts.sum(dim=1)
    minima = counts.amin(dim=1) if counts.shape[1] else totals
    maxima = counts.amax(dim=1) if counts.shape[1] else totals
    summary = torch.stack((totals, minima, maxima, torch.stack(valid)), dim=1).tolist()
    for sample, (total, minimum, maximum, ordered) in zip(conformer_samples, summary):
        available = sample[1].numel()
        # A matching sum alone can wrap even when every int64 count is positive.
        if minimum < 0 or maximum > available or total != available:
            raise ValueError("Conformer sample counts do not match the source rows")
        if not ordered:
            raise ValueError("Conformer source rows do not match ordered sample counts")

    merged_counts = counts.sum(dim=0)
    merged_offsets = merged_counts.cumsum(dim=0) - merged_counts
    # Destination start minus source start, for each sampler/considered type.
    # Inclusive scans cancel the current cell in the two exclusive prefixes.
    destination_offsets = merged_offsets + counts.cumsum(dim=0) - source_ends
    sampler_for_conformer = torch.empty(n_conformers, dtype=torch.int64, device=device)
    gbt_for_conformer = torch.empty(n_conformers, dtype=index_dtype, device=device)
    destinations = []
    for i, sample in enumerate(conformer_samples):
        indices = sample[1].to(torch.int64)
        destination = destination_offsets[i, indices] + torch.arange(
            indices.numel(), dtype=torch.int64, device=device
        )
        sampler_for_conformer[destination] = i
        gbt_for_conformer[destination] = indices.to(index_dtype)
        destinations.append(destination)

    return (
        merged_counts.to(count_dtype),
        sampler_for_conformer,
        gbt_for_conformer,
        [sampler_for_conformer == i for i in range(len(conformer_samples))],
        destinations,
    )


def _calculate_rotamer_coords(
    pbt: PackedBlockTypes,
    n_atoms_total: int,
    rot_kinforest: KinForest,
    nodes: NDArray[numpy.int32][:],
    scans: NDArray[numpy.int32][:],
    gens: NDArray[numpy.int32][:],
    rot_dofs_kto: Tensor[torch.float32][:, 9],
) -> Tensor[torch.float32][:, 3]:
    """Fold conformer DOFs and restore residue-type atom ordering.

    Args:
        pbt: Packed block types defining the target device.
        n_atoms_total: Number of real atoms across all conformers.
        rot_kinforest: Coalesced conformer kinematic forest.
        nodes: Kinematic nodes in generation order.
        scans: Segmented-scan starts.
        gens: Generation boundaries.
        rot_dofs_kto: Conformer DOFs in kinematic-tree order.

    Returns:
        Rotamer coordinates shaped ``[n_atoms_total, 3]`` in residue-type order.
    """
    from tmol.kinematics.compiled._compiled_ops import _forward_only_op

    def _p(t):
        return torch.nn.Parameter(t, requires_grad=False)

    def _t(t):
        return torch.tensor(t, dtype=torch.int32, device=pbt.device)

    def _tcpu(t):
        return torch.tensor(t, dtype=torch.int32, device=torch.device("cpu"))

    kinforest_stack = _p(
        torch.stack(
            [
                rot_kinforest.id,
                rot_kinforest.doftype,
                rot_kinforest.parent,
                rot_kinforest.frame_x,
                rot_kinforest.frame_y,
                rot_kinforest.frame_z,
            ],
            dim=1,
        ).to(pbt.device)
    )

    if torch.is_grad_enabled() and rot_dofs_kto.requires_grad:
        from tmol_core.kinematics.scan_ordering import scan_ordering_from_forward
        from tmol.kinematics.compiled._compiled_ops import _forward_kin_op

        backward = scan_ordering_from_forward(
            rot_kinforest.parent.cpu().numpy(), nodes, scans, gens
        ).backward
        new_coords_kto = _forward_kin_op(
            rot_dofs_kto,
            _t(nodes),
            _t(scans),
            _tcpu(gens),
            _t(backward.nodes),
            _t(backward.scans),
            _tcpu(backward.gens),
            kinforest_stack,
        )
    else:
        new_coords_kto = _forward_only_op(
            rot_dofs_kto, _t(nodes), _t(scans), _tcpu(gens), kinforest_stack
        )

    new_coords_rto = torch.zeros(
        (n_atoms_total, 3), dtype=torch.float32, device=pbt.device
    )

    new_coords_rto[rot_kinforest.id[1:].to(torch.int64)] = new_coords_kto[1:]
    return new_coords_rto


def _get_rotamer_origin_data(
    task: SetPackerTask, gbt_for_rot: Tensor[torch.int32][:]
) -> tuple[
    Tensor[torch.int64][:],
    Tensor[torch.int64][:],
    Tensor[torch.int64][:, :],
    Tensor[torch.int64][:, :],
    Tensor[torch.int64][:],
    Tensor[torch.int32][:],
]:
    n_poses = task.per_block_orig_block_type.shape[0]
    pose_for_gbt = task.cons_bt_pose.to(torch.int32)

    block_ind_for_rt = task.cons_bt_block.to(torch.int32)
    max_n_blocks = task.per_block_orig_block_type.shape[1]

    gbt_for_rot64 = gbt_for_rot.to(torch.int64)
    pose_for_rot = pose_for_gbt[gbt_for_rot64].to(torch.int64)
    n_rots_for_pose = torch.bincount(pose_for_rot, minlength=n_poses)
    rot_offset_for_pose = _exclusive_cumsum1d(n_rots_for_pose)
    block_ind_for_rot = block_ind_for_rt[gbt_for_rot64]
    block_ind_for_rt_global = max_n_blocks * pose_for_gbt + block_ind_for_rt
    block_ind_for_rot_global = block_ind_for_rt_global[gbt_for_rot64]
    n_rots_for_block = torch.bincount(
        block_ind_for_rot_global, minlength=n_poses * max_n_blocks
    ).reshape(n_poses, max_n_blocks)
    rot_offset_for_block = _exclusive_cumsum1d(n_rots_for_block.flatten()).reshape(
        n_poses, max_n_blocks
    )

    return (
        n_rots_for_pose,
        rot_offset_for_pose,
        n_rots_for_block,
        rot_offset_for_block,
        pose_for_rot,
        block_ind_for_rot,
    )


def _correlated_groups_for_samples(task, samples):
    from tmol_core.pack.correlation import correlated_groups_for_samples

    declarations = [
        (i, gbts)
        for i, sample in enumerate(samples)
        for gbts in sample[2].get("correlated_gbts", ())
        if gbts
    ]
    if not declarations:
        return ()
    return correlated_groups_for_samples(
        task.cons_bt_pose.cpu().numpy(),
        task.cons_bt_block.cpu().numpy(),
        numpy.stack([sample[0].cpu().numpy() for sample in samples]),
        declarations,
    )


def build_rotamers(
    poses: PoseStack, task: SetPackerTask, chem_db: ChemicalDatabase
) -> tuple[PoseStack, RotamerSet]:
    """Build coordinates and indexing for every task-allowed conformer.

    Chi construction supports first derivatives of sampled angles with fixed
    input coordinates. Coordinate derivatives through inverse kinematics and
    higher derivatives through forward kinematics are not implemented. Current
    conformers copied directly from the input retain coordinate derivatives.

    Args:
        poses: Input poses supplying backbone geometry and current conformers.
        task: Finalized packing choices and conformer samplers.
        chem_db: Chemical definitions used to annotate residue types.

    Returns:
        The input poses and a device-resident rotamer set grouped by pose and
        block.
    """
    # step 1: replace the existing PBT in the Pose w/ a new one in case
    #     there will possibly be new block types in the repacked Pose;
    #     but since PoseStack should not be altered after construction,
    #     what this really means is build an entirely new PoseStack
    # step 2: let the dunbrack library annotate the block types and the packed block types
    # step 3: get the block-type index for the "global block types" (gbts)
    # step 4: use the conformer samplers to decide how many conformers they will build for
    #         each bt/block/pose
    # step 5: merge the conformer samples from different samplers, so that different
    #         conformers for the same bt/block/pose will be in contiguous ranges in the
    #         rotamer set, and keeping track of the mapping back to the original set of
    #         conformer samples
    # step 6: allocate a n_atoms_total x 3 tensor for rotamer coordinates and
    #         create the tensor of offsets
    # step 7: build a kintree for all of the rotamers; initialize the DOFs to ideal
    # step 8: build a kintree for the PoseStack residues
    # step 9: measure the DOFs of the PoseStack residues
    # step 9a: take starting coordinates from residue roots
    # step 9b: take internal dofs from mainchain-fingerprint atoms
    # step 10: ask the samplers to set the DOFs for everything else
    # step 11: refold

    # Step 1
    samplers = tuple(task.conformer_samplers)
    pbt = poses.packed_block_types

    # Step 2
    _annotate_everything(chem_db, samplers, pbt)

    # Step 3
    # the "global block type" (gbt) is the list of all considered block types
    # at all positions in all poses.
    gbt_block_type_ind = task.cons_bt_block_type.cpu().numpy().astype(numpy.int32)

    # Step 4
    # A sampler that fills gaps left by the others must see what they actually
    # built, so it runs last and receives the measured per-block totals.
    deferred = [i for i, s in enumerate(samplers) if s.samples_after_other_samplers]
    conformer_samples = [None] * len(samplers)
    for i, sampler in enumerate(samplers):
        if i not in deferred:
            conformer_samples[i] = sampler.create_samples_for_poses(poses, task)
    if deferred:
        built = None
        for i, sample in enumerate(conformer_samples):
            if sample is None:
                continue
            counts = sample[0]
            built = counts.clone() if built is None else built + counts
        for i in deferred:
            conformer_samples[i] = samplers[i].create_samples_for_poses(
                poses, task, built_rotamer_counts=built
            )

    if (
        torch.is_grad_enabled()
        and poses.coords.requires_grad
        and any(
            "chi_for_rotamers" in sample[2] and sample[2]["chi_for_rotamers"].numel()
            for sample in conformer_samples
        )
    ):
        raise NotImplementedError(
            "Chi-conformer construction requires inverse-kinematics coordinate "
            "derivatives, which are not implemented. Current-only conformers "
            "support coordinate derivatives; sampled-angle gradients require "
            "fixed input coordinates."
        )

    correlated_groups = _correlated_groups_for_samples(task, conformer_samples)

    from tmol.pack.rotamer._chi_budget import _check_task_sample_budget

    _check_task_sample_budget(task, conformer_samples)

    # Step 5
    (
        _,
        _,
        gbt_for_conformer,
        conformer_built_by_sampler,
        new_ind_for_sampler_rotamer,
    ) = _merge_conformer_samples(conformer_samples)

    def _t(t, dtype):
        return torch.tensor(t, dtype=dtype, device=pbt.device)

    gbt_for_conformer_np = gbt_for_conformer.cpu().numpy()

    gbt_for_conformer_torch = gbt_for_conformer.to(dtype=torch.int64, device=pbt.device)

    block_type_ind_for_conformer = gbt_block_type_ind[gbt_for_conformer_np]
    block_type_ind_for_conformer_torch = _t(block_type_ind_for_conformer, torch.int64)

    # The topology plan is assembled on the host. Residue atom counts are
    # immutable metadata, so do not send conformer types to the device merely
    # to gather counts and immediately copy their prefix sum back.
    n_atoms_by_type = numpy.asarray(
        [rt.n_atoms for rt in pbt.active_block_types], dtype=numpy.int32
    )
    layout = prepare_conformer_layout(
        pbt.rotamer_kinforest,
        pbt.max_n_atoms,
        n_atoms_by_type,
        block_type_ind_for_conformer,
    )
    n_atoms_for_conformer = torch.from_numpy(layout.n_atoms).to(pbt.device)
    n_atoms_offset_for_conformer_torch = _t(layout.offsets, torch.int64)
    n_atoms_total = layout.total
    packed_forest = torch.as_tensor(layout.forest, device=pbt.device)
    conformer_kinforest = KinForest(
        **dict(zip(KINFOREST_FIELDS, packed_forest.unbind(0)))
    )
    nodes, scans, gens = layout.nodes, layout.scans, layout.gens

    # Step 8 & 9
    orig_kinforest, orig_dofs_kto = _measure_pose_dofs(poses)

    # Step 9a
    conf_dofs_kto = torch.zeros(
        (n_atoms_total + 1, 9), dtype=torch.float32, device=pbt.device
    )

    conf_dofs_kto[1:] = pbt.rotamer_kinforest.dofs_ideal.reshape(-1, 9)[
        _t(layout.ideal_indices, torch.int64)
    ].to(dtype=torch.float32, device=pbt.device)

    for i, sampler in enumerate(samplers):
        sampler.fill_dofs_for_samples(
            poses,
            task,
            orig_kinforest,
            orig_dofs_kto,
            gbt_for_conformer_torch,
            block_type_ind_for_conformer_torch,
            n_atoms_offset_for_conformer_torch,
            conformer_built_by_sampler[i],
            new_ind_for_sampler_rotamer[i],
            conformer_samples[i][0],
            conformer_samples[i][1],
            conformer_samples[i][2],
            conf_dofs_kto,
        )

    _correct_phi_c_from_measured_chi(
        pbt,
        conformer_samples,
        new_ind_for_sampler_rotamer,
        block_type_ind_for_conformer_torch,
        n_atoms_offset_for_conformer_torch,
        conformer_kinforest,
        nodes,
        scans,
        gens,
        conf_dofs_kto,
    )

    rotamer_coords = _calculate_rotamer_coords(
        pbt,
        n_atoms_total,
        conformer_kinforest,
        nodes,
        scans,
        gens,
        conf_dofs_kto,
    )

    (
        n_rots_for_pose,
        rot_offset_for_pose,
        n_rots_for_block,
        rot_offset_for_block,
        pose_for_rot,
        block_ind_for_rot,
    ) = _get_rotamer_origin_data(task, gbt_for_conformer_torch)

    # Current/fallback rows must preserve the input, including frozen blocks.
    # Gather only real copied atoms, without a padded conformer-by-atom array.
    copied = [
        indices
        for indices, samples in zip(new_ind_for_sampler_rotamer, conformer_samples)
        if samples[2].get("copy_input_coordinates", False)
    ]
    if copied:
        copied = torch.cat(copied)
        sizes = n_atoms_for_conformer[copied].long()
        if pbt.device.type != "mps":
            from ._conformer_execution import copy_conformer_rows

            pose = pose_for_rot[copied]
            source = poses.block_coord_offset[pose, block_ind_for_rot[copied]].long()
            source = source + pose * poses.max_n_pose_atoms
            copy_conformer_rows(
                sizes,
                source,
                n_atoms_offset_for_conformer_torch[copied],
                poses.coords.reshape(-1, 3),
                rotamer_coords,
            )
        else:
            owner = torch.repeat_interleave(sizes)
            atom = torch.arange(owner.numel(), device=pbt.device)
            atom -= (torch.cumsum(sizes, 0) - sizes)[owner]
            rot = copied[owner]
            pose = pose_for_rot[rot]
            source = poses.block_coord_offset[pose, block_ind_for_rot[rot]] + atom
            destination = n_atoms_offset_for_conformer_torch[rot] + atom
            rotamer_coords[destination] = poses.coords[pose, source]

    return (
        poses,
        RotamerSet(
            max_n_rots_per_pose=torch.max(n_rots_for_pose).item(),
            n_rots_for_pose=n_rots_for_pose,
            rot_offset_for_pose=rot_offset_for_pose,
            n_rots_for_block=n_rots_for_block,
            rot_offset_for_block=rot_offset_for_block,
            pose_for_rot=pose_for_rot,
            block_type_ind_for_rot=block_type_ind_for_conformer_torch,
            block_ind_for_rot=block_ind_for_rot,
            coord_offset_for_rot=n_atoms_offset_for_conformer_torch.to(torch.int32),
            coords=rotamer_coords,
            correlated_groups=correlated_groups,
        ),
    )
