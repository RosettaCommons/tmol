import torch
import attr

from typing import Tuple

from tmol.types._torch import Tensor
from tmol.types._functional import _validate_args
from tmol.chemical import RefinedResidueType
from tmol.pose._packed_block_types import PackedBlockTypes
from tmol.pose._pose_stack import PoseStack
from tmol.kinematics._datatypes import KinForest
from tmol.pack.rotamer._conformer_sampler import ConformerSampler, _sc_roots_for_chis
from tmol.pack.rotamer._single_residue_kinforest import (
    _construct_single_residue_kinforest,
)
from tmol.pack._packer_task import (
    DEFAULT_CHI_SAMPLE_EXPANDED_LIMIT,
    DEFAULT_CHI_SAMPLE_LIMIT,
)
from tmol.numeric import coord_dihedrals
from tmol.utility.tensor._common_operations import _exclusive_cumsum1d

from tmol_core.pack.opth import (  # noqa: F401
    OptHSamplerRTCache,
    OptHSamplerPackedBlockTypeCache,
    _build_empty_proton_cache,
    _compute_nhq_downstream_kfo,
    annotate_opth_residue,
    pack_opth_metadata,
    opth_builds_residue,
)


def _opth_fill_dofs(
    pose_stack: PoseStack,
    task: "PackerTask",  # noqa: F821
    gbt_for_conformer: Tensor[torch.int64][:],
    block_type_ind_for_conformer: Tensor[torch.int64][:],
    n_dof_atoms_offset_for_conformer: Tensor[torch.int64][:],
    conf_inds_for_sampler: Tensor[torch.int64][:],
    orig_dofs_kto: Tensor[torch.float32][:, 9],
    chi_atoms: Tensor[torch.int32][:, :],
    chi_vals: Tensor[torch.float32][:, :],
    conf_dofs_kto: Tensor[torch.float32][:, 9],
    flip_NHQ: bool,
) -> None:
    """Fill the packed DOF tensor for all OptHSampler rotamers.

    1. Copy DOFs from pose into conf_dofs_kto.
    2. For NHQ only: atoms that are kinematic children of the chi-defining atom
       of the flip are reset to their ideal DOF values
    3. Write the chi torsion into DOF column 3 for the chi-defining atom of
       each flip / proton-chi rotamer; correct_phi_c_from_measured_chi turns
       it into the intended dihedral.
    """
    pbt = pose_stack.packed_block_types
    dev = conf_dofs_kto.device

    # Per-sampler-rotamer lookup vectors (torch, length n_rots)
    bt_inds = block_type_ind_for_conformer[conf_inds_for_sampler]
    at_offs = n_dof_atoms_offset_for_conformer[conf_inds_for_sampler]

    # Source offsets in orig_dofs_kto: mirrors create_full_dof_inds_to_copy_...
    orig_bt_ind = (
        pose_stack.block_type_ind[pose_stack.block_type_ind != -1]
        .view(-1)
        .to(torch.int64)
    )
    orig_at_off_base = _exclusive_cumsum1d(pbt.n_atoms[orig_bt_ind]).to(torch.int64)
    flat_block_for_gbt = task.global_block_ind_for_considered_block_types
    real_idx_for_flat = torch.full(
        (pose_stack.block_type_ind.shape[0] * pose_stack.block_type_ind.shape[1],),
        -1,
        dtype=torch.int64,
        device=dev,
    )
    real_idx_for_flat[pose_stack.block_type_ind.view(-1) != -1] = torch.arange(
        orig_bt_ind.shape[0], dtype=torch.int64, device=dev
    )
    orig_at_offs = orig_at_off_base[
        real_idx_for_flat[flat_block_for_gbt[gbt_for_conformer[conf_inds_for_sampler]]]
    ]

    # Step 1: vectorised all-atom copy
    n_rots = conf_inds_for_sampler.shape[0]
    n_atoms_per_rot = pbt.n_atoms[bt_inds]
    dummy = (
        torch.arange(pbt.max_n_atoms, dtype=torch.int64, device=dev)
        .view(1, -1)
        .expand(n_rots, -1)
    )
    real_mask = dummy < n_atoms_per_rot.unsqueeze(1)
    dst = (at_offs.unsqueeze(1).expand(-1, pbt.max_n_atoms) + dummy)[real_mask]
    src = (orig_at_offs.unsqueeze(1).expand(-1, pbt.max_n_atoms) + dummy)[real_mask]
    conf_dofs_kto[dst + 1, :] = orig_dofs_kto[src + 1, :]

    # Steps 2 & 3: only for rotamers that have a chi override
    if chi_atoms.shape[0] == 0:
        return

    chi_mask = chi_atoms >= 0
    has_chi_override = chi_mask.any(dim=1)
    dofs_ideal_t = pbt.rotamer_kinforest.dofs_ideal
    opth_cache = pbt.opth_sample_cache

    # Reset NHQ atoms downstream of the flipped chi to ideal geometry. Keep
    # this entirely on-device: reading each rotamer's offsets with .item()
    # serializes large CUDA batches.
    if flip_NHQ:
        downstream = opth_cache.nhq_downstream_kfo[bt_inds]
        downstream_count = opth_cache.nhq_downstream_count[bt_inds]
        downstream_slot = torch.arange(
            downstream.shape[1], dtype=torch.int32, device=dev
        )
        reset_mask = (downstream_slot[None, :] < downstream_count[:, None]) & (
            has_chi_override[:, None]
        )
        reset_k = downstream[reset_mask].to(torch.int64)
        reset_bt = bt_inds[:, None].expand_as(downstream)[reset_mask]
        reset_kto = (downstream + at_offs[:, None].to(downstream.dtype) + 1)[
            reset_mask
        ].to(torch.int64)
        conf_dofs_kto[reset_kto] = dofs_ideal_t[reset_bt, reset_k]

    # Write every chi override in one indexed assignment.
    from tmol.pack.rotamer._build_rotamers import (
        _kinforest_device_indices,
        _build_ring_chi_phi_c_corrections,
    )

    kfidx = _kinforest_device_indices(pbt, dev)
    corrections = _build_ring_chi_phi_c_corrections(pbt)
    safe_chi_atoms = chi_atoms.clamp_min(0).to(torch.int64)
    rotamer_bt = bt_inds[:, None].expand_as(safe_chi_atoms)
    chi_kto = kfidx[rotamer_bt, safe_chi_atoms] + at_offs[:, None] + 1
    corrected_chi = chi_vals - corrections[rotamer_bt, safe_chi_atoms]
    conf_dofs_kto[chi_kto[chi_mask], 3] = corrected_chi[chi_mask]


@attr.s(auto_attribs=True, frozen=True)
class OptHSampler(ConformerSampler):
    """Build rotamers by sampling proton chi angles only, keeping all heavy
    atoms at their input-conformation positions.

    When flip_NHQ is True (default), also builds flip rotamers for:

    - ASN/GLN: current conformation + 180-degree rotation of the last chi.
    - HIS/HIS_D: {HIS, HIS_D} x {current chi2, chi2+180} = 4 rotamers.
      All atoms through CG are taken from the input; ring atoms are rebuilt
      from ideal geometry for three non-input variants.
    - Other HIS states: current chi2 + chi2+180, as ASN/GLN.

    NOTE: DunbrackChiSampler and OptHSampler must not be assigned to the
    same block (Dunbrack already samples proton chis, so both on one block
    oversamples). Assigning them to different blocks in the same task is fine.
    """

    flip_NHQ: bool = True
    # optH samples one residue at a time, so its budget is per residue: no
    #    grouping, but the same limits the full packer applies
    chi_sample_expanded_limit: int = DEFAULT_CHI_SAMPLE_EXPANDED_LIMIT
    chi_sample_limit: int = DEFAULT_CHI_SAMPLE_LIMIT

    @classmethod
    def sampler_name(cls):
        return "OptHSampler"

    @_validate_args
    def _annotate_residue_type(self, rt: RefinedResidueType):
        annotate_opth_residue(
            rt,
            self.chi_sample_expanded_limit,
            self.chi_sample_limit,
            _construct_single_residue_kinforest,
        )

    @_validate_args
    def _annotate_packed_block_types(self, packed_block_types: PackedBlockTypes):
        key = (self.chi_sample_expanded_limit, self.chi_sample_limit, self.flip_NHQ)
        if getattr(packed_block_types, "_opth_sampler_key", None) == key:
            return
        for bt in packed_block_types.active_block_types:
            self._annotate_residue_type(bt)
        host = pack_opth_metadata(
            [bt.opth_sampler_cache for bt in packed_block_types.active_block_types],
            packed_block_types.max_n_atoms,
            self.flip_NHQ,
        )
        cache = OptHSamplerPackedBlockTypeCache(
            **{
                field.name: torch.from_numpy(getattr(host, field.name)).to(
                    packed_block_types.device
                )
                for field in attr.fields(OptHSamplerPackedBlockTypeCache)
            }
        )
        setattr(packed_block_types, "opth_sample_cache", cache)
        setattr(packed_block_types, "_opth_sampler_key", key)

    @_validate_args
    def defines_rotamers_for_rt(self, rt: RefinedResidueType):
        self._annotate_residue_type(rt)
        cache = rt.opth_sampler_cache
        return opth_builds_residue(cache, self.flip_NHQ)

    def defines_rotamers_for_bts(
        self, pbt: PackedBlockTypes, bt_inds: Tensor[torch.int64]
    ) -> Tensor[torch.bool]:
        self._annotate_packed_block_types(pbt)
        return pbt.opth_sample_cache.opth_sample_for_bt[bt_inds]

    @_validate_args
    def first_sc_atoms_for_rt(self, rt: RefinedResidueType) -> Tuple[str, ...]:
        """Roots for the chis optH turns: every proton chi, plus the flipped chi."""
        self._annotate_residue_type(rt)
        chis = [cs.chi_dihedral for cs in rt.chi_samples if cs.is_proton]
        if self.flip_NHQ and rt.opth_sampler_cache.nhq_chi_col >= 0:
            chis.append(f"chi{rt.opth_sampler_cache.nhq_chi_col + 1}")
        return _sc_roots_for_chis(rt, chis)

    def _assert_no_dun_opth_conflict(self, task: "SetPackerTask"):  # noqa: F821
        self_index_in_task = task.conformer_sampler_index[id(self)]
        optH_sampler_allowed = task.per_block_conformer_sampler_allowed[
            :, :, self_index_in_task
        ]
        for i, sampler in enumerate(task.conformer_samplers):
            if sampler is self:
                continue
            if sampler.sampler_name() == "DunbrackChiSampler":
                i_allowed = task.per_block_conformer_sampler_allowed[:, :, i]
                if torch.any(torch.logical_and(optH_sampler_allowed, i_allowed)):
                    raise RuntimeError(
                        "OptHSampler and DunbrackChiSampler cannot both be assigned "
                        "to the same block. DunbrackChiSampler already samples proton "
                        "chi angles as part of each library rotamer."
                    )

    def _measure_all_nhq_flip_chis(
        self,
        pose_stack: PoseStack,
        pose_inds: Tensor[torch.int64][:],
        block_inds: Tensor[torch.int64][:],
    ) -> Tensor[torch.float64][:]:
        """Measure the current NHQ chi for each selected ``(pose, block)``."""
        offsets = pose_stack.block_coord_offset64[pose_inds, block_inds]
        block_types = pose_stack.block_type_ind64[pose_inds, block_inds]
        a4 = pose_stack.packed_block_types.opth_sample_cache.nhq_chi_4atoms[block_types]

        pose_inds_expanded = pose_inds.repeat_interleave(4)
        offsets_expanded = offsets.repeat_interleave(4)

        c = pose_stack.coords[
            pose_inds_expanded, offsets_expanded + a4.flatten()
        ]  # [4 * n_selected, 3]
        c = c.view(-1, 4, 3)  # [n_selected, 4, 3]
        c = c.to(dtype=torch.float64)
        return coord_dihedrals(c[:, 0], c[:, 1], c[:, 2], c[:, 3])

    def _count_rots_and_measure_all_flips(self, pose_stack, task):
        from tmol.pack.rotamer._conformer_compiled import opth_sampling_counts

        inputs = _opth_inputs(self, pose_stack, task)
        counts, measure = opth_sampling_counts(inputs, self.flip_NHQ)
        # Measure each physical residue once, even when several HIS tautomers
        # are allowed. Repeated indexed writes duplicate its backward gradient.
        if not self.flip_NHQ:
            return counts, inputs
        selected = torch.nonzero(measure.to(pose_stack.device), as_tuple=True)[0]
        pos_flip = inputs[14].to(pose_stack.device)
        if len(selected):
            physical = torch.unique(
                task.cons_bt_pose[selected] * pose_stack.max_n_blocks
                + task.cons_bt_block[selected],
                sorted=True,
            )
            poses = torch.div(physical, pose_stack.max_n_blocks, rounding_mode="floor")
            blocks = physical % pose_stack.max_n_blocks
            pos_flip[poses, blocks] = self._measure_all_nhq_flip_chis(
                pose_stack,
                poses,
                blocks,
            )
        inputs[14] = pos_flip.to(inputs[0].device).contiguous()
        return counts, inputs

    def _fill_proton_chi_for_all_blocks(
        self,
        pose_stack,
        task,
        rot_offset_for_gbt,
        gbt_for_rotamer,
        chi_defining_atom_for_rotamer,
        chi_for_rotamers,
    ):
        cache = pose_stack.packed_block_types.opth_sample_cache
        device = _opth_execution_device(pose_stack.device)
        inputs = [
            task.cons_bt_block_type,
            rot_offset_for_gbt.to(torch.int64),
            gbt_for_rotamer.to(torch.int32),
            cache.has_proton_chi,
            cache.chi_defining_atom,
            cache.n_samples_per_chi,
            cache.expanded_samples,
        ]
        atoms, values = _OptHProtonValues.apply(
            *[v.to(device).contiguous() for v in inputs],
            chi_for_rotamers.shape[1],
        )
        atoms, values = atoms.to(pose_stack.device), values.to(pose_stack.device)
        active = atoms >= 0
        chi_defining_atom_for_rotamer[active] = atoms[active]
        chi_for_rotamers[active] = values[active]

    def create_samples_for_poses(self, pose_stack: PoseStack, task):
        from tmol.pack.rotamer._chi_budget import (
            _sampler_for_task,
            _checked_sample_count,
        )

        configured, task = _sampler_for_task(self, task)
        if configured is not self:
            return configured.create_samples_for_poses(pose_stack, task)
        self._annotate_packed_block_types(pose_stack.packed_block_types)
        self._assert_no_dun_opth_conflict(task)
        counts, inputs = self._count_rots_and_measure_all_flips(pose_stack, task)
        total = _checked_sample_count(
            counts,
            self.chi_sample_expanded_limit,
            self.chi_sample_limit,
        )
        device = pose_stack.device
        if total == 0:
            width = inputs[13].shape[1]
            return (
                counts.to(device=device, dtype=torch.int32),
                torch.empty(0, dtype=torch.int32, device=device),
                dict(
                    chi_defining_atom_for_rotamer=torch.empty(
                        (0, width),
                        dtype=torch.int32,
                        device=device,
                    ),
                    chi_for_rotamers=torch.empty(
                        (0, width),
                        dtype=torch.float32,
                        device=device,
                    ),
                ),
            )
        owners, atoms, values = _OptHSampleValues.apply(*inputs, counts, total)
        return (
            counts.to(device=device, dtype=torch.int32),
            owners.to(device),
            dict(
                chi_defining_atom_for_rotamer=atoms.to(device),
                chi_for_rotamers=values.to(device),
            ),
        )

    def fill_dofs_for_samples(
        self,
        pose_stack: PoseStack,
        task: "PackerTask",  # noqa: F821
        orig_kinforest: KinForest,
        orig_dofs_kto: Tensor[torch.float32][:, 9],
        gbt_for_conformer: Tensor[torch.int64][:],
        block_type_ind_for_conformer: Tensor[torch.int64][:],
        n_dof_atoms_offset_for_conformer: Tensor[torch.int64][:],
        conformer_built_by_sampler: Tensor[torch.bool][:],
        conf_inds_for_sampler: Tensor[torch.int64][:],
        sampler_n_rots_for_gbt: Tensor[torch.int32][:],
        sampler_gbt_for_rotamer: Tensor[torch.int32][:],
        sample_dict: dict[str, torch.Tensor],
        conf_dofs_kto: Tensor[torch.float32][:, 9],
    ) -> None:
        if sampler_gbt_for_rotamer.shape[0] == 0:
            return

        _opth_fill_dofs(
            pose_stack,
            task,
            gbt_for_conformer,
            block_type_ind_for_conformer,
            n_dof_atoms_offset_for_conformer,
            conf_inds_for_sampler,
            orig_dofs_kto,
            sample_dict["chi_defining_atom_for_rotamer"],
            sample_dict["chi_for_rotamers"],
            conf_dofs_kto,
            self.flip_NHQ,
        )


def _opth_execution_device(device):
    return device if device.type in ("cpu", "cuda") else torch.device("cpu")


def _opth_inputs(sampler, pose, task, flip_values=None):
    cache = pose.packed_block_types.opth_sample_cache
    if flip_values is None:
        flip_values = torch.zeros(
            pose.block_type_ind64.shape, dtype=torch.float32, device=pose.device
        )
    index = task.conformer_sampler_index[id(sampler)]
    values = [
        task.cons_bt_block_type,
        task.cons_bt_pose,
        task.cons_bt_block,
        pose.block_type_ind64,
        task.is_cons_bt_allowed,
        task.per_block_conformer_sampler_allowed[:, :, index],
        cache.n_samples_for_bt_by_orig_bt,
        cache.nhq_chi_col,
        cache.nhq_chi_atom,
        cache.is_his,
        cache.has_proton_chi,
        cache.chi_defining_atom,
        cache.n_samples_per_chi,
        cache.expanded_samples,
        flip_values,
    ]
    device = _opth_execution_device(pose.device)
    return [value.to(device).contiguous() for value in values]


class _OptHSampleValues(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        from tmol.pack.rotamer._conformer_compiled import opth_sampling_fill

        owners, atoms, values, sources = opth_sampling_fill(
            list(args[:15]), args[15], args[16]
        )
        ctx.save_for_backward(sources)
        ctx.proton_shape, ctx.flip_shape = args[13].shape, args[14].shape
        ctx.proton_size, ctx.flip_size = args[13].numel(), args[14].numel()
        ctx.mark_non_differentiable(owners, atoms)
        return owners, atoms, values

    @staticmethod
    def backward(ctx, grad_owners, grad_atoms, grad_values):
        result = [None] * 17
        if grad_values is None:
            return tuple(result)
        (sources,) = ctx.saved_tensors
        active = sources >= 0
        # The native indices describe a linear lookup. Its transpose scatter
        # retains parameter/coordinate higher derivatives through the frontend.
        grad = grad_values.new_zeros(ctx.proton_size + ctx.flip_size).index_add(
            0,
            sources[active],
            grad_values[active],
        )
        result[13] = grad[: ctx.proton_size].reshape(ctx.proton_shape)
        result[14] = grad[ctx.proton_size :].reshape(ctx.flip_shape)
        return tuple(result)


class _OptHProtonValues(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        from tmol.pack.rotamer._conformer_compiled import opth_sampling_proton

        atoms, values, sources = opth_sampling_proton(list(args[:7]), args[7])
        ctx.save_for_backward(sources)
        ctx.shape, ctx.size = args[6].shape, args[6].numel()
        ctx.mark_non_differentiable(atoms)
        return atoms, values

    @staticmethod
    def backward(ctx, grad_atoms, grad_values):
        result = [None] * 8
        if grad_values is not None:
            (sources,) = ctx.saved_tensors
            active = sources >= 0
            result[6] = (
                grad_values.new_zeros(ctx.size)
                .index_add(
                    0,
                    sources[active],
                    grad_values[active],
                )
                .reshape(ctx.shape)
            )
        return tuple(result)
