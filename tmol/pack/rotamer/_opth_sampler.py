import math
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

    def _count_rots_and_measure_all_flips(
        self,
        pose_stack: PoseStack,
        task: "SetPackerTask",  # noqa: F821
    ) -> tuple[Tensor[torch.int32][:], int, Tensor[torch.float32][:, :]]:
        """Count sampler rotamers and measure NHQ chis needed for flips.

        Returns:
            Rotamer counts shaped ``[n_global_block_types]``, maximum chi
            width, and current flip chis shaped ``[n_poses, max_n_blocks]``.
        """
        # First we have to get the list of all the blocks where we are using
        # this sampler. Next we will identify the subset of the NHQ blocks
        # where we will measure the chi dihedrals for the flip. Then we will
        # count the number of rotamers for each block as well as the number
        # of chi columns.
        pbt = pose_stack.packed_block_types
        optH_cache = pbt.opth_sample_cache

        n_gbt = task.cons_bt_pose.shape[0]
        n_rots_for_gbt = torch.zeros(n_gbt, dtype=torch.int32, device=pose_stack.device)
        self_index_in_task = task.conformer_sampler_index[id(self)]

        allowed_bt_is_optH_buildable = task.per_block_conformer_sampler_allowed[
            task.allowed_bt_pose, task.allowed_bt_block, self_index_in_task
        ]  # size (n_allowed_bt,)
        nz_allowed_bt_is_optH_buildable = torch.nonzero(
            allowed_bt_is_optH_buildable, as_tuple=True
        )[
            0
        ]  # size (n_allowed_and_buildable_bt,)
        allowed_and_buildable_pose = task.allowed_bt_pose[
            nz_allowed_bt_is_optH_buildable
        ]
        allowed_and_buildable_block = task.allowed_bt_block[
            nz_allowed_bt_is_optH_buildable
        ]
        allowed_and_buildable_bt = task.allowed_bt_block_type[
            nz_allowed_bt_is_optH_buildable
        ]
        orig_bt_for_allowed_and_buildable = pose_stack.block_type_ind64[
            allowed_and_buildable_pose, allowed_and_buildable_block
        ]

        n_rots_for_allowed_and_buildable = optH_cache.n_samples_for_bt_by_orig_bt[
            1 if self.flip_NHQ else 0,
            orig_bt_for_allowed_and_buildable,
            allowed_and_buildable_bt,
        ]
        n_rots_for_gbt[task.allowed_cons_bt[nz_allowed_bt_is_optH_buildable]] = (
            n_rots_for_allowed_and_buildable
        )

        max_n_chi_cols = optH_cache.expanded_samples.shape[1]

        # now we need to figure out which residues are NHQ and measure their chi dihedrals
        if not self.flip_NHQ:
            pos_flip_chi = torch.zeros(
                (pose_stack.n_poses, pose_stack.max_n_blocks),
                dtype=torch.float32,
                device=pose_stack.device,
            )
        else:
            is_allowed_and_buildable_bt_nhq = (
                optH_cache.nhq_chi_col[allowed_and_buildable_bt] >= 0
            )
            nz_nhq = torch.nonzero(is_allowed_and_buildable_bt_nhq, as_tuple=True)[0]
            if len(nz_nhq) == 0:
                pos_flip_chi = torch.zeros(
                    (pose_stack.n_poses, pose_stack.max_n_blocks),
                    dtype=torch.float32,
                    device=pose_stack.device,
                )
            else:
                pose_inds_nhq = allowed_and_buildable_pose[nz_nhq]
                block_inds_nhq = allowed_and_buildable_block[nz_nhq]
                pos_flip_chi_nhq = self._measure_all_nhq_flip_chis(
                    pose_stack, pose_inds_nhq, block_inds_nhq
                )
                pos_flip_chi = torch.zeros(
                    (pose_stack.n_poses, pose_stack.max_n_blocks),
                    dtype=torch.float32,
                    device=pose_stack.device,
                )
                pos_flip_chi[pose_inds_nhq, block_inds_nhq] = pos_flip_chi_nhq
        return n_rots_for_gbt, max_n_chi_cols, pos_flip_chi

    def _fill_proton_chi_for_all_blocks(
        self,
        pose_stack: PoseStack,
        task: "SetPackerTask",  # noqa: F821
        rot_offset_for_gbt: Tensor[torch.int32][:],
        gbt_for_rotamer: Tensor[torch.int32][:],
        chi_defining_atom_for_rotamer: Tensor[torch.int32][:, :],
        chi_for_rotamers: Tensor[torch.float32][:, :],
    ) -> None:
        """Fill proton-chi samples in ``[n_rotamers, max_n_chi]`` tensors."""
        pbt = pose_stack.packed_block_types
        opth_cache = pbt.opth_sample_cache

        bt_for_rotamer = task.cons_bt_block_type[gbt_for_rotamer]
        rotamers_w_proton_chi_samples = torch.nonzero(
            opth_cache.has_proton_chi[bt_for_rotamer], as_tuple=True
        )[0]
        if rotamers_w_proton_chi_samples.shape[0] == 0:
            return

        bt_for_proton_rotamer = bt_for_rotamer[rotamers_w_proton_chi_samples]
        n_rotamers = gbt_for_rotamer.shape[0]
        sample_ind_for_rotamer = (
            torch.arange(n_rotamers, dtype=torch.int64, device=pose_stack.device)
            - rot_offset_for_gbt[gbt_for_rotamer]
        )
        remaining_sample_ind = sample_ind_for_rotamer[rotamers_w_proton_chi_samples]
        n_samples_per_chi = opth_cache.n_samples_per_chi[bt_for_proton_rotamer].to(
            torch.int64
        )

        n_chi_cols = min(
            chi_for_rotamers.shape[1],
            opth_cache.expanded_samples.shape[1],
        )
        for chi_col in range(n_chi_cols):
            n_samples = n_samples_per_chi[:, chi_col]
            active = n_samples > 0
            # Skipping empty columns avoids launching several empty advanced-
            # indexing kernels; this is faster than eliminating the sync.
            if not torch.any(active):
                continue

            active_rotamers = rotamers_w_proton_chi_samples[active]
            active_bt = bt_for_proton_rotamer[active]
            active_n_samples = n_samples[active]
            active_sample_ind = torch.remainder(
                remaining_sample_ind[active],
                active_n_samples,
            )
            chi_for_rotamers[active_rotamers, chi_col] = opth_cache.expanded_samples[
                active_bt,
                chi_col,
                active_sample_ind,
            ]
            chi_defining_atom_for_rotamer[active_rotamers, chi_col] = (
                opth_cache.chi_defining_atom[active_bt, chi_col]
            )
            remaining_sample_ind[active] = torch.div(
                remaining_sample_ind[active],
                active_n_samples,
                rounding_mode="floor",
            )

    def _fill_all_nhq_blocks(
        self,
        pose_stack: PoseStack,
        task: "SetPackerTask",  # noqa: F821
        gbt_for_rotamer: Tensor[torch.int32][:],
        pos_flip_chi: Tensor[torch.float32][:, :],
        chi_defining_atom_for_rotamer: Tensor[torch.int32][:, :],
        chi_for_rotamers: Tensor[torch.float32][:, :],
    ) -> None:
        """Fill NHQ flip overrides in the per-rotamer chi tensors."""
        pbt = pose_stack.packed_block_types
        opth_cache = pbt.opth_sample_cache

        bt_for_rotamer = task.cons_bt_block_type[gbt_for_rotamer]
        rotamer_nhq_chi_col = opth_cache.nhq_chi_col[bt_for_rotamer]
        rotamer_is_flippable = rotamer_nhq_chi_col >= 0
        flippable_rotamers = torch.nonzero(rotamer_is_flippable, as_tuple=True)[0]

        # NHQ samples are emitted as adjacent (native, flipped) pairs. Slicing
        # those pairs avoids constructing two full-length temporary masks.
        unflipped_flippable_rotamers = flippable_rotamers[0::2]
        flipped_rotamers = flippable_rotamers[1::2]

        flipped_rot_chi = rotamer_nhq_chi_col[flipped_rotamers]
        gbt_for_flipped_rotamer = gbt_for_rotamer[flipped_rotamers]
        pose_for_flipped_rotamer = task.cons_bt_pose[gbt_for_flipped_rotamer]
        block_for_flipped_rotamer = task.cons_bt_block[gbt_for_flipped_rotamer]
        bt_for_flipped_rotamer = bt_for_rotamer[flipped_rotamers]

        is_his_rotamer = opth_cache.is_his[bt_for_rotamer]
        is_orig_bt_rotamer = bt_for_rotamer == (
            pose_stack.block_type_ind64[
                task.cons_bt_pose[gbt_for_rotamer],
                task.cons_bt_block[gbt_for_rotamer],
            ]
        )
        is_his_taut_rotamer = torch.logical_and(is_his_rotamer, ~is_orig_bt_rotamer)
        his_taut_not_flipped_rotamers = unflipped_flippable_rotamers[
            is_his_taut_rotamer[unflipped_flippable_rotamers]
        ]
        his_taut_not_flipped_rot_chi = rotamer_nhq_chi_col[
            his_taut_not_flipped_rotamers
        ]

        gbt_for_his_taut_not_flipped_rotamer = gbt_for_rotamer[
            his_taut_not_flipped_rotamers
        ]
        pose_for_his_taut_not_flipped_rotamer = task.cons_bt_pose[
            gbt_for_his_taut_not_flipped_rotamer
        ]
        block_for_his_taut_not_flipped_rotamer = task.cons_bt_block[
            gbt_for_his_taut_not_flipped_rotamer
        ]
        bt_for_his_taut_not_flipped_rotamer = bt_for_rotamer[
            his_taut_not_flipped_rotamers
        ]

        # mark the chi for the HIS tautomer in its non-flipped state
        # because we have to rebuild the ring from ideal geometry.
        chi_for_rotamers[
            his_taut_not_flipped_rotamers, his_taut_not_flipped_rot_chi
        ] = pos_flip_chi[
            pose_for_his_taut_not_flipped_rotamer,
            block_for_his_taut_not_flipped_rotamer,
        ]
        chi_defining_atom_for_rotamer[
            his_taut_not_flipped_rotamers, his_taut_not_flipped_rot_chi
        ] = opth_cache.nhq_chi_atom[bt_for_his_taut_not_flipped_rotamer]

        # mark the flipped chi for all NHQ rotamers
        chi_for_rotamers[flipped_rotamers, flipped_rot_chi] = (
            pos_flip_chi[
                pose_for_flipped_rotamer,
                block_for_flipped_rotamer,
            ]
            + math.pi
        )
        chi_defining_atom_for_rotamer[flipped_rotamers, flipped_rot_chi] = (
            opth_cache.nhq_chi_atom[bt_for_flipped_rotamer]
        )

    def _fill_all_chi_tensors(
        self,
        pose_stack: PoseStack,
        task: "SetPackerTask",  # noqa: F821
        rot_offset_for_gbt: Tensor[torch.int32][:],
        gbt_for_rotamer: Tensor[torch.int32][:],
        pos_flip_chi: Tensor[torch.float32][:, :],
        chi_defining_atom_for_rotamer: Tensor[torch.int32][:, :],
        chi_for_rotamers: Tensor[torch.float32][:, :],
    ) -> None:
        """Fill proton-chi and NHQ overrides into shared rotamer tensors."""
        self._fill_proton_chi_for_all_blocks(
            pose_stack,
            task,
            rot_offset_for_gbt,
            gbt_for_rotamer,
            chi_defining_atom_for_rotamer,
            chi_for_rotamers,
        )
        self._fill_all_nhq_blocks(
            pose_stack,
            task,
            gbt_for_rotamer,
            pos_flip_chi,
            chi_defining_atom_for_rotamer,
            chi_for_rotamers,
        )

    def create_samples_for_poses(
        self,
        pose_stack: PoseStack,
        task: "SetPackerTask",  # noqa: F821
    ) -> tuple[
        Tensor[torch.int32][:],
        Tensor[torch.int32][:],
        dict[str, torch.Tensor],
    ]:
        from tmol.pack.rotamer._chi_budget import _sampler_for_task

        configured, task = _sampler_for_task(self, task)
        if configured is not self:
            return configured.create_samples_for_poses(pose_stack, task)
        self._annotate_packed_block_types(pose_stack.packed_block_types)

        # ensure dunbrack and optH sampler are not _both_ specified for the same block
        self._assert_no_dun_opth_conflict(task)

        # 1) compute:
        #      n_rots per GBT
        #      max chi tensor width
        #      current last-chi angle
        # for each NHQ position in the input
        n_rots_for_gbt, max_n_chi_cols, pos_flip_chi = (
            self._count_rots_and_measure_all_flips(pose_stack, task)
        )

        from tmol.pack.rotamer._chi_budget import _checked_sample_count

        n_rots_total = _checked_sample_count(
            n_rots_for_gbt, self.chi_sample_expanded_limit, self.chi_sample_limit
        )

        if n_rots_total == 0:
            empty_chi = torch.zeros(
                (0, max_n_chi_cols), dtype=torch.int32, device=pose_stack.device
            )
            return (
                n_rots_for_gbt,
                torch.zeros(0, dtype=torch.int32, device=pose_stack.device),
                dict(
                    chi_defining_atom_for_rotamer=empty_chi,
                    chi_for_rotamers=empty_chi.float(),
                ),
            )
        rot_offset_for_gbt = _exclusive_cumsum1d(n_rots_for_gbt)
        # Reuse the synchronized total and keep the existing int32 indices.
        gbt_for_rotamer = torch.repeat_interleave(
            torch.arange(
                n_rots_for_gbt.shape[0], dtype=torch.int32, device=pose_stack.device
            ),
            n_rots_for_gbt,
            output_size=n_rots_total,
        )
        chi_defining_atom_for_rotamer = torch.full(
            (n_rots_total, max_n_chi_cols),
            -1,
            dtype=torch.int32,
            device=pose_stack.device,
        )
        chi_for_rotamers = torch.zeros(
            (n_rots_total, max_n_chi_cols),
            dtype=torch.float32,
            device=pose_stack.device,
        )

        # # 2) fill chi tensors
        self._fill_all_chi_tensors(
            pose_stack,
            task,
            rot_offset_for_gbt,
            gbt_for_rotamer,
            pos_flip_chi,
            chi_defining_atom_for_rotamer,
            chi_for_rotamers,
        )

        return (
            n_rots_for_gbt,
            gbt_for_rotamer,
            dict(
                chi_defining_atom_for_rotamer=chi_defining_atom_for_rotamer,
                chi_for_rotamers=chi_for_rotamers,
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
