import numpy
import torch
import attr

from typing import Tuple

from tmol.types._torch import Tensor

from tmol.database import ParameterDatabase
from tmol.chemical import RefinedResidueType
from tmol.pose._packed_block_types import PackedBlockTypes
from tmol.pose._pose_stack import PoseStack
from tmol.pack._packer_task import (
    SetPackerTask,
    DEFAULT_CHI_SAMPLE_EXPANDED_LIMIT,
    DEFAULT_CHI_SAMPLE_LIMIT,
)
from tmol.pack.rotamer._chi_budget import (
    _sampler_for_task,
)
from tmol.pack.rotamer._single_residue_kinforest import (
    _construct_single_residue_kinforest,
)
from tmol.pack.rotamer._chi_sampler import ChiSampler
from tmol.pack.rotamer._conformer_sampler import _sc_roots_for_chis
from tmol.score.na_torsion._params import (
    NaTorsionParams,
    N_PUCKER,
    SYN_MEAN,
    N_BASE_PER_POLYMER,
)
from tmol.score.na_torsion import pucker_weights
from tmol.utility import resolve_device

from tmol_core.pack.na_sampling import (  # noqa: F401
    CHI_STEPS,
    MAX_SYN_WELL,
    annotate_na_sampler_residue,
    pack_na_sampler_params,
    expanded_samples as _expanded_samples,
    proton_combinations as _proton_combinations,
)


@attr.s(auto_attribs=True, frozen=True)
class NaChiRotamerSampler(ChiSampler):
    """Glycosidic chi rotamers for DNA and RNA, taken from the na_torsion tables.

    Every chi generated sits at a minimum of the term that will score it: the
    anti rotamer is the per-base, per-pucker mean and the syn rotamer is the
    fixed syn mean, each expanded in units of that term's own sdev_chi. The
    pucker comes from the input sugar, which is not itself a packing degree of
    freedom.

    The 2'-OH proton chi is expanded here rather than left to OptHSampler,
    which is applied to disjoint blocks; see DunbrackChiSampler, which folds
    proton chis in the same way.
    """

    params: NaTorsionParams
    element_for_atom_type: dict
    chi_sample_level: int = 0
    sample_syn: bool = True
    device: torch.device = torch.device("cpu")

    chi_sample_expanded_limit: int = DEFAULT_CHI_SAMPLE_EXPANDED_LIMIT
    chi_sample_limit: int = DEFAULT_CHI_SAMPLE_LIMIT

    @classmethod
    def from_database(
        cls,
        param_db: ParameterDatabase,
        device: torch.device,
        chi_sample_level: int = 0,
        sample_syn: bool = True,
    ) -> "NaChiRotamerSampler":
        """Create a nucleic-acid chi sampler on a concrete device.

        Args:
            param_db: Source scoring and chemical parameters.
            device: Target device. Unindexed CUDA resolves to the current GPU.
            chi_sample_level: Number of standard-deviation expansion levels.
            sample_syn: Include eligible syn rotamers when true.

        Returns:
            A sampler whose parameters and device metadata agree.
        """
        device = resolve_device(device)
        return cls(
            params=NaTorsionParams.from_database(param_db.scoring.na_torsion, device),
            element_for_atom_type={
                at.name: at.element for at in param_db.chemical.atom_types
            },
            chi_sample_level=chi_sample_level,
            sample_syn=sample_syn,
            device=device,
        )

    @classmethod
    def sampler_name(cls):
        return "NaChiRotamerSampler"

    def _annotation_key(self):
        # Keep only the most recent tables on each type/PBT, but validate all
        # inputs to those tables. A different sampler must not inherit the
        # first sampler's budget or chemical element assignments.
        return (
            self.chi_sample_expanded_limit,
            self.chi_sample_limit,
            tuple(sorted(self.element_for_atom_type.items())),
        )

    def annotate_residue_type(self, rt: RefinedResidueType):
        key = self._annotation_key()
        if getattr(rt, "_na_chi_sampler_key", None) == key and hasattr(
            rt, "na_chi_sampler_params"
        ):
            return rt.na_chi_sampler_params
        p = annotate_na_sampler_residue(
            rt,
            self.element_for_atom_type,
            self.chi_sample_expanded_limit,
            self.chi_sample_limit,
            _construct_single_residue_kinforest,
        )
        setattr(rt, "na_chi_sampler_params", p)
        setattr(rt, "_na_chi_sampler_key", key)
        return p

    def annotate_packed_block_types(self, packed_block_types: PackedBlockTypes):
        key = self._annotation_key()
        if getattr(packed_block_types, "_na_chi_sampler_key", None) == key:
            return packed_block_types.na_chi_sampler_cache
        bts = packed_block_types.active_block_types
        params = [self.annotate_residue_type(bt) for bt in bts]
        arrays = pack_na_sampler_params(params)
        cache = {
            name: (
                torch.tensor(value, device=packed_block_types.device)
                if isinstance(value, numpy.ndarray)
                else value
            )
            for name, value in arrays.items()
        }
        setattr(packed_block_types, "na_chi_sampler_cache", cache)
        setattr(packed_block_types, "_na_chi_sampler_key", key)
        return cache

    def defines_rotamers_for_rt(self, rt: RefinedResidueType):
        return self.annotate_residue_type(rt)["base"] >= 0

    def defines_rotamers_for_bts(
        self, pbt: PackedBlockTypes, bt_inds: Tensor[torch.int64]
    ) -> Tensor[torch.bool]:
        return self.annotate_packed_block_types(pbt)["builds_bt"][bt_inds]

    def first_sc_atoms_for_rt(self, rt: RefinedResidueType) -> Tuple[str, ...]:
        return _sc_roots_for_chis(
            rt, ("chi1",) + tuple(cs.chi_dihedral for cs in rt.chi_samples)
        )

    def _pucker_for_blocks(
        self,
        poses: PoseStack,
        pbt: PackedBlockTypes,
        pose_indices: Tensor[torch.int64][:],
        block_indices: Tensor[torch.int64][:],
        block_type_indices: Tensor[torch.int64][:],
        cache,
    ) -> Tensor[torch.int64][:]:
        """Return the most likely input-sugar pucker for selected blocks."""
        ring_local = cache["ring"][block_type_indices]
        offset = poses.block_coord_offset64[pose_indices, block_indices].unsqueeze(-1)
        ok = ring_local >= 0
        index = (offset + ring_local).clamp_min(0)
        max_n_atoms = poses.coords.shape[1]
        flat = pose_indices.unsqueeze(-1) * max_n_atoms + index
        if poses.device.type in ("cpu", "cuda"):
            from tmol.score.na_torsion.potentials._compiled import _na_sampling_pucker

            return _na_sampling_pucker(
                poses.coords.reshape(-1, 3), flat, ok, self.params.pucker_temperature
            )
        xyz = poses.coords.reshape(-1, 3)[flat.reshape(-1)].reshape(*index.shape, 3)
        xyz = torch.where(ok.unsqueeze(-1), xyz, torch.zeros_like(xyz))
        w = pucker_weights(
            xyz.reshape(-1, 5, 3), self.params.pucker_temperature
        ).reshape(pose_indices.shape[0], N_PUCKER)
        pucker = w.argmax(dim=-1)
        return torch.where(ok.all(-1), pucker, torch.full_like(pucker, -1))

    def sample_chi_for_poses(
        self, poses: PoseStack, task: SetPackerTask
    ) -> Tuple[
        Tensor[torch.int32][:],  # n_rots_for_gbt
        Tensor[torch.int32][:],  # gbt_for_rotamer
        Tensor[torch.int32][:, :],  # chi_defining_atom_for_rotamer
        Tensor[torch.float32][:, :],  # chi_for_rotamers
    ]:
        configured, task = _sampler_for_task(self, task)
        if configured is not self:
            return configured.sample_chi_for_poses(poses, task)
        pbt = poses.packed_block_types
        cache = self.annotate_packed_block_types(pbt)

        self_ind = task.conformer_sampler_index[id(self)]
        allowed = task.per_block_conformer_sampler_allowed[:, :, self_ind]
        allowed_for_cons = allowed[task.cons_bt_pose, task.cons_bt_block]
        builds = cache["builds_bt"][task.cons_bt_block_type]
        bt_allowed = task.per_block_is_block_type_allowed[
            task.cons_bt_pose, task.cons_bt_block, task.cons_bt_which_block_type
        ]
        active = allowed_for_cons & builds & bt_allowed

        active_gbt = torch.nonzero(active, as_tuple=True)[0]
        pucker = torch.full_like(task.cons_bt_pose, -1)
        if active_gbt.numel() > 0:
            active_pose = task.cons_bt_pose[active_gbt]
            active_block = task.cons_bt_block[active_gbt]
            active_global_block = active_pose * poses.max_n_blocks + active_block
            unique_global_block, inverse = torch.unique_consecutive(
                active_global_block, return_inverse=True
            )
            unique_pose = torch.div(
                unique_global_block, poses.max_n_blocks, rounding_mode="floor"
            )
            unique_block = unique_global_block % poses.max_n_blocks
            unique_pucker = self._pucker_for_blocks(
                poses,
                pbt,
                unique_pose,
                unique_block,
                poses.block_type_ind64[unique_pose, unique_block],
                cache,
            )
            pucker[active_gbt] = unique_pucker[inverse]
        from tmol.score.na_torsion.potentials._compiled import _na_sampling_counts
        from tmol.pack.rotamer._chi_budget import _checked_sample_count

        execution_device = (
            poses.device
            if poses.device.type in ("cpu", "cuda")
            else torch.device("cpu")
        )
        inputs = [
            value.to(execution_device)
            for value in (
                task.cons_bt_block_type,
                cache["base"],
                pucker,
                cache["n_combos"],
                active,
                cache["chi_atom"],
                cache["proton"],
                self.params.well_chi_syn,
                self.params.chi_means,
                self.params.sdev_chi,
                torch.tensor(
                    CHI_STEPS[self.chi_sample_level],
                    dtype=torch.float32,
                    device=poses.device,
                ),
            )
        ]
        policy = (self.sample_syn, N_BASE_PER_POLYMER, MAX_SYN_WELL, SYN_MEAN)
        counts = _na_sampling_counts(inputs, *policy)
        n_rots = _checked_sample_count(
            counts, self.chi_sample_expanded_limit, self.chi_sample_limit
        )
        owners, atoms, chi = _NaChiSampleValues.apply(*inputs, counts, n_rots, policy)
        return tuple(
            value.to(poses.device)
            for value in (counts.to(torch.int32), owners, atoms, chi)
        )


class _NaChiSampleValues(torch.autograd.Function):
    """Native linear sample values with differentiable parameter gathers."""

    @staticmethod
    def forward(ctx, *args):
        from tmol.score.na_torsion.potentials._compiled import _na_sampling_fill

        inputs, counts, n_rots, policy = args[:11], args[11], args[12], args[13]
        owners, atoms, chi, offsets = _na_sampling_fill(
            list(inputs), counts, n_rots, *policy
        )
        ctx.save_for_backward(*inputs, owners, offsets)
        ctx.bases_per_polymer = policy[1]
        ctx.mark_non_differentiable(owners, atoms)
        return owners, atoms, chi

    @staticmethod
    def backward(ctx, grad_owners, grad_atoms, grad_chi):
        # Transpose of linear lookup at selected discrete samples. Torch
        # scatter preserves higher derivatives of learnable parameter tables.
        (
            types,
            base,
            pucker,
            n_combos,
            active,
            atoms,
            proton,
            wells,
            means,
            sdev,
            steps,
            owners,
            offsets,
        ) = ctx.saved_tensors
        result = [None] * 14
        if grad_chi is None:
            return tuple(result)
        type_for_rot = types[owners]
        local = (
            torch.arange(len(owners), device=owners.device, dtype=torch.int64)
            - offsets[owners]
        )
        combo = local % n_combos[type_for_rot]
        slot = torch.div(local, n_combos[type_for_rot], rounding_mode="floor")
        base_for_rot = base[type_for_rot]
        angle_grad = grad_chi[:, 0] * (torch.pi / 180.0)
        if ctx.needs_input_grad[8]:
            anti = slot < len(steps)
            index = base_for_rot * means.shape[1] + pucker[owners]
            result[8] = (
                torch.zeros_like(means)
                .flatten()
                .index_add(0, index, angle_grad * anti)
                .reshape_as(means)
            )
        if ctx.needs_input_grad[9]:
            polymer = torch.div(
                base_for_rot, ctx.bases_per_polymer, rounding_mode="floor"
            )
            result[9] = torch.zeros_like(sdev).index_add(
                0, polymer, angle_grad * steps[slot % len(steps)]
            )
        if ctx.needs_input_grad[6]:
            index = type_for_rot * proton.shape[1] + combo
            result[6] = (
                torch.zeros_like(proton)
                .reshape(proton.shape[0] * proton.shape[1], proton.shape[2])
                .index_add(0, index, grad_chi[:, 1:])
                .reshape_as(proton)
            )
        return tuple(result)
