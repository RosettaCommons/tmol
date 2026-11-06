import torch
import attr
import numpy

from typing import Tuple
from tmol_core.pack.dunbrack import (  # noqa: F401
    DunSamplerRTCache,
    DunSamplerPBTCache,
    DunbrackSamplingMetadata,
    prepare_dunbrack_sampler_residue,
    pack_dunbrack_sampler_metadata,
)

from tmol_core.pack.sampling_request import (
    prepare_dunbrack_sampling_request,
    backbone_dihedral_indices,
    apply_frozen_chi,
)

from tmol.types._torch import Tensor
from tmol.types._functional import _validate_args
from tmol.score.dunbrack._params import DunbrackParamResolver
from tmol.score._annotation_cache import (
    AnnotationKey,
    _cached_annotation,
    _store_annotation,
)

from tmol.pack.rotamer._chi_sampler import ChiSampler
from tmol.pack.rotamer._single_residue_kinforest import (
    _construct_single_residue_kinforest,
)
from tmol.pack._packer_task import DEFAULT_CHI_SAMPLE_LIMIT
from tmol.pack.rotamer._chi_budget import (
    _sampler_for_task,
)

from tmol.database import ParameterDatabase

# from tmol.pack.rotamer.dunbrack import _compiled  # noqa F401
from tmol.pack._packer_task import SetPackerTask
from tmol.chemical import RefinedResidueType
from tmol.pose._packed_block_types import PackedBlockTypes
from tmol.pose._pose_stack import PoseStack


def _host_array(value):
    return value.detach().cpu().numpy()


@attr.s(auto_attribs=True, eq=False)
class DunbrackChiSampler(ChiSampler):
    """Sample amino-acid side-chain conformers from Dunbrack libraries."""

    dun_param_resolver: DunbrackParamResolver
    # Existing canonical proton grids exceed the generic expansion threshold.
    # Retain them while bounding supplemental products; explicit tasks override both.
    chi_sample_expanded_limit: int = DEFAULT_CHI_SAMPLE_LIMIT
    chi_sample_limit: int = DEFAULT_CHI_SAMPLE_LIMIT

    def __eq__(self, other):
        if not isinstance(other, DunbrackChiSampler):
            return NotImplemented
        return (
            self.dun_param_resolver is other.dun_param_resolver
            and self.chi_sample_expanded_limit == other.chi_sample_expanded_limit
            and self.chi_sample_limit == other.chi_sample_limit
        )

    def __hash__(self):
        return hash(
            (
                id(self.dun_param_resolver),
                self.chi_sample_expanded_limit,
                self.chi_sample_limit,
            )
        )

    def __attrs_post_init__(self):
        self._annotation_key()

    def _annotation_key(self):
        resolver = self.dun_param_resolver
        if getattr(self, "_annotation_resolver", None) is not resolver:
            # Resolver tables are immutable inputs. Keep small lookup metadata
            # on the host instead of creating/synchronizing a tensor per type.
            if not resolver.all_table_indices.index.is_unique:
                raise ValueError("Dunbrack residue-to-library names must be unique")
            self._library_indices = resolver.all_table_indices[
                "dun_table_name"
            ].to_dict()
            self._nchi_for_table = tuple(
                resolver.scoring_db_aux.nchi_for_table_set.cpu().tolist()
            )
            self._library_sizes = tuple(
                resolver.sampling_db.n_rotamers_for_tableset.cpu().tolist()
            )
            self._sampling_metadata = DunbrackSamplingMetadata(
                self._library_indices, self._nchi_for_table, self._library_sizes
            )
            self._annotation_resolver = resolver
            self._annotation_settings = None
        settings = (self.chi_sample_expanded_limit, self.chi_sample_limit)
        if self._annotation_settings != settings:
            self._sampler_key = AnnotationKey.from_sources(resolver, settings=settings)
            self._annotation_settings = settings
        return self._sampler_key

    @property
    def device(self):
        return self.dun_param_resolver.device

    @classmethod
    @_validate_args
    def from_database(cls, param_resolver: DunbrackParamResolver):
        return cls(dun_param_resolver=param_resolver)

    @classmethod
    def sampler_name(cls):
        return "DunbrackChiSampler"

    def _library_index(self, lib_name: str):
        """Index of a rotamer library by name, or -1 if there is none."""
        self._annotation_key()
        return int(self._library_indices.get(lib_name, -1))

    def annotate_residue_type(self, restype: RefinedResidueType):
        key = self._annotation_key()
        cached = _cached_annotation(restype, "_dun_sampler_annotation", key)
        if cached is not None:
            return cached

        cache = prepare_dunbrack_sampler_residue(
            restype,
            self._sampling_metadata,
            expanded_limit=self.chi_sample_expanded_limit,
            limit=self.chi_sample_limit,
            prepare_kinematics=_construct_single_residue_kinforest,
        )
        setattr(restype, "dun_sampler_cache", cache)
        # sample_chi_for_poses reads the alias, not the returned snapshot, so a
        # later cache hit has to restore it or one sampler reads another's tables.
        return _store_annotation(
            restype,
            "_dun_sampler_annotation",
            key,
            cache,
            fields=("dun_sampler_cache",),
        )

    def annotate_packed_block_types(self, packed_block_types: PackedBlockTypes):
        key = self._annotation_key()
        cached = _cached_annotation(packed_block_types, "_dun_sampler_annotation", key)
        if cached is not None:
            return cached

        for rt in packed_block_types.active_block_types:
            self.annotate_residue_type(rt)

        block_types = packed_block_types.active_block_types
        host = pack_dunbrack_sampler_metadata(
            block_types,
            [rt.dun_sampler_cache for rt in block_types],
            [self.defines_rotamers_for_rt(rt) for rt in block_types],
        )
        cache = attr.evolve(
            host,
            **{
                name: torch.tensor(
                    getattr(host, name), device=packed_block_types.device
                )
                for name in (
                    "bbdihe_uaids",
                    "chi_defining_atom",
                    "non_dunbrack_sample_counts",
                    "non_dunbrack_samples",
                    "defines_rotamers_for_bts",
                    "rottable_set_for_bt",
                )
            },
        )
        setattr(packed_block_types, "dun_sampler_cache", cache)
        setattr(packed_block_types, "_dun_sampler_host_metadata", host)
        return _store_annotation(
            packed_block_types,
            "_dun_sampler_annotation",
            key,
            cache,
            fields=("dun_sampler_cache", "_dun_sampler_host_metadata"),
        )

    def _library_for_rt(self, rt: RefinedResidueType):
        self._annotation_key()
        return self._sampling_metadata.library_for_type(rt)

    def defines_rotamers_for_rt(self, rt: RefinedResidueType):
        self._annotation_key()
        return self._sampling_metadata.defines_rotamers(rt)

    def defines_rotamers_for_bts(
        self, pbt: PackedBlockTypes, bt_inds: Tensor[torch.int64]
    ) -> Tensor[torch.bool]:
        self.annotate_packed_block_types(pbt)
        return pbt.dun_sampler_cache.defines_rotamers_for_bts[bt_inds]

    def first_sc_atoms_for_rt(self, rt: RefinedResidueType):
        self._annotation_key()
        return self._sampling_metadata.sidechain_roots(rt)

    def sample_chi_for_poses(
        self, pose_stack: PoseStack, task: SetPackerTask
    ) -> Tuple[
        Tensor[torch.int32][:],  # n_rots_for_rt
        Tensor[torch.int32][:],  # rt_for_rotamer
        Tensor[torch.int32][:, :],  # chi_defining_atom_for_rotamer
        Tensor[torch.float32][:, :],  # chi_for_rotamers
    ]:
        configured, task = _sampler_for_task(self, task)
        if configured is not self:
            return configured.sample_chi_for_poses(pose_stack, task)
        assert self.device == pose_stack.coords.device
        pbt = pose_stack.packed_block_types
        self.annotate_packed_block_types(pbt)
        sampler_index = task.conformer_sampler_index[id(self)]
        request = prepare_dunbrack_sampling_request(
            pbt.active_block_types,
            pbt._dun_sampler_host_metadata,
            _host_array(pose_stack.block_type_ind),
            _host_array(pose_stack.block_coord_offset),
            _host_array(pose_stack.inter_residue_connections),
            _host_array(pbt.atom_downstream_of_conn),
            pose_stack.max_n_pose_atoms,
            _host_array(task.cons_bt_pose),
            _host_array(task.cons_bt_block),
            _host_array(task.cons_bt_block_type),
            _host_array(task.cons_bt_which_block_type),
            _host_array(task.is_cons_bt_allowed),
            _host_array(task.per_block_conformer_sampler_allowed[:, :, sampler_index]),
            _host_array(task.per_block_chi_expansion),
        )
        if not request.has_allowed:
            empty_chi = torch.zeros((0, 4), dtype=torch.int32, device=self.device)
            return (
                torch.zeros(request.n_gbt, dtype=torch.int32, device=self.device),
                torch.zeros(0, dtype=torch.int32, device=self.device),
                empty_chi,
                empty_chi.float(),
            )
        measured = numpy.empty(0, dtype=numpy.float64)
        if request.frozen_requests:
            from tmol.pose._util import _measure_torsions

            with torch.no_grad():
                measured = _measure_torsions(
                    pose_stack, request.frozen_requests, degrees=False
                )
        apply_frozen_chi(request, measured)

        def device_array(value):
            result = torch.tensor(value, device=self.device)
            # NumPy empty arrays can retain zero strides even after copying.
            # Native vector views require positive consumed-axis strides.
            if not result.numel():
                result = torch.empty(
                    result.shape, dtype=result.dtype, device=self.device
                )
            return result

        sampled_chi = self.launch_rotamer_building(
            pose_stack.coords.reshape(-1, 3),
            *(
                device_array(getattr(request, name))
                for name in (
                    "n_dihe_for_block",
                    "dihedral_offset_for_block",
                    "dihedral_atom_inds",
                    "bubl_and_rottable_set_for_bbt",
                    "chi_expansion_for_bbt",
                    "non_dunbrack_expansion_for_bbt",
                    "non_dunbrack_expansion_counts_for_bbt",
                    "prob_cumsum_limit_for_bbt",
                    "n_chi_for_bbt",
                )
            ),
            max_samples_per_restype=max(
                getattr(task, "chi_sample_budget", None) or (0,)
            ),
        )
        return self.package_samples_for_output(
            pbt,
            task,
            request.n_gbt,
            device_array(request.bbt_to_gbt),
            device_array(request.block_type_ind_for_bbt),
            request.max_n_chi,
            sampled_chi,
        )

    def atom_indices_for_backbone_dihedral(
        self, pose_stack: PoseStack, bb_dihedral_ind: int
    ):
        pbt = pose_stack.packed_block_types
        self.annotate_packed_block_types(pbt)
        indices = backbone_dihedral_indices(
            pbt._dun_sampler_host_metadata,
            _host_array(pose_stack.block_type_ind),
            _host_array(pose_stack.block_coord_offset),
            _host_array(pose_stack.inter_residue_connections),
            _host_array(pbt.atom_downstream_of_conn),
            pose_stack.max_n_pose_atoms,
        )
        return torch.tensor(indices[:, :, bb_dihedral_ind], device=self.device)

    def launch_rotamer_building(
        self,
        coords,
        ndihe_for_res,
        dihedral_offset_for_res,
        dihedral_atom_inds,
        bubl_and_rottable_set_for_buildable_restype,
        chi_expansion_for_buildable_restype,
        non_dunbrack_expansion_for_buildable_restype,
        non_dunbrack_expansion_counts_for_buildable_restype,
        prob_cumsum_limit_for_buildable_restype,
        nchi_for_buildable_restype,
        max_samples_per_restype=0,
    ):
        from ._compiled import _dun_sample_chi

        return _dun_sample_chi(
            coords,
            self.dun_param_resolver.sampling_db.rotameric_prob_tables,
            self.dun_param_resolver.sampling_db.rotprob_table_sizes,
            self.dun_param_resolver.sampling_db.rotprob_table_strides,
            self.dun_param_resolver.sampling_db.rotameric_mean_tables,
            self.dun_param_resolver.sampling_db.rotameric_sdev_tables,
            self.dun_param_resolver.sampling_db.rotmean_table_sizes,
            self.dun_param_resolver.sampling_db.rotmean_table_strides,
            self.dun_param_resolver.sampling_db.rotameric_meansdev_tableset_offsets,
            self.dun_param_resolver.sampling_db.rotameric_bb_start,
            self.dun_param_resolver.sampling_db.rotameric_bb_step,
            self.dun_param_resolver.sampling_db.rotameric_bb_periodicity,
            self.dun_param_resolver.sampling_db.rotameric_bb_source_start,
            self.dun_param_resolver.sampling_db.rotameric_bb_is_mirrored,
            self.dun_param_resolver.sampling_db.semirotameric_tables,
            self.dun_param_resolver.sampling_db.semirot_table_sizes,
            self.dun_param_resolver.sampling_db.semirot_table_strides,
            self.dun_param_resolver.sampling_db.semirot_start,
            self.dun_param_resolver.sampling_db.semirot_step,
            self.dun_param_resolver.sampling_db.semirot_periodicity,
            self.dun_param_resolver.sampling_db.rotameric_rotind2tableind,
            self.dun_param_resolver.sampling_db.semirotameric_rotind2tableind,
            self.dun_param_resolver.sampling_db.all_chi_rotind2tableind,
            self.dun_param_resolver.sampling_db.all_chi_rotind2tableind_offsets,
            self.dun_param_resolver.sampling_db.n_rotamers_for_tableset,
            self.dun_param_resolver.sampling_db.n_rotamers_for_tableset_offsets,
            self.dun_param_resolver.sampling_db.sorted_rotamer_2_rotamer,
            self.dun_param_resolver.sampling_db.nchi_for_table_set,
            self.dun_param_resolver.sampling_db.rotwells,
            ndihe_for_res,
            dihedral_offset_for_res,
            dihedral_atom_inds,
            bubl_and_rottable_set_for_buildable_restype,
            chi_expansion_for_buildable_restype,
            non_dunbrack_expansion_for_buildable_restype,
            non_dunbrack_expansion_counts_for_buildable_restype,
            prob_cumsum_limit_for_buildable_restype,
            nchi_for_buildable_restype,
            max_samples_per_restype,
        )

    def package_samples_for_output(
        self,
        pbt: PackedBlockTypes,
        task: SetPackerTask,
        n_gbt_total: int,
        bbt_to_gbt: Tensor[torch.int64][:],
        block_type_ind_for_brt: Tensor[torch.int64][:],
        max_n_chi: int,
        sampled_chi,
    ):
        n_bbt = bbt_to_gbt.shape[0]
        assert sampled_chi[0].shape[0] == n_bbt
        assert sampled_chi[1].shape[0] == n_bbt

        n_rots_for_rt = torch.zeros(
            (n_gbt_total,), dtype=torch.int32, device=self.device
        )
        n_rots_for_rt[bbt_to_gbt] = sampled_chi[0]
        n_rots_for_rt_offsets = torch.zeros_like(n_rots_for_rt)
        n_rots_for_rt_offsets[bbt_to_gbt] = sampled_chi[1]

        brt_for_rotamer = sampled_chi[2]
        chi_for_rotamers = sampled_chi[3]

        # Now lets map back to the original set of rts per block type.
        # lots of reindxing below
        gbt_for_rotamer = bbt_to_gbt[brt_for_rotamer].to(torch.int32)

        pbt_cda = pbt.dun_sampler_cache.chi_defining_atom
        chi_defining_atom_for_rotamer = torch.full(
            (chi_for_rotamers.shape[0], max_n_chi),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        max_n_chi = min((max_n_chi, pbt_cda.shape[1]))

        if chi_for_rotamers.shape[1] < max_n_chi:
            # we must keep chi_for_rotamers
            # and chi_defining_atom_for_rotamer the
            # same shape
            new_chi_for_rotamers = torch.zeros(
                (chi_for_rotamers.shape[0], max_n_chi),
                dtype=chi_for_rotamers.dtype,
                device=chi_for_rotamers.device,
            )
            old_n_chi = chi_for_rotamers.shape[1]
            new_chi_for_rotamers[:, :old_n_chi] = chi_for_rotamers
            chi_for_rotamers = new_chi_for_rotamers

        chi_defining_atom_for_rotamer[:, :max_n_chi] = pbt_cda[
            block_type_ind_for_brt[brt_for_rotamer.to(torch.int64)], :max_n_chi
        ]
        return (
            n_rots_for_rt,
            gbt_for_rotamer,
            chi_defining_atom_for_rotamer,
            chi_for_rotamers,
        )


def create_dunbrack_sampler_from_database(
    param_db: ParameterDatabase, device: torch.device
) -> DunbrackChiSampler:
    """Create a DunbrackChiSampler from the default database.

    Args:
        param_db: The parameter database containing Dunbrack parameters
        device: The device to use for the sampler. An unindexed CUDA device
            resolves to the current CUDA device.

    Returns:
        DunbrackChiSampler: Configured sampler for rotamer building
    """
    param_resolver = DunbrackParamResolver.from_database(param_db.scoring.dun, device)
    return DunbrackChiSampler.from_database(param_resolver)
