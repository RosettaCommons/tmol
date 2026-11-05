from types import SimpleNamespace

import attr

import numpy
import pandas
import torch

from tmol.types._array import NDArray
from tmol.types._torch import Tensor
from tmol.types._tensor import TensorGroup
from tmol.types._attrs import ValidateAttrs, ConvertAttrs
from tmol.types._functional import _validate_args
from tmol.numeric._bspline import BSplineInterpolation
from tmol.utility import resolve_device
from tmol.utility.weak_identity_cache import WeakIdentityLRU

from tmol.database.scoring import DunbrackRotamerLibrary

from tmol_core.scoring.dunbrack import prepare_dunbrack_tables, _DunbrackBuilder


def _host_record(value):
    if isinstance(value, torch.Tensor):
        return value.numpy(force=True)
    if attr.has(type(value)):
        return SimpleNamespace(
            **{
                f.name: _host_record(getattr(value, f.name))
                for f in attr.fields(type(value))
            }
        )
    if isinstance(value, (list, tuple)):
        return type(value)(_host_record(v) for v in value)
    return value


def _fit_coefficients(tables):
    return [
        t.numpy()
        for t in BSplineInterpolation._coefficients_from_coordinate_tables(
            [torch.from_numpy(t) for t in tables]
        )
    ]


def _device_arrays(value, device, memo=None):
    if memo is None:
        memo = {}
    if isinstance(value, numpy.ndarray):
        if id(value) not in memo:
            tensor = torch.from_numpy(value)
            # NumPy gives empty arrays zero strides; native typed views require
            # the usual positive consumed-dimension strides even with no rows.
            memo[id(value)] = (
                tensor.to(device)
                if value.size
                else torch.empty(value.shape, dtype=tensor.dtype, device=device)
            )
        return memo[id(value)]
    if isinstance(value, tuple):
        return tuple(_device_arrays(v, device, memo) for v in value)
    return value


def _table_dataframe(mapping):
    return pandas.DataFrame(
        {"dun_table_name": numpy.asarray(list(mapping.values()), dtype=numpy.int64)},
        index=pandas.Index(mapping, name="residue_name"),
    )


@attr.s(auto_attribs=True)
class DunbrackParams(TensorGroup):
    ndihe_for_res: Tensor[torch.int32][:, :]
    dihedral_offset_for_res: Tensor[torch.int32][:, :]  # prev dihedral_offsets
    dihedral_atom_inds: Tensor[torch.int32][:, :, 4]  # prev dihedral_atom_indices
    rottable_set_for_res: Tensor[torch.int32][:, :]
    nchi_for_res: Tensor[torch.int32][:, :]
    nrotameric_chi_for_res: Tensor[torch.int32][:, :]  # ??needed??
    rotres2resid: Tensor[torch.int32][:, :]
    prob_table_offset_for_rotresidue: Tensor[torch.int32][:, :]
    rotmean_table_offset_for_residue: Tensor[torch.int32][:, :]
    rotind2tableind_offset_for_res: Tensor[torch.int32][:, :]
    rotameric_chi_desc: Tensor[torch.int32][:, :, 2]
    semirotameric_chi_desc: Tensor[torch.int32][:, :, 4]


@attr.s(auto_attribs=True)
class DunbrackScratch(TensorGroup):
    dihedrals: Tensor[torch.float][:, :]
    ddihe_dxyz: Tensor[torch.float][:, :, 4, 3]
    rotameric_rottable_assignment: Tensor[torch.int32][:, :]
    semirotameric_rottable_assignment: Tensor[torch.int32][:, :]


@attr.s(auto_attribs=True, slots=True, frozen=True)
class ScoringDunbrackDatabaseView(ConvertAttrs):
    """The tables for the dunbrack database needed for scoring
    stored on the device
    """

    rotameric_neglnprob_tables: Tensor[torch.float][:, :, :]
    rotprob_table_sizes: Tensor[torch.long][:, 2]
    rotprob_table_strides: Tensor[torch.long][:, 2]
    rotameric_mean_tables: Tensor[torch.float][:, :, :]
    rotameric_sdev_tables: Tensor[torch.float][:, :, :]
    rotmean_table_sizes: Tensor[torch.long][:, 2]
    rotmean_table_strides: Tensor[torch.long][:, 2]

    rotameric_bb_start: Tensor[torch.float][:, :]
    rotameric_bb_step: Tensor[torch.float][:, :]
    rotameric_bb_periodicity: Tensor[torch.float][:, :]

    rotameric_rotind2tableind: Tensor[torch.int32][:]
    semirotameric_rotind2tableind: Tensor[torch.int32][:]

    semirotameric_tables: Tensor[torch.float][:, :, :, :]
    semirot_table_sizes: Tensor[torch.long][:, 3]
    semirot_table_strides: Tensor[torch.long][:, 3]
    semirot_start: Tensor[torch.float][:, :]
    semirot_step: Tensor[torch.float][:, :]
    semirot_periodicity: Tensor[torch.float][:, :]


@attr.s(auto_attribs=True, slots=True, frozen=True)
class ScoringDunbrackDatabaseAux(ConvertAttrs):
    rotameric_prob_tableset_offsets: Tensor[torch.int32][:]
    rotameric_meansdev_tableset_offsets: Tensor[torch.int32][:]
    nchi_for_table_set: Tensor[torch.int32][:]
    rotameric_chi_ri2ti_offsets: Tensor[torch.int32][:]
    semirotameric_tableset_offsets: Tensor[torch.int32][:]


@attr.s(auto_attribs=True, slots=True, frozen=True)
class SamplingDunbrackDatabaseView(ConvertAttrs):
    """The tables that are needed in order to sample
    side-chain conformations.
    """

    rotameric_prob_tables: Tensor[torch.float][:, :, :]
    rotprob_table_sizes: Tensor[torch.long][:, 2]
    rotprob_table_strides: Tensor[torch.long][:, 2]
    rotameric_mean_tables: Tensor[torch.float][:, :, :]
    rotameric_sdev_tables: Tensor[torch.float][:, :, :]
    rotmean_table_sizes: Tensor[torch.long][:, 2]
    rotmean_table_strides: Tensor[torch.long][:, 2]
    rotameric_meansdev_tableset_offsets: Tensor[torch.int32][:]

    n_rotamers_for_tableset: Tensor[torch.long][:]
    n_rotamers_for_tableset_offsets: Tensor[torch.int32][:]
    sorted_rotamer_2_rotamer: Tensor[torch.long][:, :, :]

    rotameric_bb_start: Tensor[torch.float][:, :]
    rotameric_bb_step: Tensor[torch.float][:, :]
    rotameric_bb_periodicity: Tensor[torch.float][:, :]
    rotameric_bb_source_start: Tensor[torch.float][:, :]
    rotameric_bb_is_mirrored: Tensor[torch.bool][:]

    rotameric_rotind2tableind: Tensor[torch.int32][:]
    semirotameric_rotind2tableind: Tensor[torch.int32][:]
    all_chi_rotind2tableind: Tensor[torch.int32][:]
    all_chi_rotind2tableind_offsets: Tensor[torch.int32][:]

    semirotameric_tables: Tensor[torch.float][:, :, :, :]
    semirot_table_sizes: Tensor[torch.long][:, 3]
    semirot_table_strides: Tensor[torch.long][:, 3]
    semirot_start: Tensor[torch.float][:, :]
    semirot_step: Tensor[torch.float][:, :]
    semirot_periodicity: Tensor[torch.float][:, :]

    nchi_for_table_set: Tensor[torch.int32][:]
    rotwells: Tensor[torch.int32][:, :]


@attr.s(frozen=True, slots=True, auto_attribs=True)
class DunbrackParamResolver(ValidateAttrs):
    # Each default resolver owns about 67 MB of derived tables. Keep a small
    # working set, and release an entry when its immutable source expires.
    _from_dun_db_cache = WeakIdentityLRU(capacity=4)

    # These live on the device
    scoring_db: ScoringDunbrackDatabaseView
    scoring_db_aux: ScoringDunbrackDatabaseAux
    sampling_db: SamplingDunbrackDatabaseView

    # This will live on the CPU
    all_table_indices: pandas.DataFrame
    rotameric_table_indices: pandas.DataFrame
    semirotameric_table_indices: pandas.DataFrame

    device: torch.device

    @classmethod
    @_validate_args
    def from_database(cls, dun_database: DunbrackRotamerLibrary, device: torch.device):
        device = resolve_device(device)
        return cls._from_dun_db_cache.get_or_create(
            dun_database,
            (cls, device),
            lambda: cls._from_database(dun_database, device),
        )

    @classmethod
    def _from_database(cls, dun_database, device):
        prepared = prepare_dunbrack_tables(
            _host_record(dun_database), _fit_coefficients
        )
        memo = {}

        def record(record_type, arrays):
            return record_type(
                **{
                    name: _device_arrays(value, device, memo)
                    for name, value in arrays.items()
                }
            )

        return cls(
            scoring_db=record(ScoringDunbrackDatabaseView, prepared.scoring_db),
            scoring_db_aux=record(ScoringDunbrackDatabaseAux, prepared.scoring_db_aux),
            sampling_db=record(SamplingDunbrackDatabaseView, prepared.sampling_db),
            all_table_indices=_table_dataframe(prepared.all_table_indices),
            rotameric_table_indices=_table_dataframe(prepared.rotameric_table_indices),
            semirotameric_table_indices=_table_dataframe(
                prepared.semirotameric_table_indices
            ),
            device=device,
        )

    @classmethod
    def _create_table_indices(cls, all_table_names, dun_lookup, n_rotameric):
        return tuple(
            _table_dataframe(v)
            for v in _DunbrackBuilder(_fit_coefficients)._create_table_indices(
                _host_record(all_table_names),
                _host_record(dun_lookup),
                _host_record(n_rotameric),
            )
        )

    @classmethod
    def _create_nchi_for_table_set(cls, all_rotlibs, device):
        return _device_arrays(
            _DunbrackBuilder(_fit_coefficients)._create_nchi_for_table_set(
                _host_record(all_rotlibs)
            ),
            device,
        )

    @classmethod
    def _create_prob_table_offsets(cls, all_rotlibs, device):
        return _device_arrays(
            _DunbrackBuilder(_fit_coefficients)._create_prob_table_offsets(
                _host_record(all_rotlibs)
            ),
            device,
        )

    @classmethod
    def _compute_rotprob_coeffs(cls, all_rotlibs, device):
        return _device_arrays(
            _DunbrackBuilder(_fit_coefficients)._compute_rotprob_coeffs(
                _host_record(all_rotlibs)
            ),
            device,
        )

    @classmethod
    def _create_rot_mean_offsets(cls, all_rotlibs, device):
        return _device_arrays(
            _DunbrackBuilder(_fit_coefficients)._create_rot_mean_offsets(
                _host_record(all_rotlibs)
            ),
            device,
        )

    @classmethod
    def _calculate_rot_mean_coeffs(cls, all_rotlibs, device):
        return _device_arrays(
            _DunbrackBuilder(_fit_coefficients)._calculate_rot_mean_coeffs(
                _host_record(all_rotlibs)
            ),
            device,
        )

    @classmethod
    def _calculate_rot_sdev_coeffs(cls, all_rotlibs, device):
        return _device_arrays(
            _DunbrackBuilder(_fit_coefficients)._calculate_rot_sdev_coeffs(
                _host_record(all_rotlibs)
            ),
            device,
        )

    @classmethod
    def _create_rot_periodicities(cls, all_rotlibs, device):
        return _device_arrays(
            _DunbrackBuilder(_fit_coefficients)._create_rot_periodicities(
                _host_record(all_rotlibs)
            ),
            device,
        )

    @classmethod
    def _create_rot_source_starts(cls, all_rotlibs, target_starts, device):
        return _device_arrays(
            _DunbrackBuilder(_fit_coefficients)._create_rot_source_starts(
                _host_record(all_rotlibs), _host_record(target_starts)
            ),
            device,
        )

    @classmethod
    def _create_rotind2tableinds(cls, dun_database, device):
        return _device_arrays(
            _DunbrackBuilder(_fit_coefficients)._create_rotind2tableinds(
                _host_record(dun_database)
            ),
            device,
        )

    @classmethod
    def _create_rotameric_rotind2tableind_offsets(cls, dun_database, device):
        return _device_arrays(
            _DunbrackBuilder(
                _fit_coefficients
            )._create_rotameric_rotind2tableind_offsets(_host_record(dun_database)),
            device,
        )

    @classmethod
    def _create_all_chi_rotind2tableind_offsets(cls, dun_database, device):
        return _device_arrays(
            _DunbrackBuilder(_fit_coefficients)._create_all_chi_rotind2tableind_offsets(
                _host_record(dun_database)
            ),
            device,
        )

    @classmethod
    def _calc_semirot_coeffs(cls, dun_database, device):
        return _device_arrays(
            _DunbrackBuilder(_fit_coefficients)._calc_semirot_coeffs(
                _host_record(dun_database)
            ),
            device,
        )

    @classmethod
    def _create_semirot_periodicity(
        cls, dun_database, device, backbone_start, backbone_step
    ):
        return _device_arrays(
            _DunbrackBuilder(_fit_coefficients)._create_semirot_periodicity(
                _host_record(dun_database),
                _host_record(backbone_start),
                _host_record(backbone_step),
            ),
            device,
        )

    @classmethod
    def _create_semirot_offsets(cls, dun_database, device):
        return _device_arrays(
            _DunbrackBuilder(_fit_coefficients)._create_semirot_offsets(
                _host_record(dun_database)
            ),
            device,
        )

    @classmethod
    def create_sorted_rot_2_rot(cls, all_rotlibs, device):
        return _device_arrays(
            _DunbrackBuilder(_fit_coefficients).create_sorted_rot_2_rot(
                _host_record(all_rotlibs)
            ),
            device,
        )

    @classmethod
    def create_rotamer_well_table(cls, all_rotlibs, device):
        return _device_arrays(
            _DunbrackBuilder(_fit_coefficients).create_rotamer_well_table(
                _host_record(all_rotlibs)
            ),
            device,
        )

    # TO DO: DECIDE IF THIS IS PART OF THE API OF THIS CLASS
    # SURE SEEMS LIKE IT IS
    def _indices_from_names(
        self,
        dataframe: pandas.DataFrame,
        names: NDArray[object][:, :],
        device: torch.device,
    ) -> Tensor[torch.int64][:, :]:
        names_flat = names.ravel()
        inds = dataframe.index.get_indexer(names_flat)
        inds[inds != -1] = dataframe.iloc[inds[inds != -1]]["dun_table_name"].values
        inds = inds.reshape(names.shape)
        return torch.tensor(inds, dtype=torch.int64, device=device)
