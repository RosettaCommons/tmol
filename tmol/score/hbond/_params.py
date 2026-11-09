import attr
from typing import Union

import numpy
import pandas
import torch

import toolz

from tmol.types._torch import Tensor
from tmol.types._tensor import TensorGroup
from tmol.types._attrs import ValidateAttrs, ConvertAttrs
from tmol.types._functional import _validate_args
from tmol.database.scoring import HBondDatabase
from tmol.database.chemical import ChemicalDatabase
from tmol.database._patched_chemdb import PatchedChemicalDatabase

from .._chemical_database import AcceptorHybridization  # noqa: F401
from tmol_core.scoring.hbond import hbond_pair_parameters, native_hbond_parameters
from tmol.utility import resolve_device
from tmol.utility.weak_identity_cache import WeakIdentityLRU


@attr.s(auto_attribs=True, slots=True, frozen=True)
class HBondPolyParams(TensorGroup, ConvertAttrs):
    range: Tensor[torch.double][..., 2]
    bound: Tensor[torch.double][..., 2]
    coeffs: Tensor[torch.double][..., 11]

    def __setitem__(self, idx, value):
        self.range[idx] = value.range[idx]
        self.bound[idx] = value.bound[idx]
        self.coeffs[idx] = value.coeffs[idx]

    def to(self, device: torch.device):
        return type(self)(
            **toolz.valmap(lambda t: t.to(device), attr.asdict(self, recurse=False))
        )

    @classmethod
    def full(cls, shape, fill_value):
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        return cls(
            range=torch.full(shape + (2,), fill_value),
            bound=torch.full(shape + (2,), fill_value),
            coeffs=torch.full(shape + (11,), fill_value),
        )


@attr.s(auto_attribs=True, frozen=True, slots=True)
class HBondPairParams(TensorGroup, ValidateAttrs):
    donor_weight: Tensor[torch.float32][...]
    acceptor_weight: Tensor[torch.float32][...]
    acceptor_hybridization: Tensor[torch.int32][...]
    AHdist: HBondPolyParams
    cosBAH: HBondPolyParams
    cosAHD: HBondPolyParams

    def to(self, device: torch.device):
        return type(self)(
            **toolz.valmap(lambda t: t.to(device), attr.asdict(self, recurse=False))
        )

    @classmethod
    def full(cls, shape, fill_value):
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        return cls(
            donor_weight=torch.full(shape, fill_value, dtype=torch.float),
            acceptor_weight=torch.full(shape, fill_value, dtype=torch.float),
            acceptor_hybridization=torch.full(
                shape, numpy.nan_to_num(fill_value), dtype=torch.int32
            ),  # nan_to_num fill value for integer dtype
            AHdist=HBondPolyParams.full(shape, fill_value),
            cosBAH=HBondPolyParams.full(shape, fill_value),
            cosAHD=HBondPolyParams.full(shape, fill_value),
        )


@attr.s(auto_attribs=True, frozen=True, slots=True)
class HBondParamResolver(ValidateAttrs):
    _from_db_cache = WeakIdentityLRU()

    donor_type_index: pandas.Index = attr.ib()
    acceptor_type_index: pandas.Index = attr.ib()

    pair_params: HBondPairParams = attr.ib()
    device: torch.device = attr.ib()

    @classmethod
    @_validate_args
    def from_database(
        cls,
        chemical_database: Union[ChemicalDatabase, PatchedChemicalDatabase],
        hbond_database: HBondDatabase,
        device: torch.device,
    ):
        device = resolve_device(device)
        return cls._from_db_cache.get_or_create_many(
            (chemical_database, hbond_database),
            device,
            lambda: cls._from_database(chemical_database, hbond_database, device),
        )

    @classmethod
    def _from_database(cls, chemical_database, hbond_database, device):
        host = hbond_pair_parameters(chemical_database, hbond_database)
        # MPS polynomial coefficients stay double precision on the host; the
        # existing compact adapter transports hi/lo float32 words to Metal.
        poly_device = torch.device("cpu") if device.type == "mps" else device

        def polynomial(values):
            return HBondPolyParams(
                range=torch.tensor(values[..., :2], device=poly_device),
                bound=torch.tensor(values[..., 2:4], device=poly_device),
                coeffs=torch.tensor(values[..., 4:], device=poly_device),
            )

        return cls(
            donor_type_index=pandas.Index(host.donor_names),
            acceptor_type_index=pandas.Index(host.acceptor_names),
            pair_params=HBondPairParams(
                donor_weight=torch.tensor(host.donor_weight, device=device),
                acceptor_weight=torch.tensor(host.acceptor_weight, device=device),
                acceptor_hybridization=torch.tensor(
                    host.acceptor_hybridization, device=device
                ),
                **{
                    kind: polynomial(values)
                    for kind, values in host.polynomials.items()
                },
            ),
            device=device,
        )


@attr.s(auto_attribs=True, frozen=True, slots=True)
class CompactedHBondDatabase(ValidateAttrs):
    """Store the hbond evaluation parameters in a compact form"""

    _from_db_cache = WeakIdentityLRU()

    global_param_table: Tensor[torch.float32][:, :]
    pair_param_table: Tensor[torch.float32][:, :, :]
    # float64 on the host and CUDA; on Metal, which has no float64 tensor,
    # each coefficient is an unevaluated hi + lo float32 pair and the trailing
    # dimension is twice as long. Validated loosely for that reason.
    pair_poly_table: torch.Tensor

    @classmethod
    @_validate_args
    def from_database(
        cls,
        chemical_database: Union[ChemicalDatabase, PatchedChemicalDatabase],
        hbond_database: HBondDatabase,
        device: torch.device,
    ):
        device = resolve_device(device)
        return cls._from_db_cache.get_or_create_many(
            (chemical_database, hbond_database),
            device,
            lambda: cls._from_database(chemical_database, hbond_database, device),
        )

    @classmethod
    def _from_database(cls, chemical_database, hbond_database, device):
        host = native_hbond_parameters(chemical_database, hbond_database)

        def tensor(values, target):
            # Empty NumPy arrays may carry zero strides; native struct views
            # still require a contiguous consumed parameter dimension.
            return (
                torch.tensor(values, device=target)
                if values.size
                else torch.empty(
                    values.shape, dtype=torch.from_numpy(values).dtype, device=target
                )
            )

        global_param_table = torch.nn.Parameter(
            tensor(host["global_param_table"], device), requires_grad=False
        )
        pair_param_table = torch.nn.Parameter(
            tensor(host["pair_param_table"], device), requires_grad=False
        )
        poly_device = torch.device("cpu") if device.type == "mps" else device
        packed = tensor(host["pair_poly_table"], poly_device)

        if device.type == "mps":
            # Metal has no float64 at the tensor level either, so each
            # coefficient crosses as an unevaluated hi + lo float32 pair --
            # the layout of tmol::portable::DoubleFloat<float>, which is the
            # same eight bytes a double occupied, so the struct keeps its
            # footprint. Rounding to plain float32 instead costs a relative
            # error of 1.96 on these degree-10 fits, enough to flip the sign
            # of a near-zero value.
            hi = packed.to(torch.float32)
            lo = (packed - hi.to(torch.float64)).to(torch.float32)
            packed = torch.stack([hi, lo], dim=-1).reshape(
                packed.shape[:-1] + (2 * packed.shape[-1],)
            )
            packed = packed.to(device)

        pair_poly_table = torch.nn.Parameter(packed, requires_grad=False)

        return cls(
            global_param_table=global_param_table,
            pair_param_table=pair_param_table,
            pair_poly_table=pair_poly_table,
        )
