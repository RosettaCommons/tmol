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
from tmol_core.scoring.hbond import hbond_pair_parameters
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
        def _p(t):
            return torch.nn.Parameter(t, requires_grad=False)

        def _stack(ts):
            return torch.cat([t.unsqueeze(-1) for t in ts], -1)

        def _f32(ts):
            return [t.to(torch.float32) for t in ts]

        global_params = {
            n: torch.tensor(v, device=device).expand(1).to(dtype=torch.float32)
            for n, v in attr.asdict(hbond_database.global_parameters).items()
        }
        resolver = HBondParamResolver.from_database(
            chemical_database, hbond_database, device
        )
        distances = [
            p.xmax
            for p in hbond_database.polynomial_parameters
            if p.dimension == "hbgd_AHdist"
        ]
        if not distances and resolver.pair_params.donor_weight.numel():
            raise ValueError("HBond pair table has no AHdist polynomial range")
        max_ahdis = max(distances, default=0.0)

        # also store the distance of the longest possible hbond
        # by reading the set of hbond polynomials
        global_param_table = _p(
            (
                torch.cat(
                    [
                        global_params["hb_sp2_range_span"],
                        global_params["hb_sp2_BAH180_rise"],
                        global_params["hb_sp2_outer_width"],
                        global_params["hb_sp3_softmax_fade"],
                        global_params["threshold_distance"],
                        torch.tensor([max_ahdis], device=device, dtype=torch.float32),
                    ],
                    0,
                )
            ).unsqueeze(0)
        )

        pp = resolver.pair_params

        # Note: acceptor_hybridization is an integer, but can be exactly represented
        # as a float (since it is small). Pack it with the donor and acceptor weights
        # (which themselves could just be pre-multiplied?!) to reduce the number of
        # arguments to the hbond evaluation function
        pair_param_table = _p(
            _stack(
                _f32([pp.acceptor_hybridization, pp.acceptor_weight, pp.donor_weight])
            )
        )

        # Eigen allocates space for 12 Reals for an 11x1 matrix
        # so we need to pad out with 0s in between the coefficients
        # and the ranges.
        # On the same device as the coefficients, which is the host when the
        # requested device is MPS -- that backend has no float64. See the
        # comment on poly_device in HBondParamResolver._from_database.
        pad = torch.zeros(
            [pp.AHdist.coeffs.shape[0], pp.AHdist.coeffs.shape[1], 1],
            device=pp.AHdist.coeffs.device,
            dtype=torch.float64,
        )

        packed = torch.cat(
            [
                pp.AHdist.coeffs,
                pad,
                pp.AHdist.range,
                pp.AHdist.bound,
                pp.cosBAH.coeffs,
                pad,
                pp.cosBAH.range,
                pp.cosBAH.bound,
                pp.cosAHD.coeffs,
                pad,
                pp.cosAHD.range,
                pp.cosAHD.bound,
            ],
            2,
        )

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

        pair_poly_table = _p(packed)

        return cls(
            global_param_table=global_param_table,
            pair_param_table=pair_param_table,
            pair_poly_table=pair_poly_table,
        )
