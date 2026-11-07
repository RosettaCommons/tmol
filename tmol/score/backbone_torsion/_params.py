from tmol_core.scoring.torsion_ownership import _omega_connection  # noqa: F401
import attr

import pandas
import numpy
import torch

import toolz.functoolz

from tmol.types._torch import Tensor
from tmol.types._attrs import ValidateAttrs, ConvertAttrs
from tmol.types._functional import _validate_args
from tmol.numeric._bspline import BSplineInterpolation
from tmol_core.scoring.backbone_torsion import (
    BackboneTorsionPreparation,
    prepare_backbone_tables,
    _mirror_grid as _host_mirror_grid,
)

from tmol.database.scoring import (
    RamaDatabase,
    OmegaBBDepDatabase,
)


def _mirror_grid(table, bbstart, bbstep):
    """Reflect a periodic 2-D torsion grid through the origin.

    Bin k covers start + k*step, so the bin holding -x is (c - k) modulo the
    number of bins, with c = -2*start/step. The two backbone grids are
    registered differently -- rama starts at -180 and omega at -175 -- so the
    reflection is derived from the grid rather than assumed.
    """
    mirrored = _host_mirror_grid(table.detach().cpu().numpy(), bbstart, bbstep)
    return torch.from_numpy(numpy.ascontiguousarray(mirrored)).to(device=table.device)


# the rama database packed into a single tensor
@attr.s(auto_attribs=True, slots=True, frozen=True)
class PackedRamaDatabase(ConvertAttrs):
    tables: Tensor[torch.float][:, :, :]
    bbsteps: Tensor[torch.float][:, :]
    bbstarts: Tensor[torch.float][:, :]


# the omega bb-dep database packed into a single tensor
@attr.s(auto_attribs=True, slots=True, frozen=True)
class PackedOmegaDatabase(ConvertAttrs):
    tables: Tensor[torch.float][:, 2, :, :]
    bbsteps: Tensor[torch.float][:, :]
    bbstarts: Tensor[torch.float][:, :]


@attr.s(frozen=True, slots=True, auto_attribs=True)
class BackboneTorsionParamResolver(ValidateAttrs):
    _from_rama_db_cache = {}

    # respair -> table index mapping
    rama_lookup: pandas.DataFrame
    omega_lookup: pandas.DataFrame

    # rama tables (spline coeffs)
    rama_params: PackedRamaDatabase
    omega_params: PackedOmegaDatabase

    # tables before the mirrored copies; table i has its mirror at i + n
    n_rama_tables: int
    n_omega_tables: int

    device: torch.device
    host_preparation: object = None

    @classmethod
    @_validate_args
    @toolz.functoolz.memoize(
        cache=_from_rama_db_cache,
        key=lambda args, kwargs: (
            args[1].uniq_id,
            args[2].uniq_id,
            args[3].type,
            args[3].index,
        ),
    )
    def from_database(
        cls,
        rama_database: RamaDatabase,
        bbdep_omega_database: OmegaBBDepDatabase,
        device: torch.device,
    ):
        rama_rows = rama_database.rama_tables
        omega_rows = bbdep_omega_database.bbdep_omega_tables
        host = BackboneTorsionPreparation(
            rama_database.rama_lookup,
            bbdep_omega_database.bbdep_omega_lookup,
            [r.table_id for r in rama_rows],
            [r.table_id for r in omega_rows],
        )

        def fit(raw):
            tensor = torch.from_numpy(numpy.ascontiguousarray(raw))
            return BSplineInterpolation._coefficients_from_coordinates(tensor).numpy()

        rama, omega = prepare_backbone_tables(
            [attr.evolve(r, table=r.table.detach().cpu().numpy()) for r in rama_rows],
            [
                attr.evolve(
                    r,
                    mu=r.mu.detach().cpu().numpy(),
                    sigma=r.sigma.detach().cpu().numpy(),
                )
                for r in omega_rows
            ],
            fit,
        )

        def lookup_frame(lookup):
            rows = [
                dict(
                    res_middle=key[0],
                    res_upper=key[1],
                    table_id=value[0],
                    invert_phi=value[1],
                    invert_psi=value[2],
                )
                for key, value in lookup.items()
            ]
            return pandas.DataFrame.from_records(rows).set_index(
                ["res_middle", "res_upper"]
            )

        def on_device(data, cls):
            return cls(
                **{
                    f.name: torch.from_numpy(getattr(data, f.name)).to(device)
                    for f in attr.fields(type(data))
                }
            )

        return cls(
            rama_lookup=lookup_frame(host.rama_lookup),
            omega_lookup=lookup_frame(host.omega_lookup),
            rama_params=on_device(rama, PackedRamaDatabase),
            omega_params=on_device(omega, PackedOmegaDatabase),
            n_rama_tables=host.n_rama_tables,
            n_omega_tables=host.n_omega_tables,
            device=device,
            host_preparation=host,
        )
