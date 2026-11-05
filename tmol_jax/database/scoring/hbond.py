"""JAX compatibility view over the shared hydrogen-bond records."""

from typing import Dict, Tuple
import attr
from tmol_core.database.hbond import (  # noqa: F401
    GlobalParams as HBondGlobalParams,
    DonorAtomType,
    AcceptorAtomType,
    DonorTypeParam,
    AcceptorTypeParam,
    PolynomialParameters,
    PairParameters,
    HBondDatabaseRaw,
)


@attr.s(auto_attribs=True, frozen=True, slots=True)
class HBondDatabase:
    global_parameters: HBondGlobalParams
    donor_atom_types: Tuple[DonorAtomType, ...]
    acceptor_atom_types: Tuple[AcceptorAtomType, ...]
    don_type_params: Tuple[DonorTypeParam, ...]
    acc_type_params: Tuple[AcceptorTypeParam, ...]
    pair_parameters: Tuple[PairParameters, ...]
    polynomial_parameters: Tuple[PolynomialParameters, ...]
    # Pure-Python alternatives to the upstream pandas index mappers:
    # {atom_type_name: donor_type_name}
    don_type_map: Dict[str, str]
    acc_type_map: Dict[str, str]

    @property
    def donor_type_params(self):
        return self.don_type_params

    @property
    def acceptor_type_params(self):
        return self.acc_type_params

    @classmethod
    def from_file(cls, path: str) -> "HBondDatabase":
        rawdb = HBondDatabaseRaw.from_file(path)
        return cls(
            global_parameters=rawdb.global_parameters,
            donor_atom_types=rawdb.donor_atom_types,
            acceptor_atom_types=rawdb.acceptor_atom_types,
            don_type_params=rawdb.donor_type_params,
            acc_type_params=rawdb.acceptor_type_params,
            pair_parameters=rawdb.pair_parameters,
            polynomial_parameters=rawdb.polynomial_parameters,
            don_type_map={x.d: x.donor_type for x in rawdb.donor_atom_types},
            acc_type_map={x.a: x.acceptor_type for x in rawdb.acceptor_atom_types},
        )


# Aliases consistent with the task spec.
GlobalParams = HBondGlobalParams


__all__ = [
    "HBondGlobalParams",
    "GlobalParams",
    "DonorAtomType",
    "AcceptorAtomType",
    "DonorTypeParam",
    "AcceptorTypeParam",
    "PolynomialParameters",
    "PairParameters",
    "HBondDatabase",
]
