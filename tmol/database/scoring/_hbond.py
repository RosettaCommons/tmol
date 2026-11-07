import attr
import pandas

from typing import Tuple


from tmol_core.database.hbond import GlobalParams  # noqa: F401
from tmol_core.database.hbond import DonorAtomType  # noqa: F401
from tmol_core.database.hbond import AcceptorAtomType  # noqa: F401
from tmol_core.database.hbond import DonorTypeParam  # noqa: F401
from tmol_core.database.hbond import AcceptorTypeParam  # noqa: F401
from tmol_core.database.hbond import PolynomialParameters  # noqa: F401
from tmol_core.database.hbond import PairParameters  # noqa: F401
from tmol_core.database.hbond import HBondDatabaseRaw  # noqa: F401


@attr.s(auto_attribs=True, frozen=True, slots=True)
class HBondDatabase:
    global_parameters: GlobalParams
    donor_atom_types: Tuple[DonorAtomType, ...]
    donor_type_params: Tuple[DonorTypeParam, ...]
    donor_type_mapper: pandas.DataFrame

    acceptor_atom_types: Tuple[AcceptorAtomType, ...]
    acceptor_type_params: Tuple[AcceptorTypeParam, ...]
    acceptor_type_mapper: pandas.DataFrame

    pair_parameters: Tuple[PairParameters, ...]
    polynomial_parameters: Tuple[PolynomialParameters, ...]

    @classmethod
    def from_raw_hbond_db(cls, hbr):
        acc_ind = pandas.Index([x.a for x in hbr.acceptor_atom_types])
        don_ind = pandas.Index([x.d for x in hbr.donor_atom_types])

        atypes, acctypes = zip(
            *([(x.a, x.acceptor_type) for x in hbr.acceptor_atom_types])
        )
        dtypes, dontypes = zip(*([(x.d, x.donor_type) for x in hbr.donor_atom_types]))
        acc_df = pandas.DataFrame({"a": atypes, "acc_type": acctypes}, index=acc_ind)
        don_df = pandas.DataFrame({"d": dtypes, "don_type": dontypes}, index=don_ind)

        return cls(
            global_parameters=hbr.global_parameters,
            donor_atom_types=hbr.donor_atom_types,
            donor_type_params=hbr.donor_type_params,
            donor_type_mapper=don_df,
            acceptor_atom_types=hbr.acceptor_atom_types,
            acceptor_type_params=hbr.acceptor_type_params,
            acceptor_type_mapper=acc_df,
            pair_parameters=hbr.pair_parameters,
            polynomial_parameters=hbr.polynomial_parameters,
        )

    @classmethod
    def from_file(cls, path):
        hbr = HBondDatabaseRaw.from_file(path)
        return cls.from_raw_hbond_db(hbr)
