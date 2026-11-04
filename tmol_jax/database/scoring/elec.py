"""Shared electrostatic parameter records and loader."""

from tmol_core.database.elec import (  # noqa: F401
    GlobalParams,
    CountPairReps,
    PartialCharges,
    ElecDatabase,
)

ElecGlobalParams = GlobalParams
atom_charges = PartialCharges
