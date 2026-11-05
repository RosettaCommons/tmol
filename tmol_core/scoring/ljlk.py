"""Ordered LJ/LK host parameters and validation of the active chemistry."""

from dataclasses import dataclass

import numpy as np

LJLK_FIELDS = ("lj_radius", "lj_wdepth", "lk_dgfree", "lk_lambda", "lk_volume")


@dataclass
class LJLKTypeData:
    names: tuple
    values: np.ndarray
    is_carbon_lk: np.ndarray
    invalid_type_parameters: dict

    def validate_block_type(self, block_type):
        validate_block_type(block_type, self.invalid_type_parameters)


def validate_block_type(block_type, invalid_type_parameters):
    invalid = [
        f"{atom.name} ({atom.atom_type}): {', '.join(fields)}"
        for atom in block_type.atoms
        if (fields := invalid_type_parameters.get(atom.atom_type))
    ]
    if invalid:
        raise ValueError(
            f"Residue {block_type.name} has missing or non-finite LJLK parameters: "
            + "; ".join(invalid)
        )


def ljlk_type_parameters(names, database):
    """Pack float32 kernel values in the supplied chemical type order.

    Missing unused types and explicit padding retain NaNs. Callers validate
    active types before gathering or replacing padding with safe values.
    """
    names = tuple(names)
    records = {row.name: row for row in database.atom_type_parameters}
    if len(records) != len(database.atom_type_parameters):
        raise ValueError("LJLK atom type parameter names must be unique")
    values = np.full((len(names), len(LJLK_FIELDS)), np.nan, dtype=np.float32)
    with np.errstate(over="ignore"):
        for i, name in enumerate(names):
            if name in records:
                values[i] = [getattr(records[name], field) for field in LJLK_FIELDS]
    finite = np.isfinite(values)
    invalid = {
        name: tuple(field for field, valid in zip(LJLK_FIELDS, finite[i]) if not valid)
        for i in np.flatnonzero(~finite.all(axis=1))
        if (name := names[i]) is not None
    }
    return LJLKTypeData(
        names,
        values,
        np.asarray([name in ("CH1", "CH2", "CH3", "Caro") for name in names]),
        invalid,
    )
