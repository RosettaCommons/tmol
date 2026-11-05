"""Ordered LJ/LK host parameters and validation of the active chemistry."""

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np

LJLK_FIELDS = ("lj_radius", "lj_wdepth", "lk_dgfree", "lk_lambda", "lk_volume")

LJLK_NATIVE_TYPE_FIELDS = LJLK_FIELDS + (
    "is_donor",
    "is_hydroxyl",
    "is_polarh",
    "is_acceptor",
    "is_carbon_lk",
    "is_hydrogen",
)
LJLK_NATIVE_GLOBAL_FIELDS = (
    "max_dis",
    "lj_dlin_sigma_factor",
    "lj_dlin_sigma_factor_soft",
    "lj_hbond_dis",
    "lj_hbond_OH_donor_dis",
    "lj_hbond_hdis",
)


@dataclass
class LJLKTypeData:
    names: tuple
    values: np.ndarray
    is_carbon_lk: np.ndarray
    invalid_type_parameters: dict

    def native_parameters(self, atom_type_flags):
        values = dict(zip(LJLK_FIELDS, self.values.T))
        values.update(atom_type_flags)
        values["is_carbon_lk"] = self.is_carbon_lk
        return native_ljlk_type_parameters(values)

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


def lk_isotropic_coefficients(dgfree, lk_lambda):
    dgfree = np.asarray(dgfree, dtype=np.float32)
    lk_lambda = np.asarray(lk_lambda, dtype=np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        return -dgfree / (2 * 5.56832799683 * lk_lambda), 1 / (lk_lambda * lk_lambda)


def native_ljlk_type_parameters(parameters):
    """Pack the canonical 13-scalar native record from NumPy field arrays."""
    p = {
        name: np.asarray(value, dtype=np.float32) for name, value in parameters.items()
    }
    coeff, inv_lambda2 = lk_isotropic_coefficients(p["lk_dgfree"], p["lk_lambda"])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.stack(
            [
                p["lj_radius"],
                np.sqrt(p["lj_wdepth"]),
                p["lk_dgfree"],
                p["lk_lambda"],
                p["lk_volume"],
                p["is_donor"],
                p["is_hydroxyl"],
                p["is_polarh"],
                p["is_acceptor"],
                p["is_carbon_lk"],
                p["is_hydrogen"],
                coeff,
                inv_lambda2,
            ],
            axis=1,
        )


def native_ljlk_global_parameters(parameters, soft_repulsive=False):
    """Pack the five-scalar native LJ record, including the soft-repulsive mode."""
    get = (
        parameters.__getitem__
        if isinstance(parameters, Mapping)
        else lambda name: getattr(parameters, name)
    )
    fields = (
        "max_dis",
        "lj_dlin_sigma_factor_soft" if soft_repulsive else "lj_dlin_sigma_factor",
        "lj_hbond_dis",
        "lj_hbond_OH_donor_dis",
        "lj_hbond_hdis",
    )
    return np.stack(
        [np.atleast_1d(np.asarray(get(name), dtype=np.float32)) for name in fields],
        axis=1,
    )
