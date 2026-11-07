"""Reuse common fitted library tables within one immutable molecular palette."""

from tmol_core.scoring.dunbrack import prepare_dunbrack_tables
from tmol_jax.score._native import fit_bspline_coefficients
from tmol_jax.score.energy_term import prepared_palette


def prepare_dunbrack_parameters(database, owner=None):
    def prepare():
        return prepare_dunbrack_tables(
            database.scoring.dunbrack,
            lambda values: [fit_bspline_coefficients(value) for value in values],
        )

    if owner is None:
        return prepare()
    return prepared_palette(owner, database, "dunbrack_tables", prepare)
