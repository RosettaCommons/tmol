"""Reuse canonical fitted tables by source identity and complete parameter contents."""

from tmol_core.scoring.dunbrack import prepare_dunbrack_tables
from tmol_core.weak_identity_cache import WeakIdentityLRU
from tmol_core.database.content_hash import _content_hash
from tmol_jax.score._native import fit_bspline_coefficients

_TABLES = WeakIdentityLRU(capacity=2)


def prepare_dunbrack_parameters(database):
    library = database.scoring.dunbrack
    return _TABLES.get_or_create(
        library,
        (_content_hash(library),),
        lambda: prepare_dunbrack_tables(
            library,
            lambda values: [fit_bspline_coefficients(value) for value in values],
        ),
    )
