"""Reuse canonical fitted tables by source identity and complete parameter contents."""

import jax
import numpy as np

from tmol_core.scoring.dunbrack import prepare_dunbrack_tables
from tmol_core.weak_identity_cache import WeakIdentityLRU
from tmol_core.database.content_hash import _content_hash
from tmol_jax.score._native import fit_bspline_coefficients

_TABLES = WeakIdentityLRU(capacity=2)
_DEVICE_TABLES = WeakIdentityLRU(capacity=4)


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


def device_dunbrack_tables(tables, dtype, device):
    """Reuse device tables for one fitted snapshot, without retaining its owner."""
    return _DEVICE_TABLES.get_or_create(
        tables,
        (np.dtype(dtype), device),
        lambda: native_parameters(tables.scoring_db.values(), dtype, device),
    )


def native_parameters(arrays, dtype, device):
    """Validate native index width and preserve the coordinate precision."""
    parameters = []
    for array in arrays:
        array = np.asarray(array)
        if np.issubdtype(array.dtype, np.integer):
            if array.size and (
                array.min() < np.iinfo(np.int32).min
                or array.max() > np.iinfo(np.int32).max
            ):
                raise ValueError("native Dunbrack indices exceed int32")
            array = array.astype(np.int32)
        elif np.issubdtype(array.dtype, np.floating):
            array = array.astype(dtype, copy=False)
        parameters.append(jax.device_put(array, device=device))
    return tuple(parameters)
