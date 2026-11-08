"""Reuse canonical fitted tables by source identity and complete parameter contents."""

import attr
import jax
import numpy as np

from tmol_core.scoring.dunbrack import prepare_dunbrack_lookup, prepare_dunbrack_tables
from tmol_core.weak_identity_cache import WeakIdentityLRU
from tmol_core.database.content_hash import _content_hash
from tmol_jax.score._native import fit_bspline_coefficients

_TABLES = WeakIdentityLRU(capacity=2)
_DEVICE_TABLES = WeakIdentityLRU(capacity=4)


def prepare_dunbrack_parameters(database, *, table_indices=None):
    """Fit common tables for source library IDs, or the full sampling library."""
    library = database.scoring.dunbrack
    if table_indices is not None:
        table_indices = tuple(sorted(set(table_indices)))

    def prepare():
        selected = library
        if table_indices is not None:
            # Validate the complete declared lookup before projecting storage.
            # All fitting and table-local index preparation remain common.
            prepare_dunbrack_lookup(library)
            libraries = library.rotameric_libraries + library.semi_rotameric_libraries
            if any(i < 0 or i >= len(libraries) for i in table_indices):
                raise ValueError("Dunbrack library selection is out of range")
            names = {libraries[i].table_name for i in table_indices}
            selected = attr.evolve(
                library,
                rotameric_libraries=tuple(
                    x for x in library.rotameric_libraries if x.table_name in names
                ),
                semi_rotameric_libraries=tuple(
                    x for x in library.semi_rotameric_libraries if x.table_name in names
                ),
                dun_lookup=tuple(
                    row for row in library.dun_lookup if row.dun_table_name in names
                ),
            )
        return prepare_dunbrack_tables(
            selected,
            lambda values: [fit_bspline_coefficients(value) for value in values],
        )

    return _TABLES.get_or_create(
        library,
        (_content_hash(library), table_indices),
        prepare,
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
