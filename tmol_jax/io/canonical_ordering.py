"""Canonical-form metadata from the shared chemical ordering."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from tmol_core.io.canonical_records import (  # noqa: F401
    _polymer_gaps,
    _canonical_form_from_records,
)

from tmol_core.io.canonical_ordering import CanonicalOrdering as _CanonicalOrdering


class CanonicalOrdering(_CanonicalOrdering):
    """Shared canonical residue and atom identities with frontend lookups."""


@lru_cache(maxsize=1)
def default_canonical_ordering() -> CanonicalOrdering:
    """Return the shared atom ordering for the complete default chemical database."""
    from tmol_jax.chemical.patched_chemdb import default_patched_chemdb

    return CanonicalOrdering.from_chemdb(default_patched_chemdb())


@lru_cache(maxsize=1)
def default_packed_block_types() -> Any:
    """Return the default molecular palette, including patched residue types.

    Metal input grows this ordinary palette with the shared coordination policy.
    """
    from tmol_jax.pose.packed_block_types import PackedBlockTypes

    return PackedBlockTypes.default()


# --------------------------------------------------------------------- #
# canonical_form_from_pdb                                                #
# --------------------------------------------------------------------- #


def canonical_form_from_pdb(
    pdb_text_or_path, canonical_ordering=None, *, missing_density_distance_threshold=2.4
):
    """Convert a PDB to canonical molecular coordinates, retaining missing atoms.

    Unknown residue and atom identities fail explicitly; water is ignored. Chain labels
    and database-defined polymer connection atoms determine geometry gaps.
    """
    from .pdb_parsing import parse_pdb

    from .pose_stack_from_biotite import canonical_form_from_biotite

    return canonical_form_from_biotite(
        parse_pdb(pdb_text_or_path)["_atom_array"],
        canonical_ordering=canonical_ordering or default_canonical_ordering(),
        missing_density_distance_threshold=missing_density_distance_threshold,
    )


__all__ = [
    "CanonicalOrdering",
    "default_canonical_ordering",
    "default_packed_block_types",
    "canonical_form_from_pdb",
]
