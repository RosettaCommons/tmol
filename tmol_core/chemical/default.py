"""Default chemical identities without loading framework scoring tables."""

from functools import lru_cache

from . import ChemicalDatabase
from .patching import PatchedChemicalDatabase


@lru_cache(maxsize=1)
def default_patched_chemical_database():
    """Return the immutable default patched chemical database."""
    return PatchedChemicalDatabase.from_chem_db(ChemicalDatabase.get_default())


@lru_cache(maxsize=1)
def default_canonical_ordering():
    """Canonical identity metadata for the default structure-reader policy."""
    from tmol_core.io.canonical_ordering import CanonicalOrdering

    return CanonicalOrdering.from_chemdb(default_patched_chemical_database())
