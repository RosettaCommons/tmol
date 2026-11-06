"""Default chemical identities without loading framework scoring tables."""

from functools import lru_cache

from . import ChemicalDatabase
from .patching import PatchedChemicalDatabase


@lru_cache(maxsize=1)
def default_patched_chemical_database():
    """Return the immutable default patched chemical database."""
    return PatchedChemicalDatabase.from_chem_db(ChemicalDatabase.get_default())
