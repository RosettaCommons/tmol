"""Shared chemical records with the JAX database-location override."""

import os
import cattr
from tmol_core.chemical import *  # noqa: F401,F403
from tmol_core.chemical import ChemicalDatabase as _ChemicalDatabase
from tmol_core.chemical import (  # noqa: F401
    _normalize_bond_tuples as normalize_bond_tuples,
)
from tmol_core._yaml import _safe_load as safe_load  # noqa: F401
from tmol_core.data import chemical_directory

# Kept for callers that structure custom raw chemical records.
_structure = cattr.structure


class ChemicalDatabase(_ChemicalDatabase):
    __default = None

    @classmethod
    def get_default(cls):
        if cls.__default is None:
            path = os.environ.get("TMOL_JAX_DATABASE")
            cls.__default = cls.from_file(
                os.path.join(path, "chemical") if path else chemical_directory()
            )
        return cls.__default
