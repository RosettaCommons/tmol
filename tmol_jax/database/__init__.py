"""Immutable shared chemistry/parameter context with NumPy scoring tables."""

from __future__ import annotations

import os
from pathlib import Path

from tmol_core.chemical.default import default_patched_chemical_database
from tmol_core.database.context import (
    ParameterDatabase as _ParameterDatabase,
    _inject_residue_params,
)
from tmol_core.database.mirrored_dunbrack import _with_mirrored_libraries
from tmol_jax.database.chemical import ChemicalDatabase
from tmol_jax.database.scoring import ScoringDatabase


def default_directory():
    return Path(os.environ.get("TMOL_JAX_DATABASE", Path(__file__).parent / "default"))


class ParameterDatabase(_ParameterDatabase):
    """Shared context operations with Torch-free table loading and storage."""

    _default = None
    _dunbrack_attribute = "dunbrack"
    _load_scoring = staticmethod(ScoringDatabase.from_file)
    _with_mirrored_libraries = staticmethod(_with_mirrored_libraries)

    @classmethod
    def get_default(cls) -> "ParameterDatabase":
        if cls.__dict__.get("_default") is None:
            path = str(default_directory())
            if "TMOL_JAX_DATABASE" in os.environ:
                cls._default = cls.from_file(path)
            else:
                cls._default = cls._from_file_with_chemical(
                    path, default_patched_chemical_database()
                )
        return cls._default


__all__ = [
    "ParameterDatabase",
    "ChemicalDatabase",
    "ScoringDatabase",
    "_inject_residue_params",
]
