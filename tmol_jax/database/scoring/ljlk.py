"""Compatibility exports for the common LJ/LK parameter schema."""

from tmol_core.database.ljlk import (  # noqa: F401
    LJLKGlobalParameters,
    LJLKAtomTypeParameters,
    LJLKDatabase,
)

LJLKGlobal = LJLKGlobalParameters
AtomLJLKParam = LJLKAtomTypeParameters
