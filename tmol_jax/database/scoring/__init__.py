"""Shared scoring parameters with JAX-compatible table archive readers."""

from __future__ import annotations

import os
from typing import Optional

import attr
from tmol_core.database.numpy_libraries import NumPyScoringDatabase
from tmol_jax._compat import alias_modules

from tmol_core.database.metal_coordination import MetalCoordinationDatabase
from tmol_core.database.na_torsion import NaTorsionDatabase

from tmol_core.database.cartbonded import CartBondedDatabase
from tmol_core.database.disulfide import DisulfideDatabase
from tmol_core.database.genbonded import GenBondedDatabase
from .dunbrack_libraries import DunbrackRotamerLibrary
from .elec import ElecDatabase
from .hbond import HBondDatabase
from .ljlk import LJLKDatabase
from .omega_bbdep import OmegaBBDepDatabase
from .rama import RamaDatabase
from tmol_core.database.ref import RefDatabase

alias_modules(
    __name__, "tmol_core.database", ("cartbonded", "disulfide", "genbonded", "ref")
)


@attr.s(auto_attribs=True, slots=True, frozen=True)
class ScoringDatabase(NumPyScoringDatabase):
    """Shared NumPy scoring parameters with the JAX package's default location."""

    @classmethod
    def from_file(cls, path: Optional[str] = None) -> "ScoringDatabase":
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "..", "default", "scoring")
        return super().from_file(path)


__all__ = [
    "ScoringDatabase",
    "CartBondedDatabase",
    "GenBondedDatabase",
    "DisulfideDatabase",
    "DunbrackRotamerLibrary",
    "ElecDatabase",
    "HBondDatabase",
    "LJLKDatabase",
    "MetalCoordinationDatabase",
    "NaTorsionDatabase",
    "OmegaBBDepDatabase",
    "RamaDatabase",
    "RefDatabase",
]
