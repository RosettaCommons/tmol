"""Shared scoring parameters with JAX-compatible table archive readers."""

from __future__ import annotations

import os
from typing import Optional

import attr
from tmol_core.database.scoring import load_scoring_parameters

from tmol_core.database.metal_coordination import MetalCoordinationDatabase
from tmol_core.database.na_torsion import NaTorsionDatabase

from .cartbonded import CartBondedDatabase
from .disulfide import DisulfideDatabase
from .genbonded import GenBondedDatabase
from .dunbrack_libraries import DunbrackRotamerLibrary
from .elec import ElecDatabase
from .hbond import HBondDatabase
from .ljlk import LJLKDatabase
from .omega_bbdep import OmegaBBDepDatabase
from .rama import RamaDatabase
from .ref import RefDatabase


@attr.s(auto_attribs=True, slots=True, frozen=True)
class ScoringDatabase:
    """Aggregate scoring-parameter database (one object per term)."""

    ljlk: LJLKDatabase
    elec: ElecDatabase
    hbond: HBondDatabase
    cartbonded: CartBondedDatabase
    genbonded: GenBondedDatabase
    ref: RefDatabase
    rama: RamaDatabase
    dunbrack: DunbrackRotamerLibrary
    disulfide: DisulfideDatabase
    omega_bbdep: OmegaBBDepDatabase
    metal_coordination: MetalCoordinationDatabase
    na_torsion: NaTorsionDatabase

    @property
    def dun(self):
        """The common ligand-preparation name for the same rotamer library."""
        return self.dunbrack

    @classmethod
    def from_file(cls, path: Optional[str] = None) -> "ScoringDatabase":
        """Load every scoring sub-database from ``<path>/*.yaml``.

        ``path`` defaults to the package's bundled
        ``default/scoring/`` directory. Binary tables are read from
        ``rama.npz`` / ``omega_bbdep.npz`` / ``dunbrack.npz``. Legacy
        Torch archives must be converted at package build time.
        """
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "..", "default", "scoring")
        path = os.path.abspath(path)

        parameters = load_scoring_parameters(
            path,
            dun_loader=lambda root: DunbrackRotamerLibrary.from_file(
                os.path.join(root, "dunbrack.yaml")
            ),
            rama_loader=lambda root: RamaDatabase.from_file(
                os.path.join(root, "rama.yaml")
            ),
            omega_loader=lambda root: OmegaBBDepDatabase.from_file(
                os.path.join(root, "omega_bbdep.yaml")
            ),
        )
        parameters["dunbrack"] = parameters.pop("dun")
        return cls(**parameters)


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
