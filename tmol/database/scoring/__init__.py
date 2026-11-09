"""Typed schemas and loaders for TMol scoring parameters."""

from ._cartbonded import CartBondedDatabase  # noqa: F401
from ._disulfide import DisulfideDatabase  # noqa: F401
from ._dunbrack_libraries import DunbrackRotamerLibrary  # noqa: F401
from ._elec import ElecDatabase  # noqa: F401
from ._genbonded import GenBondedDatabase  # noqa: F401
from ._hbond import HBondDatabase  # noqa: F401
from ._ljlk import LJLKDatabase  # noqa: F401
from ._metal_coordination import MetalCoordinationDatabase  # noqa: F401
from ._na_torsion import NaTorsionDatabase  # noqa: F401
from ._omega_bbdep import OmegaBBDepDatabase  # noqa: F401
from ._rama import RamaDatabase  # noqa: F401

import os
import attr

from ._ref import RefDatabase  # noqa: F401

__all__ = [
    "CartBondedDatabase",
    "DunbrackRotamerLibrary",
    "ElecDatabase",
    "GenBondedDatabase",
    "HBondDatabase",
    "LJLKDatabase",
    "MetalCoordinationDatabase",
    "NaTorsionDatabase",
    "OmegaBBDepDatabase",
    "RamaDatabase",
]

from tmol_core.database.scoring import load_scoring_parameters


@attr.s(auto_attribs=True, slots=True, frozen=True)
class ScoringDatabase:
    """Immutable parameters for all standard TMol energy terms."""

    cartbonded: CartBondedDatabase
    genbonded: GenBondedDatabase
    disulfide: DisulfideDatabase
    metal_coordination: MetalCoordinationDatabase
    na_torsion: NaTorsionDatabase
    dun: DunbrackRotamerLibrary
    elec: ElecDatabase
    hbond: HBondDatabase
    ljlk: LJLKDatabase
    omega_bbdep: OmegaBBDepDatabase
    rama: RamaDatabase
    ref: RefDatabase

    @classmethod
    def from_file(
        cls, path: str | os.PathLike[str] = os.path.dirname(__file__)
    ) -> "ScoringDatabase":
        """Load all scoring-term databases rooted at ``path``."""

        return cls(
            **load_scoring_parameters(
                path,
                dun_loader=lambda root: DunbrackRotamerLibrary.from_file(
                    os.path.join(root, "dunbrack.bin"),
                    lookup=os.path.join(root, "dunbrack.yaml"),
                ),
                rama_loader=lambda root: RamaDatabase.from_file(
                    os.path.join(root, "rama.zip")
                ),
                omega_loader=lambda root: OmegaBBDepDatabase.from_file(
                    os.path.join(root, "omega_bbdep.zip")
                ),
                hbond_converter=HBondDatabase.from_raw_hbond_db,
            )
        )
