"""Standard scoring database assembly, independent of execution framework."""

import os

from .cartbonded import CartBondedDatabase
from .disulfide import DisulfideDatabase
from .elec import ElecDatabase
from .genbonded import GenBondedDatabase
from .hbond import HBondDatabaseRaw
from .ljlk import LJLKDatabase
from .metal_coordination import MetalCoordinationDatabase
from .na_torsion import NaTorsionDatabase
from .ref import RefDatabase
from .generated import (
    GENERATED_CARTBONDED_FILES,
    GENERATED_ELEC_FILES,
    GENERATED_REF_FILES,
)


def load_scoring_parameters(
    path, *, dun_loader, rama_loader, omega_loader, hbond_converter=None
):
    """Load standard parameters and generated overlays from a database root.

    The three table providers receive the root path and own their archive format.
    Results use the canonical ``dun`` key. An optional HBond converter adapts the
    common raw schema to a frontend's public representation. Required-file and
    optional generated-overlay validation belongs to the shared schema loaders.
    """

    def filename(name):
        return os.path.join(path, name)

    def hbond():
        return HBondDatabaseRaw.from_file(filename("hbond.yaml"))

    return dict(
        cartbonded=CartBondedDatabase.from_file(
            filename("cartbonded.yaml"),
            generated=[filename(f) for f in GENERATED_CARTBONDED_FILES],
        ),
        genbonded=GenBondedDatabase.from_file(filename("genbonded.yaml")),
        disulfide=DisulfideDatabase.from_file(filename("disulfide.yaml")),
        metal_coordination=MetalCoordinationDatabase.from_file(
            filename("metal_coordination.yaml")
        ),
        na_torsion=NaTorsionDatabase.from_file(filename("na_torsion.yaml")),
        dun=dun_loader(path),
        elec=ElecDatabase.from_file(
            filename("elec.yaml"),
            generated=[filename(f) for f in GENERATED_ELEC_FILES],
        ),
        hbond=hbond() if hbond_converter is None else hbond_converter(hbond()),
        ljlk=LJLKDatabase.from_file(filename("ljlk.yaml")),
        omega_bbdep=omega_loader(path),
        rama=rama_loader(path),
        ref=RefDatabase.from_file(
            filename("ref.yaml"),
            generated=[filename(f) for f in GENERATED_REF_FILES],
        ),
    )
