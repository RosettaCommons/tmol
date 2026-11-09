"""Explicit L/D disulfides using the committed heterochiral 6DMZ geometry."""

from dataclasses import dataclass

import biotite.structure.io.pdbx as pdbx
import numpy as np

from .molecules import DATA, MolecularCase


@dataclass(frozen=True)
class DisulfideCase(MolecularCase):
    def extra_connections(self):
        return [((0, "dslf", 1, "dslf"),) for _ in self.sequences]

    def blocks(self, database, dtype=np.float32):
        blocks = super().blocks(database, dtype)
        array = pdbx.get_structure(
            pdbx.CIFFile.read(DATA / "disulfide/6dmz_3_47_ld.cif"), model=1
        )
        for row in blocks:
            for index, (name, names, coords) in enumerate(row[:2]):
                residue = array[array.chain_id == ("A", "B")[index]]
                for atom in residue:
                    if atom.atom_name in names:
                        coords[names.index(atom.atom_name)] = atom.coord
                # The source is L/D. Reflect a changed stereocenter through
                # N/CB/SG, preserving the two sulfur-sidechain bond anchors.
                if (name == "DCYD") != (index == 1):
                    n, cb, sg = (
                        coords[names.index(a)].copy() for a in ("N", "CB", "SG")
                    )
                    normal = np.cross(cb - n, sg - n)
                    normal /= np.linalg.norm(normal)
                    coords -= 2 * ((coords - n) @ normal)[:, None] * normal
        return blocks


DISULFIDE_CASES = {
    pair: DisulfideCase(
        (), (tuple("DCYD" if side == "d" else "CYD" for side in pair),), ((1, 1),)
    )
    for pair in ("ll", "ld", "dd")
}
DISULFIDE_CASES["mixed_batch"] = DisulfideCase(
    (), (("CYD", "DCYD", "GLY"), ("DCYD", "DCYD")), ((1, 1, 1), (1, 1))
)
