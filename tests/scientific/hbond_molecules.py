"""Create actual nonzero hydrogen bonds by rigidly placing complete residues."""

from dataclasses import dataclass
import numpy as np
from .molecules import MolecularCase


@dataclass(frozen=True)
class HydrogenBondCase(MolecularCase):
    donor: tuple
    acceptor: tuple

    def blocks(self, database, dtype=np.float32):
        result = super().blocks(database, np.float64)
        for row in result:
            dn, da, dx = row[0]
            an, aa, ax = row[1]
            D, H = [dx[da.index(a)] for a in self.donor]
            A, B, B0 = [ax[aa.index(a)] for a in self.acceptor]

            def unit(x):
                return x / np.linalg.norm(x)

            u = unit(H - D)
            axis = unit(B - A)
            side = unit(B0 - A - np.dot(B0 - A, axis) * axis)
            approach = unit(-0.5 * axis - np.sqrt(0.75) * side)
            normal = unit(np.cross(axis, side))
            src = np.stack((approach, normal, np.cross(approach, normal)))
            target = -u
            trial = np.eye(3)[np.argmin(np.abs(u))]
            normal_to = unit(np.cross(target, trial))
            dest = np.stack((target, normal_to, np.cross(target, normal_to)))
            rotation = src.T @ dest
            translation = H + 1.9 * u + 0.06 * normal_to
            placed = (ax - A) @ rotation + translation
            # Preserve every internal distance, without changing identity or bonding.
            np.testing.assert_allclose(
                np.linalg.norm(ax[:, None] - ax[None, :], axis=-1),
                np.linalg.norm(placed[:, None] - placed[None, :], axis=-1),
                atol=1e-12,
            )
            row[0] = (dn, da, dx.astype(dtype))
            row[1] = (an, aa, placed.astype(dtype))
        return result


LIGAND = "protein_ligand_test/hsp90.xtal-lig.mmff94.tmol"
HBOND_CASES = {
    "ligand_pair": HydrogenBondCase(
        (LIGAND,), (("LG1", "LG1"),), ((1, 1),), ("N2", "H14"), ("N1", "C11", "C12")
    ),
    "d_amino_ligand": HydrogenBondCase(
        (LIGAND,), (("DALA", "LG1"),), ((1, 1),), ("N", "H"), ("N1", "C11", "C12")
    ),
    "beta_amino_ligand": HydrogenBondCase(
        (LIGAND, "ligand_params/noncanonical_nonstandard_aa.tmol"),
        (("B3Q", "LG1"),),
        ((1, 1),),
        ("N", "HN3"),
        ("N1", "C11", "C12"),
    ),
    "modified_dna": HydrogenBondCase(
        ("ligand_params/noncanonical_dna.tmol",),
        (("5CM", "DG"),),
        ((1, 1),),
        ("N4", "HN1"),
        ("O6", "C6", "N1"),
    ),
}

HBOND_CASES["mixed_batch"] = HydrogenBondCase(
    (LIGAND, "ligand_params/noncanonical_nonstandard_aa.tmol"),
    (("ALA", "LG1"), ("DALA", "LG1")),
    ((1, 1), (1, 1)),
    ("N", "H"),
    ("N1", "C11", "C12"),
)
