"""Framework-independent, committed molecular fixtures and parameter context."""

from dataclasses import dataclass
from pathlib import Path

import attrs
import cattr
import numpy as np

from tmol_core._yaml import _load_yaml
from tmol_core.chemical import RawResidueType, _normalize_bond_tuples
from tmol_core.chemical.restypes import refined_from_raw
from tmol_core.database.cartbonded import CartBondedDatabase, CartRes
from tmol_core.database.elec import PartialCharges

DATA = Path(__file__).parents[2] / "tmol/tests/data"


@dataclass(frozen=True)
class MolecularCase:
    parameter_files: tuple
    sequences: tuple
    chain_lengths: tuple

    def database(self, base):
        # These committed fixtures contain complete residue records. This is
        # deliberately not a general .tmol parser or ligand-preparation path.
        residues = {r.name: r for r in base.chemical.residues}
        charges = list(base.scoring.elec.atom_charge_parameters)
        bonded = dict(base.scoring.cartbonded.residue_params)
        for filename in self.parameter_files:
            record = _load_yaml(DATA / filename)
            for raw in _normalize_bond_tuples(record["chemical"])["residues"]:
                residue = cattr.structure(raw, RawResidueType)
                residues[residue.name] = residue
            charges.extend(
                cattr.structure(p, PartialCharges)
                for p in record["elec"]["atom_charge_parameters"]
            )
            bonded.update(
                {
                    name: cattr.structure(value, CartRes)
                    for name, value in record["cartbonded"]["residue_params"].items()
                }
            )
        return attrs.evolve(
            base,
            chemical=attrs.evolve(base.chemical, residues=tuple(residues.values())),
            scoring=attrs.evolve(
                base.scoring,
                elec=attrs.evolve(
                    base.scoring.elec, atom_charge_parameters=tuple(charges)
                ),
                cartbonded=CartBondedDatabase.from_cartres_dict(bonded),
            ),
        )

    def blocks(self, database, dtype=np.float32):
        types = {r.name: r for r in database.chemical.residues}
        rng = np.random.default_rng(191)
        result = []
        for sequence in self.sequences:
            row = []
            for index, name in enumerate(sequence):
                rt = refined_from_raw(types[name])
                coords = rt.ideal_coords[rt.at_to_icoor_ind].copy()
                coords += rng.normal(0, 0.025, coords.shape)
                coords += np.array([3.8 * index, 0, 0])
                row.append(
                    (name, tuple(a.name for a in rt.atoms), coords.astype(dtype))
                )
            result.append(row)
        return result

    def connections(self):
        result = []
        for lengths in self.chain_lengths:
            edges, begin = [], 0
            for length in lengths:
                edges.extend(
                    (i, "up", i + 1, "down") for i in range(begin, begin + length - 1)
                )
                begin += length
            result.append(edges)
        return result


CASES = {
    "ligand": MolecularCase(
        ("protein_ligand_test/hsp90.xtal-lig.mmff94.tmol",), (("LG1",),), ((1,),)
    ),
    "noncanonical": MolecularCase(
        ("ligand_params/noncanonical_alpha_aa.tmol",), (("HYP",),), ((1,),)
    ),
    "connected_noncanonical": MolecularCase(
        ("ligand_params/noncanonical_nonstandard_aa.tmol",), (("B3Q", "B3D"),), ((2,),)
    ),
    "mixed_batch": MolecularCase(
        (
            "protein_ligand_test/hsp90.xtal-lig.mmff94.tmol",
            "ligand_params/noncanonical_alpha_aa.tmol",
        ),
        (("GLY", "HYP", "LG1"), ("HYP",)),
        ((1, 1, 1), (1,)),
    ),
}

# Broader chemistry inputs for topology/parameter and per-term audits. These
# do not imply full scoring support or nonzero generic-bonded parameterization.
DOMAIN_CASES = {
    **CASES,
    "d_peptide": MolecularCase((), (("DALA", "DPRO", "DVAL"),), ((3,),)),
    "dna": MolecularCase((), (("DA", "DC"),), ((2,),)),
    "rna": MolecularCase((), (("RA", "RC"),), ((2,),)),
    "metal_mixed": MolecularCase((), (("HIS", "ASP", "ZN_tetrahedral"),), ((1, 1, 1),)),
    "metal_cluster": MolecularCase((), (("SF4",),), ((1,),)),
}

# Backbone fixtures require connected residues: isolated HYP has no backbone
# interaction. Its committed parameter record explicitly references PRO tables.
BACKBONE_CASES = {
    "canonical": MolecularCase((), (("ALA", "PRO", "VAL"),), ((3,),)),
    "d_peptide": DOMAIN_CASES["d_peptide"],
    "hydroxyproline": MolecularCase(
        CASES["noncanonical"].parameter_files, (("ALA", "HYP", "VAL"),), ((3,),)
    ),
    "variants": MolecularCase(
        (), (("ALA:nterm", "HIS_D", "CYD", "PRO:cterm"),), ((4,),)
    ),
    "mixed": MolecularCase(
        CASES["mixed_batch"].parameter_files,
        (("ALA", "HYP", "VAL", "LG1"), ("DALA", "DPRO", "DVAL")),
        ((3, 1), (3,)),
    ),
}
