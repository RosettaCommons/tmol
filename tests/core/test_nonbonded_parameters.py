"""Real-atom validation and parameter precision belong to the common core."""

from pathlib import Path
from types import SimpleNamespace

import attr
import numpy as np
import pytest

from tmol_core.chemical import ChemicalDatabase
from tmol_core.chemical.restypes import refined_from_raw
from tmol_core.database.hbond import HBondDatabaseRaw
from tmol_core.scoring.atom_types import atom_type_parameters, block_type_indices
from tmol_core.scoring.hbond import hbond_pair_parameters
from tmol_core.scoring.ljlk import ljlk_type_parameters


def test_real_atoms_cannot_resolve_to_padding_or_missing_lj_parameters():
    db = ChemicalDatabase.get_default()
    residue = refined_from_raw(next(r for r in db.residues if r.name == "ALA"))
    names, flags = atom_type_parameters(db.atom_types)
    indices = block_type_indices(names, residue)
    assert (indices >= 0).all() and (indices < len(names) - 1).all()
    assert not flags["is_hydrogen"][-1]
    invalid = attr.evolve(
        residue, atoms=(attr.evolve(residue.atoms[0], atom_type=None),)
    )
    with pytest.raises(ValueError, match="unregistered chemical atom types"):
        block_type_indices(names, invalid)
    table = ljlk_type_parameters(names, SimpleNamespace(atom_type_parameters=()))
    with pytest.raises(ValueError, match="ALA.*missing or non-finite LJLK"):
        table.validate_block_type(residue)


def test_complete_hbond_table_keeps_float64_and_rejects_missing_pair():
    path = Path(__file__).parents[2] / "tmol/database/default/scoring/hbond.yaml"
    database = HBondDatabaseRaw.from_file(path)
    chemical = ChemicalDatabase.get_default()
    data = hbond_pair_parameters(chemical, database)
    for values in data.polynomials.values():
        assert values.dtype == np.float64
        assert np.isfinite(values).all()
    missing = attr.evolve(database, pair_parameters=database.pair_parameters[1:])
    with pytest.raises(ValueError, match="Missing HBond pair parameters"):
        hbond_pair_parameters(chemical, missing)
