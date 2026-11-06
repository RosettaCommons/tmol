"""Host structure chemistry must remain usable without either frontend."""

import os
from pathlib import Path
import subprocess
import sys


def _assert_framework_free(source):
    root = Path(__file__).resolve().parents[2]
    script = """
import importlib.abc
import sys
class BlockFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'jax', 'tmol'}:
            raise ImportError('framework import: ' + fullname)
sys.meta_path.insert(0, BlockFrameworks())
""" + source
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=dict(os.environ, PYTHONPATH=str(root)),
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_protonation_and_terminal_policy_import_without_frameworks():
    _assert_framework_free("""
from tmol_core.chemical.default import default_patched_chemical_database
from tmol_core.io.protonation_input import _database_forms
from tmol_core.io.input_termini import _connection_hydrogen_capacities
database = default_patched_chemical_database()
assert _database_forms(database)
assert _connection_hydrogen_capacities(database)
assert not {'torch', 'jax', 'tmol'}.intersection(sys.modules)
""")


def test_ligand_preparation_preserves_supplied_charges_without_frameworks():
    _assert_framework_free("""
import importlib
from pathlib import Path
import tmol_core.ligand
for path in Path(tmol_core.ligand.__file__).parent.glob('*.py'):
    importlib.import_module('tmol_core.ligand.' + path.stem)
from tmol_core.ligand._preparation import _prepare_mol2
from tmol_core.ligand._detect import nonstandard_residue_info_from_mol2
from tmol_core.ligand import load_params_file, prepare_ligand_from_mol2
data = Path('tmol/tests/data')
source = data / 'ligand_test/ligand_ground_truth/mol2/comt_2.mol2'
info = nonstandard_residue_info_from_mol2(source, res_name='LG1')
prepared = _prepare_mol2(source, res_name='LG1', mode='keep')
assert prepared.partial_charges == info.partial_charges
assert len(prepared.residue_type.atoms) == len(info.atom_names)
assert prepared.cartbonded_params.length_parameters
assert any(abs(value) > 0.05 for value in prepared.partial_charges.values())
assert load_params_file(data / 'protein_ligand_test/vegfr2.xtal-lig.mmff94.tmol')
# A caller-supplied parameter context needs no framework for registration.
import attr
from tmol_core.database.context import ParameterDatabase
from tmol_core.database.elec import ElecDatabase
from tmol_core.database.cartbonded import CartBondedDatabase
from tmol_core.chemical.default import default_patched_chemical_database
@attr.s(auto_attribs=True, frozen=True)
class Parameters:
    elec: object
    cartbonded: object
scoring = Path('tmol/database/default/scoring')
base = ParameterDatabase(
    scoring=Parameters(
        ElecDatabase.from_file(scoring / 'elec.yaml'),
        CartBondedDatabase.from_file(scoring / 'cartbonded.yaml'),
    ),
    chemical=default_patched_chemical_database(),
)
result, ordering = prepare_ligand_from_mol2(
    source, res_name='LG1', mode='keep', param_db=base,
)
assert type(result) is type(base) and result is not base
assert all(res.name != 'LG1' for res in base.chemical.residues)
assert any(res.name == 'LG1' for res in result.chemical.residues)
assert {p.atom: p.charge for p in result.scoring.elec.atom_charge_parameters
        if p.res == 'LG1'} == info.partial_charges
assert 'LG1' in ordering.restype_io_equiv_classes

assert not {'torch', 'jax', 'tmol'}.intersection(sys.modules)
""")


def test_structure_reader_preserves_atoms_and_bonds_without_frameworks():
    _assert_framework_free("""
from pathlib import Path
import numpy as np
from tmol_core.io import atom_array_from_file, file_from_atom_array
data = Path('tmol/tests/data')
for path in (
    data / 'pdb/1ubq.pdb',
    data / 'protein_ligand_test/ace.lig.mol2',
    data / 'covalent_fixtures/lys_biotin_1bdo.cif',
):
    array = atom_array_from_file(path)
    assert len(array) > 0 and len(array.bonds.as_array()) > 0
    assert np.isfinite(array.coord).any()
    assert 'charge' in array.get_annotation_categories()
    assert file_from_atom_array(array, format='cif').startswith('data_')
assert not {'torch', 'jax', 'tmol'}.intersection(sys.modules)
""")


def test_structure_context_host_policy_imports_without_frameworks():
    _assert_framework_free("""
import numpy as np
import biotite.structure as struc
from types import SimpleNamespace
from tmol_core.chemical.default import default_patched_chemical_database
from tmol_core.io import structure_context as common
from tmol_core.ligand import nonstandard_residue_info_from_mol2
chemistry = default_patched_chemical_database()
database = SimpleNamespace(chemical=chemistry)
source = 'tmol/tests/data/ligand_test/ligand_ground_truth/mol2/comt_2.mol2'
atoms = nonstandard_residue_info_from_mol2(source, res_name='LG1').atom_array
original = atoms.coord.copy()
def unexpected(*args, **kwargs):
    raise AssertionError('preparation executor called with preparation disabled')
result = common.prepare_structure_context(
    atoms, parameter_database=database, default_database=lambda: database,
    prepare_ligands=False, ligand_ph=7.4, strict_atom_types=False,
    strict_ligands=True, ligand_params_files=None, chem_comp_types=None,
    ligand_seed=47, ligand_hydrogens='input', input_hydrogens=unexpected,
    ligand_preparer=unexpected,
)
assert result.parameter_database is database
assert not result.use_default_context
np.testing.assert_array_equal(atoms.coord, original)
assert common._get_all_residue_positions(atoms[:0]).shape == (0,)
assert not {'torch', 'jax', 'tmol'}.intersection(sys.modules)
""")
