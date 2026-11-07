from types import SimpleNamespace as S
import numpy as np
import biotite.structure as bs
from tmol_core.io.atom_array import (
    INFERRED_POLYMER_BOND,
    with_metal_coordination_typed,
    covalent_bonds_from_biotite,
    metal_coordination_from_biotite,
)


def test_bond_channels_keep_class_specific_atoms_provenance_and_input_identity():
    atoms = bs.AtomArray(6)
    atoms.element = ["S", "S", "C", "N", "ZN", "O"]
    atoms.chain_id[:] = "A"
    atoms.res_id = [1, 2, 3, 3, 4, 5]
    atoms.res_name = ["CYS", "DCY", "LIG", "LIG", "ZN", "DON"]
    atoms.set_annotation(
        INFERRED_POLYMER_BOND, np.array(["", "link", "link", "", "", ""])
    )
    bonds = np.array([[0, 1, 1], [1, 2, 1], [2, 3, 1], [4, 5, 1]])
    atoms.bonds = bs.BondList(6, bonds)
    original = atoms.bonds.as_array().copy()
    typed = with_metal_coordination_typed(atoms)
    co = S(cys_inds=S(cys_co_aa_inds=(0, 1), sg_by_class={0: 5, 1: 4}))
    residue = np.array([0, 1, 2, 2, 3, 4])
    valid = np.ones(6, bool)
    slots = np.array([5, 4, 0, 1, 0, 0])
    covalent, disulfides, inferred = covalent_bonds_from_biotite(
        typed, co, residue, np.arange(5), valid, slots
    )
    np.testing.assert_array_equal(covalent, [[1, 4, 2, 0]])
    np.testing.assert_array_equal(disulfides, [[0, 1]])
    np.testing.assert_array_equal(inferred, [True])
    np.testing.assert_array_equal(
        metal_coordination_from_biotite(typed, residue, valid, slots), [[3, -1, 4, 0]]
    )
    np.testing.assert_array_equal(atoms.bonds.as_array(), original)
    assert with_metal_coordination_typed(typed) is typed
