"""Custom L/D chemical classes may order their canonical atoms differently."""

import attr
import numpy as np
import pytest
import torch

from tmol import CanonicalOrdering, PatchedChemicalDatabase
from tmol.database.chemical import ChemicalDatabase, _l_base_name
from tmol.io.details._his_taut_resolution import _resolve_his_tautomerization
from tmol.io.details._disulfide_search import _find_disulfides


@pytest.fixture(scope="module")
def mixed_slots():
    db = PatchedChemicalDatabase.from_chem_db(ChemicalDatabase.get_default())
    families = {"HIS", "HIS_D", "HIS_POS", "CYS", "CYD"}
    db = attr.evolve(
        db,
        residues=tuple(
            (
                attr.evolve(r, atoms=tuple(r.atoms[1:]) + tuple(r.atoms[:1]))
                if _l_base_name(r) in families
                and r.io_equiv_class not in ("HIS", "CYS")
                else r
            )
            for r in db.residues
        ),
    )
    return CanonicalOrdering.from_chemdb(db)


def test_histidine_alias_and_tautomer_follow_each_class(mixed_slots):
    co = mixed_slots
    classes = co.his_inds.his_co_aa_inds
    assert len(set(co.his_inds.atom_inds_by_class)) > 1
    count = len(classes)
    types = torch.tensor([classes], dtype=torch.int32)
    variants = torch.zeros_like(types)
    coords = torch.zeros((1, count, co.max_n_canonical_atoms, 3))
    present = torch.zeros(coords.shape[:-1], dtype=torch.bool)
    for r, cls in enumerate(classes):
        index = co.restypes_atom_index_mapping[co.restype_io_equiv_classes[cls]]
        for atom in ("ND1", "NE2", "HN"):
            present[0, r, index[atom]] = True
        coords[0, r, index["NE2"], 0] = 5
        coords[0, r, index["HN"], 0] = 0.75
    _, variants, out, mask = _resolve_his_tautomerization(
        co, types, variants, coords, present
    )
    assert variants.tolist() == [[1] * count]
    for r, cls in enumerate(classes):
        index = co.restypes_atom_index_mapping[co.restype_io_equiv_classes[cls]]
        assert mask[0, r, index["HD1"]]
        torch.testing.assert_close(out[0, r, index["HD1"]], torch.tensor([0.75, 0, 0]))


def test_mixed_cysteine_pair_uses_each_sulfur_slot(mixed_slots):
    co = mixed_slots
    classes = co.cys_inds.cys_co_aa_inds
    slots = co.cys_inds.sg_atoms_for_co_cys
    assert len(set(slots)) > 1
    types = torch.tensor([[classes[0], classes[-1], -1]], dtype=torch.int32)
    coords = torch.full((1, 3, co.max_n_canonical_atoms, 3), float("nan"))
    coords[0, 0, slots[0]] = torch.tensor([0.0, 0, 0])
    coords[0, 1, slots[-1]] = torch.tensor([2.1, 0, 0])
    original = coords.clone()
    bonds, variants = _find_disulfides(co, types, coords)
    assert bonds.tolist() == [[0, 0, 1]]
    assert variants.tolist() == [[1, 1, 0]]
    np.testing.assert_array_equal(coords.numpy(), original.numpy())


def test_declared_mixed_class_disulfide_keeps_sulfur_identity(mixed_slots):
    import biotite.structure as struc
    from tmol.io._pose_stack_from_biotite import _covalent_bonds_from_biotite

    co = mixed_slots
    classes = [co.cys_inds.cys_co_aa_inds[0], co.cys_inds.cys_co_aa_inds[-1]]
    slots = [co.cys_inds.sg_by_class[c] for c in classes]
    array = struc.AtomArray(2)
    array.coord[:] = np.nan  # declared connectivity does not require resolved SGs
    array.bonds = struc.BondList(2, np.array([[0, 1, struc.BondType.SINGLE]]))

    def partition(indices):
        return _covalent_bonds_from_biotite(
            array,
            co,
            np.array([0, 1]),
            np.array(classes),
            np.ones(2, bool),
            np.array(indices),
        )

    bonds, disulfides, inferred = partition(slots)
    assert bonds.shape == (0, 4) and inferred.shape == (0,)
    assert disulfides.tolist() == [[0, 1]]
    name = co.restype_io_equiv_classes[classes[1]]
    other = co.restypes_atom_index_mapping[name]["CA"]
    bonds, disulfides, _ = partition([slots[0], other])
    assert bonds.tolist() == [[0, slots[0], 1, other]]
    assert disulfides.shape == (0, 2)
