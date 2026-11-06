from pathlib import Path

import numpy as np
import pytest

from tmol_jax.io import (
    canonical_form_from_pdb,
    default_packed_block_types,
    pose_stack_from_canonical_form,
)

PDB = Path(__file__).parents[2] / "tmol/tests/data/pdb/1ubq.pdb"


def _three_residues():
    canonical = canonical_form_from_pdb(PDB)
    return {
        key: value[:, :3].copy() if isinstance(value, np.ndarray) else value
        for key, value in canonical.items()
    }


def _variant_names(pose):
    return [pose.block_type(0, index).name for index in range(pose.max_n_blocks)]


def test_explicit_polymer_break_selects_terminal_variants_and_disconnects():
    canonical = _three_residues()
    canonical["res_not_connected"] = np.zeros((1, 3, 2), dtype=bool)
    canonical["res_not_connected"][0, 0, 1] = True
    pose = pose_stack_from_canonical_form(canonical, default_packed_block_types())
    names = _variant_names(pose)
    assert "nterm" in names[0] and "cterm" in names[0]
    assert "nterm" in names[1] and "cterm" not in names[1]
    assert np.all(np.asarray(pose.inter_residue_connections)[0, 0] == -1)
    assert 0 not in np.asarray(pose.inter_residue_connections)[0, 1, :, 0]
    assert np.isfinite(np.asarray(pose.coords)).all()


def test_internal_padding_creates_two_chain_segments():
    canonical = _three_residues()
    canonical["res_name"][0, 1] = ""
    canonical["res_types"][0, 1] = -1
    canonical["chain_id"][0, 1] = -1
    canonical["coords"][0, 1] = np.nan
    canonical["atom_is_present"][0, 1] = False
    pose = pose_stack_from_canonical_form(canonical, default_packed_block_types())
    assert pose.max_n_blocks == 2
    assert all("nterm" in name and "cterm" in name for name in _variant_names(pose))
    assert np.all(np.asarray(pose.inter_residue_connections) == -1)
    assert np.isfinite(np.asarray(pose.coords)).all()


def test_keyword_metadata_is_used_when_canonical_values_are_none():
    canonical = _three_residues()
    canonical["chain_id"] = None
    canonical["res_num"] = None
    pose = pose_stack_from_canonical_form(
        canonical,
        default_packed_block_types(),
        chain_id=[[0, 0, 1]],
        res_num=[[10, 11, 20]],
    )
    assert np.asarray(pose.pdb_info.res_num)[0].tolist() == [10, 11, 20]
    assert 2 not in np.asarray(pose.inter_residue_connections)[0, 1, :, 0]
    assert 1 not in np.asarray(pose.inter_residue_connections)[0, 2, :, 0]


def test_unknown_input_atom_is_not_silently_dropped():
    lines = PDB.read_text().splitlines()
    first = next(i for i, line in enumerate(lines) if line.startswith("ATOM"))
    lines[first] = lines[first][:12] + " ZZ " + lines[first][16:]
    with pytest.raises(ValueError, match="atom ZZ"):
        canonical_form_from_pdb("\n".join(lines))


@pytest.mark.parametrize(
    "option,value",
    (
        ("find_additional_metal_coordination", True),
        ("protonation_variants", np.zeros((1, 3), np.int32)),
    ),
)
def test_input_inference_preserves_ordinary_protein(option, value):
    canonical = _three_residues()
    pbt = default_packed_block_types()
    expected = pose_stack_from_canonical_form(canonical, pbt)
    canonical[option] = value
    actual = pose_stack_from_canonical_form(canonical, pbt)
    np.testing.assert_array_equal(actual.coords, expected.coords)
    np.testing.assert_array_equal(actual.block_type_ind, expected.block_type_ind)
    np.testing.assert_array_equal(
        actual.inter_residue_connections, expected.inter_residue_connections
    )


def test_conflicting_canonical_residue_identities_are_rejected():
    canonical = _three_residues()
    canonical["res_types"][0, 1] = canonical["res_types"][0, 0]
    with pytest.raises(ValueError, match="residue names and res_types disagree"):
        pose_stack_from_canonical_form(canonical, default_packed_block_types())


@pytest.mark.parametrize(
    "indices",
    (np.array([[0, 0, 1.5, 1, 1]]), np.array([[0, 0, 2**32, 1, 1]], np.uint64)),
)
def test_explicit_bond_indices_cannot_be_truncated_or_wrapped(indices):
    canonical = _three_residues()
    canonical["covalent_bonds"] = indices
    with pytest.raises(ValueError, match="integer indices within the int32 range"):
        pose_stack_from_canonical_form(canonical, default_packed_block_types())


@pytest.mark.parametrize("names", (("DA", "DT"), ("DALA", "DPHE")))
def test_biotite_polymer_gaps_use_chemical_ports_and_shared_ensemble(names):
    import biotite.structure as struc
    from tmol_jax.io.canonical_ordering import default_canonical_ordering
    from tmol_jax.io.pose_stack_from_biotite import canonical_form_from_biotite

    from tmol_jax.chemical.patched_chemdb import default_patched_chemdb

    raw = {r.name: r for r in default_patched_chemdb().residues}
    classes = tuple(raw[name].io_equiv_class for name in names)
    co = default_canonical_ordering()
    kinds = [co.restype_io_equiv_classes.index(c) for c in classes]
    ports = co.polymer_conn_inds
    atoms = struc.AtomArray(4)
    atoms.chain_id[:] = "A"
    atoms.res_id[:] = [1, 1, 2, 2]
    atoms.res_name[:] = np.repeat(classes, 2)
    for r, cls in enumerate(kinds):
        names = co.restypes_ordered_atom_names[classes[r]]
        atoms.atom_name[2 * r : 2 * r + 2] = [
            names[ports.down_atom_for_co_restype[cls]],
            names[ports.up_atom_for_co_restype[cls]],
        ]
    atoms.coord[:] = [[30, 0, 0], [0, 0, 0], [1.6, 0, 0], [40, 0, 0]]
    close = canonical_form_from_biotite(atoms)
    assert not close["res_gap"].any()
    distant = atoms.copy()
    distant.coord[2, 0] = 8
    broken = canonical_form_from_biotite(distant)
    np.testing.assert_array_equal(broken["res_gap"], [[[False, True], [True, False]]])
    assert not canonical_form_from_biotite(
        distant, missing_density_distance_threshold=0
    )["res_gap"].any()
    ensemble = canonical_form_from_biotite(struc.stack([distant, atoms]))
    assert not ensemble["res_gap"].any()
    distant.coord[2] = np.nan
    assert not canonical_form_from_biotite(distant)["res_gap"].any()
    atoms.bonds = struc.BondList(4, np.array([[0, 1, struc.BondType.SINGLE]]))
    assert canonical_form_from_biotite(atoms)["covalent_bonds"].shape == (0, 5)
