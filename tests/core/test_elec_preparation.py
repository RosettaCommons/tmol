"""Chemical lookup invariants independent of coordinate-execution frameworks."""

from types import SimpleNamespace as Record

import numpy as np
import pytest

from tmol_core.scoring.elec import (
    count_pair_table,
    lookup_order,
    partial_charge_table,
    partial_charges_for_block,
    representative_mapping_for_block,
)


def residue(name, atoms=("N", "H1", "H2")):
    return Record(
        name=name,
        atoms=tuple(Record(name=a) for a in atoms),
        atom_to_idx={a: i for i, a in enumerate(atoms)},
    )


def test_exact_compound_and_individual_patch_precedence():
    database = Record(
        atom_charge_parameters=tuple(
            Record(res=res, atom=atom, charge=value)
            for res, atom, value in (
                ("LIG", "N", 1.0),
                ("LIG", "H1", 2.0),
                ("LIG", "H2", 3.0),
                ("LIG:nterm:conj_x", "N", 10.0),
                ("LIG:nterm:conj_x", "H1", 20.0),
                ("LIG:foo", "H2", 30.0),
                ("LIG:nterm:conj_x:foo", "N", 99.0),
                ("LIG:nterm:conj_x:foo", "N", 100.0),
            )
        )
    )
    table = partial_charge_table(database)
    actual = partial_charges_for_block(residue("LIG:nterm:conj_x:foo"), table)
    np.testing.assert_array_equal(actual, np.array([100, 20, 30], dtype=np.float32))
    assert actual.dtype == np.float32
    # An arbitrary compound tag must not specialize a larger patch set.
    assert "foo:bar" not in lookup_order("LIG:foo:bar:other")[1]
    # Coordinate execution receives owned vectors, without a mutable global cache.
    actual[:] = 0
    np.testing.assert_array_equal(
        partial_charges_for_block(residue("LIG:nterm:conj_x:foo"), table), [100, 20, 30]
    )


def test_missing_policy_is_explicit():
    block = residue("LIG")
    with pytest.raises(KeyError, match="LIG,N"):
        partial_charges_for_block(block, {})
    np.testing.assert_array_equal(
        partial_charges_for_block(block, {}, missing=0.0), [0, 0, 0]
    )


@pytest.mark.parametrize("charge", [float("nan"), float("inf"), 1e50])
def test_nonfinite_float32_charge_is_rejected(charge):
    with pytest.raises(ValueError, match="LIG atoms N are non-finite"):
        partial_charges_for_block(residue("LIG", ("N",)), {"LIG": {"N": {"": charge}}})


@pytest.mark.parametrize("specific", [False, True])
def test_representative_specificity_and_equal_rank_atom_order(specific):
    database = Record(
        atom_cp_reps_parameters=(
            Record(
                res="LIG:nterm" if specific else "LIG", atm_outer="H1", atm_inner="N"
            ),
            Record(res="LIG", atm_outer="H2", atm_inner="N"),
        )
    )
    actual = representative_mapping_for_block(
        residue("LIG:nterm"), count_pair_table(database)
    )
    np.testing.assert_array_equal(actual, [1 if specific else 2, 1, 2])
    assert actual.dtype == np.int32


def test_representative_identity_and_invalid_target():
    block = residue("LIG")
    np.testing.assert_array_equal(
        representative_mapping_for_block(block, {}), [0, 1, 2]
    )
    with pytest.raises(KeyError, match="LIG H1->missing"):
        representative_mapping_for_block(block, {"LIG": {"H1": {"": "missing"}}})


def test_explicit_charge_overrides_are_complete_finite_and_take_precedence():
    block = residue("LIG", ("N", "H1"))
    table = {"LIG": {"N": {"": 99.0}, "H1": {"": 98.0}}}
    charges = partial_charges_for_block(block, table, overrides={"N": -0.7, "H1": 0.3})
    np.testing.assert_array_equal(charges, np.array([-0.7, 0.3], np.float32))
    with pytest.raises(KeyError, match="LIG,H1"):
        partial_charges_for_block(block, table, missing=0.0, overrides={"N": 1.0})
    with pytest.raises(ValueError, match="non-finite"):
        partial_charges_for_block(block, table, overrides={"N": np.inf, "H1": 1.0})

    from types import SimpleNamespace


    block = SimpleNamespace(name="LIG:nterm:conj_x:other", atoms=("N", "H", "V"))
    table = {
        "LIG": {
            "N": {"": 1.0, "nterm:conj_x": 2.0, "nterm:conj_x:other": 3.0},
            "H": {"": 4.0},
        }
    }
    with pytest.raises(KeyError, match="V not found"):
        partial_charges_for_block(
            block, table, overrides=getattr(block, "elec_partial_charges", None)
        )
    table["LIG"]["V"] = {"": 0.0}
    np.testing.assert_array_equal(
        partial_charges_for_block(
            block, table, overrides=getattr(block, "elec_partial_charges", None)
        ),
        [3, 4, 0],
    )
    block.elec_partial_charges = {"N": -0.5, "H": 0.5}
    with pytest.raises(KeyError, match="V not found"):
        partial_charges_for_block(
            block, table, overrides=getattr(block, "elec_partial_charges", None)
        )
    block.elec_partial_charges["V"] = 0.0
    np.testing.assert_array_equal(
        partial_charges_for_block(
            block, table, overrides=getattr(block, "elec_partial_charges", None)
        ),
        [-0.5, 0.5, 0],
    )
    block.elec_partial_charges["V"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        partial_charges_for_block(
            block, table, overrides=getattr(block, "elec_partial_charges", None)
        )


def test_packed_electrostatics_preserves_physical_axes_and_parameter_owned_geometry():
    from tmol_core.scoring.elec import prepare_elec_block, pack_elec_parameters

    block = residue("LIG", ("A", "B", "C", "D"))
    block.n_atoms = 4
    for atom in block.atoms:
        atom.atom_type = "generic_C"
    block.path_distance = np.abs(np.arange(4)[:, None] - np.arange(4)[None, :])
    charge = np.array([0.2, -0.3, 0.4, -0.5], np.float32)
    rep = np.arange(4, dtype=np.int32)
    first = prepare_elec_block(block, charge, rep, ())
    second = prepare_elec_block(block, -charge, rep.copy(), (), previous=first)
    assert second.geometry is first.geometry
    packed = pack_elec_parameters([block], [second], 6)
    np.testing.assert_array_equal(packed.charges[0, :4], -charge)
    np.testing.assert_array_equal(packed.inter[0, :4, :4], block.path_distance)
    assert packed.intra[0, 0, 3] == 5  # generic 1-4 pairs carry full weight
    assert packed.all_ligand_typed.tolist() == [1]
    assert not packed.charges[0, 4:].any()
    assert not packed.inter[0, 4:].any() and not packed.inter[0, :, 4:].any()
    # A new generic/canonical typing table changes the physical count-pair rule.
    canonical = prepare_elec_block(block, charge, rep, {"generic_C"}, previous=first)
    assert canonical.geometry is not first.geometry
    assert canonical.geometry.intra[0, 3] == 3
    omitted = pack_elec_parameters(
        [block], [second], 6, include_geometry=False, include_ligand_flags=False
    )
    np.testing.assert_array_equal(omitted.charges, packed.charges)
    assert omitted.inter is omitted.intra is omitted.all_ligand_typed is None
