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
