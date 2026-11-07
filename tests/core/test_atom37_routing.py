"""Routing invariants independent of either framework's coordinate execution."""

from types import SimpleNamespace

import numpy as np
import pytest

from tmol_core.io.atom37 import prepare_atom37_routing


def ordering():
    return SimpleNamespace(
        restype_io_equiv_classes=("LIG", "HYP"),
        restypes_ordered_atom_names={"LIG": ("C", "O"), "HYP": ("N", "CA", "O")},
        max_n_canonical_atoms=3,
    )


def test_mixed_routing_preserves_atom_identity_and_padding():
    types = np.array([[0, -1, 1], [1, 0, -1]])
    slots = np.array([[1, -1, 0], [2, 0, 1]])
    plan = prepare_atom37_routing(types, slots, 3, ordering())
    coordinates = np.arange(18).reshape(2, 3, 3)
    output = np.full(plan.canonical_shape, -1)
    output[plan.target_indices] = coordinates[plan.source_indices]
    np.testing.assert_array_equal(
        output,
        [
            [[2, 0, -1], [-1, -1, -1], [7, 8, 6]],
            [[10, 11, 9], [14, 12, -1], [-1, -1, -1]],
        ],
    )
    # Every target has exactly one source, and padding contributes neither.
    assert len(set(zip(*plan.target_indices))) == 10
    assert np.all(types[plan.source_indices[:2]] >= 0)
    empty = prepare_atom37_routing(np.full((2, 0), -1), slots, 3, ordering())
    assert empty.canonical_shape == (2, 0, 3)
    assert all(x.size == 0 for x in empty.source_indices)


@pytest.mark.parametrize(
    "types, slots, count, error",
    [
        ([[2]], [[0], [0]], 1, "outside the canonical slot map"),
        ([[-2]], [[0], [0]], 1, "outside the canonical slot map"),
        ([[0]], [[2], [0]], 1, "outside the canonical atom range"),
        ([[0]], [[0, 0], [0, 1]], 2, "same canonical atom"),
        ([[1]], [[0], [-1]], 1, r"residue type\(s\) HYP"),
        ([[0]], [[0], [0]], 2, "covers 1 slots"),
    ],
)
def test_invalid_routing_cannot_silently_fabricate_atoms(types, slots, count, error):
    with pytest.raises(ValueError, match=error):
        prepare_atom37_routing(np.array(types), np.array(slots), count, ordering())
