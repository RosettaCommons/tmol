"""Forest index invariants are independent of chemical or storage ordering."""

import numpy as np
import pytest

from tmol_core.kinematics import prepare_kinematic_forest


def inputs():
    # Node one follows node three: valid trees need not be stored topologically.
    return dict(
        parent=np.array([0, 3, 0, 2]),
        frame_x=np.array([0, 1, 2, 3]),
        frame_y=np.array([0, 3, 0, 2]),
        frame_z=np.array([0, 2, 1, 0]),
        doftype=np.array([0, 2, 1, 2]),
    )


def test_snapshot_is_independent_and_preserves_nonordered_tree():
    source = inputs()
    forest = prepare_kinematic_forest(**source)
    source["parent"][1] = 1
    np.testing.assert_array_equal(forest.parent, [0, 3, 0, 2])
    assert forest.n_nodes == 4
    for name in source:
        value = getattr(forest, name)
        assert value.dtype == np.int32 and value.flags.c_contiguous
        assert not value.flags.writeable


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("parent", [0, 3, 0, 1], "cycle"),
        ("frame_x", [0, 1, 2, 4], "outside"),
        ("parent", [0, -1, 0, 2], "outside"),
        ("doftype", [0, 2, 0, 2], "only node zero"),
        ("doftype", [0, 2, 3, 2], "doftype"),
        ("frame_z", [1, 2, 1, 0], "virtual root"),
        ("frame_y", [0.0, 3.0, 0.0, 2.0], "integer"),
        ("doftype", [False, True, True, True], "integer"),
        ("parent", [0, 2**40, 0, 2], "int32"),
        ("parent", [0, 0], "same length"),
    ],
)
def test_rejects_invalid_native_indices(field, value, reason):
    data = inputs()
    data[field] = np.array(value)
    with pytest.raises(ValueError, match=reason):
        prepare_kinematic_forest(**data)


def test_empty_forest():
    empty = np.empty(0, dtype=np.int32)
    forest = prepare_kinematic_forest(empty, empty, empty, empty, empty)
    assert forest.n_nodes == 0 and forest.parent.shape == (0,)


def test_native_node_count_checked_before_copy():
    # Broadcast storage stays one element, so rejecting the native ABI overflow
    # must not first allocate or traverse the logically oversized input.
    oversized = np.broadcast_to(np.int32(0), (2**31,))
    with pytest.raises(ValueError, match="node count"):
        prepare_kinematic_forest(oversized, oversized, oversized, oversized, oversized)
