from types import SimpleNamespace

import numpy as np
import pytest

from tmol_core.pack.conjugated import append_current_conformer, group_torsion_nodes


def test_joint_torsion_child_is_independent_of_named_axis_direction():
    forest = SimpleNamespace(id=np.array([2, 0, 1]), parent=np.array([-1, 0, 1]))
    columns = [(0, "chi1", 2, 0), (1, "linkage", 0, 2)]
    np.testing.assert_array_equal(group_torsion_nodes(forest, columns, 3), [2, 2])
    with pytest.raises(ValueError, match="kinforest does not contain"):
        group_torsion_nodes(forest, [(0, "non_tree", 2, 1)], 3)


def test_current_conformer_requires_complete_finite_measurement():
    grid = np.array([[0.2, 0.4]], dtype=np.float64)
    actual = append_current_conformer(grid, iter([0.5, 0.75]))
    np.testing.assert_array_equal(actual, [[0.2, 0.4], [0.5, 0.75]])
    assert actual.dtype == grid.dtype
    np.testing.assert_array_equal(grid, [[0.2, 0.4]])

    def unmeasurable():
        yield float("nan")
        raise AssertionError("An unmeasurable state must stop requesting torsions")

    assert append_current_conformer(grid, unmeasurable()) is grid
    with pytest.raises(ValueError, match="do not cover"):
        append_current_conformer(grid, iter([0.5]))
