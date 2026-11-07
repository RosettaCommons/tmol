import numpy as np
import pytest

from tmol_core.pose.block_selection import selection_plan, validate_repeat


def test_reordered_custom_ports_keep_reciprocity_and_clear_both_missing_fields():
    types = np.array([[0, 1, 2], [0, 1, 2]], np.int64)
    offsets = np.array([[0, 2, 5], [0, 2, 5]], np.int64)
    links = np.full((2, 3, 8, 2), -1, np.int64)
    links[:, 0, 7] = [2, 3]
    links[:, 2, 3] = [0, 7]
    links[:, 1, 4] = [2, 5]
    links[:, 2, 5] = [1, 4]
    plan = selection_plan([2, 0], types, offsets, [2, 3, 4], links)
    np.testing.assert_array_equal(plan.source_atoms, [5, 6, 7, 8, 0, 1])
    np.testing.assert_array_equal(plan.offsets, [[0, 4], [0, 4]])
    np.testing.assert_array_equal(plan.connections[:, 0, 3], [[1, 7]] * 2)
    np.testing.assert_array_equal(plan.connections[:, 1, 7], [[0, 3]] * 2)
    np.testing.assert_array_equal(plan.connections[:, 0, 5], -1)
    np.testing.assert_array_equal(links[:, 2, 5], [[1, 4]] * 2)
    with pytest.raises(ValueError, match="homogeneous"):
        selection_plan([0], [[0], [1]], [[0], [0]], [2, 3], links[:, :1])
    with pytest.raises(ValueError, match="non-empty"):
        selection_plan([], types, offsets, [2, 3, 4], links)
    with pytest.raises(ValueError, match="split-block"):
        selection_plan([0], types, offsets, [2, 3, 4], links, split_block_mapping=True)
    with pytest.raises(ValueError, match="single reference"):
        validate_repeat(2)
    with pytest.raises(ValueError, match="split-block"):
        validate_repeat(1, True)
