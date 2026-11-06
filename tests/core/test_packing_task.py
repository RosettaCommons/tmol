import numpy as np
import pytest

from tmol_core.pack.task import chi_sample_budget, prepare_packing_task_rows


def test_disabled_background_rows_and_padding_keep_stable_identity():
    considered = np.array([[[8, 4], [2, -1]], [[7, -1], [-1, -1]]])
    allowed = np.array(
        [[[False, False], [True, False]], [[True, False], [False, False]]]
    )
    rows = prepare_packing_task_rows(
        np.array([[True, True], [True, False]]), considered, allowed
    )
    np.testing.assert_array_equal(rows.cons_bt_block_type, [8, 4, 2, 7])
    np.testing.assert_array_equal(
        rows.global_block_ind_for_considered_block_types, [0, 0, 1, 2]
    )
    np.testing.assert_array_equal(rows.allowed_cons_bt, [2, 3])
    np.testing.assert_array_equal(rows.allowed_bt_block_type, [2, 7])
    np.testing.assert_array_equal(rows.cons_bt_which_block_type, [0, 1, 0, 0])
    assert rows.is_cons_bt_allowed.dtype == np.bool_


def test_empty_task_and_strict_budget():
    rows = prepare_packing_task_rows(
        np.zeros((0, 2), bool), np.zeros((0, 2, 3), np.int64), np.zeros((0, 2, 3), bool)
    )
    assert rows.cons_bt_pose.shape == rows.allowed_cons_bt.shape == (0,)
    assert chi_sample_budget(np.int64(2), 9) == (2, 9)
    for value in (True, 0, -1, 1.5, "2"):
        with pytest.raises(ValueError, match="positive integers"):
            chi_sample_budget(value, 9)
