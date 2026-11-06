from types import SimpleNamespace

import numpy as np
import pytest

from tmol_core.pack.correlation import (
    CorrelatedBlockGroup,
    collapse_group_rotamers,
    expand_group_assignments,
)


def test_joint_sources_preserve_types_empty_poses_and_exclusive_offsets():
    layout = SimpleNamespace(
        n_rots_for_block=np.array([[2, 2, 2, 0], [0, 0, 0, 0]], dtype=np.int32),
        rot_offset_for_block=np.array([[0, 2, 4, 6], [6, 6, -1, -1]], dtype=np.int32),
        n_rots_for_pose=np.array([6, 0], dtype=np.int32),
        rot_offset_for_pose=np.array([0, 6], dtype=np.int32),
        block_ind_for_rot=np.array([0, 0, 1, 1, 2, 2], dtype=np.int32),
        pose_for_rot=np.zeros(6, dtype=np.int32),
        block_type_ind_for_rot=np.array([10, 11, 20, 21, 30, 31], dtype=np.int32),
    )
    group = CorrelatedBlockGroup(0, (0, 2))
    collapse, _, _ = collapse_group_rotamers(layout, [group])
    np.testing.assert_array_equal(collapse.compact_to_orig, [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(collapse.compact_n_rots_for_pose, [5, 0])
    np.testing.assert_array_equal(
        collapse.compact_rot_offset_for_block, [[0, 2, 4, 5], [5, 5, -1, -1]]
    )
    for joint in range(2):
        for external in range(2):
            original = np.array([[joint, 2 + external, 4, -1], [-1] * 4])
            expanded = expand_group_assignments(original, collapse)
            np.testing.assert_array_equal(
                expanded[0], [joint, 2 + external, 4 + joint, -1]
            )
            np.testing.assert_array_equal(
                layout.block_type_ind_for_rot[expanded[0, :3]],
                [10 + joint, 20 + external, 30 + joint],
            )
            assert original[0, 2] == 4
    with pytest.raises(ValueError, match="invalid original"):
        expand_group_assignments(np.array([[3, 2, 4, -1], [-1] * 4]), collapse)
    with pytest.raises(ValueError, match="multiple correlated"):
        collapse_group_rotamers(layout, [group, group])
