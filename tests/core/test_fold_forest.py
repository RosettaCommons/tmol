import numpy as np
import pytest

from tmol_core.kinematics.fold_forest import EdgeType, FoldForest
from tmol_core.kinematics.check_fold_forest import validate_fold_forest


def test_database_without_connection_slots_and_padded_batch():
    forest = FoldForest.from_connectivity(
        np.array([[0, 1, -1], [1, -1, -1]], dtype=np.int64),
        np.empty((2, 3, 0, 2), dtype=np.int64),
        np.array([[0, 0, -1], [0, -1, -1]], dtype=np.int64),
        np.array([-1, -1]),
        np.array([-1, -1]),
        np.array([0, 0]),
        np.empty((2, 0), dtype=bool),
    )
    np.testing.assert_array_equal(forest.n_edges, [2, 1])
    np.testing.assert_array_equal(
        forest.edges,
        [
            [[EdgeType.root_jump, -1, 0, -1], [EdgeType.jump, 0, 1, 0]],
            [[EdgeType.root_jump, -1, 0, -1], [-1, -1, -1, -1]],
        ],
    )
    validate_fold_forest(np.array([2, 1], dtype=np.int64), forest.edges)


@pytest.mark.parametrize("n_poses", [0, 1])
def test_empty_fold_forests(n_poses):
    forest = FoldForest.from_connectivity(
        np.empty((n_poses, 0), dtype=np.int64),
        np.empty((n_poses, 0, 0, 2), dtype=np.int64),
        np.empty((n_poses, 0), dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty((0, 0), dtype=bool),
    )
    np.testing.assert_array_equal(forest.n_edges, np.zeros(n_poses, dtype=np.int64))
    validate_fold_forest(np.zeros(n_poses, dtype=np.int64), forest.edges)
