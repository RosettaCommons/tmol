import numpy as np
import pytest

from tmol_core.pose.sequences import chain_ids_from_lengths


def test_ragged_chain_partition_and_empty_pose():
    result = chain_ids_from_lengths([4, 0, 2], 5, [[1, 3], [], [2]])
    assert result.dtype == np.int32
    np.testing.assert_array_equal(
        result, [[0, 1, 1, 1, -1], [-1] * 5, [0, 0, -1, -1, -1]]
    )
    np.testing.assert_array_equal(chain_ids_from_lengths([1], 2, []), [[0, -1]])
    with pytest.raises(ValueError, match="partition"):
        chain_ids_from_lengths([4], 5, [[1, 2]])
    with pytest.raises(ValueError, match="extent"):
        chain_ids_from_lengths([6], 5)
