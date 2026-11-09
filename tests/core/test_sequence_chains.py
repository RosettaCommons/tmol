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


def test_connection_labels_are_pose_local_and_do_not_modify_input():
    from tmol_core.pose.sequences import find_connections_in_sequences

    sequences = [
        ["CYD--dslf-link", "ALA", "CYD--dslf-link"],
        ["CYD--dslf-link"],
    ]
    trimmed, pairs = find_connections_in_sequences(sequences)
    assert trimmed == [["CYD", "ALA", "CYD"], ["CYD"]]
    assert pairs == [[(0, "dslf", 2, "dslf")], []]
    assert sequences[0][0] == sequences[1][0] == "CYD--dslf-link"
    with pytest.raises(ValueError, match="more than two"):
        find_connections_in_sequences([["CYD--dslf-link"] * 3])


def test_named_connection_resolution_accepts_custom_ports_and_rejects_missing():
    from types import SimpleNamespace
    from tmol_core.pose.sequences import resolve_connection_pairs

    active_types = [SimpleNamespace(connection_to_cidx={"branch": 7, "attach": 3})]
    names = [["CUSTOM", "CUSTOM"]]
    types = np.zeros((1, 2), np.int64)
    assert resolve_connection_pairs(
        active_types, names, types, [[(0, "branch", 1, "attach")]]
    ) == [[(0, 7, 1, 3)]]
    with pytest.raises(ValueError, match="Valid connection names.*'branch', 'attach'"):
        resolve_connection_pairs(
            active_types, names, types, [[(0, "branch", 1, "missing")]]
        )
