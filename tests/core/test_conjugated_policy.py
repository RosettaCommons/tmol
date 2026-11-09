import numpy as np
from tmol_core.pack.conjugated import find_conjugated_groups


def test_group_keeps_cycle_edges_external_constraints_and_padding():
    types = np.array([[0, 1, 2, 0, -1]], np.int32)
    connections = np.full((1, 5, 3, 2), -1, np.int32)
    for first, a, second, b in ((0, 1, 1, 0), (1, 1, 2, 0), (2, 2, 0, 2), (0, 0, 3, 0)):
        connections[0, first, a] = (second, b)
        connections[0, second, b] = (first, a)
    original = connections.copy()
    conjugated = np.array(
        [[False, True, False], [True, True, False], [True, False, False]]
    )
    groups = find_conjugated_groups(
        types,
        connections,
        conjugated,
        np.array([3, 3, 3]),
        np.array([True, False, False]),
    )
    assert len(groups) == 1
    group = groups[0]
    assert group.blocks == (0, 1, 2)
    assert len(group.links) == 3
    assert group.external_links == ((0, 0, 3, 0),)
    assert group.anchor == 0
    np.testing.assert_array_equal(connections, original)
    assert (
        find_conjugated_groups(
            np.empty((2, 0), np.int32),
            np.empty((2, 0, 0, 2), np.int32),
            np.empty((0, 0), bool),
            np.empty(0, np.int32),
            np.empty(0, bool),
        )
        == []
    )
