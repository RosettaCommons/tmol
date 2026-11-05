import numpy as np
import pytest

from tmol_core.kinematics.scan_ordering import prepare_scan_ordering, validate_scan_plan


@pytest.mark.parametrize(
    "parents", [[], [0], [0, 0, 1, 1, 3], [0, 3, 0, 2, 1, 2], [0] * 129]
)
def test_scan_plans_match_independent_ancestor_sums(parents):
    parent = np.array(parents, dtype=np.int32)
    ordering = prepare_scan_ordering(parent)
    values = np.arange(len(parent), dtype=np.int64)
    for reverse, plan in [(False, ordering.forward), (True, ordering.backward)]:
        validate_scan_plan(parent, plan.nodes, plan.scans, plan.gens, reverse=reverse)
        actual = values.copy()
        for (nb, sb), (ne, se) in zip(plan.gens[:-1], plan.gens[1:]):
            for scan in range(sb, se):
                start = nb + plan.scans[scan]
                stop = nb + plan.scans[scan + 1] if scan + 1 < se else ne
                path = plan.nodes[start:stop]
                for source, target in zip(path[:-1], path[1:]):
                    actual[target] += actual[source]
        expected = values.copy()
        for node in range(1, len(parent)):
            ancestor = parent[node]
            while True:
                if reverse:
                    expected[ancestor] += values[node]
                else:
                    expected[node] += values[ancestor]
                if ancestor == 0:
                    break
                ancestor = parent[ancestor]
        np.testing.assert_array_equal(actual, expected)
        for array in (plan.nodes, plan.scans, plan.gens):
            assert array.dtype == np.int32 and array.flags.c_contiguous
            assert not array.flags.writeable


@pytest.mark.parametrize(
    "nodes,scans,gens,reason",
    [
        ([0, 1, 2, 1, 4], [0, 0], [[0, 0], [3, 1], [5, 2]], "outside"),
        ([0, 2, 1, 1, 3], [0, 0], [[0, 0], [3, 1], [5, 2]], "parent edges"),
        ([0, 1, 2, 1, 2], [0, 0], [[0, 0], [3, 1], [5, 2]], "more than once"),
        ([0, 1, 2, 1, 3], [0, 3], [[0, 0], [5, 2]], "unfinished"),
        ([0, 1, 2], [0], [[0, 0], [3, 1]], "every forest edge"),
        ([0, 1, 2, 1, 3], [0, 0], [[0, 0], [5, 2]], "disjoint"),
        ([0, 1, 2, 1, 3], [0, 0], [[0, 0], [3, 1], [4, 2]], "endpoints"),
    ],
)
def test_rejects_unsafe_native_scan_plans(nodes, scans, gens, reason):
    with pytest.raises(ValueError, match=reason):
        validate_scan_plan(
            np.array([0, 0, 1, 1]), np.array(nodes), np.array(scans), np.array(gens)
        )
