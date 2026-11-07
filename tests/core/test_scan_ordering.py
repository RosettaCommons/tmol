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


def test_scan_validation_owns_snapshots_after_caller_mutation():
    parent = np.array([0, 0, 1, 1], np.int32)
    nodes = np.array([0, 1, 2, 1, 3], np.int32)
    scans = np.array([0, 0], np.int32)
    gens = np.array([[0, 0], [3, 1], [5, 2]], np.int32)
    plan = validate_scan_plan(parent, nodes, scans, gens)
    expected = [array.copy() for array in (plan.nodes, plan.scans, plan.gens)]
    for caller in (parent, nodes, scans, gens):
        caller[:] = -1
    for snapshot, before in zip((plan.nodes, plan.scans, plan.gens), expected):
        np.testing.assert_array_equal(snapshot, before)
        assert not snapshot.flags.writeable


def test_generation_dependencies_apply_to_single_node_paths():
    # A redundant one-node path reads its start even though it writes no edge.
    # Node one becomes available only after the first generation finishes.
    parent = np.array([0, 0], np.int32)
    nodes = np.array([0, 1, 1], np.int32)
    scans = np.array([0, 2], np.int32)
    gens = np.array([[0, 0], [3, 2]], np.int32)
    with pytest.raises(ValueError, match="unfinished"):
        validate_scan_plan(parent, nodes, scans, gens)
    scans[1] = 0
    gens = np.array([[0, 0], [2, 1], [2, 1], [3, 2]], np.int32)
    plan = validate_scan_plan(parent, nodes, scans, gens)
    np.testing.assert_array_equal(plan.nodes, nodes)


def test_rejects_root_written_as_an_edge_in_either_direction():
    parent = np.array([0, 0], np.int32)
    for reverse, nodes in [(False, [0, 0, 1]), (True, [1, 0, 0])]:
        with pytest.raises(ValueError, match="more than once"):
            validate_scan_plan(
                parent,
                np.array(nodes),
                np.array([0]),
                np.array([[0, 0], [3, 1]]),
                reverse=reverse,
            )


def test_generation_offsets_are_bounded_before_integer_differences():
    # int32 subtraction alone can wrap and make these decreases look positive.
    generations = np.array([[0, 0], [2**31 - 1, 2**31 - 1], [-2, -2], [0, 0]], np.int32)
    with pytest.raises(ValueError, match="within"):
        validate_scan_plan(
            np.array([0], np.int32),
            np.empty(0, np.int32),
            np.empty(0, np.int32),
            generations,
        )
