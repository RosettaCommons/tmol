from types import SimpleNamespace

import numpy as np
import pytest

from tmol_core.pack.conformer_layout import (
    compact_conformer_forest,
    prepare_conformer_layout,
)


def test_conformer_roots_are_shared_but_coordinate_offsets_stay_external():
    ids = np.array([[0, 1, -1], [0, 1, 2]], dtype=np.int32)
    parent = np.array([[-1, 0, -1], [-1, 0, 1]], dtype=np.int32)
    data = SimpleNamespace(
        id=ids,
        doftype=np.ones_like(ids),
        parent=parent,
        frame_x=ids,
        frame_y=ids,
        frame_z=ids,
    )
    forest, ideal_indices = compact_conformer_forest(
        data,
        3,
        np.array([0, 1], np.int32),
        5,
        np.array([2, 3], np.int32),
        np.array([7, 30], np.int64),
        return_ideal_indices=True,
    )
    np.testing.assert_array_equal(forest[0], [-1, 7, 8, 30, 31, 32])
    np.testing.assert_array_equal(forest[2], [0, 0, 1, 0, 3, 4])
    # Ideal rows are per-type KFO rows, independent of external RTO offsets.
    np.testing.assert_array_equal(ideal_indices, [0, 1, 3, 4, 5])
    assert ideal_indices.dtype == np.int64
    assert forest.dtype == np.int32
    assert forest.flags.c_contiguous
    np.testing.assert_array_equal(data.parent, parent)


def test_empty_layout_and_atom_budget_guard_before_allocation():
    data = SimpleNamespace(
        **{
            name: np.empty((0, 0), np.int32)
            for name in (
                "id",
                "doftype",
                "parent",
                "frame_x",
                "frame_y",
                "frame_z",
                "scans",
                "nodes",
                "n_scans_per_gen",
            )
        },
        gens=np.empty((0, 1, 2), np.int32),
        n_nodes=np.empty(0, np.int32),
    )
    result = prepare_conformer_layout(
        data, 0, np.empty(0, np.int32), np.empty(0, np.int32)
    )
    assert result.ideal_indices.shape == (0,)
    assert result.total == 0 and result.forest.shape == (6, 1)
    assert result.forest[0, 0] == -1
    assert result.offsets.shape == result.nodes.shape == result.scans.shape == (0,)
    with pytest.raises(ValueError, match="32-bit index capacity"):
        prepare_conformer_layout(
            data, 0, np.array([2**30], np.int32), np.array([0, 0], np.int32)
        )


def test_existing_forward_schedule_preserves_paths_and_reverses_dependencies():
    from tmol_core.kinematics.scan_ordering import (
        scan_ordering_from_forward,
        validate_scan_plan,
    )

    parent = np.array([0, 0, 1, 0, 3, 4], np.int32)
    # Two conformers share only the virtual root. Preserve this producer's
    # deliberate second-conformer-first path order.
    nodes = np.array([0, 3, 4, 5, 0, 1, 2], np.int32)
    scans = np.array([0, 4], np.int32)
    gens = np.array([[0, 0], [7, 2]], np.int32)
    ordering = scan_ordering_from_forward(parent, nodes, scans, gens)
    np.testing.assert_array_equal(ordering.forward.nodes, nodes)
    np.testing.assert_array_equal(ordering.backward.nodes, [2, 1, 0, 5, 4, 3, 0])
    validate_scan_plan(
        parent,
        ordering.backward.nodes,
        ordering.backward.scans,
        ordering.backward.gens,
        reverse=True,
    )
    nodes[:] = -1
    assert ordering.forward.nodes[0] == 0
    assert not ordering.backward.nodes.flags.writeable
