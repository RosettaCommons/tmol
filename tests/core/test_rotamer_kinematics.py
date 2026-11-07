from types import SimpleNamespace

import numpy as np
import pytest

from tmol_core.kinematics.scan_ordering import validate_scan_plan
from tmol_core.pack.kinematics import (
    pack_rotamer_kinematics,
    prepare_rotamer_kinematics,
)


def test_rotamer_native_boundary_preserves_physical_identity_and_root():
    order = np.array([2, 0, 4, 3, 1], dtype=np.int32)
    predecessors = np.array([2, 4, -9999, 0, 2], dtype=np.int32)
    chemistry = SimpleNamespace(
        bfto_2_orig=order,
        preds=predecessors,
        dof_type=np.array([2, 2, 1, 2, 2], dtype=np.int32),
    )
    ideal = np.arange(15, dtype=np.float64).reshape(5, 3)
    calls = []

    def capture_boundary(coords, parent, frame_x, frame_y, frame_z, doftype):
        calls.append((coords, parent, frame_x, frame_y, frame_z, doftype))
        np.testing.assert_array_equal(coords[0], 0)
        np.testing.assert_array_equal(coords[1:], ideal[order])
        assert coords.dtype == np.float32
        for array in (parent, frame_x, frame_y, frame_z, doftype):
            assert array.dtype == np.int32 and array.shape == (6,)
            assert array[0] == 0
        assert parent[1] == 0
        # Mainchain atom a4 is the root's reference child, although it comes
        # after the non-mainchain child a0 in the breadth-first atom order.
        assert order[frame_x[1] - 1] == 4
        # This unit test checks transport, not inverse-coordinate physics.
        return np.ones((6, 9), dtype=np.float32)

    tree = prepare_rotamer_kinematics(
        chemistry,
        5,
        [f"a{i}" for i in range(5)],
        ["a2", "a4"],
        ideal,
        capture_boundary,
    )
    assert len(calls) == 1
    np.testing.assert_array_equal(tree.id[tree.kinforest_idx], np.arange(5))
    np.testing.assert_array_equal(tree.id[tree.parent[1:]], predecessors[tree.id[1:]])
    assert tree.parent[0] == -1
    validate_scan_plan(calls[0][1], tree.nodes, tree.scans, tree.gens)
    assert tree.dofs_ideal.shape == (5, 9)


def test_rotamer_inverse_boundary_rejects_wrong_precision():
    chemistry = SimpleNamespace(
        bfto_2_orig=np.array([0], dtype=np.int32),
        preds=np.array([-9999], dtype=np.int32),
        dof_type=np.array([1], dtype=np.int32),
    )
    with pytest.raises(ValueError, match="float32 DOFs"):
        prepare_rotamer_kinematics(
            chemistry,
            1,
            ["ZN"],
            [],
            np.zeros((1, 3)),
            lambda *args: np.ones((2, 9), dtype=np.float64),
        )


def test_empty_rotamer_packing_has_valid_native_shapes():
    packed = pack_rotamer_kinematics([])
    assert packed.id.shape == (0, 0)
    assert packed.gens.shape == (0, 0, 2)
    assert packed.n_scans_per_gen.shape == (0, 0)
    assert packed.dofs_ideal.shape == (0, 0, 9)
    assert packed.id.dtype == np.int32
    assert packed.dofs_ideal.dtype == np.float32
