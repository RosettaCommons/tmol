import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tmol_core.kinematics.move_map import MoveMap


def test_host_movement_selection_and_explicit_disable():
    mm = MoveMap(2, 3, 5, 7)
    assert not any(bool(np.any(a)) for a in mm.native_arguments)
    mm.move_all_sc = True
    selected = np.array([[False, True, False], [True, False, False]])
    mm.set_move_all_sc_tors_for_blocks(selected, value=False)
    np.testing.assert_array_equal(mm.move_scs_mask, selected)
    assert not mm.move_scs.any()
    mm.set_move_named_torsion_for_blocks(
        np.array([0, 1]), np.array([2, 0]), np.array([4, 1])
    )
    assert mm.move_named_torsion.sum() == 2
    assert mm.move_named_torsion[0, 2, 4] and mm.move_named_torsion[1, 0, 1]
    mm.set_move_atom_dof_for_blocks(1, 2, 6, 3, value=False)
    assert mm.move_atom_dof_mask[1, 2, 6, 3]
    assert not mm.move_atom_dof[1, 2, 6, 3]
    mm.set_move_jump_dof_for_root_jumps(1, 0, 5)
    assert mm.move_root_jump_dof.sum() == 1
    assert not mm.move_jump_dof.any()


def test_host_movement_shape_and_empty_capacity():
    mm = MoveMap(0, 3, 0, 7)
    assert mm.move_atom_dof.shape == (0, 3, 7, 4)
    assert mm.move_root_jump_dof.shape == (0, 3, 6)
    mm = MoveMap(2, 3, 5, 7)
    with pytest.raises(AssertionError):
        mm.set_move_all_mc_tors_for_blocks(np.zeros((2, 2), dtype=bool))
    with pytest.raises(AssertionError):
        mm.set_move_named_torsion_for_blocks(np.zeros((2, 3, 4), dtype=bool))


def test_host_movement_imports_no_execution_framework():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from tmol_core.kinematics.move_map import MoveMap; "
            "import sys; MoveMap(1, 2, 3, 4); "
            "assert not any(x in sys.modules for x in ('torch', 'jax', 'tmol'))",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
        env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2])),
    )
