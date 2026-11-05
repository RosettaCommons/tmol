"""JAX-specific movement snapshots, dtype and optimizer diagnostics."""

import jax
import numpy as np
import pytest

from tests.scientific.execution import Execution
from tests.scientific.kinematics import MOLECULAR_KINEMATIC_TERMS
from tests.scientific.molecules import DOMAIN_CASES
from tmol_jax import MoveMap, PoseStackKinematics, run_kin_min


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_internal_minimizer_fixed_dofs_and_disabled_identity(dtype):
    with jax.enable_x64():
        backend = Execution("jax-cpu").molecule(
            DOMAIN_CASES["mixed_batch"], terms=MOLECULAR_KINEMATIC_TERMS
        )
        pose = backend.pose.replace(coords=backend.pose.coords.astype(dtype))
        move_map = MoveMap.from_pose_stack(pose)
        module = PoseStackKinematics(pose)
        mapping = module.bind()
        unchanged, result = run_kin_min(
            pose, backend.score_function, move_map, return_result=True
        )
        np.testing.assert_array_equal(unchanged.coords, pose.coords)
        assert result["n_iter"] == 0 and result["converged"]
        move_map.move_all_named_torsions = True
        mask = np.asarray(module.resolve_move_map(move_map)).reshape(-1)
        minimized, result = run_kin_min(
            pose, backend.score_function, move_map, max_iter=3, return_result=True
        )
        assert minimized.coords.dtype == dtype
        assert result["fun"] < result["initial_fun"]
        np.testing.assert_array_equal(
            np.asarray(result["x"])[~mask],
            np.asarray(mapping.initial_dofs).reshape(-1)[~mask],
        )
        np.testing.assert_array_equal(np.asarray(result["grad"])[~mask], 0)
        move_map.move_all_named_torsions = False
        assert mask.any() and not module.resolve_move_map(move_map).any()
        wrong = MoveMap(1, 1, 0, 1)
        with pytest.raises(ValueError, match="dimensions"):
            module.resolve_move_map(wrong)
