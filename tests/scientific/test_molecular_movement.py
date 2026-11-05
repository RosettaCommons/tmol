"""Shared chemical movement priorities and internal minimization invariants."""

import numpy as np
import pytest

from .kinematics import MolecularKinematicExecution, MOLECULAR_KINEMATIC_TERMS
from .molecules import DOMAIN_CASES


@pytest.mark.parametrize("name", DOMAIN_CASES)
def test_molecular_movement_defaults_and_explicit_overrides(backend, name):
    backend.molecule(DOMAIN_CASES[name], terms=("fa_ljrep",))
    execution = MolecularKinematicExecution(backend, backend.coords())
    move_map = execution.move_map()
    assert not execution.movement_mask(move_map).any()
    move_map.move_all_root_jumps = True
    mask = execution.movement_mask(move_map)
    assert mask[:, :3].any() and not mask[0].any()
    move_map.set_move_all_jump_dofs_for_root_jump(0, value=False)
    disabled = execution.movement_mask(move_map)
    atoms_in_first_pose = execution.atom_ids // backend.coords().shape[1] == 0
    assert not disabled[atoms_in_first_pose].any()
    np.testing.assert_array_equal(
        disabled[~atoms_in_first_pose], mask[~atoms_in_first_pose]
    )
    move_map.move_all_named_torsions = True
    named = execution.movement_mask(move_map)
    # Named torsions can rotate bonds; they never enable bond lengths or angles.
    newly_moving = named & ~disabled
    assert not newly_moving[:, 1:3].any()
    if name not in ("metal_cluster",):
        assert newly_moving.any()


@pytest.mark.parametrize("name", ("ligand", "connected_noncanonical", "d_peptide"))
def test_internal_minimization_decreases_real_molecular_energy(backend, name):
    backend.molecule(DOMAIN_CASES[name], terms=MOLECULAR_KINEMATIC_TERMS)
    initial = backend.coords().copy()
    execution = MolecularKinematicExecution(backend, initial)
    move_map = execution.move_map()
    move_map.move_all_named_torsions = True
    before = backend.scores(initial).sum()
    result = execution.minimize(move_map, 5)
    after = backend.scores(result).sum()
    assert np.isfinite(result).all() and np.isfinite(after)
    assert after < before - 1e-3
    assert np.linalg.norm(result - initial) > 1e-4
    np.testing.assert_array_equal(backend.coords(), initial)
