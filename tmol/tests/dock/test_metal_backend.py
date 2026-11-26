"""Metal coordination on the continuous docking backend."""

from __future__ import annotations

import torch

from tmol.dock import ContinuousBackend
from tmol.dock.ligand_dofs import LigandVariantDOFs, build_ligand_docking_dofs
from tmol.dock.scoring_backend import MinimizationResult, ScoringBackend
from tmol.score._score_types import ScoreType
from tmol.tests.dock.test_ligand_dof_roundtrip import _ligand_block


class _BareBackend(ScoringBackend):
    """A backend implementing only the two abstract methods."""

    ligand_block = 0

    def score(self, pose_stack):
        return torch.zeros(pose_stack.n_poses)

    def minimize(self, pose_stack, docking_dofs, **kwargs):
        return MinimizationResult(pose_stack=pose_stack, genes=None)


def test_variant_hooks_are_optional_for_a_backend():
    """The GA calls these every generation; a backend need not implement them."""
    backend = _BareBackend()
    assert backend.set_ligand_variants(None, None) is None
    assert backend.apply_pose_variants(None) is None
    assert backend.set_metal_term(None, 1.0) is None


def test_continuous_scoring_already_carries_the_metal_term(cdk2_pose_and_db):
    """A bond the search declares is scored without the backend rendering a term."""
    _, param_db = cdk2_pose_and_db
    backend = ContinuousBackend(ligand_block=0, param_db=param_db)

    sfxn = backend._score_function(torch.device("cpu"))

    assert sfxn.score_type_covered_by_contained_term(ScoreType.metal_coordination)
    assert float(sfxn.get_weight(ScoreType.metal_coordination)) != 0.0


def test_one_ligand_form_minimizes_on_the_dofs_it_was_handed(cdk2_pose_and_db):
    """Recorded variants do not disturb a population that agrees on its form."""
    pose_stack, param_db = cdk2_pose_and_db
    ligand_block = _ligand_block(pose_stack)
    backend = ContinuousBackend(ligand_block=ligand_block, param_db=param_db)
    docking_dofs = build_ligand_docking_dofs(pose_stack, ligand_block)

    assert backend._minimized_variants(pose_stack, docking_dofs) is None

    block_type_index = int(pose_stack.block_type_ind64[0, ligand_block])
    variants = LigandVariantDOFs.single(
        docking_dofs,
        block_type_index,
        pose_stack.packed_block_types.active_block_types[block_type_index],
        pose_stack.device,
    )
    backend.set_ligand_variants(variants, pose_stack)

    assert backend.ligand_variant_dofs is variants
    assert backend._minimized_variants(pose_stack, docking_dofs) is None
