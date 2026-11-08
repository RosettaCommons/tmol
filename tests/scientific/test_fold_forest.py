"""Chemical fold edges preserve independent components across molecular domains."""

import numpy as np
import pytest

from tmol_core.kinematics.fold_forest import EdgeType
from tmol_core.kinematics.check_fold_forest import validate_fold_forest
from .molecules import DOMAIN_CASES


@pytest.mark.parametrize("name", DOMAIN_CASES)
def test_molecular_fold_forest_preserves_components_and_polymer_edges(backend, name):
    case = DOMAIN_CASES[name]
    backend.molecule(case, terms=("ref",))
    if backend.framework == "jax":
        from tmol_jax import FoldForest
    else:
        from tmol import FoldForest
    forest = FoldForest.reasonable_fold_forest(backend.pose)
    n_blocks = np.array([len(row) for row in case.sequences], np.int64)
    validate_fold_forest(n_blocks, forest.edges)
    for pose, lengths in enumerate(case.chain_lengths):
        edges = forest.edges[pose, : forest.n_edges[pose]]
        roots = edges[edges[:, 0] == EdgeType.root_jump]
        assert len(roots) == len(lengths)
        polymers = edges[edges[:, 0] == EdgeType.polymer]
        assert np.abs(polymers[:, 2] - polymers[:, 1]).sum() == n_blocks[pose] - len(
            lengths
        )

    if backend.name == "jax-cpu" and name == "ligand":
        _check_explicit_chain_labels(backend)


def _check_explicit_chain_labels(backend):
    jax = backend.module

    with jax.default_device(backend.device):
        from tmol_jax import FoldForest
        from tests.scientific.execution import Execution
        from tests.scientific.molecules import MolecularCase
        from tmol_core.kinematics.fold_forest import EdgeType

        pose = (
            Execution("jax-cpu")
            .molecule(MolecularCase((), (("ALA", "GLY"),), ((1, 1),)), terms=("ref",))
            .pose
        )
        separate = FoldForest.reasonable_fold_forest(pose)
        assert (separate.edges[0, :, 0] == EdgeType.root_jump).sum() == 2
        without_labels = pose.replace(pdb_info=None)
        with pytest.raises(ValueError, match="chain_id is required"):
            FoldForest.reasonable_fold_forest(without_labels)
        with pytest.raises(ValueError, match="shape"):
            FoldForest.reasonable_fold_forest(without_labels, chain_id=["A", "A"])
        with pytest.raises(ValueError, match="every real block"):
            FoldForest.reasonable_fold_forest(without_labels, chain_id=[[None, None]])
        empty = FoldForest.reasonable_fold_forest(pose[:0])
        assert empty.edges.shape == (0, 1, 4) and empty.n_edges.size == 0
        gap = FoldForest.reasonable_fold_forest(without_labels, chain_id=[["A", "A"]])
        assert (gap.edges[0, :, 0] == EdgeType.root_jump).sum() == 1
        np.testing.assert_array_equal(gap.edges[0, 1], [EdgeType.jump, 0, 1, 0])
