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
