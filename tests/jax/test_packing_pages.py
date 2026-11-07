"""Paged execution retains molecular sparse values without retaining full tables."""

import os

import jax
import numpy as np
import pytest

from tests.scientific.execution import Execution
from tests.scientific.rotamer_execution import RotamerExecution
from tests.scientific.test_complete_preset import CASES


@pytest.mark.parametrize(
    "name,sampled",
    [(name, False) for name in ("ligand", "d_peptide", "mixed_batch", "dna")]
    + [("d_peptide", True)],
)
def test_packing_pages_conserve_all_terms(name, sampled):
    backend = Execution(
        "jax-cuda"
        if "jax-cuda" in os.environ.get("TMOL_TEST_BACKENDS", "")
        else "jax-cpu"
    )
    with jax.enable_x64():
        backend.molecule(CASES[name], terms=())
        execution = RotamerExecution(backend, family="genbonded", sampled=sampled)
        backend.use_beta2016()
        scorer = backend.score_function.render_rotamer_scoring_module(
            backend.pose, execution.rotamers, sum_terms=False, apply_weights=False
        )
        coords = jax.device_put(
            np.asarray(execution.rotamers.coords, np.float64), backend.device
        )
        values, indices = map(np.asarray, scorer(coords))
        expected = np.zeros((len(scorer.score_types), backend.pose.n_poses))
        np.add.at(expected.T, indices[0], values.T)

        def accumulated(pages):
            result = {}
            for vals, ids in pages:
                for key, val in zip(map(tuple, np.asarray(ids).T), np.asarray(vals).T):
                    result[key] = result.get(key, 0) + val
            return result

        expected_entries = accumulated(((values, indices),))
        actual_entries = {}
        total = np.zeros_like(expected)
        topology = []
        for values, indices in scorer.iter_packing_entries(
            coords, block_pair_window=1, score_window=2
        ):
            assert indices.shape[1] <= 2
            assert values.device == coords.device and indices.device == coords.device
            values, indices = map(np.asarray, (values, indices))
            np.add.at(total.T, indices[0], values.T)
            topology.extend(map(tuple, indices.T))
            for key, value in accumulated(((values, indices),)).items():
                actual_entries[key] = actual_entries.get(key, 0) + value
        np.testing.assert_allclose(total, expected, rtol=2e-10, atol=2e-10)
        assert actual_entries.keys() == expected_entries.keys()
        for key in expected_entries:
            np.testing.assert_allclose(
                actual_entries[key], expected_entries[key], rtol=2e-10, atol=2e-10
            )
        if not sampled:
            whole = backend.scores(
                backend.coords().astype(np.float64),
                sum_terms=False,
                apply_weights=False,
            )
            np.testing.assert_allclose(total, whole, rtol=3e-6, atol=3e-5)
        else:
            assert np.max(np.asarray(execution.rotamers.n_rots_for_block)) > 2
        from tmol_jax.pack.interaction_graph import build_interaction_graph
        from tmol_jax.pack.streaming import streaming_interaction_graph

        scalar = backend.score_function.render_rotamer_scoring_module(
            backend.pose, execution.rotamers
        )
        values, indices = scalar(coords)
        full_graph = build_interaction_graph(
            execution.rotamers, values, indices, chunk_size=2
        )
        streamed = streaming_interaction_graph(
            scalar, coords, chunk_size=2, block_pair_window=2, score_window=7
        )
        for left, right in zip(jax.tree.leaves(full_graph), jax.tree.leaves(streamed)):
            np.testing.assert_allclose(left, right, rtol=2e-10, atol=2e-10)
        topology_only = []
        for values, indices in scorer.iter_packing_entries(
            coords, block_pair_window=2, score_window=3, topology_only=True
        ):
            assert values.size == 0 and indices.shape[1] <= 3
            topology_only.extend(map(tuple, np.asarray(indices).T))
        assert sorted(topology) == sorted(topology_only)
        for invalid in (0, -1, 1.5, True):
            with pytest.raises(ValueError, match="positive integers"):
                next(
                    scorer.iter_packing_entries(
                        coords, block_pair_window=invalid, score_window=2
                    )
                )
        with pytest.raises(ValueError, match="outside JAX transformations"):
            jax.jit(
                lambda x: next(
                    scorer.iter_packing_entries(x, block_pair_window=1, score_window=2)
                )[0]
            )(coords)
