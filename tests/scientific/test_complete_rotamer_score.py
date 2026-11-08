"""Complete sparse scores conserve whole-pose science for input conformers."""

from contextlib import nullcontext

import numpy as np
import pytest

from .test_complete_preset import CASES, CHANNELS
from .rotamer_execution import RotamerExecution


def complete_rotamer_scorer(backend, *, sampled=False):
    execution = RotamerExecution(backend, family="genbonded", sampled=sampled)
    backend.use_beta2016()
    scorer = backend.score_function.render_rotamer_scoring_module(
        backend.pose, execution.rotamers
    )
    if backend.framework == "jax":
        return execution, scorer.prepare(execution.rotamers.coords)

    def evaluate(coords):
        result = scorer(coords).coalesce()
        return result.values(), result.indices()

    return execution, evaluate


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_complete_rotamer_score_conserves_whole_pose(backend, name, dtype):
    precision = (
        backend.module.enable_x64() if backend.framework == "jax" else nullcontext()
    )
    with precision:
        backend.molecule(CASES[name], terms=("ref",))
        e, score = complete_rotamer_scorer(backend)
        xyz = backend.numpy(e.rotamers.coords).astype(dtype)
        x = e.array(xyz)
        if backend.framework == "jax":
            values, indices = backend.module.jit(score)(x)
            dx = backend.module.jit(backend.module.grad(lambda x: score(x)[0].sum()))(x)
        else:
            x = x.detach().requires_grad_()
            values, indices = score(x)
            dx = backend.module.autograd.grad(values.sum(), x)[0]
        values, indices, dx = map(backend.numpy, (values, indices, dx))
        total = np.zeros(backend.pose.n_poses)
        np.add.at(total, indices[0], values)
        whole, whole_gradient = backend.value_and_grad(backend.coords().astype(dtype))
        np.testing.assert_allclose(total, whole, rtol=3e-6, atol=3e-5)
        assert np.isfinite(values).all() and np.any(np.abs(dx) > 1e-3)
        pbt = backend.pose.packed_block_types
        for r, (p, b, t, begin) in enumerate(
            zip(
                *[
                    backend.numpy(getattr(e.rotamers, field))
                    for field in (
                        "pose_for_rot",
                        "block_ind_for_rot",
                        "block_type_ind_for_rot",
                        "coord_offset_for_rot",
                    )
                ]
            )
        ):
            count = len(pbt.active_block_types[t].atoms)
            offset = int(backend.numpy(backend.pose.block_coord_offset)[p, b])
            np.testing.assert_allclose(
                dx[begin : begin + count],
                whole_gradient[p, offset : offset + count],
                rtol=3e-5,
                atol=3e-5,
            )

        if backend.framework == "jax":
            if dtype == np.float32:
                _check_named_sparse_channels(backend, e)
            elif name in ("ligand", "d_peptide", "mixed_batch", "dna"):
                _check_packing_pages(backend, e, sampled=False)
                if name == "d_peptide":
                    sampled = RotamerExecution(
                        backend, family="genbonded", sampled=True
                    )
                    _check_packing_pages(backend, sampled, sampled=True)


def _check_named_sparse_channels(b, e):
    import jax

    scorer = b.score_function.render_rotamer_scoring_module(
        b.pose, e.rotamers, sum_terms=False, apply_weights=False
    )
    assert {st.name for st in scorer.score_types} == CHANNELS
    values, indices = jax.jit(scorer.prepare(e.rotamers.coords))(e.rotamers.coords)
    totals = np.zeros((25, b.pose.n_poses))
    for c in range(25):
        np.add.at(totals[c], np.asarray(indices)[0], np.asarray(values)[c])
    np.testing.assert_allclose(
        totals,
        b.scores(b.coords(), sum_terms=False, apply_weights=False),
        rtol=3e-5,
        atol=3e-4,
    )


def _check_packing_pages(backend, execution, *, sampled):
    import jax

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
