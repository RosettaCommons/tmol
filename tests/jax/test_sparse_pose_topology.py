"""Sparse native topology preserves dense compatibility and pose transformations."""

import jax
import numpy as np
import pytest

from tests.scientific.execution import Execution
from tests.scientific.molecules import DOMAIN_CASES


@pytest.mark.parametrize(
    "name", ("connected_noncanonical", "mixed_batch", "metal_cluster")
)
def test_sparse_and_explicit_dense_topology_have_identical_energy_gradients(name):
    backend = Execution("jax-cpu").molecule(
        DOMAIN_CASES[name], terms=("fa_ljatr", "fa_ljrep", "fa_lk", "fa_elec")
    )
    pose = backend.pose
    assert pose._inter_block_bondsep is None
    assert not hasattr(pose, "_dense_connections")
    scorer = backend.score_function.render_whole_pose_scoring_module(pose)
    evaluate = jax.jit(jax.value_and_grad(lambda x: scorer(x).sum()))
    value, gradient = evaluate(pose.coords)
    assert value.device == gradient.device == pose.device
    assert not hasattr(pose, "_dense_connections")
    dense = pose.inter_block_bondsep
    assert pose.inter_block_bondsep is dense
    explicit = pose.replace(inter_block_bondsep=dense)
    other = backend.score_function.render_whole_pose_scoring_module(explicit)
    expected, derivatives = jax.jit(jax.value_and_grad(lambda x: other(x).sum()))(
        pose.coords
    )
    assert expected.device == derivatives.device == pose.device
    np.testing.assert_array_equal(value, expected)
    np.testing.assert_array_equal(gradient, derivatives)
    # Dense replacements override any retained sparse graph, including under jit.
    changed = explicit.replace(inter_block_bondsep=dense + 1)
    np.testing.assert_array_equal(changed.inter_block_bondsep, dense + 1)
    np.testing.assert_array_equal(
        jax.jit(lambda p: p.inter_block_bondsep)(changed), dense + 1
    )


def test_sparse_pose_slices_and_pytree_keep_graph_ownership():
    pose = (
        Execution("jax-cpu").molecule(DOMAIN_CASES["mixed_batch"], terms=("ref",)).pose
    )
    expected = np.asarray(pose.inter_block_bondsep)
    selected = pose[[1, 0, 1]]
    assert selected._inter_block_bondsep is None
    assert not hasattr(selected, "_dense_connections")
    np.testing.assert_array_equal(selected.inter_block_bondsep, expected[[1, 0, 1]])
    moved = selected.to(jax.devices("cpu")[0])
    assert not hasattr(moved, "_dense_connections")
    result = jax.jit(lambda p: p.inter_block_bondsep)(moved)
    np.testing.assert_array_equal(result, expected[[1, 0, 1]])
    # Tracing must not cache a tracer on the source pose.
    assert not hasattr(moved, "_dense_connections")
    np.testing.assert_array_equal(moved.inter_block_bondsep, result)
    empty = pose[:0]
    assert empty.inter_block_bondsep.shape[0] == 0
