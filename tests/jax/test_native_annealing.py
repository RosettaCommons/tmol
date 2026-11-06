"""Independent graph minima, seed handling and native CPU execution contracts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tmol_jax.pack.annealing import anneal_cpu


def independent_graph(empty=False):
    # Pose zero is empty. Pose one has two residues with two and three states.
    # Every nonzero state adds its index to both the one-body and pair energy,
    # so the unique minimum is (0, 0), with energy 5 + 5. Chunk size two also
    # exercises the trailing partial pair chunks in both directions.
    host = (
        [0, 2],
        [0, 5],
        [0, 0],
        [[0, 0], [2, 3]],
        [[0, 0], [0, 2]],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 2],
        [1, 0],
        [0, 2],
        [0, 4, 6, 10],
        [5, 6, 5, 6, 7],
        [0, 1, 1, 2, 2, 3, 0, 1, 1, 2, 2, 3],
    )
    if empty:
        host = (
            [0, 2],
            [0, 0],
            [0, 0],
            [[0, 0], [0, 0]],
            [[0, 0], [0, 0]],
            [],
            [0, 0, 0, 0, 0],
            [],
            [],
            [],
            [],
            [],
        )
    arrays = []
    for index, value in enumerate(host):
        if index in (6, 8, 9):
            array = np.asarray(value, np.int64).view(np.uint8).reshape(-1, 8)
        else:
            array = np.asarray(value, np.float32 if index >= 10 else np.int32)
        arrays.append(jax.device_put(array, jax.devices("cpu")[0]))
    return tuple(arrays)


@pytest.mark.parametrize("empty", (False, True))
def test_native_cpu_annealer_known_minimum_and_transformations(empty):
    arguments = independent_graph(empty)
    original = tuple(np.asarray(value).copy() for value in arguments)
    for seed in (0, 42, (1 << 64) - 1):
        scores, assignments = anneal_cpu(*arguments, chunk_size=2, seed=seed)
        np.testing.assert_array_equal(scores, [[0], [0 if empty else 10]])
        np.testing.assert_array_equal(assignments[0], 0)
        np.testing.assert_array_equal(assignments[1], -1 if empty else 0)
        assert scores.dtype == np.float32 and assignments.dtype == np.int32
        assert scores.device.platform == assignments.device.platform == "cpu"
        again = anneal_cpu(*arguments, chunk_size=2, seed=seed)
        for expected, actual in zip((scores, assignments), again):
            np.testing.assert_array_equal(expected, actual)
    for before, after in zip(original, arguments):
        np.testing.assert_array_equal(before, after)

    def run(onebody):
        return anneal_cpu(
            *arguments[:-2], onebody, arguments[-1], chunk_size=2, seed=42
        )

    batched_scores, batched_assignments = jax.jit(jax.vmap(run))(
        jnp.stack((arguments[-2], arguments[-2] + 1))
    )
    np.testing.assert_array_equal(batched_scores[0], scores)
    np.testing.assert_array_equal(batched_scores[1], [[0], [0 if empty else 12]])
    np.testing.assert_array_equal(batched_assignments, np.stack((assignments,) * 2))
    with jax.enable_x64(), pytest.raises(TypeError, match="float32"):
        run(arguments[-2].astype(np.float64))
    with pytest.raises(ValueError, match="64-bit"):
        anneal_cpu(*arguments, chunk_size=2, seed=1 << 64)
    with pytest.raises(ValueError, match="chunk size"):
        anneal_cpu(*arguments, chunk_size=0, seed=0)
    with pytest.raises(ValueError, match="differentiated"):
        jax.grad(lambda x: run(x)[0].sum())(arguments[-2])
