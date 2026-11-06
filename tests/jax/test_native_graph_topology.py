"""Sparse graph topology retains edge order, partial chunks and empty poses."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tmol_jax.pack.graph import topology_stage, prefix_counts, load_balance


@pytest.mark.parametrize("platform", ("cpu", "cuda"))
@pytest.mark.parametrize("empty", (False, True))
def test_native_graph_topology_partial_chunks_and_empty_rows(platform, empty):
    try:
        device = jax.devices(platform)[0]
    except RuntimeError:
        pytest.skip(f"{platform} unavailable")
    with jax.enable_x64(False), jax.default_device(device):

        def integer(value):
            return jnp.asarray(value, jnp.int32)

        def unpack(value):
            array = np.asarray(value)
            return array.reshape(-1, 8).copy().view(np.int64).reshape(array.shape[:-1])

        def result(shape, dtype=np.uint8):
            return jax.ShapeDtypeStruct(shape, dtype)

        counts = integer([[0, 1, 0], [3, 1, 2]])
        mapping = topology_stage((counts,), operation=0, outputs=result((2, 3, 8)))
        np.testing.assert_array_equal(unpack(mapping), [[-1, -1, -1], [0, -1, 1]])
        molten = integer([[0, 0], [3, 2]])
        adjacency = integer([[[0], [0]], [[0 if empty else 2], [0 if empty else 1]]])

        def count_neighbors(x):
            return topology_stage((x,), operation=1, outputs=result((5, 8)))

        counts = count_neighbors(adjacency)
        expected = [0, 0, 0 if empty else 1, 0 if empty else 1, 0]
        np.testing.assert_array_equal(unpack(counts), expected)
        batched = jax.jit(jax.vmap(count_neighbors))(jnp.stack((adjacency, adjacency)))
        np.testing.assert_array_equal(unpack(batched), [expected, expected])
        rows = prefix_counts(counts[:-1])
        np.testing.assert_array_equal(
            unpack(rows), [0, 0, 0, 0, 0] if empty else [0, 0, 0, 1, 2]
        )
        edges = 0 if empty else 2
        neighbors, pairs = topology_stage(
            (molten, adjacency, rows),
            operation=2,
            chunk_size=2,
            outputs=(result((edges,), np.int32), result((edges, 8))),
        )
        np.testing.assert_array_equal(neighbors, [] if empty else [1, 0])
        np.testing.assert_array_equal(unpack(pairs), [] if empty else [2, 2])
        words = topology_stage((pairs,), operation=3, outputs=result((edges, 8)))
        np.testing.assert_array_equal(unpack(words), [] if empty else [1, 1])
        recounted = topology_stage(
            (molten, rows, neighbors),
            operation=4,
            chunk_size=2,
            outputs=result((edges, 8)),
        )
        np.testing.assert_array_equal(recounted, pairs)
        # First directed edge retains both chunks; the reverse edge retains
        # only the trailing 2-by-1 chunk. Unobserved chunks have no energy rows.
        edge_offsets = prefix_counts(pairs)[:-1]
        bitset_offsets = prefix_counts(words)[:-1]
        bits = integer([] if empty else [3, 2])
        chunks = 0 if empty else 4
        sizes = topology_stage(
            (molten, rows, neighbors, edge_offsets, bitset_offsets, bits),
            operation=5,
            chunk_size=2,
            outputs=result((chunks, 8)),
        )
        np.testing.assert_array_equal(unpack(sizes), [] if empty else [4, 2, 0, 2])
        offsets = prefix_counts(sizes)[:-1]
        marked = topology_stage(
            (sizes, offsets), operation=6, outputs=result((chunks, 8))
        )
        np.testing.assert_array_equal(unpack(marked), [] if empty else [0, 4, -1, 6])
        np.testing.assert_array_equal(unpack(offsets), [] if empty else [0, 4, 6, 6])
        assert mapping.device == marked.device == device


@pytest.mark.parametrize("platform", ("cpu", "cuda"))
def test_native_graph_adjacency_spans_bitset_words(platform):
    try:
        device = jax.devices(platform)[0]
    except RuntimeError:
        pytest.skip(f"{platform} unavailable")
    with jax.enable_x64(False):
        adjacency = np.zeros((1, 35, 2), np.int32)
        adjacency[0, 0, 0] = -(1 << 31)  # neighbor 31, signed high bit
        adjacency[0, 31, 0] = 1  # neighbor 0
        adjacency[0, 0, 1] = 2  # neighbor 33
        adjacency[0, 33, 0] = 1  # neighbor 0
        result = topology_stage(
            (jax.device_put(adjacency, device),),
            operation=1,
            outputs=jax.ShapeDtypeStruct((36, 8), np.uint8),
        )
        expected = np.zeros(36, np.int64)
        expected[[31, 33]] = 1
        expected[0] = 2
        np.testing.assert_array_equal(
            np.asarray(result).view(np.int64).ravel(), expected
        )


@pytest.mark.parametrize("platform", ("cpu", "cuda"))
def test_native_graph_prefix_retains_wide_counts_without_x64(platform):
    try:
        device = jax.devices(platform)[0]
    except RuntimeError:
        pytest.skip(f"{platform} unavailable")
    with jax.enable_x64(False), jax.default_device(device):
        for count in (0, 257, 40000):
            values = np.random.default_rng(42).integers(
                0, 1 << 40, count, dtype=np.int64
            )
            opaque = values.view(np.uint8).reshape(count, 8)
            argument = jax.device_put(opaque, device)
            expected = np.asarray([0, *np.cumsum(values, dtype=np.int64)], np.int64)
            actual = prefix_counts(argument)
            np.testing.assert_array_equal(
                np.asarray(actual).view(np.int64).ravel(), expected
            )
            batched = jax.jit(jax.vmap(prefix_counts))(jnp.stack((argument, argument)))
            np.testing.assert_array_equal(
                np.asarray(batched).copy().view(np.int64).reshape(2, count + 1),
                np.stack((expected, expected)),
            )
            np.testing.assert_array_equal(argument, opaque)
            assert actual.device == device


@pytest.mark.parametrize("platform", ("cpu", "cuda"))
def test_load_balance_preserves_empty_generators_and_partition_boundaries(platform):
    try:
        device = jax.devices(platform)[0]
    except RuntimeError:
        pytest.skip(f"{platform} unavailable")
    with jax.enable_x64(False), jax.default_device(device):
        for counts in (
            np.zeros(0, np.int64),
            np.zeros(17, np.int64),
            np.array([0, 3, 0, 1, 0], np.int64),
            np.random.default_rng(73).integers(0, 19, 40000, dtype=np.int64),
        ):
            total = int(counts.sum())
            offsets = np.cumsum(counts, dtype=np.int64) - counts
            opaque = offsets.view(np.uint8).reshape(-1, 8)
            argument = jax.device_put(opaque, device)
            expected = np.repeat(np.arange(len(counts), dtype=np.int32), counts)

            def expand(x):
                return load_balance(x, total=total)

            actual = expand(argument)
            np.testing.assert_array_equal(actual, expected)
            batched = jax.jit(jax.vmap(expand))(jnp.stack((argument, argument)))
            np.testing.assert_array_equal(batched, np.stack((expected, expected)))
            np.testing.assert_array_equal(argument, opaque)
            assert actual.device == device


def test_load_balance_rejects_unrepresentable_extents_before_allocation():
    offsets = jnp.zeros((2, 8), np.uint8)
    with pytest.raises(OverflowError, match="int32"):
        load_balance(offsets, total=np.iinfo(np.int32).max)
    with pytest.raises(ValueError, match="requires generators"):
        load_balance(offsets[:0], total=1)
    with pytest.raises(TypeError, match="opaque int64"):
        load_balance(jnp.zeros(2, np.int32), total=0)
