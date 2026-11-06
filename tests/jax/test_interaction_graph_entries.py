"""Native graph energies equal independently enumerated sparse assignments."""

import itertools
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.scientific.execution import Execution
from tmol_jax.pack.interaction_graph import build_interaction_graph


def synthetic_rotamers(counts, device):
    counts = np.asarray(counts, np.int32)
    offsets = (
        (np.cumsum(counts) - counts.ravel()).reshape(counts.shape).astype(np.int32)
    )
    pose_counts = counts.sum(axis=1, dtype=np.int32)
    poses, blocks = np.indices(counts.shape)
    arrays = dict(
        n_rots_for_block=counts,
        rot_offset_for_block=offsets,
        rot_offset_for_pose=np.cumsum(pose_counts, dtype=np.int32) - pose_counts,
        pose_for_rot=np.repeat(poses.ravel(), counts.ravel()).astype(np.int32),
        block_ind_for_rot=np.repeat(blocks.ravel(), counts.ravel()).astype(np.int32),
    )
    return SimpleNamespace(
        **{k: jax.device_put(v, device) for k, v in arrays.items()},
        device=device,
        n_rotamers_total=int(counts.sum()),
        correlated_groups=(),
    )


def unpack(value):
    value = np.asarray(value)
    return value.copy().view(np.int64).reshape(value.shape[:-1])


@pytest.mark.parametrize("platform", ("cpu", "cuda"))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_native_graph_energy_for_every_assignment(platform, dtype):
    device = Execution("jax-" + platform).device
    with jax.enable_x64(dtype == np.float64), jax.default_device(device):
        counts = np.array([[2, 1, 3], [1, 2, 0]], np.int32)
        rotamers = synthetic_rotamers(counts, device)
        offsets = np.asarray(rotamers.rot_offset_for_block)
        entries = []
        for p, row in enumerate(counts):
            for b1, n1 in enumerate(row):
                for b2 in range(b1, len(row)):
                    for r1 in range(n1):
                        for r2 in range(row[b2]):
                            if b1 != b2 or r1 == r2:
                                entries.append(
                                    (p, offsets[p, b1] + r1, offsets[p, b2] + r2)
                                )
        entries += entries[::3]  # Duplicated terms must accumulate exactly once each.
        indices = np.asarray(entries, np.int32).T
        values = (np.arange(len(entries)) % 11 - 4).astype(dtype) / 8
        graph = build_interaction_graph(
            rotamers, jnp.asarray(values), jnp.asarray(indices), chunk_size=2
        )
        retained = graph["retained"]
        inv_blocks = unpack(retained["inverse_blocks"])
        block_pose_offsets = np.asarray(retained["block_pose_offsets"])
        block_counts = np.asarray(retained["block_pose_counts"])
        oneb_offsets = np.asarray(retained["block_rotamer_offsets"])
        molten_counts = np.asarray(retained["block_rotamer_counts"])
        rows = unpack(graph["neighbor_row_offsets"])
        neighbors = np.asarray(graph["neighbor_blocks"])
        edge_offsets = unpack(graph["neighbor_chunk_offset_offsets"])
        chunk_offsets = unpack(graph["chunk_offsets"])
        background = np.asarray(graph["background"])
        one_body, two_body = map(np.asarray, (graph["energy1b"], graph["energy2b"]))
        for p, row in enumerate(counts):
            real = np.flatnonzero(row)
            for selected in itertools.product(*(range(row[b]) for b in real)):
                assignment = {b: offsets[p, b] + r for b, r in zip(real, selected)}
                expected = sum(
                    v
                    for (q, r1, r2), v in zip(entries, values)
                    if q == p
                    and r1 in assignment.values()
                    and r2 in assignment.values()
                )
                actual = background[p]
                for m in range(block_counts[p]):
                    original_block = (
                        inv_blocks[block_pose_offsets[p] + m] % counts.shape[1]
                    )
                    r = assignment[original_block] - offsets[p, original_block]
                    actual += one_body[oneb_offsets[p, m] + r]
                    csr = p * molten_counts.shape[1] + m
                    for edge in range(rows[csr], rows[csr + 1]):
                        other = neighbors[edge]
                        if other <= m:
                            continue
                        b2 = inv_blocks[block_pose_offsets[p] + other] % counts.shape[1]
                        r2 = assignment[b2] - offsets[p, b2]
                        nchunks2 = (molten_counts[p, other] + 1) // 2
                        chunk = edge_offsets[edge] + (r // 2) * nchunks2 + r2 // 2
                        start = chunk_offsets[chunk]
                        if start >= 0:
                            width = min(2, molten_counts[p, other] - (r2 // 2) * 2)
                            actual += two_body[start + (r % 2) * width + r2 % 2]
                np.testing.assert_equal(actual, expected)
        assert graph["energy1b"].dtype == dtype
        assert all(
            a.device == device
            for a in jax.tree.leaves(graph)
            if isinstance(a, jax.Array)
        )


@pytest.mark.parametrize("platform", ("cpu", "cuda"))
@pytest.mark.parametrize("counts", ([[1, 1], [0, 0]], [[], []], []))
def test_native_graph_empty_entries_and_fixed_blocks(platform, counts):
    device = Execution("jax-" + platform).device
    with jax.default_device(device):
        counts = np.asarray(counts, np.int32)
        if counts.ndim == 1:
            counts = counts.reshape(0, 0)
        rotamers = synthetic_rotamers(counts, device)
        graph = build_interaction_graph(
            rotamers, jnp.zeros(0), jnp.zeros((3, 0), jnp.int32)
        )
        assert graph["energy1b"].shape == graph["energy2b"].shape == (0,)
        np.testing.assert_array_equal(graph["background"], np.zeros(counts.shape[0]))


@pytest.mark.parametrize("platform", ("cpu", "cuda"))
def test_graph_rejects_energy_between_alternatives_of_one_block(platform):
    device = Execution("jax-" + platform).device
    with jax.default_device(device):
        rotamers = synthetic_rotamers([[2, 1]], device)
        with pytest.raises(ValueError, match="distinct alternatives of one block"):
            build_interaction_graph(
                rotamers, jnp.array([-100.0]), jnp.array([[0], [0], [1]], jnp.int32)
            )
