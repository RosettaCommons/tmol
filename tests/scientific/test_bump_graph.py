"""Bump pruning preserves energies and original identities through survivor holes."""

import itertools
from types import SimpleNamespace
import contextlib

import numpy as np
import pytest


def fixture():
    counts = np.array([[0, 0, 0, 0], [5, 1, 3, 2], [0, 0, 0, 0]], np.int32)
    offsets = (
        np.r_[0, np.cumsum(counts.ravel())][:-1].reshape(counts.shape).astype(np.int32)
    )
    p, b = np.indices(counts.shape)
    pose_counts = counts.sum(1, dtype=np.int32)
    fields = dict(
        n_rots_for_block=counts,
        rot_offset_for_block=offsets,
        n_rots_for_pose=pose_counts,
        rot_offset_for_pose=np.cumsum(pose_counts, dtype=np.int32) - pose_counts,
        pose_for_rot=np.repeat(p.ravel(), counts.ravel()),
        block_ind_for_rot=np.repeat(b.ravel(), counts.ravel()),
        block_type_ind_for_rot=np.zeros(counts.sum(), np.int32),
    )
    rows = [(1, r, r) for r in range(11)]
    values = [0, 12, 1, 13, 2, 0, 9, 10, 0, 12, 13]
    # One duplicate pair contribution and sparse missing pairs are intentional.
    for left in range(5):
        for right in (9, 10):
            rows.append((1, left, right))
            values.append(0.25 * (right - 8))
    rows += [(1, 2, 9), (1, 2, 9)]
    values += [0.125, 0.125]
    return fields, np.asarray(rows, np.int32).T, np.asarray(values)


def execute_graph(backend, fields, indices, values):
    names = (
        "background",
        "energy1b",
        "neighbor_row_offsets",
        "neighbor_blocks",
        "neighbor_chunk_offset_offsets",
        "chunk_offsets",
        "energy2b",
    )
    if backend.framework == "jax":
        from tmol_jax.pack.interaction_graph import build_interaction_graph

        rots = SimpleNamespace(
            **{
                k: backend.module.device_put(v.astype(np.int32), backend.device)
                for k, v in fields.items()
            },
            device=backend.device,
            correlated_groups=(),
        )
        graph = build_interaction_graph(
            rots,
            backend.module.device_put(values, backend.device),
            backend.module.device_put(indices, backend.device),
            chunk_size=2,
            bump_check=True,
        )

        def wide(a):
            a = backend.numpy(a)
            return a.copy().view(np.int64).reshape(a.shape[:-1])

        result = {k: backend.numpy(graph[k]) for k in names}
        for k in (
            "neighbor_row_offsets",
            "neighbor_chunk_offset_offsets",
            "chunk_offsets",
        ):
            result[k] = wide(graph[k])
        result.update(
            fixed=wide(graph["retained"]["fixed"]),
            original=wide(graph["retained"]["inverse_rotamers"]),
            counts=backend.numpy(graph["retained"]["block_rotamer_counts"]),
            offsets=backend.numpy(graph["retained"]["block_rotamer_offsets"]),
        )
        return result
    from tmol.pack.compiled._compiled import _build_interaction_graph

    torch = backend.module

    def array(v, dtype):
        return torch.as_tensor(v, dtype=dtype, device=backend.device)

    graph = _build_interaction_graph(
        True,
        2,
        1,
        *[
            array(fields[k], torch.int32 if k == "block_ind_for_rot" else torch.int64)
            for k in (
                "n_rots_for_pose",
                "rot_offset_for_pose",
                "n_rots_for_block",
                "rot_offset_for_block",
                "pose_for_rot",
                "block_type_ind_for_rot",
                "block_ind_for_rot",
            )
        ],
        array(indices, torch.int32),
        torch.as_tensor(values, device=backend.device),
        0,
        False,
    )
    result = dict(zip(names, map(backend.numpy, graph[9:])))
    result.update(
        fixed=backend.numpy(graph[7]),
        original=backend.numpy(graph[8]),
        counts=backend.numpy(graph[4]),
        offsets=backend.numpy(graph[5]),
    )
    return result


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_bump_graph_retains_holes_fixed_survivors_and_assignment_energies(
    backend, dtype
):
    fields, indices, values = fixture()
    values = values.astype(dtype)
    with (
        backend.module.enable_x64()
        if backend.framework == "jax"
        else contextlib.nullcontext()
    ):
        graph = execute_graph(backend, fields, indices, values)
    np.testing.assert_array_equal(graph["original"], [0, 2, 4, 9, 10])
    np.testing.assert_array_equal(graph["fixed"], [[-1] * 4, [-1, 5, 8, -1], [-1] * 4])
    np.testing.assert_array_equal(graph["counts"], [[0, 0], [3, 2], [0, 0]])
    rows, neighbors = graph["neighbor_row_offsets"], graph["neighbor_blocks"]
    edge = next(e for e in range(rows[2], rows[3]) if neighbors[e] == 1)
    pair_offset = graph["neighbor_chunk_offset_offsets"][edge]
    for left, right in itertools.product(range(3), range(2)):
        chosen = {int(graph["original"][left]), 5, 8, int(graph["original"][3 + right])}
        expected = sum(
            v for (_, a, b), v in zip(indices.T, values) if a in chosen and b in chosen
        )
        chunk = pair_offset + (left // 2)
        offset = graph["chunk_offsets"][chunk]
        actual = (
            graph["background"][1]
            + graph["energy1b"][left]
            + graph["energy1b"][3 + right]
            + graph["energy2b"][offset + (left % 2) * 2 + right]
        )
        np.testing.assert_equal(actual, expected)
    assert graph["energy1b"].dtype == dtype
