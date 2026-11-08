"""Bump pruning preserves energies and original identities through survivor holes."""

import itertools
from types import SimpleNamespace
import contextlib

import numpy as np
import pytest


def rotamer_fields(counts):
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
    return fields


def fixture():
    counts = np.array([[0, 0, 0, 0], [5, 1, 3, 2], [0, 0, 0, 0]], np.int32)
    fields = rotamer_fields(counts)
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

        result = {k: backend.numpy(graph[k]) for k in names}
        for k in (
            "neighbor_row_offsets",
            "neighbor_chunk_offset_offsets",
            "chunk_offsets",
        ):
            result[k] = unpack(graph[k])
        result.update(
            fixed=unpack(graph["retained"]["fixed"]),
            original=unpack(graph["retained"]["inverse_rotamers"]),
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


def unpack(value):
    array = np.asarray(value)
    return array.copy().view(np.int64).reshape(array.shape[:-1])


@pytest.fixture
def graph_backend(backend):
    if backend.framework != "jax":
        pytest.skip("Opaque graph-stage ABI is provided by the JAX frontend")
    return backend.module, backend.module.numpy, backend.device


@pytest.mark.parametrize(
    "mode", ("bump", "all", "empty", "fixed", "no_poses", "no_blocks")
)
def test_native_retention_conserves_conformer_identity(graph_backend, mode):
    jax, jnp, device = graph_backend
    from tmol_jax.pack.graph import retained_topology

    counts = np.asarray([[0, 0, 0, 0], [1, 3, 2, 2], [0, 0, 0, 0]], np.int32)
    if mode == "no_poses":
        counts = np.zeros((0, 4), np.int32)
    elif mode == "no_blocks":
        counts = np.zeros((3, 0), np.int32)
    elif mode == "empty":
        counts[:] = 0
    elif mode == "fixed":
        counts[:] = counts > 0
    prefix = np.r_[0, np.cumsum(counts.ravel())].astype(np.int32)
    offsets = prefix[:-1].reshape(counts.shape)
    pose_offsets = prefix[np.arange(counts.shape[0]) * counts.shape[1]]
    origins = [
        (p, b)
        for p, row in enumerate(counts)
        for b, n in enumerate(row)
        for _ in range(n)
    ]
    poses = np.asarray([p for p, _ in origins], np.int32)
    blocks = np.asarray([b for _, b in origins], np.int32)
    energies = np.asarray([0, 1, 8, 2, 6, 8, 5, 6], np.float64)[: len(origins)]
    best = np.zeros(counts.shape, np.float64)
    selected, fixed, retained_blocks = [], {}, []
    for p, row in enumerate(counts):
        for b, n in enumerate(row):
            start = offsets[p, b]
            states = list(range(start, start + n))
            if n and mode == "bump":
                best[p, b] = min(energies[states])
                states = [i for i in states if energies[i] <= 5 or best[p, b] >= 5]
            if len(states) == 1:
                fixed[p, b] = states[0]
            elif len(states) > 1:
                retained_blocks.append((p, b, states))
                selected.extend(states)
    with jax.enable_x64(True):
        args = tuple(
            jax.device_put(x, device)
            for x in (counts, offsets, pose_offsets, poses, blocks)
        )
        kwargs = (
            {}
            if mode != "bump"
            else dict(
                best_energy_for_rotamer=jax.device_put(energies, device),
                best_energy_for_block=jax.device_put(best, device),
            )
        )
        result = retained_topology(*args, **kwargs)
        np.testing.assert_array_equal(unpack(result["inverse_rotamers"]), selected)
        np.testing.assert_array_equal(
            unpack(result["inverse_blocks"]),
            [p * counts.shape[1] + b for p, b, _ in retained_blocks],
        )
        actual_fixed = unpack(result["fixed"])
        assert {
            (p, b): int(actual_fixed[p, b]) for p, b in np.argwhere(actual_fixed >= 0)
        } == fixed
        # Every accepted state occurs once in either the graph or fixed sites.
        assert len(set(selected) | set(fixed.values())) == len(selected) + len(fixed)
        for p in range(len(counts)):
            retained = [(b, states) for pp, b, states in retained_blocks if p == pp]
            assert result["block_pose_counts"][p] == len(retained)
            assert result["rotamer_pose_counts"][p] == sum(len(s) for _, s in retained)
            for local, (b, states) in enumerate(retained):
                offset = int(result["block_rotamer_offsets"][p, local])
                count = int(result["block_rotamer_counts"][p, local])
                assert selected[offset : offset + count] == states
                np.testing.assert_array_equal(
                    result["block_for_rotamer"][offset : offset + count], local
                )
        for value in result.values():
            assert value.device == device
        for actual, original in zip(
            args, (counts, offsets, pose_offsets, poses, blocks)
        ):
            np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_native_retention_threshold_and_transforms(graph_backend, dtype):
    jax, jnp, device = graph_backend
    from tmol_jax.pack.graph import retention_stage

    with jax.enable_x64(True), jax.default_device(device):
        pose = jax.numpy.zeros(2, np.int32)
        block = jax.numpy.zeros(2, np.int32)
        energies = jax.numpy.asarray([[5, 5], [5, 6], [6, 6], [4, 6]], dtype)

        def retain(e):
            return retention_stage(
                (pose, block, e, e.min().reshape(1, 1)),
                operation=0,
                outputs=jax.ShapeDtypeStruct((2, 8), np.uint8),
            )

        result = jax.jit(jax.vmap(retain))(energies)
        np.testing.assert_array_equal(unpack(result), [[1, 1], [1, 1], [1, 1], [1, 0]])
        np.testing.assert_array_equal(energies, [[5, 5], [5, 6], [6, 6], [4, 6]])


@pytest.mark.parametrize("empty", (False, True))
def test_native_graph_topology_partial_chunks_and_empty_rows(graph_backend, empty):
    jax, jnp, device = graph_backend
    from tmol_jax.pack.graph import topology_stage, prefix_counts

    with jax.enable_x64(False), jax.default_device(device):

        def integer(value):
            return jnp.asarray(value, jnp.int32)

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


def test_native_graph_adjacency_spans_bitset_words(graph_backend):
    jax, jnp, device = graph_backend
    from tmol_jax.pack.graph import topology_stage

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


@pytest.mark.parametrize("maximum", (False, True))
def test_native_graph_prefix_retains_wide_counts_without_x64(graph_backend, maximum):
    jax, jnp, device = graph_backend
    from tmol_jax.pack.graph import prefix_counts

    with jax.enable_x64(False), jax.default_device(device):
        for count in (0, 257, 40000):
            values = np.random.default_rng(42).integers(
                0, 1 << 40, count, dtype=np.int64
            )
            opaque = values.view(np.uint8).reshape(count, 8)
            argument = jax.device_put(opaque, device)
            reduction = (
                np.maximum.accumulate(values)
                if maximum
                else np.cumsum(values, dtype=np.int64)
            )
            expected = np.asarray([0, *reduction], np.int64)

            def prefix(x):
                return prefix_counts(x, maximum=maximum)

            actual = prefix(argument)
            np.testing.assert_array_equal(
                np.asarray(actual).view(np.int64).ravel(), expected
            )
            batched = jax.jit(jax.vmap(prefix))(jnp.stack((argument, argument)))
            np.testing.assert_array_equal(
                np.asarray(batched).copy().view(np.int64).reshape(2, count + 1),
                np.stack((expected, expected)),
            )
            np.testing.assert_array_equal(argument, opaque)
            assert actual.device == device


def test_load_balance_preserves_empty_generators_and_partition_boundaries(
    graph_backend,
):
    jax, jnp, device = graph_backend
    from tmol_jax.pack.graph import load_balance

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


def test_load_balance_rejects_unrepresentable_extents_before_allocation(graph_backend):
    jax, jnp, device = graph_backend
    from tmol_jax.pack.graph import load_balance

    offsets = jnp.zeros((2, 8), np.uint8)
    with pytest.raises(OverflowError, match="int32"):
        load_balance(offsets, total=np.iinfo(np.int32).max)
    with pytest.raises(ValueError, match="requires generators"):
        load_balance(offsets[:0], total=1)
    with pytest.raises(TypeError, match="opaque int64"):
        load_balance(jnp.zeros(2, np.int32), total=0)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("paged", (False, True))
def test_native_graph_energy_for_every_assignment(graph_backend, dtype, paged):
    jax, jnp, device = graph_backend
    from tmol_jax.pack.interaction_graph import build_interaction_graph

    with jax.enable_x64(dtype == np.float64), jax.default_device(device):
        counts = np.array([[2, 1, 3], [1, 2, 0]], np.int32)
        rotamers = synthetic_rotamers(counts, jax, device)
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
        if paged:
            from tmol_jax.pack.interaction_graph import additive_entry_stage
            from tmol_jax.pack.graph import topology_stage, _wide

            retained = graph["retained"]
            mapping = topology_stage(
                (rotamers.n_rots_for_block,),
                operation=0,
                outputs=_wide(*counts.shape),
                chunk_size=2,
            )
            initial = tuple(
                jnp.zeros_like(graph[k]) for k in ("background", "energy1b", "energy2b")
            )
            energies = initial
            for begin in range(0, len(values), 3):
                energies = additive_entry_stage(
                    (
                        jnp.asarray(indices[:, begin : begin + 3]),
                        jnp.asarray(values[begin : begin + 3]),
                        rotamers.block_ind_for_rot,
                        mapping,
                        rotamers.rot_offset_for_block,
                        retained["fixed"],
                        retained["block_rotamer_counts"],
                        retained["block_rotamer_offsets"],
                        graph["neighbor_row_offsets"],
                        graph["neighbor_blocks"],
                        graph["neighbor_chunk_offset_offsets"],
                        graph["chunk_offsets"],
                    ),
                    energies,
                    operation=2,
                    extent=graph["energy1b"].size,
                    chunk_size=2,
                )
            for key, value, original in zip(
                ("background", "energy1b", "energy2b"), energies, initial
            ):
                np.testing.assert_array_equal(value, graph[key])
                np.testing.assert_array_equal(original, np.zeros(original.shape))
                graph[key] = value
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


@pytest.mark.parametrize("counts", ([[1, 1], [0, 0]], [[], []], []))
@pytest.mark.parametrize("bump_check", (False, True))
def test_native_graph_empty_entries_and_fixed_blocks(graph_backend, counts, bump_check):
    jax, jnp, device = graph_backend
    from tmol_jax.pack.interaction_graph import build_interaction_graph

    with jax.default_device(device):
        counts = np.asarray(counts, np.int32)
        if counts.ndim == 1:
            counts = counts.reshape(0, 0)
        rotamers = synthetic_rotamers(counts, jax, device)
        graph = build_interaction_graph(
            rotamers, jnp.zeros(0), jnp.zeros((3, 0), jnp.int32), bump_check=bump_check
        )
        assert graph["energy1b"].shape == graph["energy2b"].shape == (0,)
        np.testing.assert_array_equal(graph["background"], np.zeros(counts.shape[0]))


def test_graph_rejects_energy_between_alternatives_of_one_block(graph_backend):
    jax, jnp, device = graph_backend
    from tmol_jax.pack.interaction_graph import build_interaction_graph

    with jax.default_device(device):
        rotamers = synthetic_rotamers([[2, 1]], jax, device)
        with pytest.raises(ValueError, match="distinct alternatives of one block"):
            build_interaction_graph(
                rotamers, jnp.array([-100.0]), jnp.array([[0], [0], [1]], jnp.int32)
            )


def test_bump_bounds_accumulate_duplicate_entries_without_lost_updates(graph_backend):
    jax, jnp, device = graph_backend
    from tmol_jax.pack.bump import bump_bounds

    with jax.default_device(device):
        rotamers = synthetic_rotamers([[2]], jax, device)
        # More entries than a CPU dispatch grain, all targeting one location.
        indices = jnp.zeros((3, 40960), jnp.int32)
        values = jnp.full(40960, 1 / 4096, jnp.float32)
        for _ in range(3):
            bounds, best = bump_bounds(rotamers, values, indices)
            np.testing.assert_array_equal(bounds, [10, 0])
            np.testing.assert_array_equal(best, [[0]])


def synthetic_rotamers(counts, jax, device):
    counts = np.asarray(counts, np.int32)
    return SimpleNamespace(
        **{
            k: jax.device_put(v.astype(np.int32), device)
            for k, v in rotamer_fields(counts).items()
        },
        device=device,
        n_rotamers_total=int(counts.sum()),
        correlated_groups=(),
    )
