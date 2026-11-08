"""Joint conformers retain exactly the energies of their legal molecular states."""

from types import SimpleNamespace

import numpy as np
import pytest

from tmol_core.pack.correlation import CorrelatedBlockGroup, correlation_indices


def joint_fixture(backend, include_incompatible=True):
    counts = np.array([[2, 2, 2, 0], [0, 0, 0, 0]], np.int32)
    fields = dict(
        n_rots_for_block=counts,
        rot_offset_for_block=np.array([[0, 2, 4, 6], [6, 6, 6, 6]], np.int32),
        n_rots_for_pose=np.array([6, 0], np.int32),
        rot_offset_for_pose=np.array([0, 6], np.int32),
        block_ind_for_rot=np.array([0, 0, 1, 1, 2, 2], np.int32),
        pose_for_rot=np.zeros(6, np.int32),
        block_type_ind_for_rot=np.array([10, 11, 20, 21, 30, 31], np.int32),
        coords=np.zeros((6, 3), np.float32),
    )

    def tensor(value):
        if backend.framework == "jax":
            return backend.module.device_put(value, backend.device)
        return backend.module.as_tensor(value, device=backend.device)

    rots = SimpleNamespace(
        **{name: tensor(value) for name, value in fields.items()},
        correlated_groups=(CorrelatedBlockGroup(0, (0, 2)),),
        device=backend.device,
    )
    rots.group_for_block = tensor(
        correlation_indices(counts.shape, rots.correlated_groups, counts)
    )
    rows, values = [], []
    for state in range(2):
        for rot, value in ((state, 1 + state), (4 + state, 3 + 2 * state)):
            rows.append((0, rot, rot))
            values.append(value)
        for value in (5 + state, 7 + state):
            rows.append((0, state, 4 + state))
            values.append(value)
        if include_incompatible:
            rows.append((0, state, 5 - state))
            values.append(-100000)  # Impossible joint states must never win.
        for external in range(2):
            rows.extend(((0, state, 2 + external), (0, 2 + external, 4 + state)))
            values.extend((11 + state + external, 13 + 2 * state + external))
    return (
        rots,
        tensor(np.asarray(rows, np.int32).T),
        tensor(np.asarray(values, np.float32)),
    )


def graph_arrays(backend, rots, indices, values):
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

        graph = build_interaction_graph(rots, values, indices, chunk_size=2)
        arrays = {name: backend.numpy(graph[name]) for name in names}
        for name in (
            "neighbor_row_offsets",
            "neighbor_chunk_offset_offsets",
            "chunk_offsets",
        ):
            x = arrays[name]
            arrays[name] = x.copy().view(np.int64).reshape(x.shape[:-1])
        return arrays, graph["retained"]["group_collapse"]
    from tmol.pack.rotamer._conjugated_groups import (
        _collapse_group_rotamers,
        _collapse_group_energies,
    )
    from tmol.pack.compiled._compiled import _build_interaction_graph

    collapse, group, conformer = _collapse_group_rotamers(
        None, rots, rots.correlated_groups
    )
    import attr

    energies = backend.module.sparse_coo_tensor(indices.long(), values, (2, 6, 6))
    energies = _collapse_group_energies(energies, collapse, group, conformer)
    graph = _build_interaction_graph(
        False,
        2,
        32,
        *(
            getattr(collapse, "compact_" + name).to(
                backend.module.int32
                if name == "block_ind_for_rot"
                else backend.module.int64
            )
            for name in (
                "n_rots_for_pose",
                "rot_offset_for_pose",
                "n_rots_for_block",
                "rot_offset_for_block",
                "pose_for_rot",
                "block_type_ind_for_rot",
                "block_ind_for_rot",
            )
        ),
        energies.indices().to(backend.module.int32),
        energies.values(),
        -1,
        False,
    )
    host = attr.evolve(
        collapse,
        **{
            field.name: backend.numpy(getattr(collapse, field.name))
            for field in attr.fields(type(collapse))
            if field.name != "members"
        },
    )
    return dict(zip(names, map(backend.numpy, graph[9:]))), host


@pytest.mark.parametrize("include_incompatible", (True, False))
def test_joint_graph_excludes_incompatible_states_and_preserves_all_contributions(
    backend,
    include_incompatible,
):
    from tmol_core.pack.correlation import expand_group_assignments

    rots, indices, values = joint_fixture(backend, include_incompatible)
    graph, collapse = graph_arrays(backend, rots, indices, values)
    np.testing.assert_array_equal(collapse.compact_n_rots_for_pose, [5, 0])
    np.testing.assert_array_equal(
        collapse.compact_rot_offset_for_block, [[0, 2, 4, 5], [5] * 4]
    )
    np.testing.assert_array_equal(graph["background"], [0, 0])
    edge = next(
        e
        for e in range(
            graph["neighbor_row_offsets"][0], graph["neighbor_row_offsets"][1]
        )
        if graph["neighbor_blocks"][e] == 1
    )
    start = graph["chunk_offsets"][graph["neighbor_chunk_offset_offsets"][edge]]
    for state in range(2):
        for external in range(2):
            energy = graph["energy1b"][state] + graph["energy1b"][2 + external]
            energy += graph["energy2b"][start + 2 * state + external]
            assert energy == 40 + 8 * state + 2 * external
            original = np.array([[state, 2 + external, 4, -1], [-1] * 4])
            expanded = expand_group_assignments(original, collapse)
            np.testing.assert_array_equal(
                expanded, [[state, 2 + external, 4 + state, -1], [-1] * 4]
            )
            np.testing.assert_array_equal(
                backend.numpy(rots.block_type_ind_for_rot)[expanded[0, :3]],
                [10 + state, 20 + external, 30 + state],
            )

    if backend.framework != "jax" or not include_incompatible:
        return
    for dtype in (np.float32, np.float64):
        for correlated in (False, True):
            _assert_dispatch_geometry(backend, dtype, correlated)

    import jax.numpy as jnp
    from tmol_jax.pack.dispatch import rotamer_spheres, block_spheres, candidate_pairs

    jax, device = backend.module, backend.device
    with jax.enable_x64(False), jax.default_device(device):
        for shape in ((0, 3), (3, 0), (2, 3)):
            counts = jnp.zeros(shape, jnp.int32)
            spheres = rotamer_spheres(
                jnp.zeros((0, 3), jnp.float32),
                jnp.zeros(0, jnp.int32),
                jnp.zeros(0, jnp.int32),
                jnp.asarray([2], jnp.int32),
            )
            sites = block_spheres(spheres, counts, counts)
            pairs = candidate_pairs(
                jnp.full(shape, -1, jnp.int32),
                spheres,
                sites,
                counts,
                counts,
                jnp.full(shape, -1, jnp.int32),
                reach=6.0,
            )
            assert pairs.shape == (3, 0)
            assert pairs.device == device


def _assert_dispatch_geometry(backend, dtype, correlated):
    import jax.numpy as jnp
    from tmol_jax.pack.dispatch import (
        rotamer_spheres,
        block_spheres,
        candidate_pairs,
        candidate_pair_groups,
    )

    jax, device = backend.module, backend.device
    centers = np.zeros((5, 3), dtype)
    centers[:, 0] = [0, 2, 1, 3, 20]
    atoms = (
        centers[:, None, :] + np.asarray([[-0.25, 0, 0], [0.25, 0, 0]], dtype)
    ).reshape(-1, 3)
    counts = np.asarray([[0, 0, 0], [2, 2, 1], [0, 0, 0]], np.int32)
    offsets = (
        np.r_[0, np.cumsum(counts.ravel())][:-1].astype(np.int32).reshape(counts.shape)
    )
    types = np.where(counts > 0, 0, -1).astype(np.int32)
    groups = np.full(counts.shape, -1, np.int32)
    if correlated:
        groups[1, :2] = 7
    tolerance = 2e-6 if dtype == np.float32 else 5e-14
    with jax.enable_x64(True), jax.default_device(device):
        xyz = jax.device_put(atoms, device)
        atom_offsets = jnp.arange(5, dtype=jnp.int32) * 2
        rot_types = jnp.zeros(5, jnp.int32)
        n_atoms = jnp.asarray([2], jnp.int32)
        spheres = rotamer_spheres(xyz, atom_offsets, rot_types, n_atoms)
        np.testing.assert_allclose(
            np.asarray(spheres)[:, :3], centers, rtol=0, atol=tolerance
        )
        np.testing.assert_allclose(
            np.asarray(spheres)[:, 3], 0.25, rtol=0, atol=tolerance
        )
        counts_d, offsets_d = jnp.asarray(counts), jnp.asarray(offsets)
        sites = block_spheres(spheres, counts_d, offsets_d)
        for block in range(3):
            start, n = offsets[1, block], counts[1, block]
            expected_center = centers[start : start + n].mean(0)
            expected_radius = max(
                np.linalg.norm(c - expected_center) + 0.25
                for c in centers[start : start + n]
            )
            np.testing.assert_allclose(
                sites[1, block],
                [*expected_center, expected_radius],
                rtol=0,
                atol=tolerance,
            )

        def pairs(**window):
            return candidate_pairs(
                jnp.asarray(types),
                spheres,
                sites,
                counts_d,
                offsets_d,
                jnp.asarray(groups),
                reach=1.2,
                **window,
            )

        result = pairs()
        expected = {(1, r, r) for r in range(5)}
        for a in range(2):
            for b in range(2, 4):
                if (
                    a == b - 2
                    if correlated
                    else np.linalg.norm(centers[a] - centers[b]) < 1.7
                ):
                    expected.add((1, a, b))
        actual = [tuple(row) for row in np.asarray(result).T]
        assert set(actual) == expected and len(actual) == len(expected)
        grouped = candidate_pair_groups(
            jnp.asarray(types),
            spheres,
            sites,
            counts_d,
            offsets_d,
            jnp.asarray(groups),
            reaches=(1.2, 0.0),
        )
        np.testing.assert_array_equal(grouped[0], result)
        assert set(map(tuple, np.asarray(grouped[1]).T)) == (
            expected if correlated else {(1, r, r) for r in range(5)}
        )
        paged = [
            pairs(begin=start, count=min(5, 18 - start)) for start in range(0, 18, 5)
        ]
        np.testing.assert_array_equal(jnp.concatenate(paged, axis=1), result)
        np.testing.assert_array_equal(pairs(begin=18, count=0).shape, (3, 0))
        translated = jnp.stack((xyz, xyz + jnp.asarray([2, -3, 4], dtype)))
        values = jax.jit(
            jax.vmap(lambda x: rotamer_spheres(x, atom_offsets, rot_types, n_atoms))
        )(translated)
        np.testing.assert_allclose(
            np.asarray(values)[1, :, :3], centers + [2, -3, 4], rtol=0, atol=tolerance
        )
        np.testing.assert_allclose(
            np.asarray(values)[1, :, 3], 0.25, rtol=0, atol=tolerance
        )
        np.testing.assert_array_equal(xyz, atoms)
        assert result.device == device
        with pytest.raises(ValueError, match="window"):
            pairs(begin=19)
        with pytest.raises(TypeError, match="outside jit"):
            jax.jit(
                lambda x: candidate_pairs(
                    jnp.asarray(types),
                    x,
                    sites,
                    counts_d,
                    offsets_d,
                    jnp.asarray(groups),
                    reach=1.2,
                )
            )(spheres)
