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
