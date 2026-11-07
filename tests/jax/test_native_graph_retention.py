"""Retained graph coordinates conserve selected identities across empty poses."""

import jax
import numpy as np
import pytest

from tmol_jax.pack.graph import retained_topology, retention_stage


def unpack(value):
    data = np.asarray(value)
    return data.copy().view(np.int64).reshape(data.shape[:-1])


@pytest.mark.parametrize("platform", ("cpu", "cuda"))
@pytest.mark.parametrize(
    "mode", ("bump", "all", "empty", "fixed", "no_poses", "no_blocks")
)
def test_native_retention_conserves_conformer_identity(platform, mode):
    try:
        device = jax.devices(platform)[0]
    except RuntimeError:
        pytest.skip(f"{platform} unavailable")
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


@pytest.mark.parametrize("platform", ("cpu", "cuda"))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_native_retention_threshold_and_transforms(platform, dtype):
    try:
        device = jax.devices(platform)[0]
    except RuntimeError:
        pytest.skip(f"{platform} unavailable")
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
