"""Sparse native candidates preserve geometry, lockstep states and paging."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tmol_jax.pack.dispatch import (
    rotamer_spheres,
    block_spheres,
    candidate_pairs,
    candidate_pair_groups,
)


@pytest.mark.parametrize("platform", ("cpu", "cuda"))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("correlated", (False, True))
def test_native_rotamer_dispatch_geometric_pairs(platform, dtype, correlated):
    try:
        device = jax.devices(platform)[0]
    except RuntimeError:
        pytest.skip(f"{platform} unavailable")
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


@pytest.mark.parametrize("platform", ("cpu", "cuda"))
def test_native_rotamer_dispatch_empty_batches(platform):
    try:
        device = jax.devices(platform)[0]
    except RuntimeError:
        pytest.skip(f"{platform} unavailable")
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
