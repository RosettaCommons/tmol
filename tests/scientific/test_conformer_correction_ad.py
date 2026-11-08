"""Measured chi correction preserves repeated writes, gradients and curvature."""

from contextlib import nullcontext

import numpy as np
import pytest

from .test_conformer_coordinates import coordinate_dihedral


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("wrap", (False, True))
def test_measured_chi_correction_derivatives(backend, dtype, wrap):
    b = backend
    precision = (
        b.module.enable_x64(dtype == np.float64 or wrap)
        if b.framework == "jax"
        else nullcontext()
    )
    with precision:
        _check_correction(b, dtype, wrap)


def _check_correction(b, dtype, wrap):
    def array(value, integer=False):
        value = np.asarray(value, dtype=np.int32 if integer else dtype)
        return (
            b.module.device_put(value, b.device)
            if b.framework == "jax"
            else b.module.as_tensor(value, device=b.device)
        )

    if wrap:
        xyz = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 0, 1]], dtype)
        chi = np.array([[4.8, 0.8, 0.2]], dtype)
        dofs = np.full((5, 9), -2, dtype)
        four = np.full((1, 4, 4), -1, np.int32)
        four[0, 2] = [0, 1, 2, 3]
        metadata = ([0], [0], [0], [[2, -1, 3]], four, [[0, 1, 2, 3]])
    else:
        xyz = np.array(
            [
                [8, 9, 7],
                [1.2, 0.1, -0.2],
                [0.1, -0.3, 0.2],
                [0.3, 1.1, -0.1],
                [0.8, 1.3, 1.2],
            ],
            dtype,
        )
        chi = np.array([[0.4, 0.9], [0.7, 0.8]], dtype)
        dofs = np.linspace(0.1, 0.9, 63, dtype=dtype).reshape(7, 9)
        four = np.full((1, 4, 4), -1, np.int32)
        four[0, 3] = np.arange(4)
        metadata = ([0], [1], [0, 0], [[3, -1], [3, 2]], four, [[3, 0, 2, 1]])
    metadata = tuple(array(x, True) for x in metadata)
    weights = np.linspace(0.2, 0.8, 63, dtype=dtype).reshape(7, 9)

    # Two entries write the same torsion; absent and undefined torsions do not.
    def reference(d, c, x):
        result = d.astype(np.float64).copy()
        delta = (c[:, 0] - coordinate_dihedral(x[1:]) + np.pi) % (2 * np.pi) - np.pi
        result[3, 3] += delta.sum()
        return result

    def apply(d, c, x):
        if b.framework == "jax":
            from tmol_jax.pack._native import correct_conformer_chi

            return correct_conformer_chi(d, c, x, *metadata)
        from tmol.pack.rotamer._conformer_execution import correct_chi_dofs

        result = d.clone()
        types, offsets, selected, atoms, four, kinematic = metadata
        correct_chi_dofs(
            types.long(),
            offsets.long(),
            selected.long(),
            atoms,
            c,
            x,
            four.long(),
            kinematic.long(),
            result,
        )
        return result

    w = array(weights)
    inputs = (dofs, chi, xyz)
    values = tuple(map(array, inputs))
    if wrap:
        # Positive pi/2 geometry crosses the wrap boundary; skipped and undefined
        # torsions leave every other initialized component untouched.
        expected = dofs.copy()
        delta = float(chi[0, 0]) - float(dtype(np.pi / 2))
        expected[3, 3] += dtype((delta + np.pi) % (2 * np.pi) - np.pi)
        snapshots = tuple(value.copy() for value in inputs)
        actual = apply(*values)
        np.testing.assert_array_equal(b.numpy(actual), expected)
        assert b.numpy(actual).dtype == dtype and actual.device == b.device
        if b.framework == "jax":
            batched = b.module.jit(
                b.module.vmap(apply, in_axes=(0,) + (None,) * (len(values) - 1))
            )(b.module.numpy.stack((values[0], values[0])), *values[1:])
            np.testing.assert_array_equal(
                b.numpy(batched), np.stack((expected, expected))
            )
        for value, original in zip(values, snapshots):
            np.testing.assert_array_equal(b.numpy(value), original)
        return
    direction = tuple(np.sin(x) for x in inputs)
    if b.framework == "jax":
        jax = b.module
        grad = jax.jit(
            jax.grad(lambda *v: (w * apply(*v) ** 2 / 2).sum(), argnums=(0, 1, 2))
        )

        def product(*v):
            return sum((g * array(d)).sum() for g, d in zip(grad(*v), direction))

        hvp = jax.jit(jax.grad(product, argnums=(0, 1, 2)))(*values)
        gradient = grad(*values)
        actual = jax.jit(apply)(*values)
        batched = jax.jit(jax.vmap(apply))(*(jax.numpy.stack((v, v)) for v in values))
        np.testing.assert_array_equal(b.numpy(batched[0]), b.numpy(actual))
    else:
        torch = b.module

        def grad(*v):
            v = tuple(x.detach().requires_grad_() for x in v)
            return torch.autograd.grad((w * apply(*v).square() / 2).sum(), v)

        tracked = tuple(x.detach().requires_grad_() for x in values)
        actual = apply(*tracked)
        gradient = torch.autograd.grad(
            (w * actual.square() / 2).sum(), tracked, create_graph=True
        )
        hvp = torch.autograd.grad(
            sum((g * array(d)).sum() for g, d in zip(gradient, direction)), tracked
        )
    tolerance = 3e-5 if dtype == np.float32 else 2e-8
    np.testing.assert_allclose(
        b.numpy(actual), reference(*inputs), rtol=tolerance, atol=tolerance
    )
    # Full gradients against independently evaluated plane-normal geometry.
    eps = 2e-5
    for arg, g in enumerate(gradient):
        numerical = np.empty(inputs[arg].shape, np.float64)
        for index in np.ndindex(numerical.shape):
            plus, minus = [v.astype(np.float64) for v in inputs], [
                v.astype(np.float64) for v in inputs
            ]
            plus[arg][index] += eps
            minus[arg][index] -= eps
            numerical[index] = (
                (weights * reference(*plus) ** 2).sum()
                - (weights * reference(*minus) ** 2).sum()
            ) / (4 * eps)
        np.testing.assert_allclose(
            b.numpy(g), numerical, rtol=tolerance, atol=tolerance
        )
    eps = 0.002 if dtype == np.float32 else 2e-5
    plus = grad(*(array(x + eps * d) for x, d in zip(inputs, direction)))
    minus = grad(*(array(x - eps * d) for x, d in zip(inputs, direction)))
    for actual_h, gp, gm in zip(hvp, plus, minus):
        np.testing.assert_allclose(
            b.numpy(actual_h),
            (b.numpy(gp) - b.numpy(gm)) / (2 * eps),
            rtol=8e-4 if dtype == np.float32 else 2e-7,
            atol=8e-5 if dtype == np.float32 else 2e-8,
        )
