"""Native conformer writes preserve repeated-source and untouched-input derivatives."""

from contextlib import nullcontext
import numpy as np
import pytest


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("operation", ("rows", "chi", "reset"))
def test_native_conformer_linear_value_gradient_and_curvature(
    backend, dtype, operation
):
    b = backend
    precision = (
        b.module.enable_x64(dtype == np.float64)
        if b.framework == "jax"
        else nullcontext()
    )
    with precision:

        def array(x, integer=False):
            x = np.asarray(x, dtype=np.int32 if integer else dtype)
            return (
                b.module.device_put(x, b.device)
                if b.framework == "jax"
                else b.module.as_tensor(x, device=b.device)
            )

        target = np.arange(63, dtype=dtype).reshape(7, 9) / 8
        if operation == "rows":
            source = np.arange(45, dtype=dtype).reshape(5, 9) / 4
            so, to, prefix = map(
                lambda x: array(x, True), ([1, 0, 1], [0, 2, 4], [0, 2, 2, 4])
            )
            inputs = (array(target), array(source))
            wanted = target.copy()
            wanted[[0, 1, 4, 5]] = source[[1, 2, 1, 2]]
            expected = [2 * target, np.zeros_like(source)]
            expected[0][[0, 1, 4, 5]] = 0
            expected[1][[1, 2]] = 4 * source[[1, 2]]
            curvature = [np.full_like(target, 0.5), np.zeros_like(source)]
            curvature[0][[0, 1, 4, 5]] = 0
            curvature[1][[1, 2]] = 1

            def apply(t, s):
                if b.framework == "jax":
                    from tmol_jax.pack._native import copy_conformer_rows

                    return copy_conformer_rows(t, s, so, to, prefix, n_rows=4)
                from tmol.pack.rotamer._conformer_execution import copy_conformer_rows

                result = t.clone()
                copy_conformer_rows(
                    prefix[1:] - prefix[:-1], so.long(), to.long(), s, result
                )
                return result

        elif operation == "chi":
            types, offsets, selected, atoms, kinematic = map(
                lambda x: array(x, True),
                ([0, 1], [0, 3], [0, 1], [[1, -1], [0, 2]], [[0, 1, 2], [0, 1, 2]]),
            )
            chi = np.array([[1, 99], [2, 3]], dtype)
            corrections = np.arange(6, dtype=dtype).reshape(2, 3) / 8
            inputs = (array(target), array(chi), array(corrections))
            wanted = target.copy()
            wanted[[2, 4, 6], 3] = (
                chi[[0, 1, 1], [0, 0, 1]] - corrections[[0, 1, 1], [1, 0, 2]]
            )
            expected = [2 * target, np.zeros_like(chi), np.zeros_like(corrections)]
            expected[0][[2, 4, 6], 3] = 0
            expected[1][[0, 1, 1], [0, 0, 1]] = 2 * wanted[[2, 4, 6], 3]
            expected[2][[0, 1, 1], [1, 0, 2]] = -2 * wanted[[2, 4, 6], 3]
            curvature = [
                np.full_like(target, 0.5),
                np.zeros_like(chi),
                np.zeros_like(corrections),
            ]
            curvature[0][[2, 4, 6], 3] = 0

            # Equal tangent in chi and its correction cancels exactly.
            def apply(t, c, r):
                if b.framework == "jax":
                    from tmol_jax.pack._native import assign_conformer_chi

                    return assign_conformer_chi(
                        t, c, types, offsets, selected, atoms, kinematic, r
                    )
                from tmol.pack.rotamer._conformer_execution import assign_chi_dofs

                result = t.clone()
                assign_chi_dofs(
                    types.long(),
                    offsets.long(),
                    selected.long(),
                    atoms,
                    c,
                    kinematic.long(),
                    r,
                    result,
                )
                return result

        else:
            from types import SimpleNamespace

            types, offsets, selected, atoms, downstream, counts = map(
                lambda x: array(x, True),
                ([1, 1], [0, 3], [0, 1], [[-1], [2]], [[0, 1], [1, 2]], [0, 2]),
            )
            ideal = np.arange(54, dtype=dtype).reshape(2, 3, 9) / 4
            inputs = (array(target), array(ideal))
            wanted = target.copy()
            wanted[[5, 6]] = ideal[1, [1, 2]]
            expected = [2 * target, np.zeros_like(ideal)]
            expected[0][[5, 6]] = 0
            expected[1][1, [1, 2]] = 2 * ideal[1, [1, 2]]
            curvature = [np.full_like(target, 0.5), np.zeros_like(ideal)]
            curvature[0][[5, 6]] = 0
            curvature[1][1, [1, 2]] = 0.5

            def apply(t, source):
                if b.framework == "jax":
                    from tmol_jax.pack._native import reset_opth_dofs

                    return reset_opth_dofs(
                        t,
                        source,
                        types,
                        offsets,
                        selected,
                        atoms,
                        downstream,
                        counts,
                        flip=True,
                    )
                from tmol.pack.rotamer._conformer_execution import reset_opth_dofs

                result = t.clone()
                cache = SimpleNamespace(
                    nhq_downstream_kfo=downstream, nhq_downstream_count=counts
                )
                reset_opth_dofs(
                    types.long(),
                    offsets.long(),
                    selected.long(),
                    atoms,
                    cache,
                    source,
                    result,
                    True,
                )
                return result

        if b.framework == "jax":
            jax = b.module

            def loss(*x):
                return (apply(*x) ** 2).sum()

            grad = jax.grad(loss, argnums=tuple(range(len(inputs))))
            value = jax.jit(apply)(*inputs)
            g = jax.jit(grad)(*inputs)
            _, hvp = jax.jit(
                lambda *x: jax.jvp(
                    grad, x, tuple(jax.numpy.full_like(v, 0.25) for v in x)
                )
            )(*inputs)
        else:
            values = tuple(x.detach().requires_grad_() for x in inputs)
            value = apply(*values)
            g = b.module.autograd.grad(value.square().sum(), values, create_graph=True)
            hvp = b.module.autograd.grad(sum((x * 0.25).sum() for x in g), values)
        np.testing.assert_array_equal(b.numpy(value), wanted)
        for actual, want in zip(g, expected):
            np.testing.assert_array_equal(b.numpy(actual), want)
        for actual, want in zip(hvp, curvature):
            np.testing.assert_array_equal(b.numpy(actual), want)
