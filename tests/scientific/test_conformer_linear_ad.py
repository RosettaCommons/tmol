"""Shared conformer value/AD contracts, including weighted nonzero curvature."""

from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest


@dataclass
class LinearCase:
    inputs: tuple
    metadata: tuple
    destinations: np.ndarray
    sources: tuple
    weights: np.ndarray
    directions: tuple
    root_shift: int = 0

    def value(self, inputs):
        """Independent explicit row identities, not the kernel's topology plan."""
        out = inputs[0].copy().reshape(-1)
        written = np.zeros(self.destinations.size, out.dtype)
        for argument, indices, sign in self.sources:
            written += sign * inputs[argument].reshape(-1)[indices]
        out[self.destinations] = written
        return out.reshape(inputs[0].shape)

    def transpose(self, cotangent):
        result = [np.zeros_like(value) for value in self.inputs]
        result[0][...] = cotangent
        result[0].reshape(-1)[self.destinations] = 0
        for argument, indices, sign in self.sources:
            np.add.at(
                result[argument].reshape(-1),
                indices,
                sign * cotangent.reshape(-1)[self.destinations],
            )
        return result


def _rows(rows, width=9, components=None):
    components = np.arange(width) if components is None else np.asarray(components)
    return (np.asarray(rows)[:, None] * width + components).reshape(-1)


def _case(name, dtype):
    operation, _, variant = name.partition("-")
    weighted = bool(variant)
    shift = int(variant == "root")
    target = np.arange(63, dtype=dtype).reshape(7, 9) / 8
    if operation == "rows":
        width = 3 if variant == "xyz" else 9
        if weighted:
            target = np.arange(6 * width, dtype=dtype).reshape(6, width) / 7
            source = np.arange(6 * width, dtype=dtype).reshape(6, width) / 10
            metadata = ([1, 1], [0, 3], [0, 2, 3])
            destinations = _rows(np.array([0, 1, 3]) + shift, width)
            selected = _rows(np.array([1, 2, 1]) + shift, width)
        else:
            source = np.arange(45, dtype=dtype).reshape(5, 9) / 4
            metadata = ([1, 0, 1], [0, 2, 4], [0, 2, 2, 4])
            destinations, selected = _rows([0, 1, 4, 5]), _rows([1, 2, 1, 2])
        inputs = (target, source)
        sources = ((1, selected, 1),)
        directions = (np.sin(target), np.cos(source))
    elif operation == "chi":
        if weighted:
            target = np.linspace(0.1, 0.9, 90, dtype=dtype).reshape(10, 9)
            chi = np.array([[0.2, 0.9], [0.4, 0.7]], dtype)
            correction = np.linspace(0.01, 0.06, 6, dtype=dtype).reshape(2, 3)
            metadata = (
                [1, 1],
                [1, 4],
                [1, 0],
                [[1, -1], [0, 2]],
                [[0, 1, 2], [2, 1, 0]],
            )
            destinations, corrections = np.array([57, 39, 21]), [4, 3, 5]
        else:
            chi = np.array([[1, 99], [2, 3]], dtype)
            correction = np.arange(6, dtype=dtype).reshape(2, 3) / 8
            metadata = (
                [0, 1],
                [0, 3],
                [0, 1],
                [[1, -1], [0, 2]],
                [[0, 1, 2], [0, 1, 2]],
            )
            destinations, corrections = np.array([21, 39, 57]), [1, 3, 5]
        inputs = (target, chi, correction)
        sources = ((1, [0, 2, 3], 1), (2, corrections, -1))
        directions = (np.cos(target), np.cos(chi), np.sin(correction))
    elif operation == "reset":
        if weighted:
            target = np.linspace(0.2, 0.9, 90, dtype=dtype).reshape(10, 9)
            ideal = np.linspace(0.1, 0.8, 72, dtype=dtype).reshape(2, 4, 9)
            metadata = (
                [1, 1],
                [0, 4],
                [0, 1],
                [[-1, -1], [2, -1]],
                [[-1, -1], [1, 2]],
                [0, 2],
            )
            destinations, selected = _rows([6, 7]), _rows([5, 6])
        else:
            ideal = np.arange(54, dtype=dtype).reshape(2, 3, 9) / 4
            metadata = ([1, 1], [0, 3], [0, 1], [[-1], [2]], [[0, 1], [1, 2]], [0, 2])
            destinations, selected = _rows([5, 6]), _rows([4, 5])
        inputs, sources = (target, ideal), ((1, selected, 1),)
        directions = (np.sin(target), np.cos(ideal))
    else:
        assert operation == "mainchain"
        if weighted:
            target = np.linspace(0.2, 1.0, 81, dtype=dtype).reshape(9, 9)
            source = np.linspace(0.1, 0.9, 27, dtype=dtype).reshape(3, 9)
        else:
            source = np.arange(27, dtype=dtype).reshape(3, 9) / 4
        source[2, 4:7] = (np.nan, np.inf, -np.inf)
        metadata = (
            [0, 0, 1],
            [0, 0, 0],
            [1, 1, 1],
            [2, 4, 0],
            [0, -1],
            [[0, 1], [-1, -1]],
            [[[[-1, -1], [1, -1]]]],
            [[1, 0], [0, 1]],
        )
        components = [0, 1, 2, 3, 7, 8]
        destinations = _rows([4, 6], components=components)
        sources = ((1, _rows([2, 2], components=components), 1),)
        inputs = (target, source)
        directions = (
            np.linspace(0.4, 0.8, target.size, dtype=dtype).reshape(target.shape),
            np.linspace(0.2, 0.7, source.size, dtype=dtype).reshape(source.shape),
        )
    if weighted:
        low, high = (0.3, 0.9) if operation == "mainchain" else (0.2, 0.8)
        weights = np.linspace(low, high, target.size, dtype=dtype).reshape(target.shape)
    else:
        weights = np.full_like(target, 2)
        directions = tuple(np.full_like(value, 0.25) for value in inputs)
    return LinearCase(
        inputs, metadata, destinations, sources, weights, directions, shift
    )


def _operation(backend, operation, case, array):
    metadata = tuple(array(value, True) for value in case.metadata)
    if backend.framework == "jax":
        from tmol_jax.pack import _native

        def apply(target, source, *rest):
            if operation == "rows":
                return _native.copy_conformer_rows(
                    target,
                    source,
                    *metadata,
                    n_rows=int(case.metadata[-1][-1]),
                    root_shift=case.root_shift,
                )
            if operation == "chi":
                return _native.assign_conformer_chi(target, source, *metadata, *rest)
            if operation == "reset":
                return _native.reset_opth_dofs(target, source, *metadata, flip=True)
            return _native.copy_conformer_dofs(target, source, *metadata, sampler=0)

    else:
        from tmol.pack.rotamer import _conformer_execution as native

        def apply(target, source, *rest):
            result = target.clone()
            if operation == "rows":
                src, dst, prefix = metadata
                native.copy_conformer_rows(
                    prefix[1:] - prefix[:-1],
                    src.long(),
                    dst.long(),
                    source,
                    result,
                    root_shift=case.root_shift,
                )
            elif operation == "chi":
                types, offsets, selected, atoms, kinematic = metadata
                native.assign_chi_dofs(
                    types.long(),
                    offsets.long(),
                    selected.long(),
                    atoms,
                    source,
                    kinematic.long(),
                    rest[0],
                    result,
                )
            elif operation == "reset":
                types, offsets, selected, atoms, downstream, counts = metadata
                cache = SimpleNamespace(
                    nhq_downstream_kfo=downstream, nhq_downstream_count=counts
                )
                native.reset_opth_dofs(
                    types.long(),
                    offsets.long(),
                    selected.long(),
                    atoms,
                    cache,
                    source,
                    result,
                    True,
                )
            else:
                native.copy_mainchain_dofs(
                    *(a.long() for a in metadata), source, result, 0
                )
            return result

    return apply


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize(
    "case_name",
    (
        "rows",
        "chi",
        "reset",
        "mainchain",
        "rows-xyz",
        "rows-root",
        "chi-weighted",
        "reset-weighted",
        "mainchain-weighted",
    ),
)
def test_native_conformer_linear_value_gradient_and_curvature(
    backend, dtype, case_name
):
    b = backend
    precision = (
        b.module.enable_x64(dtype == np.float64)
        if b.framework == "jax"
        else nullcontext()
    )
    with precision:

        def array(value, integer=False):
            value = np.asarray(value, dtype=np.int32 if integer else dtype)
            return (
                b.module.device_put(value, b.device)
                if b.framework == "jax"
                else b.module.as_tensor(value, device=b.device)
            )

        case = _case(case_name, dtype)
        operation = case_name.split("-")[0]
        apply = _operation(b, operation, case, array)
        inputs, directions = tuple(map(array, case.inputs)), tuple(
            map(array, case.directions)
        )
        weights = array(case.weights)
        if b.framework == "jax":

            def loss(*values):
                return (weights * apply(*values) ** 2 / 2).sum()

            grad = b.module.grad(loss, argnums=tuple(range(len(inputs))))
            value = b.module.jit(apply)(*inputs)
            gradient = b.module.jit(grad)(*inputs)
            _, hvp = b.module.jit(
                lambda *values: b.module.jvp(grad, values, directions)
            )(*inputs)
        else:
            values = tuple(value.detach().requires_grad_() for value in inputs)
            value = apply(*values)
            gradient = b.module.autograd.grad(
                (weights * value.square() / 2).sum(), values, create_graph=True
            )
            hvp = b.module.autograd.grad(
                sum((g * d).sum() for g, d in zip(gradient, directions)), values
            )
        for actual, original in zip(
            (value, *gradient, *hvp), (inputs[0], *inputs, *inputs)
        ):
            assert actual.dtype == original.dtype
            assert actual.device == original.device
        wanted = case.value(case.inputs)
        expected = case.transpose(case.weights * wanted)
        curvature = case.transpose(case.weights * case.value(case.directions))
        # Keep exact binary-valued fixtures and the original weighted test tolerances.
        if "-" not in case_name:
            check = np.testing.assert_array_equal
        elif b.framework == "torch":
            check = lambda actual, expected: b.module.testing.assert_close(
                array(actual), array(expected)
            )
        else:
            rtol, atol = (1.3e-6, 1e-5) if dtype == np.float32 else (1e-12, 1e-12)
            check = lambda actual, expected: np.testing.assert_allclose(
                actual, expected, rtol=rtol, atol=atol
            )
        check(b.numpy(value), wanted)
        for actual, expected in zip(gradient, expected):
            check(b.numpy(actual), expected)
        for actual, expected in zip(hvp, curvature):
            check(b.numpy(actual), expected)
        if case_name == "chi-weighted":
            assert np.any(curvature[1]) and np.any(curvature[2])
        if operation == "mainchain":
            _check_repaired_mainchain_mask(b, case.inputs[1], inputs[0], array, apply)


def _check_repaired_mainchain_mask(b, source, target, array, apply):
    # Reusing topology after repairing a nonfinite input must recompute its mask.
    repaired = np.nan_to_num(source, nan=0.75, posinf=0.75, neginf=0.75)
    if b.framework == "jax":
        grad = b.module.grad(lambda x: (apply(target, x) ** 2).sum())
        changed = b.module.jit(grad)(array(repaired))
    else:
        x = array(repaired).requires_grad_()
        changed = b.module.autograd.grad(apply(target, x).square().sum(), x)[0]
    expected_source = np.zeros_like(repaired)
    expected_source[2] = 4 * repaired[2]
    np.testing.assert_array_equal(b.numpy(changed), expected_source)
