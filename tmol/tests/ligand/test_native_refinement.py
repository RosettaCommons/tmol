"""Check typed native refinement against chemistry, autograd and invariants."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from tmol.ligand import _conformer_generation as torch_execution
from tmol.ligand._native_refinement import native_conformer_execution
from tmol_core.ligand import _conformer_generation as geometry, refinement
from tmol_core.ligand.native_refinement import (
    prepare_native_planar,
    prepare_native_refinement,
)


@pytest.fixture(
    scope="module",
    params=[
        "[NH3+][C@H](C)C(=O)[O-]",
        "[NH3+][C@@H](C)C(=O)[O-]",
        "CC(=O)Oc1ccccc1C(=O)O",
        pytest.param("O=C([O-])CCCC[C@H]1SC[C@H]2NC(=O)N[C@@H]12", id="biotin"),
    ],
)
def chemical_plan(request):
    molecule = geometry._obmol_to_rdkit(geometry._smiles_to_obmol(request.param))
    targets = geometry._geometry_targets(
        molecule, geometry._double_bond_stereo(request.param, molecule)
    )
    lower0, upper0 = geometry._bounds_matrix(targets, execution=torch_execution)
    lower, upper = geometry._smooth_bounds(lower0, upper0)
    xyz = geometry._metric_embed(lower, upper, 47)
    xyz += np.random.default_rng(713).normal(0, 0.05, xyz.shape)
    return xyz, (
        lower0,
        upper0,
        lower,
        upper,
        targets["planar_sets"],
        targets["chirals"],
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_native_refinement_chemical_gradient(chemical_plan, dtype):
    initial, plan = chemical_plan
    backend = native_conformer_execution().native
    parameters = prepare_native_refinement(*plan, dtype)
    torch_dtype = torch.float32 if dtype == np.float32 else torch.float64
    expected_parameters = refinement.prepare_refinement(
        *plan,
        index_array=lambda a: torch.as_tensor(a, dtype=torch.long),
        float_array=lambda a: torch.as_tensor(a, dtype=torch_dtype),
    )
    rng = np.random.default_rng(71)
    for dimensions in (3, 4):
        xyz = np.ascontiguousarray(
            (
                initial
                if dimensions == 3
                else np.c_[initial, rng.normal(0, 0.2, len(initial))]
            ),
            dtype=dtype,
        )
        options = dict(four_dimensional=dimensions == 4, fourth_weight=0.5)
        value, gradient = backend.value_and_grad(xyz, parameters, **options)
        x = torch.tensor(xyz, requires_grad=True)
        expected = refinement.refinement_loss(
            x,
            expected_parameters,
            torch_execution._OPERATIONS,
            fourth_dimension=0.5 if dimensions == 4 else None,
        )
        expected_gradient = torch.autograd.grad(expected, x)[0].numpy()
        rtol, atol = (3e-6, 3e-5) if dtype == np.float32 else (2e-10, 2e-10)
        np.testing.assert_allclose(value, expected.detach(), rtol=rtol, atol=atol)
        np.testing.assert_allclose(gradient, expected_gradient, rtol=rtol, atol=atol)
        if dtype == np.float64:
            direction = rng.normal(size=xyz.shape)
            direction /= np.linalg.norm(direction)
            step = 1e-5
            difference = (
                backend.value_and_grad(xyz + step * direction, parameters, **options)[0]
                - backend.value_and_grad(xyz - step * direction, parameters, **options)[
                    0
                ]
            ) / (2 * step)
            np.testing.assert_allclose(
                difference, (gradient * direction).sum(), rtol=2e-7, atol=2e-7
            )
            # Translation of physical coordinates leaves every objective invariant;
            # the fourth-dimensional tether is intentionally excluded.
            shifted = xyz.copy()
            shifted[:, :3] += [1.1, -0.7, 2.4]
            other, other_gradient = backend.value_and_grad(
                shifted, parameters, **options
            )
            np.testing.assert_allclose(other, value, rtol=2e-10, atol=2e-10)
            np.testing.assert_allclose(other_gradient, gradient, rtol=2e-10, atol=2e-10)
            np.testing.assert_allclose(gradient[:, :3].sum(0), 0, atol=2e-10)


def test_native_refinement_abi_rejects_invalid_buffers():
    backend = native_conformer_execution().native
    xyz = np.array([[0.0, 0.0], [1.2, 0.1]], dtype=np.float64)
    parameters = prepare_native_planar([(0, 1, 1)])
    with pytest.raises(ValueError, match="shape"):
        backend.value_and_grad(xyz, replace(parameters, distances=np.empty(0)))
    with pytest.raises(TypeError, match="dtype"):
        backend.value_and_grad(
            xyz, replace(parameters, pairs=parameters.pairs.astype(np.int64))
        )
    with pytest.raises(ValueError, match="status 2"):
        backend.value_and_grad(
            xyz, replace(parameters, pairs=np.array([[0, 2]], np.int32))
        )
    with pytest.raises(ValueError, match="offsets"):
        backend.value_and_grad(
            xyz, replace(parameters, component_offsets=np.array([1], np.int32))
        )
    unaligned = np.ndarray(
        xyz.shape, dtype=xyz.dtype, buffer=np.zeros(xyz.nbytes + 1, np.uint8), offset=1
    )
    assert unaligned.flags.c_contiguous and not unaligned.flags.aligned
    with pytest.raises(TypeError, match="aligned"):
        backend.value_and_grad(unaligned, parameters)
    pairs = np.ndarray(
        parameters.pairs.shape,
        dtype=np.int32,
        buffer=np.zeros(parameters.pairs.nbytes + 1, np.uint8),
        offset=1,
    )
    with pytest.raises(TypeError, match="aligned"):
        backend.value_and_grad(xyz, replace(parameters, pairs=pairs))
    value, gradient = backend.value_and_grad(xyz, parameters)
    assert value > 0
    np.testing.assert_allclose(gradient.sum(0), 0, atol=1e-14)
    other = backend.value_and_grad(xyz, parameters)[1]
    assert not np.shares_memory(gradient, other)


def test_native_refinement_denormal_scope_restores_on_exception():
    execution = native_conformer_execution()
    previous = execution.native.denormal_mode()
    with pytest.raises(RuntimeError, match="intentional scope exit"):
        with execution.flush_denormals():
            assert execution.native.denormal_mode() == (3 if previous >= 0 else -1)
            raise RuntimeError("intentional scope exit")
    assert execution.native.denormal_mode() == previous
