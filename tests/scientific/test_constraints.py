"""Independent constraint values, derivatives and molecular atom identity."""

import numpy as np

from .constraint_execution import Constraints
from .molecules import CASES


def test_harmonic_coordinate_and_parameter_derivatives(backend):
    adapter = Constraints(backend)
    xyz = np.array([[[0, 0, 0], [3, 4, 0]]], np.float32)
    params = np.array([[3, 2]], np.float32)
    energy, dx, dp = adapter.evaluate("harmonic", xyz, params)
    np.testing.assert_allclose(energy, [1])
    np.testing.assert_allclose(dx, [[[-0.6, -0.8, 0], [0.6, 0.8, 0]]], atol=1e-7)
    np.testing.assert_allclose(dp, [[-1, -1]])
    energy, dx, dp = adapter.evaluate(
        "harmonic_coordinate", xyz[:, :1], np.array([[3, 3, 4, 0, 2]], np.float32)
    )
    np.testing.assert_allclose(energy, [1])
    np.testing.assert_allclose(dx, [[[-0.6, -0.8, 0]]], atol=1e-7)
    np.testing.assert_allclose(dp, [[-1, 0.6, 0.8, 0, -1]], atol=1e-7)
    _, zero_dx, _ = adapter.evaluate("harmonic", xyz * 0, params)
    np.testing.assert_array_equal(zero_dx, 0)


def test_bounded_distance_regions(backend):
    xyz = np.zeros((4, 2, 3), np.float32)
    xyz[:, 1, 0] = [1, 3, 4.5, 7]
    params = np.tile(np.array([2, 4, 2, 0.7], np.float32), (4, 1))
    energy, dx, _ = Constraints(backend).evaluate("bounded", xyz, params)
    np.testing.assert_allclose(energy, [0.25, 0, 0.0625, 1.61], atol=2e-7)
    np.testing.assert_allclose(dx[:, 1, 0], [-0.5, 0, 0.25, 0.7], atol=1e-7)
    np.testing.assert_array_equal(dx.sum(axis=1), 0)


def test_circular_constraint_periodicity_and_signed_derivative(backend):
    adapter = Constraints(backend)
    xyz = np.array([[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 0, 1]]], np.float32)
    params = np.array([[np.pi / 2 - 0.4, 0.5, 0.1]], np.float32)
    energy, dx, dp = adapter.evaluate("circularharmonic", xyz, params)
    np.testing.assert_allclose(energy, [0.74], atol=5e-7)
    np.testing.assert_allclose(dp, [[-3.2, -2.56, 1]], atol=2e-6)
    params[:, 0] += 2 * np.pi
    periodic, periodic_dx, _ = adapter.evaluate("circularharmonic", xyz, params)
    np.testing.assert_allclose(periodic, energy, atol=2e-6)
    np.testing.assert_allclose(periodic_dx, dx, atol=4e-6)
    direction = np.random.default_rng(918).normal(size=xyz.shape).astype(np.float32)
    h = 1e-3
    plus = adapter.evaluate("circularharmonic", xyz + h * direction, params)[0]
    minus = adapter.evaluate("circularharmonic", xyz - h * direction, params)[0]
    np.testing.assert_allclose(
        (plus - minus) / (2 * h), (dx * direction).sum(), rtol=2e-3
    )


def test_mixed_molecular_constraint_atom_identity(backend):
    backend.molecule(CASES["mixed_batch"], terms=("constraint",))
    coords = backend.coords()
    offsets = backend.numpy(backend.pose.block_coord_offset)
    atoms = np.array(
        [[[0, 2, 0], [0, 2, 1]], [[1, 0, 0], [1, 0, 1]], [[0, 0, 0], [0, 2, 3]]],
        np.int32,
    )
    locations = [(p, int(offsets[p, b]) + a) for p, b, a in atoms.reshape(-1, 3)]
    gathered = np.array([coords[p, a] for p, a in locations]).reshape(3, 2, 3)
    distances = np.linalg.norm(gathered[:, 0] - gathered[:, 1], axis=1)
    sd = np.array([0.2, 0.2, 0.25], np.float32)
    params = np.column_stack((distances - 2 * sd, sd))
    Constraints(backend).attach([("harmonic", atoms, params)])
    values, gradient = backend.value_and_grad(coords)
    np.testing.assert_allclose(values, [8, 4], atol=2e-4)
    touched = np.zeros(coords.shape[:2], bool)
    for p, a in locations:
        touched[p, a] = True
    np.testing.assert_array_equal(gradient[~touched], 0)
    np.testing.assert_allclose(gradient.sum(axis=1), 0, atol=1e-6)
    assert np.linalg.norm(gradient[touched]) > 1
