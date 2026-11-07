"""Independent constraint values, derivatives and molecular atom identity."""

import numpy as np
import pytest

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


def multi_block_constraint(atoms, parameters):
    """Independent anchored squared distances over the callback's physical atoms."""
    return parameters[:, 0] * ((atoms[:, 1:] - atoms[:, :1]) ** 2).sum(axis=(1, 2))


def attach_multi_block_constraint(backend, count=4):
    from .molecules import MolecularCase

    case = MolecularCase(
        CASES["mixed_batch"].parameter_files,
        (("GLY", "HYP", "LG1", "DALA"), ("HYP",)),
        ((1, 1, 1, 1), (1,)),
    )
    backend.molecule(case, terms=("constraint",))
    atoms = np.array([[[0, 2, 0], [0, 0, 0], [0, 3, 0], [0, 1, 0]]], np.int32)
    atoms = atoms[:, :count]

    def callback(coordinates, parameters):
        return multi_block_constraint(coordinates[:, :count], parameters)

    Constraints(backend).attach([(callback, atoms, [[0.75]])])
    return atoms


@pytest.mark.parametrize("count", [3, 4])
def test_multi_block_whole_pose_constraint_values_and_gradients(backend, count):
    atoms = attach_multi_block_constraint(backend, count)
    xyz = backend.coords()
    offsets = backend.numpy(backend.pose.block_coord_offset)
    locations = [(p, int(offsets[p, b]) + a) for p, b, a in atoms[0]]
    points = np.array([xyz[p, a] for p, a in locations])
    differences = points[1:] - points[0]
    expected = 0.75 * (differences**2).sum()
    gradient = np.zeros_like(xyz)
    for (p, a), delta in zip(locations[1:], differences):
        gradient[p, a] += 1.5 * delta
        gradient[locations[0]] -= 1.5 * delta
    values, actual_gradient = backend.value_and_grad(xyz)
    assert expected > 1
    np.testing.assert_allclose(values, [expected, 0], rtol=2e-6, atol=2e-5)
    np.testing.assert_allclose(actual_gradient, gradient, rtol=2e-6, atol=2e-5)


@pytest.mark.parametrize("count", [3, 4])
def test_multi_block_minimization_preserves_frozen_trailing_pose(backend, count):
    atoms = attach_multi_block_constraint(backend, count)
    xyz = backend.coords()
    offsets = backend.numpy(backend.pose.block_coord_offset)
    locations = [(p, int(offsets[p, b]) + a) for p, b, a in atoms[0]]
    mask = np.zeros(xyz.shape[:2], dtype=bool)
    for location in locations[1:]:
        mask[location] = True
    initial = backend.scores(xyz).sum()
    result = backend.minimize(mask, iterations=5)
    assert backend.scores(result).sum() < initial - 1
    np.testing.assert_array_equal(result[~mask], xyz[~mask])
    np.testing.assert_array_equal(backend.coords(), xyz)
