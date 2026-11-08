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

    if backend.framework == "jax":
        import jax
        import jax.numpy as jnp

        with jax.enable_x64(True), jax.default_device(backend.device):
            for dtype in (np.float32, np.float64):
                x, p = adapter.array(xyz.astype(dtype)), adapter.array(
                    params.astype(dtype)
                )
                fn = adapter.term_type.harmonic
                batched = jax.jit(jax.vmap(fn, in_axes=(0, None)))(jnp.stack((x, x)), p)
                np.testing.assert_allclose(batched, [[1], [1]])
                _, tangent = jax.jit(lambda x: jax.jvp(lambda z: fn(z, p), (x,), (x,)))(
                    x
                )
                np.testing.assert_allclose(tangent, [5])
                hessian = jax.jit(jax.hessian(lambda p: fn(x, p).sum()))(p)
                np.testing.assert_allclose(hessian.reshape(2, 2), [[0.5, 1], [1, 1.5]])
        from .constraint_execution import check_legacy_constraint_inputs

        check_legacy_constraint_inputs(backend)


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

    if backend.framework == "jax":
        import jax
        import jax.numpy as jnp

        with jax.enable_x64(True), jax.default_device(backend.device):
            for dtype in (np.float32, np.float64):
                x = adapter.array(xyz.astype(dtype))
                fn = adapter.term_type.get_torsion_angle_test
                values = jax.jit(jax.vmap(fn))(jnp.stack((x, x)))
                np.testing.assert_allclose(values, np.pi / 2, rtol=1e-6)
                direction = jnp.arange(12, dtype=dtype).reshape(x.shape) / 12
                value, tangent = jax.jit(lambda x, v: jax.jvp(fn, (x,), (v,)))(
                    x, direction
                )
                gradient = jax.jit(jax.grad(lambda x: fn(x).sum()))(x)
                np.testing.assert_allclose(
                    tangent, jnp.sum(gradient * direction), atol=1e-7
                )
                np.testing.assert_allclose(value, np.pi / 2, rtol=1e-6)
                assert jax.jit(fn)(x[:0]).shape == (0,)

    # The planar and undefined-geometry oracle is common to both native torsions.
    angles = np.array([0, 1e-5, np.pi - 1e-5, np.pi], dtype=np.float32)
    coords = np.tile(
        np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]]), (4, 1, 1)
    ).astype(np.float32)
    coords[:, 3, 1] = np.cos(angles)
    coords[:, 3, 2] = np.sin(angles)
    torsion = adapter.term_type.get_torsion_angle_test
    if backend.framework == "jax":
        gradient = jax.jit(jax.grad(lambda x: jnp.cos(3 * torsion(x) - 0.4).sum()))
    else:

        def gradient(x):
            x = x.clone().requires_grad_()
            value = backend.module.cos(3 * torsion(x) - 0.4).sum()
            return backend.module.autograd.grad(value, x)[0]

    expected = -3 * np.sin(3 * angles.astype(np.float64) - 0.4) * np.cos(angles)
    np.testing.assert_allclose(
        backend.numpy(gradient(adapter.array(coords)))[:, 3, 2], expected, atol=3e-6
    )
    assert np.isnan(backend.numpy(gradient(adapter.array(np.zeros_like(coords))))).all()


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

    if backend.framework == "jax":
        import jax
        import jax.numpy as jnp
        from attrs import evolve

        adapter = Constraints(backend)
        cs = adapter.attach([("harmonic", [[[0, 0], [0, 1]]], [[1.0, 0.5]])])
        scores = backend.scores(backend.coords())
        selected = backend.pose[[1, 0, 1]].to(backend.device)
        score = backend.score_function.render_whole_pose_scoring_module(selected)
        np.testing.assert_allclose(jax.jit(score)(selected.coords), scores[[1, 0, 1]])
        empty = backend.pose[:0]
        assert empty.constraint_set.n_poses == 0
        assert empty.constraint_set.n_constraints == 0
        score = backend.score_function.render_whole_pose_scoring_module(empty)
        assert jax.jit(score)(empty.coords).shape == (0,)
        invalid = evolve(
            cs, constraint_function_inds=jnp.ones_like(cs.constraint_function_inds)
        )
        backend.pose = backend.pose.replace(constraint_set=invalid)
        with pytest.raises(ValueError, match="registry"):
            backend.scores(backend.coords())
        adapter.attach([("harmonic", [[[1, 2, 0], [1, 2, 1]]], [[1.0, 0.5]])])
        with pytest.raises(ValueError, match="padding"):
            backend.scores(backend.coords())


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

    if backend.framework == "jax" and count == 4:
        import jax
        import jax.numpy as jnp
        from .rotamer_execution import RotamerExecution

        scorer = backend.score_function.render_whole_pose_scoring_module(backend.pose)
        coords = backend.pose.coords
        expected = scorer(coords)
        batched = jax.jit(jax.vmap(scorer))(jnp.stack((coords, coords + 0.25)))
        np.testing.assert_allclose(batched, jnp.stack((expected, expected)), rtol=2e-6)
        gradient = jax.jit(jax.grad(lambda x: scorer(x).sum()))(coords)
        np.testing.assert_allclose(gradient.sum(axis=1), 0, atol=3e-6)
        assert np.any(np.abs(gradient) > 1)
        with pytest.raises(NotImplementedError, match="sparse constraints"):
            RotamerExecution(backend, family="constraint")


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
