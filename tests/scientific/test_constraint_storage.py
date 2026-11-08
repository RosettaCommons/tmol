"""Public constraint transport preserves identities and parameter derivatives."""

import numpy as np

from .constraint_execution import Constraints
from .molecules import BACKBONE_CASES


def test_constraint_replication_clone_and_transport_preserve_parameter_history(backend):
    adapter = Constraints(backend)
    cls = adapter.set_type
    atoms = adapter.array(np.array([[[0, 1, 0], [0, 0, 2]]], np.int32))
    np.testing.assert_array_equal(backend.numpy(cls.count_unique_blocks(atoms)), [2])
    parameters = adapter.array(np.array([[1.25, 0.5]], np.float32))

    def copied(values):
        repeated_atoms, repeated_params = cls.replicate_constraints(3, atoms, values)
        constraints = cls.create_empty(backend.device, 3).add_constraints(
            adapter.term_type.harmonic, repeated_atoms, repeated_params
        )
        return constraints.clone().to(backend.device)

    result = copied(parameters)
    np.testing.assert_array_equal(
        backend.numpy(result.constraint_atoms[:, 0, 0]), [0, 1, 2]
    )
    np.testing.assert_array_equal(
        backend.numpy(result.constraint_atoms[:, :2, 1:]),
        np.broadcast_to(backend.numpy(atoms)[0, :, 1:], (3, 2, 2)),
    )
    assert result.device == backend.device
    if backend.framework == "torch":
        parameters = parameters.clone().requires_grad_(True)
        objective = copied(parameters).constraint_params.square().sum()
        gradient = backend.module.autograd.grad(objective, parameters)[0]
    else:
        gradient = backend.module.grad(
            lambda values: (copied(values).constraint_params ** 2).sum()
        )(parameters)
    np.testing.assert_allclose(backend.numpy(gradient), 6 * backend.numpy(parameters))
    _, empty = cls.replicate_constraints(3, atoms, None)
    assert empty is None

    if backend.framework == "jax":
        import jax
        import jax.numpy as jnp

        def build(parameters):
            constraints = cls.create_empty(backend.device, n_poses=3).add_constraints(
                lambda atoms, p: (atoms[:, 0, 0] - p[:, 0]) ** 2,
                np.array([[[0, 0]]], np.int32),
                parameters,
            )
            return cls.concatenate([constraints.split(2), constraints.split(0)])

        parameters = jnp.array([[2.0, 3.0]])

        def objective(p):
            return jnp.sum(build(p).constraint_params ** 2)

        np.testing.assert_array_equal(jax.grad(objective)(parameters), 4 * parameters)
        np.testing.assert_array_equal(
            jax.hessian(objective)(parameters).reshape(2, 2), np.eye(2) * 4
        )
        constraints = build(parameters)
        np.testing.assert_array_equal(
            jax.jit(lambda cs: cs.constraint_params)(constraints), [[2, 3], [2, 3]]
        )
        np.testing.assert_array_equal(constraints.constraint_atoms[:, 0, 0], [0, 1])


def test_constraint_clone_and_device_copy_keep_molecular_atom_identity(backend):
    b = backend.molecule(BACKBONE_CASES["mixed"], terms=("constraint",))
    adapter = Constraints(b)
    atoms = np.array([[[p, 0, 0], [p, 1, 1]] for p in range(2)], np.int32)
    coords = b.coords()
    offsets = b.numpy(b.pose.block_coord_offset)
    first = offsets[:, 0]
    second = offsets[:, 1] + 1
    rows = np.arange(2)
    separation = coords[rows, first] - coords[rows, second]
    distance = np.linalg.norm(separation, axis=1)
    params = np.column_stack((distance - 0.5, np.full(2, 0.25, np.float32)))
    original = adapter.attach([("harmonic", atoms, params)])
    copied = original.clone().to(b.device)
    assert copied is not original
    if b.framework == "jax":
        b.pose = b.pose.replace(constraint_set=copied)
    else:
        b.pose.constraint_set = copied
        b._render_torch()
    values, gradient = b.value_and_grad(coords)
    np.testing.assert_allclose(values, [4, 4], rtol=2e-5, atol=2e-5)
    expected = np.zeros_like(coords)
    expected[rows, first] = 16 * separation / distance[:, None]
    expected[rows, second] = -expected[rows, first]
    np.testing.assert_allclose(gradient, expected, rtol=2e-5, atol=2e-5)
