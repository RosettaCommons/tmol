"""Execution adapters for shared constraint formula and molecular contracts."""

import numpy as np


class Constraints:
    def __init__(self, backend):
        self.backend = backend
        if backend.framework == "torch":
            from tmol import ConstraintSet
            from tmol.score import ConstraintEnergyTerm
        else:
            from tmol_jax.pose import ConstraintSet
            from tmol_jax.score.constraint import ConstraintEnergyTerm
        self.set_type = ConstraintSet
        self.term_type = ConstraintEnergyTerm

    def array(self, values):
        b = self.backend
        values = np.asarray(values)
        if b.framework == "torch":
            return b.module.as_tensor(values, device=b.device)
        return b.module.device_put(values, b.device)

    def evaluate(self, name, atoms, parameters):
        b = self.backend
        fn = getattr(self.term_type, name)
        xyz, params = self.array(atoms), self.array(parameters)
        if b.framework == "torch":
            xyz = xyz.clone().requires_grad_()
            params = params.clone().requires_grad_()
            values = fn(xyz, params)
            gradients = b.module.autograd.grad(values.sum(), (xyz, params))
        else:
            values = b.module.jit(fn)(xyz, params)
            gradients = b.module.jit(
                b.module.grad(lambda x, p: fn(x, p).sum(), argnums=(0, 1))
            )(xyz, params)
        return tuple(b.numpy(x) for x in (values, *gradients))

    def attach(self, entries):
        b = self.backend
        constraints = self.set_type.create_empty(b.device, b.pose.n_poses)
        for name, atoms, parameters in entries:
            constraints = constraints.add_constraints(
                getattr(self.term_type, name) if isinstance(name, str) else name,
                self.array(np.asarray(atoms, np.int32)),
                self.array(np.asarray(parameters, np.float32)),
            )
        if b.framework == "torch":
            b.pose.constraint_set = constraints
            b._render_torch()
        else:
            b.pose = b.pose.replace(constraint_set=constraints)
        return constraints


def check_legacy_constraint_inputs(backend):
    """The coordinate-only compatibility API has no Torch equivalent."""
    from types import SimpleNamespace
    import jax
    import jax.numpy as jnp
    import pytest
    from tmol_jax.score.constraint.constraint_energy_term import (
        ConstraintEnergyTerm,
        ConstraintSet,
    )
    from tmol_jax.score.score_types import ScoreType

    def pose():
        return SimpleNamespace(
            coords=backend.module.device_put(
                np.zeros((1, 2, 3), np.float32), backend.device
            )
        )

    def constraint(indices):
        return ConstraintSet(
            target_coords=jnp.zeros((1, 3)),
            atom_indices=indices,
            k=jnp.ones((1,)),
            tol=jnp.zeros((1,)),
        )

    for attached in (False, True):
        p = pose()
        options = {}
        if attached:
            p.constraint_set = object()
        else:
            options["constraint_set"] = object()
        with pytest.raises(TypeError, match="ConstraintSet"):
            ConstraintEnergyTerm().setup_for_pose_stack(p, **options)
    for indices in (
        np.array([[0.0, 1.5]], dtype=np.float32),
        np.array([[False, True]]),
    ):
        with pytest.raises(TypeError, match="integer"):
            ConstraintEnergyTerm().setup_for_pose_stack(
                pose(), constraint_set=constraint(indices)
            )
    p = pose()
    term = ConstraintEnergyTerm()
    state = term.setup_for_pose_stack(
        p, constraint_set=constraint(jnp.array([[0, 1]], dtype=jnp.int32))
    )
    coordinates = p.coords.at[0, 1, 0].set(2.0)
    evaluate = jax.jit(
        jax.value_and_grad(lambda x: term(p, state, x)[ScoreType.constraint].sum())
    )
    value, gradient = evaluate(coordinates)
    np.testing.assert_allclose(value, 2.0)
    expected = np.zeros((1, 2, 3), dtype=np.float32)
    expected[0, 1, 0] = 2.0
    np.testing.assert_allclose(gradient, expected)
    state = term.setup_for_pose_stack(p)
    np.testing.assert_array_equal(term(p, state)[ScoreType.constraint], [0.0])
