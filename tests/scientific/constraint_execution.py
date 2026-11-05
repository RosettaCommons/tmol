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
                getattr(self.term_type, name),
                self.array(np.asarray(atoms, np.int32)),
                self.array(np.asarray(parameters, np.float32)),
            )
        if b.framework == "torch":
            b.pose.constraint_set = constraints
            b._render_torch()
        else:
            b.pose = b.pose.replace(constraint_set=constraints)
        return constraints
