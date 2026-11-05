"""Shared branched-forest fixture and small native execution adapters."""

import numpy as np


def branched_forest(dtype):
    parent = np.array([0, 0, 1, 1, 2, 3], np.int32)
    kinds = np.array([0, 1, 2, 2, 2, 2], np.int32)
    dofs = np.zeros((6, 9), dtype)
    dofs[1] = [1, 2, 3, 0.1, 0.2, -0.15, -0.13, 0.17, 0.21]
    dofs[2:, :4] = [
        [0.3, 1.2, 1.4, -0.2],
        [0.4, 1.4, 1.3, 0.1],
        [-0.7, 1.1, 1.5, 0.3],
        [0.2, 1.3, 1.1, -0.4],
    ]
    return parent, kinds, dofs


class KinematicExecution:
    def __init__(self, backend, parent, kinds, reference):
        self.backend = backend
        if backend.requested_device == "mps":
            raise NotImplementedError("native Metal kinematic scans are not exposed")
        zeros = np.zeros(len(parent), np.int32)
        if backend.framework == "jax":
            from tmol_jax.kinematics import KinematicForest

            forest = KinematicForest(
                parent, zeros, zeros, zeros, kinds, device=backend.device
            )
            self.mapping = forest.bind(reference)
            self.forward = backend.module.jit(self.mapping)
        else:
            from tmol_core.kinematics.scan_ordering import prepare_scan_ordering
            from tmol.kinematics.compiled._compiled_ops import _forward_kin_op

            torch = backend.module
            ordering = prepare_scan_ordering(parent)
            plans = tuple(
                torch.tensor(
                    value.copy(), device="cpu" if field == "gens" else backend.device
                )
                for plan in (ordering.forward, ordering.backward)
                for field in ("nodes", "scans", "gens")
                for value in (getattr(plan, field),)
            )
            tree = torch.tensor(
                np.column_stack(
                    (np.arange(len(parent)), kinds, parent, zeros, zeros, zeros)
                ),
                dtype=torch.int32,
                device=backend.device,
            )
            reference_angles = torch.tensor(
                reference[:, 6:].copy(), device=backend.device
            )

            def forward(dofs):
                return _forward_kin_op(
                    torch.cat((dofs, reference_angles), dim=1), *plans, tree
                )

            self.forward = forward

    def evaluate(self, dofs, weights=None):
        backend = self.backend
        if backend.framework == "jax":
            x = backend.module.device_put(dofs, backend.device)
            coords = self.forward(x)
            if weights is None:
                return np.asarray(coords)
            weights = backend.module.device_put(weights, backend.device)
            gradient = backend.module.jit(
                backend.module.grad(lambda value: (self.forward(value) * weights).sum())
            )(x)
        else:
            x = backend.module.tensor(
                dofs, device=backend.device, requires_grad=weights is not None
            )
            coords = self.forward(x)
            if weights is None:
                return backend.numpy(coords)
            weights = backend.module.tensor(weights, device=backend.device)
            gradient = backend.module.autograd.grad((coords * weights).sum(), x)[0]
        return backend.numpy(coords), backend.numpy(gradient)
