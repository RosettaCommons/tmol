"""Prepared kinematic indices with JAX-owned device buffers."""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from tmol_core.kinematics import (
    prepare_kinematic_forest,
    n_movable_jump_dof_types,
)
from ._native import inverse_coordinates, forward_coordinates


@dataclass(frozen=True, eq=False, init=False)
class KinematicForest:
    """Validate a forest once and initialize DOFs for changing coordinates.

    Indices and input coordinates use kinematic order, including virtual root
    zero. The shared core validates topology; this adapter owns device arrays.
    Capture the prepared forest in a jitted function, or jit ``forest.inverse``.
    Bind initialized DOFs to obtain a differentiable forward coordinate map.
    A supplied common ``ScanOrdering`` is validated and retained for execution.
    Automatic pose-to-forest construction is not provided by this operation.
    """

    _indices: object = field(repr=False)
    _arrays: tuple = field(repr=False)
    _ordering: object = field(repr=False)

    def __init__(
        self,
        parent,
        frame_x,
        frame_y,
        frame_z,
        doftype,
        *,
        device=None,
        scan_ordering=None,
    ):
        indices = prepare_kinematic_forest(parent, frame_x, frame_y, frame_z, doftype)
        if scan_ordering is not None:
            from tmol_core.kinematics.scan_ordering import (
                ScanOrdering,
                validate_scan_plan,
            )

            scan_ordering = ScanOrdering(
                *(
                    validate_scan_plan(
                        indices.parent,
                        plan.nodes,
                        plan.scans,
                        plan.gens,
                        reverse=reverse,
                    )
                    for plan, reverse in (
                        (scan_ordering.forward, False),
                        (scan_ordering.backward, True),
                    )
                )
            )
        arrays = tuple(
            jax.device_put(getattr(indices, name), device)
            for name in ("parent", "frame_x", "frame_y", "frame_z", "doftype")
        )
        object.__setattr__(self, "_indices", indices)
        object.__setattr__(self, "_arrays", arrays)
        object.__setattr__(self, "_ordering", scan_ordering)

    @property
    def n_nodes(self):
        return self._indices.n_nodes

    @property
    def device(self):
        return self._arrays[0].device

    def bind(self, reference_dofs):
        """Fix reference rotations and return a differentiable six-column DOF map.

        ``reference_dofs`` is a concrete [nodes,9] initialization, typically from
        ``inverse``. Binding snapshots its three reference-angle columns; those
        constants are outside the differentiated arguments. The returned map
        exposes ``initial_dofs`` with shape [nodes,6] and accepts that layout.
        """
        return KinematicCoordinates(self, reference_dofs)

    def inverse(self, coords):
        """Return float32/64 DOFs [nodes,9], without differentiating initialization."""
        return inverse_coordinates(jnp.asarray(coords), self._arrays)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False, init=False)
class KinematicCoordinates:
    """A fixed forest and reference frame, mapping movable DOFs to coordinates.

    All six columns move jump nodes; bond nodes use the first four columns.
    Root and unused columns have zero derivatives. First reverse derivatives,
    jit and vmap are supported; forward-mode and higher derivatives are not.
    """

    initial_dofs: object
    _reference_angles: object = field(repr=False)
    _tree: object = field(repr=False)
    _forward: tuple = field(repr=False)
    _backward: tuple = field(repr=False)

    def __init__(self, forest, reference_dofs):
        from tmol_core.kinematics.scan_ordering import prepare_scan_ordering

        reference = np.array(reference_dofs, copy=True)
        if reference.dtype not in (np.float32, np.float64):
            raise TypeError("kinematic initialization requires float32 or float64")
        if reference.shape != (forest.n_nodes, 9):
            raise ValueError("kinematic initialization must have shape (nodes, 9)")
        initial = jax.device_put(reference[:, :n_movable_jump_dof_types], forest.device)
        if initial.dtype != reference.dtype:
            raise ValueError("enable JAX float64 before binding float64 DOFs")
        indices = forest._indices
        tree = np.column_stack(
            (
                np.arange(forest.n_nodes, dtype=np.int32),
                indices.doftype,
                indices.parent,
                indices.frame_x,
                indices.frame_y,
                indices.frame_z,
            )
        )
        ordering = forest._ordering or prepare_scan_ordering(indices.parent)
        object.__setattr__(self, "initial_dofs", initial)
        object.__setattr__(
            self,
            "_reference_angles",
            jax.device_put(reference[:, n_movable_jump_dof_types:], forest.device),
        )
        object.__setattr__(self, "_tree", jax.device_put(tree, forest.device))
        for name in ("forward", "backward"):
            plan = getattr(ordering, name)
            arrays = (
                jax.device_put(plan.nodes, forest.device),
                jax.device_put(plan.scans, forest.device),
                plan.gens,
            )
            object.__setattr__(self, "_" + name, arrays)

    def __call__(self, dofs):
        """Return [nodes,3] coordinates for [nodes,6] movable DOFs."""
        dofs = jnp.asarray(dofs)
        if dofs.shape != self.initial_dofs.shape:
            raise ValueError("movable DOFs must have shape (nodes, 6)")
        if dofs.dtype != self.initial_dofs.dtype:
            raise TypeError("movable DOF dtype must match the bound initialization")
        packed = jnp.concatenate((dofs, self._reference_angles), axis=1)
        return forward_coordinates(packed, self._tree, self._forward, self._backward)

    def tree_flatten(self):
        # Generation boundaries configure native launches; forest indices and
        # reference frames remain operands, including for a fresh binding.
        plans = (self._forward, self._backward)
        generations = tuple(tuple(map(tuple, plan[2].tolist())) for plan in plans)
        return (
            self.initial_dofs,
            self._reference_angles,
            self._tree,
            *self._forward[:2],
            *self._backward[:2],
        ), generations

    @classmethod
    def tree_unflatten(cls, generations, children):
        result = object.__new__(cls)
        for name, value in zip(
            ("initial_dofs", "_reference_angles", "_tree"), children[:3]
        ):
            object.__setattr__(result, name, value)
        for index, name in enumerate(("_forward", "_backward")):
            plan = (
                *children[3 + 2 * index : 5 + 2 * index],
                np.asarray(generations[index], dtype=np.int32).reshape(-1, 2),
            )
            object.__setattr__(result, name, plan)
        return result
