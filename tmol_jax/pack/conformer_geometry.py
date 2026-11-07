"""Device forests for shared conformer layout plans."""

import jax
import jax.numpy as jnp
import numpy as np

from tmol_core.pack.conformer_layout import prepare_conformer_layout

from tmol_core.kinematics.scan_ordering import scan_ordering_from_forward
from tmol_jax.kinematics._native import inverse_coordinates, forward_coordinates


@jax.tree_util.register_pytree_node_class
class ConformerGeometry:
    def __init__(self, layout, device):
        self.total = layout.total
        self.device = device
        ordering = scan_ordering_from_forward(
            layout.forest[2], layout.nodes, layout.scans, layout.gens
        )
        plans = (ordering.forward, ordering.backward)
        # Inverse order is parent/frame_x/frame_y/frame_z/doftype. Transfer the
        # existing host indices directly instead of gathering them on device.
        self.ids, self.forest, self.inverse_indices, scans = jax.device_put(
            (
                layout.forest[0],
                layout.forest.T,
                tuple(layout.forest[i] for i in (2, 3, 4, 5, 1)),
                tuple((plan.nodes, plan.scans) for plan in plans),
            ),
            device,
        )
        self.plans = tuple(
            (nodes, scan, plan.gens) for (nodes, scan), plan in zip(scans, plans)
        )

    def tree_flatten(self):
        arrays = (
            self.ids,
            self.forest,
            self.inverse_indices,
            tuple((nodes, scans) for nodes, scans, _ in self.plans),
        )
        metadata = (
            self.total,
            self.device,
            tuple(tuple(map(tuple, plan[2])) for plan in self.plans),
        )
        return arrays, metadata

    @classmethod
    def tree_unflatten(cls, metadata, arrays):
        obj = cls.__new__(cls)
        obj.total, obj.device, generations = metadata
        obj.ids, obj.forest, obj.inverse_indices, plans = arrays
        obj.plans = tuple(
            (nodes, scans, np.asarray(gens, np.int32).reshape(-1, 2))
            for (nodes, scans), gens in zip(plans, generations)
        )
        return obj

    def inverse(self, coordinates):
        with jax.default_device(self.device):
            gathered = jnp.concatenate(
                (
                    jnp.zeros((1, 3), coordinates.dtype),
                    coordinates.reshape(-1, 3)[self.ids[1:]],
                )
            )
        return inverse_coordinates(gathered, self.inverse_indices)

    def forward(self, dofs):
        coordinates = forward_coordinates(dofs, self.forest, *self.plans)
        with jax.default_device(self.device):
            return (
                jnp.zeros((self.total, 3), dofs.dtype)
                .at[self.ids[1:]]
                .set(coordinates[1:])
            )


def geometry_for_task(
    task, name, packed, max_n_atoms, atom_counts, types, device, coordinate_offsets=None
):
    """Reuse one validated plan per unchanged concrete task/identity layout.

    The caller validates the task against pose topology first. Dynamic sampling
    may change conformer counts/types, in which case its previous plan is replaced.
    Coordinates are never cached.
    """
    previous = getattr(task, name, None)
    if previous is not None and np.array_equal(previous[0], types):
        return previous[1:]
    layout = prepare_conformer_layout(
        packed, max_n_atoms, atom_counts, types, coordinate_offsets
    )
    geometry = ConformerGeometry(layout, device)
    setattr(task, name, (np.array(types, copy=True), layout, geometry))
    return layout, geometry
