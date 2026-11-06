"""Device forests for shared conformer layout plans."""

import jax
import jax.numpy as jnp
import numpy as np

from tmol_core.pack.conformer_layout import prepare_conformer_layout

from tmol_core.kinematics.scan_ordering import scan_ordering_from_forward
from tmol_jax.kinematics._native import inverse_coordinates, forward_coordinates


class ConformerGeometry:
    def __init__(self, layout, device):
        self.total = layout.total
        self.ids = jax.device_put(layout.forest[0], device)
        self.forest = jax.device_put(layout.forest.T, device)
        # Inverse order is parent/frame_x/frame_y/frame_z/doftype.
        self.inverse_indices = tuple(self.forest[:, i] for i in (2, 3, 4, 5, 1))
        ordering = scan_ordering_from_forward(
            layout.forest[2], layout.nodes, layout.scans, layout.gens
        )
        self.plans = tuple(
            (
                jax.device_put(plan.nodes, device),
                jax.device_put(plan.scans, device),
                plan.gens,
            )
            for plan in (ordering.forward, ordering.backward)
        )

    def inverse(self, coordinates):
        with jax.default_device(coordinates.device):
            gathered = jnp.concatenate(
                (
                    jnp.zeros((1, 3), coordinates.dtype),
                    coordinates.reshape(-1, 3)[self.ids[1:]],
                )
            )
        return inverse_coordinates(gathered, self.inverse_indices)

    def forward(self, dofs):
        coordinates = forward_coordinates(dofs, self.forest, *self.plans)
        with jax.default_device(dofs.device):
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
