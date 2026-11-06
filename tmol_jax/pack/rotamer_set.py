"""Immutable JAX coordinates and identities for a prepared conformer ensemble."""

import jax

from tmol_core.pack.correlation import CorrelatedBlockGroup
from tmol_jax.types import pytree_dataclass


@pytree_dataclass
class RotamerSet:
    n_rots_for_pose: jax.Array
    rot_offset_for_pose: jax.Array
    n_rots_for_block: jax.Array
    rot_offset_for_block: jax.Array
    pose_for_rot: jax.Array
    block_type_ind_for_rot: jax.Array
    block_ind_for_rot: jax.Array
    coord_offset_for_rot: jax.Array
    coords: jax.Array
    group_for_block: jax.Array
    first_rot_block_type: jax.Array
    pose_ind_for_atom: jax.Array
    max_n_rots_per_pose: int
    correlated_groups: tuple[CorrelatedBlockGroup, ...] = ()

    @property
    def n_rotamers_total(self):
        return self.block_ind_for_rot.shape[0]

    @property
    def device(self):
        return self.coords.device
