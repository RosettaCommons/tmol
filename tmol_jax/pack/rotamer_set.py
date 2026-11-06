"""Immutable JAX coordinates and identities for a prepared conformer ensemble."""

import jax
import numpy as np

from tmol_core.pack.correlation import CorrelatedBlockGroup
from tmol_jax.types import pytree_dataclass


class ConformerContext:
    """Static provenance; coordinate replacement does not change topology."""

    fields = (
        "block_type_ind",
        "inter_residue_connections",
        "_inter_block_bondsep",
        "near_blocks",
        "bondsep",
    )

    def __init__(self, pose):
        self.packed_block_types = pose.packed_block_types
        self.topology = tuple(getattr(pose, name) for name in self.fields)

    def validate(self, pose):
        if pose.packed_block_types is not self.packed_block_types or any(
            old is not new and not np.array_equal(old, new)
            for old, new in zip(
                self.topology, (getattr(pose, name) for name in self.fields)
            )
        ):
            raise ValueError("conformers belong to a different pose topology")


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
    context: ConformerContext | None = None

    @property
    def n_rotamers_total(self):
        return self.block_ind_for_rot.shape[0]

    @property
    def device(self):
        return self.coords.device
