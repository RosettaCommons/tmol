"""Immutable pose coordinates and topology as a JAX pytree."""

from __future__ import annotations

from typing import Any, Optional

import attr
import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.types import pytree_dataclass


@pytree_dataclass
class PoseStack:
    """A batch (stack) of molecular systems — see module docstring."""

    packed_block_types: Any  # PackedBlockTypes with array leaves.
    coords: jax.Array  # float32[n_poses, max_n_atoms_per_pose, 3]
    block_coord_offset: jax.Array  # int32[n_poses, max_n_blocks]
    inter_residue_connections: jax.Array  # int32[P, B, C, 2]
    inter_block_bondsep: jax.Array  # int32[P, B, B, C, C]
    block_type_ind: jax.Array  # int32[n_poses, max_n_blocks]
    pdb_info: Optional[Any] = None  # PDBInfo; aux.

    # Properties

    @property
    def n_poses(self) -> int:
        return int(self.block_type_ind.shape[0])

    @property
    def max_n_blocks(self) -> int:
        return int(self.block_type_ind.shape[1])

    @property
    def max_n_atoms_per_pose(self) -> int:
        """Largest number of atoms packed into any pose's coord slab."""
        return int(self.coords.shape[1])

    # Upstream alias
    @property
    def max_n_atoms_per_block(self) -> int:
        """Maximum atom count over all block types in the palette."""
        return int(self.packed_block_types.max_n_atoms)

    # Upstream aliases
    @property
    def max_n_atoms(self) -> int:
        return self.max_n_atoms_per_block

    @property
    def atom_is_real(self):
        """Mask packed atoms by topology, independent of their positions."""
        n_atoms = self.packed_block_types.n_atoms[jnp.maximum(self.block_type_ind, 0)]
        counts = jnp.where(self.block_type_ind >= 0, n_atoms, 0).sum(axis=1)
        return jnp.arange(self.coords.shape[1])[None, :] < counts[:, None]

    @property
    def device(self):
        """Device holding the pose coordinates."""
        return self.coords.device

    # Slicing / indexing

    def __len__(self) -> int:
        return self.n_poses

    def __getitem__(self, i) -> "PoseStack":
        """Slice along the pose axis. Single-index returns a 1-pose stack."""

        # Host indexing keeps metadata and arrays aligned, including bounds,
        # negative indices, boolean masks, and repeated/reordered selections.
        indices = np.atleast_1d(np.arange(self.n_poses)[i])
        if indices.ndim != 1:
            raise IndexError("pose selection must be one dimensional")

        def _take(x):
            return x[jnp.asarray(indices)]

        return PoseStack(
            packed_block_types=self.packed_block_types,
            coords=_take(self.coords),
            block_coord_offset=_take(self.block_coord_offset),
            inter_residue_connections=_take(self.inter_residue_connections),
            inter_block_bondsep=_take(self.inter_block_bondsep),
            block_type_ind=_take(self.block_type_ind),
            pdb_info=None if self.pdb_info is None else self.pdb_info.take(indices),
        )

    # Copy semantics

    def to(self, device) -> "PoseStack":
        """Move pose and packed-block arrays to a JAX device."""
        return jax.device_put(self, device)

    def clone(self) -> "PoseStack":
        """JAX arrays are immutable — ``clone`` returns ``self``.

        Downstream mutation patterns should produce a new PoseStack
        via :meth:`replace` / functional updates instead.
        """
        return self

    def replace(self, **updates) -> "PoseStack":
        """Functional field update helper."""
        return attr.evolve(self, **updates)

    # Block helpers (mirror the upstream Python-level conveniences)

    def block_type(self, pose_ind: int, block_ind: int):
        """Look up the ``RefinedResidueType``-equivalent block type.

        Delegates to ``packed_block_types.active_block_types``.
        ``is_real_block`` must be True.
        """
        return self.packed_block_types.active_block_types[
            int(self.block_type_ind[pose_ind, block_ind])
        ]

    # Constructors


__all__ = [
    "PoseStack",
]
