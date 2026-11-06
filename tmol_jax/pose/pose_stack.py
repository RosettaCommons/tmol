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
    _inter_block_bondsep: Any = attr.field(alias="inter_block_bondsep")
    block_type_ind: jax.Array  # int32[n_poses, max_n_blocks]
    pdb_info: Optional[Any] = None  # PDBInfo; aux.
    split_block_mapping: Optional[Any] = None
    constraint_set: Optional[Any] = None
    near_blocks: Optional[jax.Array] = None  # int32[P,B,K,2], common sparse graph
    bondsep: Optional[jax.Array] = None  # int8[P,B,K,C,C]

    # Properties

    @property
    def inter_block_bondsep(self):
        """Dense capped connection distances, materialized only when requested.

        Native scorers consume the common sparse graph directly. Explicit dense
        constructor/replacement values remain authoritative for compatibility.
        """
        if self._inter_block_bondsep is not None:
            return self._inter_block_bondsep
        if hasattr(self, "_dense_connections"):
            return self._dense_connections
        from tmol_core.chemical.constants import MAX_SIG_BOND_SEPARATION

        if self.near_blocks is None or self.bondsep is None:
            raise ValueError("pose requires dense or sparse connection distances")
        poses, blocks, _, ports, _ = self.bondsep.shape
        with jax.default_device(getattr(self.bondsep, "device", None)):
            dense = jnp.full(
                (poses, blocks, blocks, ports, ports),
                MAX_SIG_BOND_SEPARATION,
                dtype=jnp.int32,
            )
            p = jnp.arange(poses)[:, None, None]
            b = jnp.arange(blocks)[None, :, None]
            partner = jnp.where(
                self.near_blocks[..., 0] >= 0, self.near_blocks[..., 0], blocks
            )
            result = dense.at[p, b, partner].set(self.bondsep, mode="drop")
        if not isinstance(result, jax.core.Tracer):
            object.__setattr__(self, "_dense_connections", result)
        return result

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
        real = self.block_type_ind >= 0
        counts = jnp.where(real, n_atoms, 0)
        starts = jnp.maximum(self.block_coord_offset, 0)
        pose = jnp.arange(self.n_poses)[:, None]
        edges = jnp.zeros((self.n_poses, self.coords.shape[1] + 1), jnp.int32)
        edges = edges.at[pose, starts].add(real.astype(jnp.int32))
        edges = edges.at[pose, starts + counts].add(-real.astype(jnp.int32))
        return jnp.cumsum(edges[:, :-1], axis=1) > 0

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

        constraints = None
        if self.constraint_set is not None:
            from tmol_jax.pose.constraint_set import ConstraintSet

            constraints = (
                ConstraintSet.concatenate(
                    [self.constraint_set.split(int(index)) for index in indices]
                )
                if len(indices)
                else ConstraintSet.create_empty(self.coords.device, 0)
            )
        return PoseStack(
            constraint_set=constraints,
            packed_block_types=self.packed_block_types,
            coords=_take(self.coords),
            block_coord_offset=_take(self.block_coord_offset),
            inter_residue_connections=_take(self.inter_residue_connections),
            inter_block_bondsep=(
                None
                if self._inter_block_bondsep is None
                else _take(self._inter_block_bondsep)
            ),
            near_blocks=None if self.near_blocks is None else _take(self.near_blocks),
            bondsep=None if self.bondsep is None else _take(self.bondsep),
            block_type_ind=_take(self.block_type_ind),
            pdb_info=None if self.pdb_info is None else self.pdb_info.take(indices),
            split_block_mapping=(
                None
                if self.split_block_mapping is None
                else self.split_block_mapping.take(indices)
            ),
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
