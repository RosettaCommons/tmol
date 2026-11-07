"""Immutable pose coordinates and topology as a JAX pytree."""

from __future__ import annotations

from typing import Any, Optional
from functools import partial

import attr
import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.types import pytree_dataclass


@partial(jax.jit, static_argnames=("capacity",))
def _atom_is_real(type_atom_counts, block_types, offsets, capacity):
    """Compile the existing interval mask as one device computation."""
    n_atoms = type_atom_counts[jnp.maximum(block_types, 0)]
    real = block_types >= 0
    counts = jnp.where(real, n_atoms, 0)
    starts = jnp.maximum(offsets, 0)
    pose = jnp.arange(block_types.shape[0])[:, None]
    edges = jnp.zeros((block_types.shape[0], capacity + 1), jnp.int32)
    edges = edges.at[pose, starts].add(real.astype(jnp.int32))
    edges = edges.at[pose, starts + counts].add(-real.astype(jnp.int32))
    return jnp.cumsum(edges[:, :-1], axis=1) > 0


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

    _chain_id: Optional[jax.Array] = attr.field(default=None, alias="chain_id")

    # Properties

    @property
    def chain_id(self):
        """Logical chain indices, distinct from author chain labels.

        Builders preserve explicit logical chains. Directly constructed poses
        without them use the shared physical polymer-connection policy eagerly.
        """
        if self._chain_id is not None:
            return self._chain_id
        from tmol_core.io.chain_deduction import chain_indices, connection_masks

        pbt = self.packed_block_types
        chains = chain_indices(
            np.asarray(self.block_type_ind),
            np.asarray(self.inter_residue_connections),
            connection_masks(pbt.active_block_types, pbt.max_n_conn),
        )
        return jax.device_put(chains.astype(np.int32), pbt.n_atoms.device)

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
    def max_n_block_atoms(self) -> int:
        return self.max_n_atoms_per_block

    @property
    def max_n_pose_atoms(self) -> int:
        return self.max_n_atoms_per_pose

    @property
    def n_ats_per_block(self):
        counts = self.packed_block_types.n_atoms[jnp.maximum(self.block_type_ind, 0)]
        return jnp.where(self.block_type_ind >= 0, counts, 0)

    @property
    def real_atoms(self):
        return self.atom_is_real

    @property
    def n_res_per_pose(self):
        return jnp.sum(self.block_type_ind >= 0, axis=1)

    def is_real_block(self, pose_ind: int, block_ind: int):
        return self.block_type_ind[pose_ind, block_ind] >= 0

    def get_constraint_set(self):
        return self.constraint_set

    def block_identity_map(self):
        return jnp.broadcast_to(
            jnp.arange(
                self.max_n_blocks,
                dtype=self.block_coord_offset.dtype,
                device=getattr(self.coords, "device", None),
            ),
            self.block_coord_offset.shape,
        )

    def expand_coords(self):
        """Return residue-major coordinates and a mask, preserving gradients."""
        atom = jnp.arange(
            self.max_n_block_atoms, device=getattr(self.coords, "device", None)
        )
        real = atom < self.n_ats_per_block[..., None]
        if not self.max_n_pose_atoms:
            return (
                jnp.zeros(
                    real.shape + (3,),
                    self.coords.dtype,
                    device=getattr(self.coords, "device", None),
                ),
                real,
            )
        indices = jnp.where(real, self.block_coord_offset[..., None] + atom, 0)
        coords = self.coords[jnp.arange(self.n_poses)[:, None, None], indices]
        return jnp.where(real[..., None], coords, 0), real

    @property
    def atom_is_real(self):
        """Mask packed atoms by topology, independent of their positions."""
        return _atom_is_real(
            self.packed_block_types.n_atoms,
            self.block_type_ind,
            self.block_coord_offset,
            self.coords.shape[1],
        )

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
            chain_id=None if self._chain_id is None else _take(self._chain_id),
            pdb_info=None if self.pdb_info is None else self.pdb_info.take(indices),
            split_block_mapping=(
                None
                if self.split_block_mapping is None
                else self.split_block_mapping.take(indices)
            ),
        )

    def split(self, index: int) -> "PoseStack":
        """Select one pose, retaining its constraints and input metadata."""
        return self[index]

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

    def clone_sharing_topology(self) -> "PoseStack":
        """Share immutable arrays, as with :meth:`clone`."""
        return self.clone()

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
            int(np.asarray(self.block_type_ind)[pose_ind, block_ind])
        ]

    # Constructors


__all__ = [
    "PoseStack",
]
