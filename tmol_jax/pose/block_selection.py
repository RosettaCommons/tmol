"""Coordinate and topology gathers around the shared molecular selection plan."""

import copy

import attrs
import jax
import jax.numpy as jnp
import numpy as np

from tmol_core.pose import block_selection as common
from .pdb_info import PDBInfo


def _metadata(pose, plan=None, repeats=None):
    pdb = pose.pdb_info
    base = pdb._as_common()
    result = (
        common.select_pdb_info(base, plan)
        if plan is not None
        else common.repeat_pdb_info(base, repeats)
    )
    extra = {}
    for name in ("res_name", "atom_record_type"):
        value = getattr(pdb, name, None)
        if value is not None:
            value = (
                [[copy.deepcopy(row[b]) for b in plan.blocks] for row in value]
                if plan is not None
                else [copy.deepcopy(row) for _ in range(repeats) for row in value]
            )
        extra[name] = value
    return PDBInfo(**attrs.asdict(result, recurse=False), **extra)


def _restrict_bondsep(pose, blocks, device):
    # Preserve pre-existing paths through removed blocks, as the mature API does.
    if pose._inter_block_bondsep is not None or len(
        np.unique(blocks % pose.max_n_blocks)
    ) != len(blocks):
        from tmol_jax.score.nonbonded import _compact_connections

        dense = pose.inter_block_bondsep[:, blocks[:, None], blocks[None, :]]
        return jax.device_put(_compact_connections(dense), device)
    keep = jax.device_put(blocks.astype(np.int32), device)
    inverse = (
        jnp.full((pose.max_n_blocks,), -1, jnp.int32)
        .at[keep]
        .set(jnp.arange(len(blocks)))
    )
    old = pose.near_blocks[:, keep]
    mapped = inverse[jnp.maximum(old[..., 0], 0)]
    valid = (old[..., 0] >= 0) & (mapped >= 0)
    partner = jnp.where(valid, mapped, -1)
    order = jnp.argsort(jnp.where(valid, mapped, len(blocks)), axis=2, stable=True)
    near = jnp.stack((partner, jnp.where(valid, old[..., 1], 6)), axis=-1)
    distances = jnp.where(valid[..., None, None], pose.bondsep[:, keep], 6)
    return (
        jnp.take_along_axis(near, order[..., None], axis=2),
        jnp.take_along_axis(distances, order[..., None, None], axis=2),
    )


def select_blocks(pose_stack, blocks):
    """Select an ordered homogeneous block layout, preserving coordinate AD.

    Connections to omitted blocks clear; retained connections and metadata use
    the common plan. As in mature TMol, constraints are dropped, original bond
    separations are restricted, and split-block metadata is rejected.
    """
    pose, pbt = pose_stack, pose_stack.packed_block_types
    device = pbt.n_atoms.device
    plan = common.selection_plan(
        blocks,
        np.asarray(pose.block_type_ind),
        np.asarray(pose.block_coord_offset),
        np.asarray(pbt.n_atoms),
        np.asarray(pose.inter_residue_connections),
        split_block_mapping=pose.split_block_mapping,
    )
    with jax.default_device(device):
        keep, source, kinds, offsets, connections = jax.device_put(
            tuple(
                a.astype(np.int32)
                for a in (
                    plan.blocks,
                    plan.source_atoms,
                    plan.block_types,
                    plan.offsets,
                    plan.connections,
                )
            ),
            device,
        )
        near, bondsep = _restrict_bondsep(pose, plan.blocks, device)
        return pose.replace(
            coords=pose.coords[:, source],
            block_type_ind=kinds,
            block_coord_offset=offsets,
            inter_residue_connections=connections,
            inter_block_bondsep=None,
            near_blocks=near,
            bondsep=bondsep,
            chain_id=pose.chain_id[:, keep],
            pdb_info=_metadata(pose, plan=plan),
            constraint_set=None,
            split_block_mapping=None,
        )


def repeat_pose_stack(pose_stack, n_poses):
    """Repeat a single pose, retaining coordinate dtype, derivatives and constraints."""
    common.validate_repeat(pose_stack.n_poses, pose_stack.split_block_mapping)
    # The existing batch selection adapter already preserves every topology and
    # constraint field, including empty batches and repeated pose identities.
    result = pose_stack[np.zeros(n_poses, np.int32)]
    return result.replace(pdb_info=_metadata(pose_stack, repeats=n_poses))
