"""JAX buffer assembly over shared pose concatenation plans."""

import copy
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from tmol_core.pose import concatenation as common
from tmol_jax.chemical.constants import MAX_SIG_BOND_SEPARATION
from .constraint_set import ConstraintSet
from .packed_block_types import PackedBlockTypes
from .pdb_info import PDBInfo
from .pose_stack import PoseStack


def _stack(values, shape, fill=0):
    return jnp.concatenate(
        [
            jnp.pad(
                value,
                ((0, 0), *((0, n - d) for n, d in zip(shape, value.shape[1:]))),
                constant_values=fill,
            )
            for value in values
        ],
        axis=0,
    )


def _bondsep(poses, blocks, ports):
    if any(p._inter_block_bondsep is not None for p in poses):
        return dict(
            inter_block_bondsep=_stack(
                [p.inter_block_bondsep for p in poses],
                (blocks, blocks, ports, ports),
                MAX_SIG_BOND_SEPARATION,
            )
        )
    slots = max(p.near_blocks.shape[2] for p in poses)
    partners = _stack([p.near_blocks[..., 0] for p in poses], (blocks, slots), -1)
    distances = _stack(
        [p.near_blocks[..., 1] for p in poses], (blocks, slots), MAX_SIG_BOND_SEPARATION
    )
    return dict(
        inter_block_bondsep=None,
        near_blocks=jnp.stack((partners, distances), axis=-1),
        bondsep=_stack(
            [p.bondsep for p in poses],
            (blocks, slots, ports, ports),
            MAX_SIG_BOND_SEPARATION,
        ),
    )


def _metadata(poses, layout):
    # JAX's public label alias also accepts ragged Python rows. Present the
    # common metadata planner with its rectangular host-array contract.
    views = []
    for pose in poses:
        info = pose.pdb_info._as_common()
        views.append(
            SimpleNamespace(pdb_info=info, n_poses=pose.n_poses, coords=pose.coords)
        )
    return common.concatenate_pdb_info(
        views, layout.n_poses, layout.pose_offsets, layout.max_n_blocks
    )


def from_poses(pose_stacks, device=None):
    """Preserve chemistry, constraints and source derivatives in a padded batch."""
    from tmol_jax.io.structure import _device

    poses = list(pose_stacks)
    if not poses:
        raise ValueError("pose_stacks must contain at least one pose stack")
    device = _device(
        device if device is not None else poses[0].packed_block_types.n_atoms.device
    )
    poses = [
        (
            p
            if p.packed_block_types.n_atoms.device == device
            else jax.device_put(p, device)
        )
        for p in poses
    ]
    widest = common.widest_packed_block_types(poses)
    with jax.default_device(device):
        pbt = (
            widest
            if widest.n_atoms.device == device
            and all(
                common.is_leading_run(
                    p.packed_block_types.active_block_types, widest.active_block_types
                )
                for p in poses
            )
            else PackedBlockTypes.from_restypes(
                widest.chem_db, common.merged_block_types(poses)
            )
        )
        layout = common.concatenation_layout(poses)
        by_name = {bt.name: i for i, bt in enumerate(pbt.active_block_types)}
        types = []
        for pose in poses:
            mapping = np.asarray(
                [by_name[t.name] for t in pose.packed_block_types.active_block_types],
                np.int32,
            )
            types.append(
                jnp.where(
                    pose.block_type_ind >= 0,
                    jax.device_put(mapping, device)[
                        jnp.maximum(pose.block_type_ind, 0)
                    ],
                    -1,
                )
            )
        fields = _metadata(poses, layout)
        labels = {
            name: copy.deepcopy(
                [row for pose in poses for row in getattr(pose.pdb_info, name)]
            )
            for name in ("res_name", "atom_record_type")
        }
        constraints = ConstraintSet.concatenate(
            [
                (
                    None
                    if p.constraint_set is None
                    else (
                        p.constraint_set
                        if p.constraint_set.device == device
                        else jax.device_put(p.constraint_set, device)
                    )
                )
                for p in poses
            ],
            n_poses=layout.n_poses,
            ps_offset=layout.pose_offsets,
        )
        result = PoseStack(
            packed_block_types=pbt,
            coords=_stack([p.coords for p in poses], (layout.max_n_atoms, 3)),
            block_coord_offset=_stack(
                [p.block_coord_offset for p in poses], (layout.max_n_blocks,)
            ),
            block_type_ind=_stack(types, (layout.max_n_blocks,), -1),
            chain_id=_stack([p.chain_id for p in poses], (layout.max_n_blocks,), -1),
            inter_residue_connections=_stack(
                [p.inter_residue_connections for p in poses],
                (layout.max_n_blocks, pbt.max_n_conn, 2),
                -1,
            ),
            pdb_info=PDBInfo(**fields, **labels),
            constraint_set=constraints,
            split_block_mapping=common.concatenate_split_mapping(pbt, poses),
            **_bondsep(poses, layout.max_n_blocks, pbt.max_n_conn),
        )
        # Arrays were assembled on this device above. Rebuilding the pytree
        # here would replace the shared PackedBlockTypes object unnecessarily.
        return result
