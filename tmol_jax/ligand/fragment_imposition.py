"""JAX pose storage around common fragment identity and connection plans."""

import jax
import numpy as np

from tmol_core.ligand import fragment_imposition as common


def apply_fragment_connections(pose, mapping):
    from tmol_jax.chemical.restypes import _refined_from_raw
    from tmol_jax.pose.packed_block_types import PackedBlockTypes
    from tmol_jax.pose.pose_stack_builder import _bond_separation

    pbt = pose.packed_block_types
    types = np.asarray(pose.block_type_ind)
    mapping = common.resolve_fragment_mapping(
        mapping,
        pose.pdb_info.residue_labels,
        types,
        pbt.active_block_types,
    )
    connections = common.connect_fragment_ports(
        mapping,
        types,
        pbt.active_block_types,
        np.asarray(pose.inter_residue_connections),
    )
    missing = common.missing_original_type_names(mapping, types, pbt.active_block_types)
    if missing:
        by_name = {r.name: r for r in pbt.chem_db.residues}
        for name in missing:
            if name not in by_name:
                raise ValueError(f"Original block type {name!r} not found in chemistry")
        with jax.default_device(pose.device):
            pbt = PackedBlockTypes.from_restypes(
                pbt.chem_db,
                (
                    *pbt.active_block_types,
                    *(_refined_from_raw(by_name[name]) for name in missing),
                ),
            )
        extra = pbt.max_n_conn - connections.shape[2]
        if extra > 0:
            connections = np.pad(
                connections, ((0, 0), (0, 0), (0, extra), (0, 0)), constant_values=-1
            )
    split = common.build_split_block_mapping(mapping, types, pbt.active_block_types)
    distances = _bond_separation(pbt, types, connections)
    return pose.replace(
        packed_block_types=pbt,
        inter_residue_connections=jax.device_put(connections, pose.device),
        inter_block_bondsep=None,
        near_blocks=jax.device_put(distances.near_blocks, pose.device),
        bondsep=jax.device_put(distances.bondsep, pose.device),
        split_block_mapping=split,
    )


def recombine_fragmented_ligands(structure, pose_stack):
    """Restore original author identities using the common fragment map."""
    return common.recombine_fragmented_ligands(
        structure,
        pose_stack.split_block_mapping,
        pose_stack.pdb_info.residue_labels,
        np.asarray(pose_stack.block_type_ind),
        pose_stack.packed_block_types.active_block_types,
    )


def unsplit_pose_stack(pose):
    """Restore original block layout without moving coordinates to the host."""
    import attrs
    from types import SimpleNamespace
    import jax.numpy as jnp
    from tmol_core.io.pdb_info import PDBInfo as CommonPDBInfo
    from tmol_core.ligand.fragment_reconstruction import (
        reconstruction_plan,
        remap_constraints,
    )
    from tmol_jax.pose.pdb_info import PDBInfo
    from tmol_jax.pose.pose_stack_builder import _bond_separation

    mapping = pose.split_block_mapping
    if mapping is None or not mapping.entries:
        return pose.replace(split_block_mapping=None)
    pbt = pose.packed_block_types
    device = getattr(pose.coords, "device", pbt.n_atoms.device)
    metadata = CommonPDBInfo(
        **{
            field.name: (
                None
                if getattr(pose.pdb_info, field.name) is None
                else np.asarray(getattr(pose.pdb_info, field.name))
            )
            for field in attrs.fields(CommonPDBInfo)
        }
    )
    types = np.asarray(pose.block_type_ind)
    chains = np.asarray(pose.chain_id)
    host = SimpleNamespace(
        block_type_ind64=types,
        block_coord_offset=np.asarray(pose.block_coord_offset),
        inter_residue_connections64=np.asarray(pose.inter_residue_connections),
        pdb_info=metadata,
        chain_id=chains,
    )
    palette = SimpleNamespace(
        n_atoms=np.asarray(pbt.n_atoms), active_block_types=pbt.active_block_types
    )
    plan = reconstruction_plan(host, palette, mapping)
    constraints = pose.constraint_set
    if constraints is not None:
        constraints = type(constraints)._from_topology(
            remap_constraints(constraints._topology(), plan.atom_destinations),
            constraints.constraint_params,
            constraints.constraint_functions,
            device,
        )
    distances = _bond_separation(pbt, plan.block_types, plan.connections)
    metadata = PDBInfo(
        **{
            field.name: getattr(plan.pdb_info, field.name)
            for field in attrs.fields(CommonPDBInfo)
        }
    )
    with jax.default_device(device):
        source = jnp.asarray(plan.coordinate_sources)
        coords = pose.coords[jnp.arange(len(pose))[:, None], jnp.maximum(source, 0)]
        coords = jnp.where((source >= 0)[..., None], coords, 0)
    return pose.replace(
        coords=coords,
        block_type_ind=jax.device_put(plan.block_types, device),
        chain_id=jax.device_put(plan.chain_ids, device),
        block_coord_offset=jax.device_put(plan.block_offsets, device),
        inter_residue_connections=jax.device_put(
            plan.connections.astype(np.int32), device
        ),
        inter_block_bondsep=None,
        near_blocks=jax.device_put(distances.near_blocks, device),
        bondsep=jax.device_put(distances.bondsep, device),
        pdb_info=metadata,
        split_block_mapping=None,
        constraint_set=constraints,
    )
