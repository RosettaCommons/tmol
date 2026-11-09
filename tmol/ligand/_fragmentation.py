"""Torch pose assembly over shared ligand fragment chemistry."""

from tmol_core.ligand import _fragmentation as _core
from tmol_core.ligand._fragmentation import FragmentedLigandPoseMapping


def __getattr__(name):
    return getattr(_core, name)


def _apply_fragment_connections(pose_stack, mapping: FragmentedLigandPoseMapping):
    """Execute shared fragment identity and named bond plans on a Torch pose."""
    import attr
    import torch
    from tmol.pose._pose_stack_builder import PoseStackBuilder
    from tmol_core.ligand import fragment_imposition as common

    pbt = pose_stack.packed_block_types
    types = pose_stack.block_type_ind64.cpu().numpy()
    mapping = common.resolve_fragment_mapping(
        mapping, pose_stack.pdb_info.residue_labels, types, pbt.active_block_types
    )
    if mapping.connection_pairs:
        connections = common.connect_fragment_ports(
            mapping,
            types,
            pbt.active_block_types,
            pose_stack.inter_residue_connections64.cpu().numpy(),
        )
        connections = torch.as_tensor(connections, device=pose_stack.device)
        distances = PoseStackBuilder._inter_block_bondsep_from_connections(
            pbt,
            pose_stack.block_type_ind64,
            pose_stack.block_type_ind64 >= 0,
            connections,
        )
        pose_stack = attr.evolve(
            pose_stack,
            coords=pose_stack.coords.clone(),
            inter_residue_connections=connections.to(torch.int32),
            inter_residue_connections64=connections,
            inter_block_bondsep=distances,
        )
    pose_stack, mapping = _build_split_block_mapping(pose_stack, mapping)
    return attr.evolve(pose_stack, split_block_mapping=mapping)


def _build_split_block_mapping(pose_stack, resolved_mapping):
    """Extend the Torch palette and bind the shared original-atom map."""
    import attr
    from tmol.pose._packed_block_types import PackedBlockTypes
    from tmol_core.ligand import fragment_imposition as common

    pbt = pose_stack.packed_block_types
    types = pose_stack.block_type_ind64.cpu().numpy()
    missing = common.missing_original_type_names(
        resolved_mapping, types, pbt.active_block_types
    )
    if missing:
        by_name = {r.name: r for r in pbt.restype_set.residue_types}
        for name in missing:
            if name not in by_name:
                raise ValueError(
                    f"Original block type {name!r} not found in ResidueTypeSet"
                )
        extended = list(pbt.active_block_types) + [by_name[name] for name in missing]
        pbt = PackedBlockTypes.from_restype_list(
            pbt.chem_db, pbt.restype_set, extended, pose_stack.device
        )
        pose_stack = attr.evolve(pose_stack, packed_block_types=pbt)
    return pose_stack, common.build_split_block_mapping(
        resolved_mapping, types, pbt.active_block_types
    )


def _host_reconstruction_pose(pose):
    from types import SimpleNamespace

    values = {
        name: getattr(pose, name).cpu().numpy()
        for name in (
            "block_type_ind64",
            "block_coord_offset",
            "inter_residue_connections64",
            "chain_id",
        )
        if hasattr(pose, name)
    }
    values["pdb_info"] = getattr(pose, "pdb_info", None)
    return SimpleNamespace(**values)


def _host_reconstruction_palette(pbt):
    from types import SimpleNamespace

    return SimpleNamespace(
        n_atoms=pbt.n_atoms.cpu().numpy(), active_block_types=pbt.active_block_types
    )


def _unsplit_group_entries(sbm):
    from tmol_core.ligand.fragment_reconstruction import _unsplit_group_entries

    return _unsplit_group_entries(sbm)


def _unsplit_per_pose_blocks(pose_stack, split_block_set, entry_lookup):
    from tmol_core.ligand.fragment_reconstruction import _unsplit_per_pose_blocks

    return _unsplit_per_pose_blocks(
        _host_reconstruction_pose(pose_stack), split_block_set, entry_lookup
    )


def _unsplit_build_coords(
    per_pose_blocks, pbt, groups, pose_stack, new_bco, new_max_n_atoms, n_poses
):
    from tmol_core.ligand.fragment_reconstruction import _unsplit_build_coords

    pose = _host_reconstruction_pose(pose_stack)
    pose.coords = pose_stack.coords.detach().cpu().numpy()
    return _unsplit_build_coords(
        per_pose_blocks,
        _host_reconstruction_palette(pbt),
        groups,
        pose,
        new_bco.cpu().numpy(),
        new_max_n_atoms,
        n_poses,
    )


def _unsplit_old_to_new(per_pose_blocks, pose_stack, split_block_set, entry_lookup):
    from tmol_core.ligand.fragment_reconstruction import _unsplit_old_to_new

    return _unsplit_old_to_new(
        per_pose_blocks,
        _host_reconstruction_pose(pose_stack),
        split_block_set,
        entry_lookup,
    )


def _unsplit_connections(
    pose_stack,
    pbt,
    n_poses,
    new_max_n_blocks,
    split_block_set,
    entry_lookup,
    old_to_new,
    device,
):
    import torch
    from tmol_core.ligand.fragment_reconstruction import _unsplit_connections

    return torch.as_tensor(
        _unsplit_connections(
            _host_reconstruction_pose(pose_stack),
            _host_reconstruction_palette(pbt),
            n_poses,
            new_max_n_blocks,
            split_block_set,
            entry_lookup,
            old_to_new,
            None,
        ),
        device=device,
    )


def _unsplit_chain_and_pdb(
    pose_stack,
    old_to_new,
    pbt,
    per_pose_blocks,
    new_bco_np,
    n_poses,
    new_max_n_blocks,
    new_max_n_atoms,
    device,
):
    import torch
    from tmol_core.ligand.fragment_reconstruction import _unsplit_chain_and_pdb

    chain, pdb = _unsplit_chain_and_pdb(
        _host_reconstruction_pose(pose_stack),
        old_to_new,
        _host_reconstruction_palette(pbt),
        per_pose_blocks,
        new_bco_np,
        n_poses,
        new_max_n_blocks,
        new_max_n_atoms,
        None,
    )
    return torch.as_tensor(chain, device=device), pdb


def _unsplit_pose_stack(pose_stack):
    """Reconstruct original blocks with a shared plan and differentiable gather."""
    import attr
    import torch
    from tmol.pose._pose_stack_builder import PoseStackBuilder
    from tmol_core.ligand.fragment_reconstruction import (
        reconstruction_plan,
        remap_constraints,
    )

    mapping = pose_stack.split_block_mapping
    if mapping is None or not mapping.entries:
        return attr.evolve(pose_stack, split_block_mapping=None)
    plan = reconstruction_plan(
        _host_reconstruction_pose(pose_stack),
        _host_reconstruction_palette(pose_stack.packed_block_types),
        mapping,
    )
    constraints = pose_stack.constraint_set
    if constraints is not None:
        constraints = type(constraints)._from_topology(
            remap_constraints(constraints._topology(), plan.atom_destinations),
            constraints.constraint_params,
            constraints.constraint_functions,
            pose_stack.device,
        )

    def tensor(value):
        return torch.as_tensor(value, device=pose_stack.device)

    source = tensor(plan.coordinate_sources)
    coords = pose_stack.coords[
        torch.arange(len(pose_stack), device=pose_stack.device)[:, None],
        source.clamp_min(0),
    ]
    coords = torch.where((source >= 0)[..., None], coords, 0)
    types = tensor(plan.block_types).to(torch.int64)
    connections = tensor(plan.connections).to(torch.int64)
    offsets = tensor(plan.block_offsets)
    chains = tensor(plan.chain_ids)
    distances = PoseStackBuilder._inter_block_bondsep_from_connections(
        pose_stack.packed_block_types, types, types >= 0, connections
    )
    return attr.evolve(
        pose_stack,
        coords=coords,
        block_type_ind=types.to(torch.int32),
        block_type_ind64=types,
        block_coord_offset=offsets.to(torch.int32),
        block_coord_offset64=offsets.to(torch.int64),
        inter_residue_connections=connections.to(torch.int32),
        inter_residue_connections64=connections,
        inter_block_bondsep=distances,
        chain_id=chains.to(torch.int32),
        chain_id64=chains.to(torch.int64),
        pdb_info=plan.pdb_info,
        split_block_mapping=None,
        constraint_set=constraints,
    )


def recombine_fragmented_ligands(structure, pose_stack):
    """Restore original author identities using the common fragment map."""
    from tmol_core.ligand.fragment_imposition import (
        recombine_fragmented_ligands as restore,
    )

    return restore(
        structure,
        pose_stack.split_block_mapping,
        pose_stack.pdb_info.residue_labels,
        pose_stack.block_type_ind.cpu().numpy(),
        pose_stack.packed_block_types.active_block_types,
    )
