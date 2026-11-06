"""Framework storage and validation around common missing-leaf reconstruction."""

import numpy as np


def complete_missing_atoms(pose, available):
    """Arrange missing masks and execute the shared chemical reconstruction."""
    import jax
    from tmol_core.chemical.packed import PackedBlockTypesData
    from tmol_core.io.missing_atoms import prepare_leaf_atom_metadata
    from ._native import generate_leaf_atoms

    if all(mask.all() for row in available for mask in row):
        return pose
    pbt = pose.packed_block_types
    missing = np.zeros((*pose.block_type_ind.shape, pbt.max_n_atoms), bool)
    flat_missing = np.zeros(pose.coords.shape[:2], bool)
    offsets = np.asarray(pose.block_coord_offset)
    for p, row in enumerate(available):
        for r, mask in enumerate(row):
            missing[p, r, : len(mask)] = ~mask
            start = offsets[p, r]
            flat_missing[p, start : start + len(mask)] = ~mask
    info = prepare_leaf_atom_metadata(
        pbt.active_block_types, np.asarray(pbt.atom_is_hydrogen), pbt.max_n_atoms
    ).icoors
    downstream = PackedBlockTypesData.join_atom_downstream_of_conn(
        pbt.active_block_types
    )
    downstream = np.pad(
        downstream,
        ((0, 0), (0, pbt.max_n_conn - downstream.shape[1]), (0, 0)),
        constant_values=-1,
    )
    dtype = pose.coords.dtype
    metadata = (
        missing,
        flat_missing,
        pose.block_coord_offset,
        pose.block_type_ind,
        pose.inter_residue_connections,
        pbt.n_atoms,
        downstream,
        info.anc_uaids,
        info.geom.astype(dtype),
        info.anc_uaids_backup,
        info.geom_backup.astype(dtype),
    )
    coords = generate_leaf_atoms(
        pose.coords, tuple(jax.device_put(a, pose.device) for a in metadata)
    )
    if not np.isfinite(np.asarray(coords)).all():
        raise ValueError(
            "missing atoms could not be reconstructed from available chemical frames"
        )
    return pose.replace(coords=coords)
