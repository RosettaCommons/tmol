"""Framework storage and validation around common missing-leaf reconstruction."""

import numpy as np


def complete_missing_atoms(pose, available, *, build_context=False, return_plan=False):
    """Arrange missing masks and execute the shared chemical reconstruction."""
    import jax
    from ._native import generate_leaf_atoms

    if all(mask.all() for row in available for mask in row):
        return (pose, ()) if return_plan else pose
    plans = ()
    pbt = pose.packed_block_types
    missing = np.zeros((*pose.block_type_ind.shape, pbt.max_n_atoms), bool)
    flat_missing = np.zeros(pose.coords.shape[:2], bool)
    offsets = np.asarray(pose.block_coord_offset)
    for p, row in enumerate(available):
        for r, mask in enumerate(row):
            missing[p, r, : len(mask)] = ~mask
            start = offsets[p, r]
            flat_missing[p, start : start + len(mask)] = ~mask
    if build_context:
        from ._context import prepare_context_completion

        coords, built, plans = prepare_context_completion(
            pose, flat_missing, flat_missing
        )
        pose = pose.replace(coords=coords)
        flat_missing &= ~built
        for p, row in enumerate(available):
            for r, mask in enumerate(row):
                start = offsets[p, r]
                missing[p, r, : len(mask)] &= ~built[p, start : start + len(mask)]
        if not flat_missing.any():
            return (pose, plans) if return_plan else pose
    metadata = leaf_execution_metadata(pose, missing, flat_missing)
    coords = generate_leaf_atoms(pose.coords, jax.device_put(metadata, pose.device))
    from ._hydrogen import reorient_rebuilt_hydrogens

    coords = reorient_rebuilt_hydrogens(pose, missing, coords)
    if not np.isfinite(np.asarray(coords)).all():
        raise ValueError(
            "missing atoms could not be reconstructed from available chemical frames"
        )
    result = pose.replace(coords=coords)
    return (result, plans) if return_plan else result


def leaf_execution_metadata(pose, missing, flat_missing):
    """Arrange common chemical frames for native coordinate reconstruction."""
    from tmol_core.chemical.packed import PackedBlockTypesData
    from tmol_core.io.missing_atoms import prepare_leaf_icoor_metadata

    pbt = pose.packed_block_types
    info = prepare_leaf_icoor_metadata(
        pbt.active_block_types, np.asarray(pbt.atom_is_hydrogen), pbt.max_n_atoms
    )
    downstream = PackedBlockTypesData.join_atom_downstream_of_conn(
        pbt.active_block_types
    )
    downstream = np.pad(
        downstream,
        ((0, 0), (0, pbt.max_n_conn - downstream.shape[1]), (0, 0)),
        constant_values=-1,
    )
    dtype = pose.coords.dtype
    return (
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
