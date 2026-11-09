"""Transport shared source annotations into the selected JAX pose layout."""

import attrs
import numpy as np


def canonical_atom_mappings(pose, ordering):
    """Map each real pose atom to the left-justified canonical identity."""
    import jax
    from tmol_core.io.canonical import _create_bt_to_canonical_atom_mapping

    pbt = pose.packed_block_types
    _, mapping = _create_bt_to_canonical_atom_mapping(ordering, pbt)
    types = np.asarray(pose.block_type_ind)
    local = np.arange(pbt.max_n_atoms)
    real = (types[..., None] >= 0) & (
        local < np.asarray(pbt.n_atoms)[np.maximum(types, 0), None]
    )
    p, b, a = np.nonzero(real)
    canonical = np.stack((p, b, mapping[types[p, b], a]), axis=-1)
    packed = np.stack((p, np.asarray(pose.block_coord_offset)[p, b] + a), axis=-1)
    return {
        name: jax.device_put(value.astype(np.int32), pbt.n_atoms.device)
        for name, value in (
            ("can_atom_mapping", canonical),
            ("ps_atom_mapping", packed),
        )
    }


def canonical_construction_result(
    pose,
    ordering,
    chains,
    real,
    missing,
    *,
    return_chain_ind,
    return_atom_mapping,
    return_block_has_missing_atoms,
):
    import jax

    details = {}
    if return_block_has_missing_atoms:
        details["block_has_missing_atoms"] = jax.device_put(missing, pose.device)
    if return_chain_ind:
        details["chain_ind"] = jax.device_put(
            np.where(real, chains, -1).astype(np.int32), pose.device
        )
    if return_atom_mapping:
        details.update(canonical_atom_mappings(pose, ordering))
    return (pose, details) if details else pose


def canonical_metadata_for_pose(pose, canonical, order, rows, atom_indices, authors):
    info = attrs.evolve(pose.pdb_info, chain_id=authors)
    width = pose.max_n_blocks
    for field, key in (
        ("residue_insertion_codes", "res_ins_codes"),
        ("metal_origins", "metal_origins"),
        ("residue_annotations", "residue_annotations"),
    ):
        source = canonical.get(key)
        if source is not None:
            source = np.asarray(source)
            if source.shape != order.shape:
                raise ValueError(f"{key} must have the canonical residue shape")
            setattr(
                info, field, np.take_along_axis(source, order, axis=1)[:, :width].copy()
            )
    offsets = np.asarray(pose.block_coord_offset)
    for field in ("atom_occupancy", "atom_b_factor"):
        source = canonical.get(field)
        if source is None:
            continue
        source = np.asarray(source)
        if source.ndim != 3 or source.shape[:2] != order.shape:
            raise ValueError(f"{field} must have the canonical atom shape")
        target = getattr(info, field).copy()
        for p, row in enumerate(rows):
            for b, kind in enumerate(row):
                mapping = atom_indices[kind]
                atoms = np.flatnonzero(mapping >= 0)
                target[p, offsets[p, b] + atoms] = source[
                    p, order[p, b], mapping[atoms]
                ]
        setattr(info, field, target)
    return info
