"""Transport shared source annotations into the selected JAX pose layout."""

import attrs
import numpy as np


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
