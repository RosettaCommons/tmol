"""Validate and order explicit molecular residue input on the host."""

import numpy as np


def prepare_residue_block_coordinates(
    blocks, active_block_types, *, leaves=None, build_missing_context=False
):
    """Return type indices, ordered coordinates and optional atom-presence masks.

    Coordinates accept NumPy-compatible array-like input. Preserve each block's
    named atom identity and reject missing atoms unless a completion plan allows
    them; this function does not construct a frontend pose or execute that plan.
    """
    complete_missing = leaves is not None
    type_indices = {rt.name: i for i, rt in enumerate(active_block_types)}
    rows, coordinates, availability = [], [], []
    for pose in blocks:
        kinds, xyz, masks = [], [], []
        for block in pose:
            if block.residue_type not in type_indices:
                raise ValueError(
                    f"residue type absent from context: {block.residue_type}"
                )
            kind = type_indices[block.residue_type]
            expected = [atom.name for atom in active_block_types[kind].atoms]
            names = list(block.atom_names)
            missing_names, extra_names = set(expected) - set(names), set(names) - set(
                expected
            )
            if (
                len(set(names)) != len(names)
                or extra_names
                or (missing_names and not complete_missing)
            ):
                raise ValueError(
                    f"atoms must match {block.residue_type} exactly; "
                    f"missing={sorted(set(expected) - set(names))}, "
                    f"extra={sorted(set(names) - set(expected))}, "
                    f"duplicates={len(names) != len(set(names))}"
                )
            coords = np.asarray(block.coords)
            if coords.shape != (len(names), 3) or not np.isfinite(coords).all():
                raise ValueError(
                    "block coordinates must be finite with shape [atoms, 3]"
                )
            indices = {name: i for i, name in enumerate(names)}
            if complete_missing:
                present = np.array([name in indices for name in expected], bool)
                nonleaf = ~present & ~leaves[kind, : len(expected)]
                if nonleaf.any() and not build_missing_context:
                    raise ValueError(
                        f"missing non-leaf atoms on {block.residue_type}: "
                        + ", ".join(np.asarray(expected)[nonleaf])
                    )
                packed = np.zeros((len(expected), 3), coords.dtype)
                packed[present] = coords[
                    [indices[name] for name in expected if name in indices]
                ]
                xyz.append(packed)
                masks.append(present)
            else:
                xyz.append(coords[[indices[name] for name in expected]])
            kinds.append(kind)
        rows.append(kinds)
        availability.append(masks)
        coordinates.append(np.concatenate(xyz) if xyz else np.empty((0, 3), np.float32))
    return rows, coordinates, availability
