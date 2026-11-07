"""Framework storage and coordinate execution for shared metal input plans."""

import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def execute_metal_placement(coords, rows, offset_bytes, wide):
    """Execute fixed site fits, retaining metal-translation coordinate derivatives."""
    parent = coords[rows[:, 0], rows[:, 2]]
    # Clusters add in float64 before casting; ions cast the offset first. Keep
    # that existing rounding contract even with public JAX x64 mode disabled.
    with jax.enable_x64(True):
        offsets = jax.lax.bitcast_convert_type(offset_bytes, jnp.float64)
        widened = (parent.astype(jnp.float64) + offsets).astype(coords.dtype)
        ordinary = parent + offsets.astype(coords.dtype)
    values = jnp.where(wide[:, None], widened, ordinary)
    return coords.at[rows[:, 0], rows[:, 1]].set(values)


def prepare_metal_completion(pose, available, assignments, *, return_plan=False):
    """Execute common virtual-site placements and update per-residue availability."""
    from tmol_core.io.metal_detection import metal_virtual_placements

    if not assignments:
        return (pose, available, None) if return_plan else (pose, available)
    pbt = pose.packed_block_types
    types, offsets = map(np.asarray, (pose.block_type_ind, pose.block_coord_offset))
    missing = np.ones((*types.shape, pbt.max_n_atoms), bool)
    for p, row in enumerate(available):
        for r, mask in enumerate(row):
            missing[p, r, : len(mask)] = ~mask
    placements = list(
        metal_virtual_placements(pbt.active_block_types, types, missing, assignments)
    )
    if not placements:
        return (pose, available, None) if return_plan else (pose, available)
    rows, shifts, wide = [], [], []
    updated = [[mask.copy() for mask in row] for row in available]
    for p, r, atom, metal, shift, widened in placements:
        rows.append((p, offsets[p, r] + atom, offsets[p, r] + metal))
        shifts.append(shift)
        wide.append(widened)
        updated[p][r][atom] = True
    shifts = np.asarray(shifts, np.float64)
    data = jax.device_put(
        (
            np.asarray(rows, np.int32),
            shifts.view(np.uint8).reshape((-1, 3, 8)),
            np.asarray(wide, bool),
        ),
        pose.device,
    )
    coords = execute_metal_placement(pose.coords, *data)
    result = pose.replace(coords=coords)
    return (result, updated, data) if return_plan else (result, updated)


def grow_palette(pbt, co, patches, metal_classes):
    """Materialize common patched chemistry as a framework-owned residue palette."""
    from tmol_core.cache import AnnotationKey
    from tmol_jax.chemical.restypes import _refined_from_raw
    from tmol_jax.pose.packed_block_types import PackedBlockTypes

    active_classes = {t.io_equiv_class for t in pbt.active_block_types}
    known = {patch.name for patch in pbt.chem_db.variants}
    extra = tuple(patch for patch in patches if patch.name not in known)
    if not extra and set(metal_classes) <= active_classes:
        return pbt
    key = AnnotationKey.from_sources(
        co,
        pbt.chem_db,
        *pbt.active_block_types,
        settings=(
            tuple(p.name for p in extra),
            tuple(sorted(metal_classes)),
            pbt.n_atoms.device,
        ),
    )
    cached = getattr(pbt, "_canonical_grown_palette", None)
    if cached is not None and cached[0].matches(key):
        return cached[1]
    chemistry = pbt.chem_db.with_variants_applied(extra)
    with jax.default_device(pbt.n_atoms.device):
        expanded = PackedBlockTypes.from_restypes(
            chemistry, tuple(_refined_from_raw(t) for t in chemistry.residues)
        )
    object.__setattr__(pbt, "_canonical_grown_palette", (key, expanded))
    return expanded
