"""Build molecular poses through shared canonical chemistry and native policies."""

from __future__ import annotations
from typing import Any, Dict, List
import numpy as np


def _normalise_res_names(
    cf: Dict[str, Any],
    n_poses: int,
    max_n_res: int,
) -> List[List[str]]:
    """Pull per-residue 3-letter names from the canonical-form dict.

    Prefers ``"res_name"`` / ``"res_names"`` when provided. Falls back to
    resolving ``"res_types"`` (integer indices) through a
    :class:`CanonicalOrdering` stored under ``"canonical_ordering"``.
    Padding positions come back as the empty string.
    """

    if "res_name" in cf or "res_names" in cf:
        raw = cf.get("res_name", cf.get("res_names"))
        if isinstance(raw, np.ndarray):
            res_names = [
                [str(x) if x is not None else "" for x in row] for row in raw.tolist()
            ]
        else:
            res_names = [[str(x) if x is not None else "" for x in row] for row in raw]
        # Pad rows to max_n_res.
        for row in res_names:
            while len(row) < max_n_res:
                row.append("")
        return res_names

    if "res_types" in cf:
        types = np.asarray(cf["res_types"])
        co = cf.get("canonical_ordering", None)
        if co is None or not hasattr(co, "restype_io_equiv_classes"):
            raise ValueError(
                "pose_stack_from_canonical_form: got res_types but no "
                "canonical_ordering to resolve them. Either pass "
                "'res_name' directly or include 'canonical_ordering'."
            )
        classes = list(co.restype_io_equiv_classes)
        out: List[List[str]] = []
        for p in range(n_poses):
            row: List[str] = []
            for r in range(max_n_res):
                idx = int(types[p, r])
                row.append(classes[idx] if 0 <= idx < len(classes) else "")
            out.append(row)
        return out

    raise ValueError(
        "pose_stack_from_canonical_form: canonical_form_dict must include "
        "'res_name' (list[list[str]]) or 'res_types' (+ 'canonical_ordering')."
    )


def _canonical_metadata(pbt, co):
    """Retain one preparation for live immutable chemistry and ordering sources."""
    from tmol_core.cache import AnnotationKey
    from tmol_core.io.canonical import prepare_canonical_selection
    from tmol_core.io.missing_atoms import prepare_leaf_atom_flags

    key = AnnotationKey.from_sources(
        co, pbt.chem_db, *pbt.active_block_types, settings=(pbt.max_n_atoms,)
    )
    cached = getattr(pbt, "_canonical_input_metadata", None)
    if cached is not None and cached[0].matches(key):
        return cached[1:]
    prepared = prepare_canonical_selection(co, pbt.active_block_types, pbt.max_n_atoms)
    leaves = prepare_leaf_atom_flags(
        pbt.active_block_types, np.asarray(pbt.atom_is_hydrogen), pbt.max_n_atoms
    ).is_leaf_atom
    object.__setattr__(pbt, "_canonical_input_metadata", (key, prepared, leaves))
    return prepared, leaves


def _canonical_rows(value, shape, default, name):
    array = np.asarray(default if value is None else value, dtype=object)
    if array.ndim == 1:
        array = array[None, :]
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}")
    return array


def _map_residue_atoms(residue, source, present, mapping, leaf, allow_nonleaf=False):
    mapping = mapping[: residue.n_atoms]
    valid = (mapping >= 0) & (mapping < source.shape[0])
    safe = np.where(valid, mapping, 0)
    available = valid & present[safe]
    coords = np.where(available[:, None], source[safe], 0)
    if not allow_nonleaf:
        _require_rebuildable(residue, available, leaf)
    return coords, available


def _require_rebuildable(residue, available, leaf):
    missing = ~available & ~leaf[: residue.n_atoms]
    if missing.any():
        atoms = ", ".join(
            residue.atoms[index].name for index in np.flatnonzero(missing)
        )
        raise ValueError(f"missing non-leaf atoms on {residue.name}: {atoms}")


def _compact_palette(pbt, used):
    from tmol_core.cache import AnnotationKey
    from tmol_jax.pose.packed_block_types import PackedBlockTypes
    import jax

    types = tuple(pbt.active_block_types[k] for k in used)
    device = jax.config.jax_default_device or jax.devices()[0]
    key = AnnotationKey.from_sources(pbt.chem_db, *types, settings=(device,))
    cached = getattr(pbt, "_canonical_compact_palette", None)
    if cached is not None and cached[0].matches(key):
        return cached[1]
    compact = PackedBlockTypes.from_restypes(pbt.chem_db, types)
    object.__setattr__(pbt, "_canonical_compact_palette", (key, compact))
    return compact


def _canonical_coordinates(cf, co):
    import jax

    coords = np.asarray(cf["coords"])
    if coords.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError("canonical coordinates require float32 or float64")
    if coords.dtype == np.dtype("float64") and not jax.config.x64_enabled:
        raise ValueError("float64 coordinates require jax_enable_x64=True")
    if coords.ndim != 4 or coords.shape[-1] != 3 or min(coords.shape[:3]) == 0:
        raise ValueError("coords must have nonempty shape [poses, residues, atoms, 3]")
    shape = coords.shape[:2]
    present = np.asarray(
        cf.get("atom_is_present", np.isfinite(coords).all(axis=-1)), bool
    )
    if present.shape != coords.shape[:-1] or np.any(
        present & ~np.isfinite(coords).all(axis=-1)
    ):
        raise ValueError("atom_is_present must match finite canonical coordinates")
    if "res_types" in cf:
        source_types = np.asarray(cf["res_types"])
        if (
            source_types.shape != shape
            or source_types.dtype.kind not in "iu"
            or np.any(source_types < -1)
            or np.any(source_types >= co.n_restype_io_equiv_classes)
        ):
            raise ValueError("res_types must contain canonical indices or -1 padding")
    if coords.shape[2] != co.max_n_canonical_atoms:
        raise ValueError("canonical atom extent must match the supplied ordering")
    return coords, present


def pose_stack_from_canonical_form(
    canonical_form_dict,
    packed_block_types,
    chain_id=None,
    res_num=None,
    *,
    compact_padding=False,
    build_missing_context=False,
    ligand_hydrogens=None,
    require_all_metal_donors=None,
):
    """Select chemical variants and build a pose from canonical atom identities.

    The palette and canonical ordering may describe noncanonical polymers,
    ligands and mixed batches. Missing leaf atoms use shared construction frames;
    missing non-leaf atoms raise unless ``build_missing_context=True`` selects
    the shared anchored context reconstruction. Explicit disulfides, cycles and
    covalent and metal connections use common chemistry and port validation.
    Metal donors grow the palette; protonation uses observed hydrogen evidence
    and optional explicit states. Missing metal sites use shared placement plans.

    ``res_not_connected`` retains the JAX input convention of a terminated chain
    break; ``res_gap`` suppresses bonds without forcing terminal chemistry.
    Interior padding breaks chains by default. Set ``compact_padding=True`` to
    remove virtual slots without breaking adjacency, matching Torch canonical input.
    Complete input preparation is a host boundary, not a jittable operation.
    """
    import attrs
    from .canonical_ordering import default_canonical_ordering
    from ._selection import select_and_connect
    from ._missing import complete_missing_atoms
    from tmol_jax.pose.pose_stack_builder import PoseStackBuilder

    from tmol_core.io.missing_atoms import validate_ligand_hydrogens
    from ._hydrogen import apply_input_policy

    cf = dict(canonical_form_dict)
    if require_all_metal_donors is not None:
        cf["require_all_metal_donors"] = require_all_metal_donors
    mode = (
        cf.get("ligand_hydrogens", "input")
        if ligand_hydrogens is None
        else ligand_hydrogens
    )
    validate_ligand_hydrogens(mode)
    pbt = packed_block_types
    co = cf.get("canonical_ordering") or default_canonical_ordering()
    coords, present = _canonical_coordinates(cf, co)
    shape = coords.shape[:2]
    names = _normalise_res_names(dict(cf, canonical_ordering=co), *shape)
    if len(names) != shape[0] or any(len(row) != shape[1] for row in names):
        raise ValueError("residue names must match the canonical coordinate shape")
    classes = {name: i for i, name in enumerate(co.restype_io_equiv_classes)}
    unknown = {name for row in names for name in row if name and name not in classes}
    if unknown:
        raise NotImplementedError("unsupported residues: " + ", ".join(sorted(unknown)))
    types = np.array(
        [[classes.get(name, -1) for name in row] for row in names], np.int32
    )
    if "res_types" in cf and not np.array_equal(types, np.asarray(cf["res_types"])):
        raise ValueError("canonical residue names and res_types disagree")
    real = types >= 0
    if np.any(~real.any(axis=1)):
        raise ValueError("each pose must contain at least one molecular residue")
    chains = _canonical_rows(
        cf.get("chain_id") if cf.get("chain_id") is not None else chain_id,
        shape,
        np.zeros(shape, np.int32),
        "chain_id",
    )
    labels = _canonical_rows(cf.get("chain_labels"), shape, chains, "chain_labels")
    numbers = _canonical_rows(
        cf.get("res_num") if cf.get("res_num") is not None else res_num,
        shape,
        np.broadcast_to(np.arange(shape[1]) + 1, shape),
        "res_num",
    )
    disconnected = np.asarray(
        cf.get("res_not_connected", np.zeros((*shape, 2), bool)), bool
    )
    gaps = np.asarray(cf.get("res_gap", np.zeros((*shape, 2), bool)), bool)
    if disconnected.shape != (*shape, 2) or gaps.shape != (*shape, 2):
        raise ValueError("res_not_connected and res_gap require [poses, residues, 2]")
    # Stable layout compaction preserves source atom and residue identities.
    order = np.argsort(~real, axis=1, kind="stable")
    inverse = np.argsort(order, axis=1)
    for key, width, endpoints in (
        ("disulfides", 3, (1, 2)),
        ("cyclic_bonds", 3, (1, 2)),
        ("covalent_bonds", 5, (1, 3)),
        ("metal_connections", 5, (1, 3)),
        ("metal_sites", 3, (1,)),
        ("metal_coordination", 5, (1, 3)),
    ):
        if cf.get(key) is not None:
            from ._selection import _rows

            rows = _rows(cf, key, width, shape, endpoints).copy()
            for endpoint in endpoints:
                if np.any(~real[rows[:, 0], rows[:, endpoint]]):
                    raise ValueError(f"{key} cannot refer to residue padding")
                rows[:, endpoint] = inverse[rows[:, 0], rows[:, endpoint]]
            cf[key] = rows

    def gather(array):
        index = order[(...,) + (None,) * (array.ndim - 2)]
        return np.take_along_axis(array, index, axis=1)

    if cf.get("protonation_variants") is not None:
        state = np.asarray(cf["protonation_variants"])
        if (
            state.shape != shape
            or state.dtype.kind not in "iu"
            or np.any(state > np.iinfo(np.int32).max)
            or (state.dtype.kind == "i" and np.any(state < -1))
        ):
            raise ValueError(
                "protonation_variants requires int32 states or -1 with the residue shape"
            )
        cf["protonation_variants"] = gather(state).astype(np.int32)

    types, coords, present, chains, labels, numbers, disconnected, gaps = map(
        gather, (types, coords, present, chains, labels, numbers, disconnected, gaps)
    )
    real = types >= 0
    # Named explicit breaks are input labels; common chemistry determines which
    # resulting boundaries actually carry polymer terminal variants.
    changed = (
        (chains[:, 1:] != chains[:, :-1])
        | disconnected[:, :-1, 1]
        | disconnected[:, 1:, 0]
    )
    if not compact_padding:
        changed |= np.diff(order, axis=1) != 1
    chain_indices = (
        np.cumsum(
            np.concatenate((np.ones((shape[0], 1), bool), changed), axis=1), axis=1
        ).astype(np.int32)
        - 1
    )
    chain_indices[~real] = -1
    selected, connections, coords, present, pbt, assignments = select_and_connect(
        co, pbt, cf, types, chain_indices, coords, present, gaps
    )
    prepared, leaves = _canonical_metadata(pbt, co)
    rows, xyz, masks, authors, residue_numbers, residue_names, chain_rows = (
        [] for _ in range(7)
    )
    for p in range(shape[0]):
        indices = np.flatnonzero(real[p])
        kinds = selected[p, indices]
        mapped = [
            _map_residue_atoms(
                pbt.active_block_types[kind],
                coords[p, r],
                present[p, r],
                prepared.bt_canonical_atom_ind_map[kind],
                leaves[kind],
                allow_nonleaf=build_missing_context or bool(assignments),
            )
            for r, kind in zip(indices, kinds)
        ]
        rows.append(kinds)
        xyz.append(np.concatenate([x for x, _ in mapped]))
        masks.append([mask for _, mask in mapped])
        authors.append([str(labels[p, r]) for r in indices])
        residue_numbers.append([int(numbers[p, r]) for r in indices])
        residue_names.append(
            [co.restype_io_equiv_classes[types[p, r]] for r in indices]
        )
        chain_rows.append(chain_indices[p, indices])
    used = np.unique(np.concatenate(rows))
    remap = np.full(pbt.n_types, -1, np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    compact = _compact_palette(pbt, used)
    table = connections[:, : max(map(len, rows)), : compact.max_n_conn]
    result = PoseStackBuilder.from_block_types_and_coords(
        compact,
        [remap[row] for row in rows],
        xyz,
        chain_rows,
        residue_numbers,
        residue_names,
        _connection_table=table,
    )
    from ._metal import prepare_metal_completion

    result, masks = prepare_metal_completion(result, masks, assignments)
    if assignments and not build_missing_context:
        for p, row in enumerate(rows):
            for r, kind in enumerate(row):
                _require_rebuildable(
                    pbt.active_block_types[kind], masks[p][r], leaves[kind]
                )
    masks = apply_input_policy(result, masks, mode)
    result = complete_missing_atoms(result, masks, build_context=build_missing_context)
    return result.replace(pdb_info=attrs.evolve(result.pdb_info, chain_id=authors))


__all__ = ["pose_stack_from_canonical_form"]
