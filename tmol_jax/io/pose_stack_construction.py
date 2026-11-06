"""Build canonical protein poses with explicit topology and icoor reconstruction."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


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


def _bt_name_index(pbt: Any) -> Dict[str, int]:
    """Map every block-type ``name`` (including variants like
    ``"MET:nterm"``) to its index in ``pbt.active_block_types``.
    """
    mapping: Dict[str, int] = {}
    for i, bt in enumerate(getattr(pbt, "active_block_types", ()) or ()):
        name = getattr(bt, "name", None)
        if name is not None:
            mapping[str(name)] = i
    return mapping


def _bt_name3_index(pbt: Any) -> Dict[str, int]:
    """Map ``name3`` -> first-matching *base* block-type index in ``pbt``.

    For the default palette (which now includes termini variants) we
    prefer the base residue (``name == name3``) over any variant so
    non-terminal residues resolve to the un-patched block type.
    """
    mapping: Dict[str, int] = {}
    # First pass: base (un-patched) residues, which have name == name3.
    for i, bt in enumerate(getattr(pbt, "active_block_types", ()) or ()):
        n3 = getattr(bt, "name3", None) or getattr(bt, "name", None)
        name = getattr(bt, "name", None)
        if n3 is None or name is None:
            continue
        if str(name) == str(n3):
            mapping.setdefault(str(n3), i)
    # Second pass: anything else that hasn't been registered yet.
    for i, bt in enumerate(getattr(pbt, "active_block_types", ()) or ()):
        n3 = getattr(bt, "name3", None) or getattr(bt, "name", None)
        if n3 is not None and str(n3) not in mapping:
            mapping[str(n3)] = i
    return mapping


def _build_variant_lookup(
    pbt: Any, bt_name_index: Dict[str, int]
) -> Dict[str, Dict[str, int]]:
    """Return ``{base_name: {variant_tag: bt_ind}}`` where ``variant_tag``
    is one of ``""``, ``"nterm"``, ``"cterm"``, ``"cterm:nterm"``.

    ``base_name`` is the colon-prefix of the block type name, so
    ``HIS_POS:cterm`` contributes to ``{"HIS_POS": {"cterm": ...}}``.
    Missing variants simply don't appear in the inner dict -- callers
    fall back to the base block type.
    """
    out: Dict[str, Dict[str, int]] = {}
    for name, i in bt_name_index.items():
        parts = name.split(":")
        base = parts[0]
        tag = ":".join(parts[1:])  # "" for base, "cterm:nterm" for both, etc.
        out.setdefault(base, {})[tag] = i
    return out


def _build_name3_special_case_lookup(pbt: Any) -> Dict[str, Dict[str, str]]:
    """Return ``{name3: {special_case_key: base_name}}``.

    Used to pick a base-name (``HIS_POS``, ``HIS_D``, ``HIS``, ``CYS``,
    ``CYD``, ...) for a residue whose 3-letter code is ``name3``. The
    inner dict keys mirror upstream's ``_map_spcase_var_to_int``:

    - ``"default"``       -- base type (e.g. HIS, CYS)
    - ``"cyd"``           -- CYD (disulfide CYS)
    - ``"hisd"``          -- HIS_D (ND1 protonated)
    - ``"hispos"``        -- HIS_POS (both NE2 and ND1 protonated)

    Entries without an explicit key inherit the ``"default"`` base.
    For residues that only have one base (ALA, GLY, ...) the inner
    dict will just have ``{"default": "ALA"}`` etc.
    """
    # Map from base_name -> name3 (3-letter code that the base belongs to)
    # using the block-type metadata that ships with the palette.
    bases_for_name3: Dict[str, List[str]] = {}
    for bt in getattr(pbt, "active_block_types", ()) or ():
        n3 = getattr(bt, "name3", None) or getattr(bt, "name", None)
        base = getattr(bt, "base_name", None) or getattr(bt, "name", None)
        if n3 is None or base is None:
            continue
        n3 = str(n3)
        base = str(base)
        # Only track *base* (no-termini) rows so we don't double-add.
        if ":" in str(getattr(bt, "name", "")):
            continue
        bases_for_name3.setdefault(n3, [])
        if base not in bases_for_name3[n3]:
            bases_for_name3[n3].append(base)

    def _spcase_key(base: str, name3: str) -> str:
        if base == "CYD":
            return "cyd"
        if base == "HIS_D":
            return "hisd"
        if base == "HIS_POS":
            return "hispos"
        # The "default" base is whichever one matches the name3 literally
        # (HIS for HIS, CYS for CYS, ALA for ALA, ...).
        if base == name3:
            return "default"
        return "other:" + base

    out: Dict[str, Dict[str, str]] = {}
    for n3, bases in bases_for_name3.items():
        entry: Dict[str, str] = {}
        for base in bases:
            entry[_spcase_key(base, n3)] = base
        # Ensure a default exists: fall back to the name3 itself or the
        # first base-name we saw.
        if "default" not in entry:
            entry["default"] = n3 if n3 in bases else bases[0]
        out[n3] = entry
    return out


def _resolve_his_spcase(
    res_name: str,
    co_atom_names: Tuple[str, ...],
    atom_is_present_row: np.ndarray,
    coords_row: np.ndarray,
) -> str:
    """Return ``"default"`` / ``"hisd"`` / ``"hispos"`` for a HIS residue.

    Reproduces upstream's ``resolve_his_tautomerization`` branching:

    1. HD1 present and HE2 absent       -> HIS_D (``"hisd"``)
    2. HE2 present and HD1 absent       -> HIS   (``"default"``)
    3. Both HD1 and HE2 present         -> HIS_POS (``"hispos"``)
    4. HN present with ND1 and NE2      -> assign to whichever of ND1/NE2
       is closer to HN (HIS_D vs HIS).
    5. NH/NN/HN/CG all present          -> assign to whichever of NH/NN is
       closer to CG (HIS_D vs HIS).
    6. Otherwise                        -> HIS (arbitrary default).

    ``co_atom_names`` is the canonical-ordering atom-name tuple for the
    HIS equivalence class (includes HD1 / HE2 / HN / NH / NN / CG).
    """
    if res_name != "HIS":
        return "default"
    name_to_slot = {n: i for i, n in enumerate(co_atom_names)}

    def _has(atom: str) -> bool:
        slot = name_to_slot.get(atom, -1)
        if slot < 0 or slot >= atom_is_present_row.shape[0]:
            return False
        return bool(atom_is_present_row[slot])

    hd1 = _has("HD1")
    he2 = _has("HE2")
    hn = _has("HN")
    nd1 = _has("ND1")
    ne2 = _has("NE2")
    nh = _has("NH")
    nn = _has("NN")
    cg = _has("CG")

    if hd1 and he2:
        return "hispos"
    if hd1 and not he2:
        return "hisd"
    if he2 and not hd1:
        return "default"
    if hn and not hd1 and not he2 and nd1 and ne2:
        s_hn = name_to_slot["HN"]
        s_nd1 = name_to_slot["ND1"]
        s_ne2 = name_to_slot["NE2"]
        d_nd1 = float(np.sum((coords_row[s_nd1] - coords_row[s_hn]) ** 2))
        d_ne2 = float(np.sum((coords_row[s_ne2] - coords_row[s_hn]) ** 2))
        return "hisd" if d_nd1 < d_ne2 else "default"
    if nh and nn and hn and cg:
        s_nh = name_to_slot["NH"]
        s_nn = name_to_slot["NN"]
        s_cg = name_to_slot["CG"]
        d_nh = float(np.sum((coords_row[s_nh] - coords_row[s_cg]) ** 2))
        d_nn = float(np.sum((coords_row[s_nn] - coords_row[s_cg]) ** 2))
        # If NH is closer to CG, NH is ND1 (ND1-protonated -> HIS_D).
        # If NN is closer to CG, NN is ND1 (NE2-protonated -> HIS).
        return "hisd" if d_nh < d_nn else "default"
    # No discriminators -> NE2-protonated HIS.
    return "default"


def _resolve_disulfide_pairs(
    real_r_inds: List[int],
    real_name3s: List[str],
    co_atom_names_for_cys: Tuple[str, ...],
    atom_is_present: np.ndarray,
    coords: np.ndarray,
    pose_idx: int,
    cutoff_dis: float = 2.5,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Return ``(paired_flags, pairs)`` for CYS residues at this pose.

    * ``paired_flags`` is a list-of-ints (0/1) indexed by ``real_r_inds``
      that is True for CYS residues that should be promoted to CYD.
    * ``pairs`` is a list of ``(k_i, k_j)`` tuples giving the
      block-index-among-real-residues of the two halves of each
      disulfide. These are needed downstream to wire the ``dslf``
      inter-residue connection so the cart-bonded SG-SG bond term sees
      the disulfide as a real bond (otherwise 1cgi etc. blow up the
      ``cart_lengths`` term with ~1.95 Å SG-SG distances treated as
      "unbonded").

    Greedy nearest-neighbour matching: walks the CYS residues in order
    and pairs each unpaired CYS with the closest unpaired CYS whose SG
    is within ``cutoff_dis``. Mirrors upstream's ``find_disulf_numba``.
    """
    paired = [0] * len(real_r_inds)
    pairs: List[Tuple[int, int]] = []
    if "SG" not in co_atom_names_for_cys:
        return paired, pairs
    sg_slot = co_atom_names_for_cys.index("SG")
    cys_ks = [k for k, n3 in enumerate(real_name3s) if n3 == "CYS"]
    if len(cys_ks) < 2:
        return paired, pairs

    cutoff2 = cutoff_dis * cutoff_dis
    for i_idx, k_i in enumerate(cys_ks):
        if paired[k_i]:
            continue
        r_i = real_r_inds[k_i]
        if not bool(atom_is_present[pose_idx, r_i, sg_slot]):
            continue
        c_i = coords[pose_idx, r_i, sg_slot]
        best_j = -1
        best_d2 = None
        for j_idx in range(i_idx + 1, len(cys_ks)):
            k_j = cys_ks[j_idx]
            if paired[k_j]:
                continue
            r_j = real_r_inds[k_j]
            if not bool(atom_is_present[pose_idx, r_j, sg_slot]):
                continue
            c_j = coords[pose_idx, r_j, sg_slot]
            d2 = float(np.sum((c_i - c_j) ** 2))
            if d2 < cutoff2 and (best_d2 is None or d2 < best_d2):
                best_d2 = d2
                best_j = k_j
        if best_j >= 0:
            paired[k_i] = 1
            paired[best_j] = 1
            pairs.append((k_i, best_j))
    return paired, pairs


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


def _choose_residue_types(pbt, names, coords, present, starts, ends, co):
    variants = _build_variant_lookup(pbt, _bt_name_index(pbt))
    special_cases = _build_name3_special_case_lookup(pbt)
    keys = ["default"] * len(names)
    for index, name in enumerate(names):
        if name == "HIS":
            keys[index] = _resolve_his_spcase(
                name,
                co.restypes_ordered_atom_names[name],
                present[index],
                coords[index],
            )
    flags, disulfides = _resolve_disulfide_pairs(
        list(range(len(names))),
        names,
        co.restypes_ordered_atom_names.get("CYS", ()),
        present[None],
        coords[None],
        0,
    )
    for index in np.flatnonzero(flags):
        keys[index] = "cyd"
    indices = []
    for index, name in enumerate(names):
        cases = special_cases.get(name, {})
        base = cases.get(keys[index], cases.get("default", name))
        tags = [
            tag
            for enabled, tag in ((ends[index], "cterm"), (starts[index], "nterm"))
            if enabled
        ]
        choices = variants.get(base, {})
        selected = choices.get(":".join(tags), choices.get(":".join(reversed(tags))))
        if selected is None:
            raise ValueError(
                f"residue palette lacks required variant {base}:{':'.join(tags)}"
            )
        indices.append(selected)
    return np.asarray(indices, dtype=np.int32), disulfides


def _map_residue_atoms(residue, source, present, mapping, leaf):
    mapping = mapping[: residue.n_atoms]
    valid = (mapping >= 0) & (mapping < source.shape[0])
    safe = np.where(valid, mapping, 0)
    available = valid & present[safe]
    coords = np.where(available[:, None], source[safe], 0)
    missing = ~available & ~leaf[: residue.n_atoms]
    if missing.any():
        atoms = ", ".join(
            residue.atoms[index].name for index in np.flatnonzero(missing)
        )
        raise ValueError(f"missing non-leaf atoms on {residue.name}: {atoms}")
    return coords, available


def _chain_segments(indices, chains, disconnected, gaps):
    """Chain starts/ends (which pick terminal variants) and connected segments.

    A ``gap`` (missing density) disconnects residues but, as in TMol's loader,
    leaves their block types un-terminated; an explicit break also terminates.
    """
    changed = chains[indices[1:]] != chains[indices[:-1]]
    broken = (
        changed
        | (np.diff(indices) != 1)
        | disconnected[indices[:-1], 1]
        | disconnected[indices[1:], 0]
    )
    cut = broken | gaps[indices[:-1], 1] | gaps[indices[1:], 0]
    return (
        np.r_[True, broken],
        np.r_[broken, True],
        np.r_[True, cut],
        np.r_[cut, True],
        np.cumsum(np.r_[True, cut]) - 1,
    )


def pose_stack_from_canonical_form(
    canonical_form_dict, packed_block_types, chain_id=None, res_num=None
):
    """Build canonical proteins, respecting explicit chain breaks and atom names.

    Coordinates are [poses, residues, canonical atoms, 3]. Missing leaf atoms
    are reconstructed; missing non-leaf atoms and unavailable variants raise.
    res_not_connected[..., 0/1] breaks the preceding/following polymer bond and
    selects terminal variants; res_gap does the same without terminal variants.
    """
    import attrs
    from tmol_jax.io.canonical_ordering import default_canonical_ordering
    from tmol_jax.pose.packed_block_types import PackedBlockTypes
    from tmol_jax.pose.pose_stack_builder import PoseStackBuilder

    cf = canonical_form_dict
    pbt = packed_block_types
    coords = np.asarray(cf["coords"])
    if coords.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError("canonical coordinates require float32 or float64")
    if coords.ndim != 4 or coords.shape[-1] != 3 or min(coords.shape[:3]) == 0:
        raise ValueError("coords must have nonempty shape [poses, residues, atoms, 3]")
    shape = coords.shape[:2]
    present = np.asarray(
        cf.get("atom_is_present", np.isfinite(coords).all(axis=-1)), dtype=bool
    )
    if present.shape != coords.shape[:-1] or np.any(
        present & ~np.isfinite(coords).all(axis=-1)
    ):
        raise ValueError("atom_is_present must match finite canonical coordinates")
    co = cf.get("canonical_ordering") or default_canonical_ordering()
    prepared, leaves = _canonical_metadata(pbt, co)
    names = _normalise_res_names(dict(cf, canonical_ordering=co), *shape)
    if len(names) != shape[0] or any(len(row) != shape[1] for row in names):
        raise ValueError("residue names must match the canonical coordinate shape")
    chain_values = cf.get("chain_id")
    if chain_values is None:
        chain_values = chain_id
    chains = _canonical_rows(
        chain_values, shape, np.zeros(shape, dtype=np.int32), "chain_id"
    )
    labels = _canonical_rows(cf.get("chain_labels"), shape, chains, "chain_labels")
    number_values = cf.get("res_num")
    if number_values is None:
        number_values = res_num
    numbers = _canonical_rows(
        number_values,
        shape,
        np.broadcast_to(np.arange(shape[1]) + 1, shape),
        "res_num",
    )
    disconnected = np.asarray(
        cf.get("res_not_connected", np.zeros(shape + (2,), dtype=bool)), dtype=bool
    )
    if disconnected.shape != shape + (2,):
        raise ValueError("res_not_connected must have shape [poses, residues, 2]")
    gaps = np.asarray(cf.get("res_gap", np.zeros(shape + (2,), dtype=bool)), dtype=bool)
    known = set(_bt_name3_index(pbt))
    (
        types,
        coordinates,
        segments,
        authors,
        residue_numbers,
        residue_names,
        disulfides,
        atom_availability,
    ) = ([] for _ in range(8))
    for pose, row in enumerate(names):
        indices = np.array(
            [index for index, name in enumerate(row) if name], dtype=np.int32
        )
        selected_names = [row[index] for index in indices]
        unknown = set(selected_names) - known
        if unknown:
            raise NotImplementedError(
                "unsupported residues: " + ", ".join(sorted(unknown))
            )
        if not len(indices):
            raise ValueError("each pose must contain at least one protein residue")
        starts, ends, _, _, chain_segments = _chain_segments(
            indices, chains[pose], disconnected[pose], gaps[pose]
        )
        selected, pairs = _choose_residue_types(
            pbt,
            selected_names,
            coords[pose, indices],
            present[pose, indices],
            starts,
            ends,
            co,
        )
        residues = [pbt.active_block_types[index] for index in selected]
        mapped = [
            _map_residue_atoms(residue, xyz, mask, mapping, leaf)
            for residue, xyz, mask, mapping, leaf in zip(
                residues,
                coords[pose, indices],
                present[pose, indices],
                prepared.bt_canonical_atom_ind_map[selected],
                leaves[selected],
            )
        ]
        coordinates.append(np.concatenate([xyz for xyz, _ in mapped]))
        atom_availability.append([mask for _, mask in mapped])
        types.append(selected)
        segments.append(chain_segments)
        authors.append([str(labels[pose, index]) for index in indices])
        residue_numbers.append([int(numbers[pose, index]) for index in indices])
        residue_names.append(selected_names)
        disulfides.append(pairs)
    used = np.unique(np.concatenate(types))
    remap = np.full(pbt.n_types, -1, np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    compact = PackedBlockTypes.from_restypes(
        pbt.chem_db, [pbt.active_block_types[index] for index in used]
    )
    result = PoseStackBuilder.from_block_types_and_coords(
        compact,
        [remap[row] for row in types],
        coordinates,
        segments,
        residue_numbers,
        residue_names,
        disulfides,
    )
    from ._missing import complete_missing_atoms

    result = complete_missing_atoms(result, atom_availability)
    return result.replace(pdb_info=attrs.evolve(result.pdb_info, chain_id=authors))


__all__ = ["pose_stack_from_canonical_form"]
