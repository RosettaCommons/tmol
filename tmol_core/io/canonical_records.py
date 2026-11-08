"""Strict canonical mapping of host atom records, independent of execution."""

from collections import OrderedDict
from typing import Dict, Tuple

import numpy as np


def _polymer_gaps(co, types, chains, coords, threshold):
    """Apply the common chemical-port policy to a same-topology ensemble."""
    from tmol_core.io.polymer_gaps import break_polymer_gaps

    flags = np.zeros((coords.shape[1], 2), bool)
    break_polymer_gaps(
        flags,
        np.empty((0, 4), np.int64),
        np.empty(0, bool),
        coords,
        types[0],
        chains[0],
        co,
        threshold,
    )
    return np.broadcast_to(flags, (*coords.shape[:2], 2)).copy()


def _canonical_form_from_records(
    records,
    canonical_ordering,
    *,
    missing_density_distance_threshold=2.4,
    return_atom_mapping=False,
    prepared_atom_indices=None,
):
    atom_names = records["atom_name"]
    res_names = records["res_name"]
    chain_ids_raw = records["chain_id"]
    res_nums_raw = records["res_num"]
    insert_raw = records["insert"]
    coords_raw = records["coords"]
    occupancy_raw = records["occupancy"]
    b_factor_raw = records["b_factor"]
    record_name = records.get(
        "record_name", np.asarray(["ATOM"] * len(atom_names), dtype=object)
    )
    alt_loc_raw = records.get(
        "alt_loc", np.asarray([""] * len(atom_names), dtype=object)
    )

    n_records = len(atom_names)

    # -------- Index distinct residues and chains -------------------- #
    unique_residues: "OrderedDict[Tuple[str, int, str], int]" = OrderedDict()
    unique_chains: "OrderedDict[str, int]" = OrderedDict()

    # Per-atom: which residue row / chain slot it belongs to.
    res_ind_per_atom = np.full(n_records, -1, dtype=np.int32)
    keep_atom = np.zeros(n_records, dtype=bool)

    skipped_residue_names: Dict[str, int] = {}

    for i in range(n_records):
        rec = str(record_name[i]).strip() if record_name is not None else "ATOM"
        if rec not in ("ATOM", "HETATM", ""):
            continue
        if str(res_names[i]).strip() in {"HOH", "WAT", "DOD"}:
            continue
        alt = str(alt_loc_raw[i]).strip() if alt_loc_raw is not None else ""
        # Keep only the primary alt-loc: blank, "A", or "1".
        if alt not in ("", "A", "1"):
            continue
        resn = str(res_names[i]).strip()
        if canonical_ordering.restype_index(resn) < 0:
            skipped_residue_names[resn] = skipped_residue_names.get(resn, 0) + 1
            continue

        chain_key = str(chain_ids_raw[i])
        insert_key = str(insert_raw[i]) if insert_raw is not None else ""
        res_key = (chain_key, int(res_nums_raw[i]), insert_key)

        if chain_key not in unique_chains:
            unique_chains[chain_key] = len(unique_chains)
        if res_key not in unique_residues:
            unique_residues[res_key] = len(unique_residues)
        res_ind_per_atom[i] = unique_residues[res_key]
        keep_atom[i] = True

    if skipped_residue_names:
        raise NotImplementedError(
            "The canonical ordering does not contain residues: "
            + ", ".join(sorted(skipped_residue_names))
        )

    n_res = len(unique_residues)
    if n_res == 0:
        raise ValueError("PDB contains no supported residues")
    max_n_atoms = max(1, canonical_ordering.max_n_canonical_atoms)

    # -------- Allocate canonical-form arrays ------------------------ #
    coords = np.full((1, n_res, max_n_atoms, 3), np.nan, dtype=np.float32)
    atom_present = np.zeros((1, n_res, max_n_atoms), dtype=bool)
    chain_id = np.full((1, n_res), -1, dtype=np.int32)
    res_types = np.full((1, n_res), -1, dtype=np.int32)
    res_num_arr = np.zeros((1, n_res), dtype=np.int32)
    res_name_arr = np.empty((1, n_res), dtype=object)
    chain_labels = np.empty((1, n_res), dtype=object)
    res_ins_codes = np.empty((1, n_res), dtype=object)
    occupancy_arr = np.ones((1, n_res, max_n_atoms), dtype=np.float32)
    b_factor_arr = np.zeros((1, n_res, max_n_atoms), dtype=np.float32)

    # Initialise object arrays so NumPy doesn't leave them as None.
    res_name_arr.fill("")
    chain_labels.fill("")
    res_ins_codes.fill("")

    # Fill in per-residue metadata first, using the first atom seen for
    # each residue as the source of truth.
    residue_initialised = np.zeros(n_res, dtype=bool)
    for i in range(n_records):
        if not keep_atom[i]:
            continue
        r = int(res_ind_per_atom[i])
        if residue_initialised[r]:
            continue
        resn = str(res_names[i]).strip()
        chain_key = str(chain_ids_raw[i])
        res_types[0, r] = canonical_ordering.restype_index(resn)
        chain_id[0, r] = unique_chains[chain_key]
        res_num_arr[0, r] = int(res_nums_raw[i])
        res_name_arr[0, r] = resn
        chain_labels[0, r] = chain_key
        res_ins_codes[0, r] = str(insert_raw[i]) if insert_raw is not None else ""
        residue_initialised[r] = True

    canonical_atom_indices = np.full(n_records, -1, np.int64)
    if prepared_atom_indices is not None:
        keep_atom &= prepared_atom_indices >= 0
    # Now splat atoms into their canonical slots.
    for i in range(n_records):
        if not keep_atom[i]:
            continue
        r = int(res_ind_per_atom[i])
        resn = str(res_names[i]).strip()
        atom_name = str(atom_names[i]).strip()
        aind = (
            canonical_ordering.atom_index(resn, atom_name)
            if prepared_atom_indices is None
            else prepared_atom_indices[i]
        )
        if aind < 0:
            raise ValueError(f"{resn} atom {atom_name} is not in the chemical atom set")
        canonical_atom_indices[i] = aind
        coords[0, r, aind, 0] = coords_raw[i, 0]
        coords[0, r, aind, 1] = coords_raw[i, 1]
        coords[0, r, aind, 2] = coords_raw[i, 2]
        atom_present[0, r, aind] = np.isfinite(coords_raw[i]).all()
        occupancy_arr[0, r, aind] = float(occupancy_raw[i])
        b_factor_arr[0, r, aind] = float(b_factor_raw[i])

    form = {
        "res_gap": _polymer_gaps(
            canonical_ordering,
            res_types,
            chain_id,
            coords,
            missing_density_distance_threshold,
        ),
        "coords": coords,
        "atom_is_present": atom_present,
        "chain_id": chain_id,
        "res_types": res_types,
        "res_num": res_num_arr,
        "res_name": res_name_arr,
        "chain_labels": chain_labels,
        "res_ins_codes": res_ins_codes,
        "atom_occupancy": occupancy_arr,
        "atom_b_factor": b_factor_arr,
    }
    if return_atom_mapping:
        return form, (res_ind_per_atom, keep_atom, canonical_atom_indices[keep_atom])
    return form
