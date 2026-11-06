"""Canonical-form metadata from the shared chemical ordering."""

from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
from typing import Any, Dict, Tuple

import numpy as np

from tmol_core.io.canonical_ordering import CanonicalOrdering as _CanonicalOrdering


class CanonicalOrdering(_CanonicalOrdering):
    """Shared canonical residue and atom identities with frontend lookups."""

    def restype_index(self, name3):
        try:
            return self.restype_io_equiv_classes.index(name3)
        except ValueError:
            return -1

    def atom_index(self, name3, atom_name):
        return self.restypes_atom_index_mapping.get(name3, {}).get(atom_name, -1)


@lru_cache(maxsize=1)
def default_canonical_ordering() -> CanonicalOrdering:
    """Return the shared atom ordering for the complete default chemical database."""
    from tmol_jax.chemical.patched_chemdb import default_patched_chemdb

    return CanonicalOrdering.from_chemdb(default_patched_chemdb())


@lru_cache(maxsize=1)
def default_packed_block_types() -> Any:
    """Return the default molecular palette, including patched residue types.

    Standalone metal sites use an explicitly prepared palette so their connection
    widths do not enlarge every ordinary structure.
    """
    from tmol_jax.pose.packed_block_types import PackedBlockTypes

    return PackedBlockTypes.default()


# --------------------------------------------------------------------- #
# canonical_form_from_pdb                                                #
# --------------------------------------------------------------------- #


def canonical_form_from_pdb(pdb_text_or_path, canonical_ordering=None):
    """Convert a PDB to canonical molecular coordinates, retaining missing atoms.

    Unknown residue and atom identities fail explicitly; water is ignored. Chain labels
    and C-N distances (gaps beyond 2.4 A) determine polymer connectivity.
    """
    from .pdb_parsing import parse_pdb

    return _canonical_form_from_records(
        parse_pdb(pdb_text_or_path),
        canonical_ordering or default_canonical_ordering(),
    )


_PEPTIDE_GAP = 2.4  # Angstrom; matches tmol's missing_density_distance_threshold


def _peptide_gaps(canonical_ordering, res_names, chain_id, coords, present):
    """Flag polymer bonds whose resolved C and N atoms are farther than a peptide bond."""
    n_res = chain_id.shape[1]
    not_connected = np.zeros((1, n_res, 2), dtype=bool)
    for r in range(n_res - 1):
        if chain_id[0, r] != chain_id[0, r + 1]:
            continue
        c = canonical_ordering.atom_index(res_names[0, r], "C")
        n = canonical_ordering.atom_index(res_names[0, r + 1], "N")
        if c < 0 or n < 0 or not (present[0, r, c] and present[0, r + 1, n]):
            continue
        if np.linalg.norm(coords[0, r, c] - coords[0, r + 1, n]) > _PEPTIDE_GAP:
            not_connected[0, r, 1] = not_connected[0, r + 1, 0] = True
    return not_connected


def _canonical_form_from_records(records, canonical_ordering):
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

    # Now splat atoms into their canonical slots.
    for i in range(n_records):
        if not keep_atom[i]:
            continue
        r = int(res_ind_per_atom[i])
        resn = str(res_names[i]).strip()
        atom_name = str(atom_names[i]).strip()
        aind = canonical_ordering.atom_index(resn, atom_name)
        if aind < 0:
            raise ValueError(f"{resn} atom {atom_name} is not in the chemical atom set")
        coords[0, r, aind, 0] = coords_raw[i, 0]
        coords[0, r, aind, 1] = coords_raw[i, 1]
        coords[0, r, aind, 2] = coords_raw[i, 2]
        atom_present[0, r, aind] = True
        occupancy_arr[0, r, aind] = float(occupancy_raw[i])
        b_factor_arr[0, r, aind] = float(b_factor_raw[i])

    return {
        "res_gap": _peptide_gaps(
            canonical_ordering, res_name_arr, chain_id, coords, atom_present
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


__all__ = [
    "CanonicalOrdering",
    "default_canonical_ordering",
    "default_packed_block_types",
    "canonical_form_from_pdb",
]
