"""Canonical electrostatic charge and count-pair lookup on host chemistry.

Tables belong to their parameter-database owner. Each lookup returns a fresh
NumPy array; frontends own device materialization and topology-aware caching.
"""

import math

from collections.abc import Mapping
from typing import TypeAlias

import numpy as np

PartialChargeTable: TypeAlias = Mapping[str, Mapping[str, Mapping[str, float]]]
CountPairTable: TypeAlias = Mapping[str, Mapping[str, Mapping[str, str]]]


def _variant_table(rows, atom_field, value_field):
    table = {}
    for row in rows:
        base, _, variant = row.res.partition(":")
        table.setdefault(base, {}).setdefault(getattr(row, atom_field), {})[variant] = (
            getattr(row, value_field)
        )
    return table


def partial_charge_table(database) -> PartialChargeTable:
    """Index charge rows by residue, atom and exact patch tag; last row wins."""
    return _variant_table(database.atom_charge_parameters, "atom", "charge")


def count_pair_table(database) -> CountPairTable:
    """Index outer -> inner rows; the outer atom represents the inner atom."""
    return _variant_table(database.atom_cp_reps_parameters, "atm_outer", "atm_inner")


def lookup_order(name: str) -> tuple[str, tuple[str, ...]]:
    """Exact variant, terminal-conjugation tag, individual patches, then base.

    Only terminal N-conjugation pairs apply inside a longer variant name;
    arbitrary compound patch rows require an exact match.
    """
    base, _, patches = name.partition(":")
    tags = patches.split(":") if patches else []
    terminal_conjugations = (
        ":".join(tags[i : i + 2])
        for i in range(len(tags) - 1)
        if "nterm" in tags[i : i + 2]
        and any(tag.startswith("conj_") for tag in tags[i : i + 2])
    )
    return base, tuple(dict.fromkeys((patches, *terminal_conjugations, *tags, "")))


def partial_charges_for_block(
    block_type, partial_charges: PartialChargeTable, *, missing: float | None = None
) -> np.ndarray:
    """Return float32 charges in atom order, rejecting non-finite values.

    Missing charges raise by default. A frontend preserving zero-filled missing
    charges must explicitly pass ``missing=0.0``. Restype-carried overrides are a
    frontend contract and should be resolved before this database lookup.
    """
    residue, variants = lookup_order(block_type.name)
    residue_charges = partial_charges.get(residue, {})

    def lookup(atom):
        name = getattr(atom, "name", atom)
        charges = residue_charges.get(name, {})
        for variant in variants:
            if variant in charges:
                return charges[variant]
        if missing is not None:
            return missing
        raise KeyError(f"Elec charge for atom {block_type.name},{name} not found")

    with np.errstate(over="ignore"):
        charges = np.fromiter(
            (lookup(atom) for atom in block_type.atoms),
            dtype=np.float32,
            count=len(block_type.atoms),
        )
    invalid = ~np.isfinite(charges)
    if np.any(invalid):
        atoms = ", ".join(
            getattr(block_type.atoms[i], "name", block_type.atoms[i])
            for i in np.flatnonzero(invalid)
        )
        raise ValueError(
            f"Elec charges for {block_type.name} atoms {atoms} are non-finite in float32"
        )
    return charges


def representative_mapping_for_block(block_type, cp_reps: CountPairTable) -> np.ndarray:
    """Return int32 representative indices with identity for unmapped atoms.

    Prefer the most specific row across competing outer atoms. Equal-specificity
    nominations retain the historical last-outer-in-atom-order rule.
    """
    mapping = np.arange(len(block_type.atoms), dtype=np.int32)
    residue, variants = lookup_order(block_type.name)
    residue_reps = cp_reps.get(residue)
    if residue_reps is None:
        return mapping
    atom_to_idx = getattr(block_type, "atom_to_idx", None)
    if atom_to_idx is None:
        atom_to_idx = {
            getattr(atom, "name", atom): i for i, atom in enumerate(block_type.atoms)
        }
    best_rank = {}
    for outer in atom_to_idx:
        candidates = residue_reps.get(outer)
        if candidates is None:
            continue
        for rank, variant in enumerate(variants):
            if variant not in candidates:
                continue
            inner = candidates[variant]
            if inner not in atom_to_idx:
                raise KeyError(f"Invalid elec cp mapping: {residue} {outer}->{inner}")
            if rank <= best_rank.get(inner, len(variants)):
                best_rank[inner] = rank
                mapping[atom_to_idx[inner]] = atom_to_idx[outer]
            break
    return mapping


def native_elec_global_parameters(parameters):
    """Pack dielectric and cutoff-spline constants in native field order."""
    D = float(parameters.elec_sigmoidal_die_D)
    D0 = float(parameters.elec_sigmoidal_die_D0)
    S = float(parameters.elec_sigmoidal_die_S)
    min_dis = float(parameters.elec_min_dis)
    max_dis = float(parameters.elec_max_dis)

    def eps(dist):
        return D - 0.5 * (D - D0) * (2 + 2 * dist * S + dist * dist * S * S) * math.exp(
            -dist * S
        )

    def deps(dist):
        return 0.5 * (D - D0) * dist * dist * S * S * S * math.exp(-dist * S)

    eps_at_cutoff = eps(max_dis)
    cutoff_offset = 322.0637 / (max_dis * eps_at_cutoff)
    min_score = 322.0637 / (min_dis * eps(min_dis)) - cutoff_offset

    low_end = min_dis + 0.25
    low_eps = eps(low_end)
    low_score = 322.0637 / (low_end * low_eps) - cutoff_offset
    low_deriv = (
        -322.0637
        * (low_eps + low_end * deps(low_end))
        / (low_end * low_end * low_eps * low_eps)
    )

    high_start = max_dis - 1.0
    high_eps = eps(high_start)
    high_score = 322.0637 / (high_start * high_eps) - cutoff_offset
    high_deriv = (
        -322.0637
        * (high_eps + high_start * deps(high_start))
        / (high_start * high_start * high_eps * high_eps)
    )

    return np.asarray(
        [
            D,
            D0,
            S,
            min_dis,
            max_dis,
            cutoff_offset,
            min_score,
            low_score,
            low_deriv,
            high_score,
            high_deriv,
        ],
        dtype=np.float32,
    )[None, :]
