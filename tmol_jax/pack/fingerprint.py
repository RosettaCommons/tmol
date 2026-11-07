"""JAX storage for the common sampler-owned retained-atom copy plan."""

import attr
import jax
from dataclasses import replace

from tmol_core.cache import PreparationCache
from tmol_core.pack.fingerprint import (
    annotate_residue_type_with_sampler_fingerprints,
    fingerprint_mapping_layout,
    pack_fingerprint_rows,
)
from .kinematics import prepare_rotamer_kinematics


def prepare_atom_copy_plan(packed_block_types, samplers):
    """Preserve chemical atom identities across sampler/residue alternatives.

    Samplers define ``sampler_name``, ``defines_rotamers_for_rt`` and
    ``first_sc_atoms_for_rt``. This prepares a copy plan, not new coordinates.
    """
    pbt = packed_block_types
    samplers = tuple(samplers)
    prepare_rotamer_kinematics(pbt)
    for block in pbt.active_block_types:
        annotate_residue_type_with_sampler_fingerprints(block, samplers, pbt.chem_db)
    rows, sampler_mapping, key = fingerprint_mapping_layout(pbt.active_block_types)
    key = replace(key, settings=(*key.settings, pbt.n_atoms.device))
    if not hasattr(pbt, "_atom_copy_cache"):
        pbt._atom_copy_cache = PreparationCache()

    def prepare():
        host = pack_fingerprint_rows(rows, sampler_mapping)
        return attr.evolve(
            host,
            **{
                name: jax.device_put(getattr(host, name), pbt.n_atoms.device)
                for name in (
                    "atom_mapping",
                    "source_atom_mapping",
                    "source_fingerprint",
                )
            },
        )

    data = pbt._atom_copy_cache.get_or_create("mapping", key, prepare)
    # Equivalent new sampler instances reuse arrays, retaining their own aliases.
    return attr.evolve(data, sampler_mapping=sampler_mapping)
