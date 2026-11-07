"""JAX storage for the common sampler-owned retained-atom copy plan."""

import attr
import jax

from tmol_core.pack.fingerprint import (
    annotate_residue_type_with_sampler_fingerprints,
    prepare_packed_fingerprints,
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
    host = prepare_packed_fingerprints(pbt.active_block_types)
    return attr.evolve(
        host,
        **{
            name: jax.device_put(getattr(host, name), pbt.n_atoms.device)
            for name in ("atom_mapping", "source_atom_mapping", "source_fingerprint")
        },
    )
