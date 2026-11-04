"""Electrostatic globals, partial charges and count-pair representatives.

Setup resolves residue/patch-specific charge rows and representative atoms,
then gathers the resulting host tables onto the JAX pose layout. Custom
restype-carried charges take precedence. Cached tables must retain their
parameter and topology ownership."""

from __future__ import annotations

from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.types import pytree_dataclass
from tmol_core.scoring.elec import (
    partial_charge_table as _partial_charges_dict,
    count_pair_table as _cp_reps_dict,  # noqa: F401 (legacy preparation import)
    partial_charges_for_block,
    representative_mapping_for_block as representative_mapping_for_block_type,
)


@pytree_dataclass
class ElecGlobalParams:
    """Sigmoidal-dielectric constants and distance cutoffs.

    All fields are 0-D ``jax.Array`` scalars so the whole object is a
    pytree leaf bundle that flows naturally through ``jit`` / ``vmap``.
    """

    D: jax.Array  # dielectric asymptote at large distance
    D0: jax.Array  # dielectric limit at zero distance
    S: jax.Array  # sigmoid steepness
    min_dis: jax.Array  # lower cutoff for the flat / short-fade region
    max_dis: jax.Array  # upper cutoff (energy = 0 beyond here)


def elec_global_params_from_database(elec_db: Any) -> ElecGlobalParams:
    """Construct :class:`ElecGlobalParams` from an ``ElecDatabase``.

    Accepts either the :mod:`tmol_jax.database.scoring.elec`
    ``ElecGlobalParams`` attrs class or any duck-typed object with the
    same five attributes.
    """

    gp = elec_db.global_parameters
    return ElecGlobalParams(
        D=jnp.float32(gp.elec_sigmoidal_die_D),
        D0=jnp.float32(gp.elec_sigmoidal_die_D0),
        S=jnp.float32(gp.elec_sigmoidal_die_S),
        min_dis=jnp.float32(gp.elec_min_dis),
        max_dis=jnp.float32(gp.elec_max_dis),
    )


def _charges_for_block_type(
    block_type: Any,
    partial_charges: Dict[str, Dict[str, Dict[str, float]]],
) -> np.ndarray:
    """Per-atom partial charges for one block type, as ``float32[n_atoms]``.

    Preserve the JAX zero-fill policy for absent charges; the shared lookup
    otherwise follows canonical compound-patch precedence and validation.
    """

    atoms = tuple(block_type.atoms)
    out = np.zeros(len(atoms), dtype=np.float32)

    # Restype-carried charges (e.g. a ligand RefinedResidueType built
    # from a ``.tmol`` params file) take precedence over the elec-DB
    # lookup. Canonical residues never set this attribute, so their
    # charges are byte-unchanged. Mirrors upstream's per-block
    # ``elec_partial_charge`` annotation.
    restype_charges = getattr(block_type, "elec_partial_charges", None)
    if restype_charges is not None:
        for i, atm in enumerate(atoms):
            atm_name = getattr(atm, "name", atm)
            if atm_name in restype_charges:
                out[i] = np.float32(restype_charges[atm_name])
        return out

    return partial_charges_for_block(block_type, partial_charges, missing=0.0)


def _per_block_type_charges(
    packed_block_types: Any,
    partial_charges: Dict[str, Dict[str, Dict[str, float]]],
) -> np.ndarray:
    """``float32[n_types, max_n_atoms]`` of charges, zero-padded."""

    n_types = int(packed_block_types.n_types)
    max_n_atoms = int(packed_block_types.max_n_atoms)
    out = np.zeros((n_types, max_n_atoms), dtype=np.float32)
    for i, bt in enumerate(packed_block_types.active_block_types):
        row = _charges_for_block_type(bt, partial_charges)
        out[i, : row.shape[0]] = row
    return out


def atom_charges_for_pose_stack(
    pose_stack: Any,
    packed_block_types: Any,
    elec_db: Any,
) -> jax.Array:
    """Gather float32 charges [poses, atoms], zeroing padding.

    Reuse a palette annotation only for the same database/topology owners;
    otherwise resolve charges from elec_db. Restype-carried charges take precedence."""
    # Reuse only charges annotated for this exact parameter/topology pair.
    owners = getattr(packed_block_types, "_elec_owners", ())
    if (
        len(owners) == 2
        and owners[0] is elec_db
        and owners[1] is packed_block_types.active_block_types
    ):
        per_type_charges = packed_block_types.elec_partial_charge
    else:
        partial_charges = _partial_charges_dict(elec_db)
        per_type_charges = _per_block_type_charges(packed_block_types, partial_charges)
    per_type_charges = jnp.asarray(per_type_charges, dtype=jnp.float32)

    # Use the pose-level index cache shared across every term so a
    # single ``pose_atom_block_index`` trace is reused.
    from tmol_jax.score.common.count_pair import scatter_many_per_atom

    (charges,) = scatter_many_per_atom(
        (per_type_charges,), pose_stack, packed_block_types, defaults=(0.0,)
    )
    return charges


__all__ = [
    "ElecGlobalParams",
    "elec_global_params_from_database",
    "atom_charges_for_pose_stack",
    "representative_mapping_for_block_type",
]
