"""Sequence construction through common chemistry, topology and torsion policy."""

from dataclasses import dataclass
from functools import lru_cache

import jax
import numpy as np

from tmol_core.io import extended
from tmol_core.pose.sequence import (
    _resolve_block_type_names,
    _smiles_in_tokens,
    _tokenize_sequences,
)
from tmol_jax.chemical.restypes import ResidueTypeSet
from tmol_jax.database import ParameterDatabase
from tmol_jax.pose import PackedBlockTypes, PoseStackBuilder
from .canonical_ordering import CanonicalOrdering, default_packed_block_types
from .residue_blocks import PoseBuildContext
from .structure import _device


@dataclass(frozen=True)
class SequenceBuildContext(PoseBuildContext):
    canonical_ordering: object
    restype_set: object
    ligand_names: dict


def _make_context(database, device, ligand_names):
    is_default = database is ParameterDatabase.get_default()
    restypes = (
        ResidueTypeSet.get_default()
        if is_default
        else ResidueTypeSet.from_database(database.chemical)
    )
    with jax.default_device(device):
        palette = (
            default_packed_block_types()
            if is_default
            else PackedBlockTypes.from_restypes(
                restypes.chem_db, restypes.residue_types
            )
        )
    return SequenceBuildContext(
        database,
        palette,
        CanonicalOrdering.from_chemdb(database.chemical),
        restypes,
        ligand_names,
    )


@lru_cache(maxsize=8)
def _default_context(device):
    # The singleton default database is immutable and process-lived. Custom
    # databases are never keyed by equality or silently substituted here.
    return _make_context(ParameterDatabase.get_default(), device, {})


def _sequence_inputs(seqs, device, param_db, context, termini):
    if context is not None and param_db is not None:
        raise ValueError("pass either context= or param_db=, not both")
    tokens, lengths = _tokenize_sequences(seqs)
    if context is not None:
        if context.packed_block_types.n_atoms.device != device:
            raise ValueError("context was built for a different device")
        restypes = context.restype_set
        ligand_names = context.ligand_names
    else:
        param_db = param_db or ParameterDatabase.get_default()
        smiles = _smiles_in_tokens(tokens)
        ligand_names = {}
        if smiles:
            from tmol_core.ligand._preparation import _prepare_ligands_from_smiles
            from tmol_jax.ligand._execution import with_jax_execution

            param_db, ligand_names = with_jax_execution(_prepare_ligands_from_smiles)(
                smiles, param_db=param_db
            )
        context = (
            _default_context(device)
            if param_db is ParameterDatabase.get_default() and not ligand_names
            else _make_context(param_db, device, ligand_names)
        )
        restypes = context.restype_set
    names, lengths = _resolve_block_type_names(
        tokens, lengths, restypes, ligand_names, termini
    )
    return names, lengths, context


def extended_pose_stack_from_sequences(
    seqs, device=None, param_db=None, termini=True, context=None, return_context=False
):
    """Build ideal extended molecules from the mature sequence grammar.

    Uppercase letters select amino acids, lowercase letters select DNA,
    ``X[NAME]`` selects an exact residue type, ``X(SMILES)`` prepares a ligand,
    and ``:`` separates chains. Explicit variants and connection labels retain
    their identities. The returned context contains any extended parameters;
    use its database for scoring and reuse it for subsequent construction.
    """
    device = _device(device)
    names, lengths, context = _sequence_inputs(seqs, device, param_db, context, termini)
    with jax.default_device(device):
        pose = PoseStackBuilder.from_block_type_names(
            context.packed_block_types, names, lengths
        )
    types = context.packed_block_types.active_block_types
    kinds = np.asarray(pose.block_type_ind)
    offsets = np.asarray(pose.block_coord_offset)
    coords = extended.ideal_chained_coords(
        types,
        kinds,
        offsets,
        np.asarray(pose.inter_residue_connections),
        pose.coords.shape,
    )
    pose = pose.replace(coords=jax.device_put(coords.astype(np.float32), device))
    requests, values = extended.extended_torsion_requests(types, kinds)
    if requests:
        from tmol_jax.pose.query import _measure_torsions
        from tmol_jax.kinematics.torsion_updates import _apply_torsion_deltas

        deltas = np.radians(np.asarray(values)) - _measure_torsions(
            pose, requests, False
        )
        defined = ~np.isnan(deltas)
        pose = _apply_torsion_deltas(
            pose, [r for r, keep in zip(requests, defined) if keep], deltas[defined]
        )
    chain_ids = np.asarray(pose.chain_id)
    shifts = extended.chain_separation_shifts(
        types, kinds, offsets, chain_ids, np.asarray(pose.coords)
    )
    coords = pose.coords
    for p, indices, shift in shifts:
        coords = coords.at[p, jax.device_put(indices, device), 0].add(shift)
    pose = pose.replace(coords=coords)
    return (pose, context) if return_context else pose
