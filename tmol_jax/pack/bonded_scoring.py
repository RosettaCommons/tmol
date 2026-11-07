"""Sparse connected conformer execution from common native preparation."""

from functools import partial

import jax
import numpy as np

from tmol_jax.score.genbonded.genbonded_energy_term import GenBondedEnergyTerm
from tmol_jax.score.cartbonded.cartbonded_energy_term import CartBondedEnergyTerm
from .graph import prefix_counts, load_balance
from .scoring import validate_rotamer_context, sparse_nonbonded_values

from ._ffi import stage


@partial(jax.jit, static_argnames=("operation", "outputs"))
def _connected_stage(arguments, *, operation, outputs):
    target = "tmol_connected_rotamer_dispatch"
    return stage(arguments, target, outputs, operation)


def connected_candidates(pose, rotamers):
    """Prepare discrete intra/connected identities, preserving lockstep policy."""
    validate_rotamer_context(pose, rotamers)
    if isinstance(rotamers.coords, jax.core.Tracer):
        raise TypeError("dynamic connected candidates must be prepared outside jit")
    connections = pose.inter_residue_connections
    arguments = (
        rotamers.block_ind_for_rot,
        rotamers.pose_for_rot,
        rotamers.block_type_ind_for_rot,
        rotamers.n_rots_for_block,
        rotamers.group_for_block,
        connections,
    )
    count = rotamers.n_rotamers_total * (connections.shape[2] + 1)
    if count >= np.iinfo(np.int32).max:
        raise OverflowError("connected conformer count exceeds native capacity")
    shape = jax.ShapeDtypeStruct
    counts = _connected_stage(
        arguments, operation=0, outputs=shape((count, 8), np.uint8)
    )
    offsets = prefix_counts(counts)
    total = int(np.asarray(offsets[-1]).copy().view(np.int64)[0])
    owners = load_balance(offsets[:-1], total=total)
    indices = _connected_stage(
        (*arguments, rotamers.rot_offset_for_block, offsets[:-1], owners),
        operation=1,
        outputs=shape((3, total), np.int32),
    )
    return owners, indices


def prepare_rotamer_bonded(pose, rotamers, database, *, cartesian=False, term=None):
    validate_rotamer_context(pose, rotamers)
    with jax.default_device(pose.device):
        cls = CartBondedEnergyTerm if cartesian else GenBondedEnergyTerm
        options = {} if cartesian else {"include_empty": True}
        term = cls(database) if term is None else term
        parameters = term.setup_for_pose_stack(pose, **options)
        owners, indices = connected_candidates(pose, rotamers)
    parameters = (
        rotamers.coord_offset_for_rot,
        rotamers.block_ind_for_rot,
        rotamers.pose_for_rot,
        *parameters[3:],
        rotamers.block_type_ind_for_rot,
        owners,
    )
    target = "tmol_rotamer_cartbonded" if cartesian else "tmol_rotamer_genbonded"
    return (target, 5 if cartesian else 1, parameters, 0.0), indices


def prepare_rotamer_genbonded(pose, rotamers, database, *, term=None):
    return prepare_rotamer_bonded(pose, rotamers, database, term=term)


def prepare_rotamer_cartbonded(pose, rotamers, database, *, term=None):
    return prepare_rotamer_bonded(pose, rotamers, database, cartesian=True, term=term)


def sparse_bonded_scores(coords, state):
    """Score fixed connected candidates with native values and coordinate VJP."""
    prepared, indices = state
    return sparse_nonbonded_values(coords, prepared, indices), indices


sparse_genbonded_scores = sparse_bonded_scores
