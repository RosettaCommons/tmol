"""Sparse backbone execution with shared native candidate and physics kernels."""

from functools import partial

import jax
import numpy as np

from tmol_jax.score.backbone_torsion.backbone_torsion_energy_term import (
    BackboneTorsionEnergyTerm,
)
from .graph import prefix_counts, load_balance
from .scoring import validate_rotamer_context
from .bonded_scoring import sparse_bonded_scores

from ._ffi import stage


@partial(jax.jit, static_argnames=("operation", "outputs"))
def _stage(arguments, *, operation, outputs):
    target = "tmol_backbone_rotamer_topology"
    return stage(arguments, target, outputs, operation)


def prepare_rotamer_backbone(pose, rotamers, database, *, term=None):
    validate_rotamer_context(pose, rotamers)
    term = BackboneTorsionEnergyTerm(database) if term is None else term
    with jax.default_device(pose.device):
        parameters = term.setup_for_pose_stack(pose).parameters
        arguments = (
            rotamers.n_rots_for_block,
            rotamers.first_rot_block_type,
            pose.inter_residue_connections,
            parameters[8],
        )
        shape = jax.ShapeDtypeStruct
        counts = _stage(
            (*arguments, rotamers.group_for_block),
            operation=0,
            outputs=shape((*rotamers.n_rots_for_block.shape, 8), np.uint8),
        )
        offsets = prefix_counts(counts.reshape(-1, 8))
        total = int(np.asarray(offsets[-1]).copy().view(np.int64)[0])
        owners = load_balance(offsets[:-1], total=total)
        indices = _stage(
            (
                *arguments,
                rotamers.rot_offset_for_block,
                offsets[:-1].reshape(counts.shape),
                owners,
                rotamers.group_for_block,
            ),
            operation=1,
            outputs=shape((3, total), np.int32),
        )
    parameters = (
        rotamers.coord_offset_for_rot,
        rotamers.rot_offset_for_block,
        rotamers.first_rot_block_type,
        *parameters[3:],
        rotamers.block_ind_for_rot,
        rotamers.block_type_ind_for_rot,
        rotamers.group_for_block,
    )
    return ("tmol_rotamer_backbone", 2, parameters, 0.0), indices


sparse_backbone_scores = sparse_bonded_scores
