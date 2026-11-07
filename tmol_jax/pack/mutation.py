"""Public mutation execution over shared selection and task restrictions."""

import jax
import numpy as np

from tmol_core.pack.mutation import configure_mutation_task, sequence_mutations
from tmol_jax.score import beta2016_score_function
from .palette import PackerPalette
from .task import PackerTask
from .dunbrack import DunbrackChiSampler
from .fixed_amino_acid import FixedAAChiSampler
from .current import IncludeCurrentSampler
from .packing import pack_rotamers


def _mutate_blocks(pose_stack, pose_index, plan, score_function, param_db, seed):
    if score_function is None:
        score_function = beta2016_score_function(param_db)
    if param_db is None:
        param_db = score_function.param_db
    task = PackerTask(pose_stack, PackerPalette())
    configure_mutation_task(
        task,
        pose_stack.block_type_ind.shape,
        pose_index,
        plan,
        lambda mask: jax.device_put(mask, pose_stack.device),
    )
    task.add_conformer_sampler(DunbrackChiSampler.from_database(param_db))
    task.add_conformer_sampler(FixedAAChiSampler())
    task.add_conformer_sampler(IncludeCurrentSampler())
    return pack_rotamers(pose_stack, score_function, task, seed=seed)


def mutate_residue(
    pose_stack,
    pose_index,
    block,
    new_identity,
    score_function=None,
    *,
    param_db=None,
    seed=None,
):
    """Design one residue, freezing all other blocks and poses."""
    return _mutate_blocks(
        pose_stack, pose_index, {block: new_identity}, score_function, param_db, seed
    )


def mutate_to_sequence(
    pose_stack, pose_index, sequence, score_function=None, *, param_db=None, seed=None
):
    """Apply the shared one-letter mutation contract to one pose."""
    plan = sequence_mutations(
        pose_stack.packed_block_types.active_block_types,
        np.asarray(pose_stack.block_type_ind[pose_index]),
        sequence,
        pose_index,
    )
    if not plan:
        return pose_stack
    return _mutate_blocks(pose_stack, pose_index, plan, score_function, param_db, seed)
