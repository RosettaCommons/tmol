"""JAX storage and native packing execution for common input completion."""

import jax
import numpy as np

from tmol_core.pack.input_completion import prepare_input_completion_task
from tmol_jax.pack import (
    PackerTask,
    FixedAAChiSampler,
    OptHSampler,
    pack_rotamers,
)


def _supported_blocks(sampler, pose):
    types = np.asarray(pose.block_type_ind)
    supported = np.asarray(
        [
            sampler.defines_rotamers_for_rt(rt)
            for rt in pose.packed_block_types.active_block_types
        ]
    )
    return jax.device_put(supported[np.maximum(types, 0)], pose.device)


def complete_input(pose, missing, context, *, no_optH, seed, return_frozen=False):
    has_missing = bool(np.asarray(missing).any())
    if not has_missing and no_optH:
        return (pose, pose.block_type_ind >= 0) if return_frozen else pose
    task = prepare_input_completion_task(
        pose,
        missing,
        task=PackerTask(pose),
        dunbrack=context._dunbrack_sampler if has_missing else None,
        fixed=FixedAAChiSampler(),
        opth=None if no_optH else OptHSampler(),
        nucleic_acid=context._na_sampler if has_missing else None,
        has_missing_atoms=has_missing,
        supported_blocks=_supported_blocks,
        any_block=lambda mask: bool(np.asarray(mask).any()),
    )
    score = (
        context._packing_score_function if has_missing else context._opth_score_function
    )
    if seed is None:
        seed = int(np.random.PCG64().random_raw())
    result = pack_rotamers(pose, score, task, seed=seed)
    if not return_frozen:
        return result
    frozen = task.is_real_block & ~task.per_block_is_block_type_allowed.any(axis=-1)
    return result, frozen


def input_packer_atoms(context):
    from functools import partial
    from tmol_core.pack.input_completion import packer_atom_mask
    from tmol_core.pack.kinematics import annotate_residue_rotamer_kinematics
    from tmol_jax.kinematics._host import NativePosePlanner

    prepare = partial(
        annotate_residue_rotamer_kinematics,
        inverse_kinematics=NativePosePlanner().inverse_kinematics,
    )
    return partial(
        packer_atom_mask,
        samplers=(context._dunbrack_sampler, FixedAAChiSampler()),
        prepare_kinematics=prepare,
    )
