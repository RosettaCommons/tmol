"""Eager selection of input conformations using the common packing policy."""

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from tmol_core.pack.conformer_policy import fallback_mask, include_current_mask
from .task import PackerTask, SetPackerTask


@partial(jax.jit, static_argnames=("fallback",))
def _selection(
    original, allowed_types, allowed_samplers, rows, sampler, counts, *, fallback
):
    allowed = allowed_samplers[rows[0], rows[1], sampler]
    if not fallback:
        return include_current_mask(original[rows], allowed_types[rows], allowed)
    covered = jnp.zeros(original.shape[:2], jnp.int32)
    if counts is not None:
        covered = covered.at[rows[:2]].add((counts > 0).astype(jnp.int32))
    return fallback_mask(
        original[rows],
        ~allowed_types.any(axis=-1)[rows[:2]],
        ~allowed_samplers.any(axis=-1)[rows[:2]],
        covered[rows[:2]] > 0,
        allowed,
    )


class IncludeCurrentSampler:
    """Select allowed original identities for subsequent coordinate assembly.

    ``create_samples_for_poses`` returns considered-choice counts and indices,
    plus the input-coordinate copy marker used by conformer builders.
    Selection is eager because the number of selected conformers is dynamic.
    """

    samples_after_other_samplers = False

    @classmethod
    def sampler_name(cls):
        return cls.__name__

    def defines_rotamers_for_rt(self, restype):
        return True

    def first_sc_atoms_for_rt(self, restype):
        return (restype.default_jump_connection_atom,)

    def _selection_inputs(self, pose, task):
        if isinstance(pose.coords, jax.core.Tracer):
            raise TypeError("dynamic conformer selection requires eager coordinates")
        if isinstance(task, PackerTask):
            task = SetPackerTask.from_packer_task(task)
        task._validate_pose(pose)
        if id(self) not in task.conformer_sampler_index:
            raise ValueError("sampler must be registered with the packing task")
        rows = (task.cons_bt_pose, task.cons_bt_block, task.cons_bt_which_block_type)
        sampler = task.conformer_sampler_index[id(self)]
        return (
            task,
            rows,
            sampler,
        )

    @staticmethod
    def _result(selected):
        return (
            selected.astype(jnp.int32),
            jax.device_put(
                np.flatnonzero(np.asarray(selected)).astype(np.int32), selected.device
            ),
            {"copy_input_coordinates": True},
        )

    def create_samples_for_poses(self, pose_stack, task):
        task, rows, sampler = self._selection_inputs(pose_stack, task)
        return self._result(
            _selection(
                task.per_block_considered_block_types_is_orig,
                task.per_block_is_block_type_allowed,
                task.per_block_conformer_sampler_allowed,
                rows,
                sampler,
                None,
                fallback=False,
            )
        )


class FallbackSampler(IncludeCurrentSampler):
    """Select input conformations for fixed or actually uncovered positions.

    ``built_rotamer_counts`` contains counts per considered choice from all
    preceding samplers. Coverage is aggregated over identities at a position,
    so designing another identity never reintroduces the original as fallback.
    """

    samples_after_other_samplers = True

    def create_samples_for_poses(self, pose_stack, task, *, built_rotamer_counts=None):
        task, rows, sampler = self._selection_inputs(pose_stack, task)
        if built_rotamer_counts is not None:
            with jax.default_device(pose_stack.coords.device):
                built_rotamer_counts = jnp.asarray(built_rotamer_counts)
        if (
            built_rotamer_counts is not None
            and built_rotamer_counts.shape != task.cons_bt_pose.shape
        ):
            raise ValueError("built counts must have one row per considered choice")
        return self._result(
            _selection(
                task.per_block_considered_block_types_is_orig,
                task.per_block_is_block_type_allowed,
                task.per_block_conformer_sampler_allowed,
                rows,
                sampler,
                built_rotamer_counts,
                fallback=True,
            )
        )
