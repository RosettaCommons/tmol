"""Eager selection of input conformations using the common packing policy."""

import jax
import jax.numpy as jnp

from tmol_core.pack.conformer_policy import fallback_mask, include_current_mask
from .task import PackerTask, SetPackerTask


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
            task.per_block_conformer_sampler_allowed[rows[0], rows[1], sampler],
        )

    @staticmethod
    def _result(selected):
        return (
            selected.astype(jnp.int32),
            jnp.flatnonzero(selected).astype(jnp.int32),
            {"copy_input_coordinates": True},
        )

    def create_samples_for_poses(self, pose_stack, task):
        task, rows, allowed = self._selection_inputs(pose_stack, task)
        return self._result(
            include_current_mask(
                task.per_block_considered_block_types_is_orig[rows],
                task.per_block_is_block_type_allowed[rows],
                allowed,
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
        task, rows, allowed = self._selection_inputs(pose_stack, task)
        with jax.default_device(pose_stack.coords.device):
            covered = jnp.zeros(task.is_real_block.shape, dtype=jnp.int32)
            if built_rotamer_counts is not None:
                counts = jnp.asarray(built_rotamer_counts)
                if counts.shape != task.cons_bt_pose.shape:
                    raise ValueError(
                        "built counts must have one row per considered choice"
                    )
                covered = covered.at[rows[:2]].add((counts > 0).astype(jnp.int32))
            return self._result(
                fallback_mask(
                    task.per_block_considered_block_types_is_orig[rows],
                    ~task.per_block_is_block_type_allowed.any(axis=-1)[rows[:2]],
                    ~task.per_block_conformer_sampler_allowed.any(axis=-1)[rows[:2]],
                    covered[rows[:2]] > 0,
                    allowed,
                )
            )
