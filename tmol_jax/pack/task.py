"""Eager task masks and device storage for common packing preparation."""

from dataclasses import fields

import jax
import jax.numpy as jnp
import numpy as np

from tmol_core.pack.task import chi_sample_budget, prepare_packing_task_rows
from .palette import PackerPalette


class PackerTask:
    """Select residue choices and samplers for a pose.

    Task construction and mutation are eager host operations. Samplers must be
    registered explicitly; this interface does not yet execute rotamer packing
    or add a fallback conformation for residues outside a sampler's domain.
    """

    def __init__(self, systems, palette=None):
        palette = palette or PackerPalette()
        self.pose_stack = systems
        self.pbt = systems.packed_block_types
        self.device = systems.coords.device
        self.per_block_orig_block_type = systems.block_type_ind
        self.is_real_block = systems.block_type_ind >= 0
        (
            self.per_block_n_considered_block_types,
            self.per_block_considered_block_types,
            self.per_block_considered_block_types_is_orig,
        ) = palette.block_types_from_original(self.pbt, systems.block_type_ind)
        self.restrict_to_repacking_masks = palette.create_restrict_to_repacking_mask(
            self.pbt, systems.block_type_ind
        )
        with jax.default_device(self.device):
            self.per_block_is_block_type_allowed = jnp.ones_like(
                self.per_block_considered_block_types, dtype=bool
            )
            self.per_block_chi_expansion = jnp.zeros(
                self.per_block_considered_block_types.shape + (4,), dtype=jnp.int32
            )
            self.per_block_conformer_sampler_allowed = jnp.zeros(
                self.is_real_block.shape + (0,), dtype=bool
            )
        self.conformer_samplers = []
        self.conformer_sampler_index = {}
        self.chi_sample_budget = None

    def _mask(self, value):
        value = np.asarray(value)
        if value.shape != self.is_real_block.shape or value.dtype != np.bool_:
            raise ValueError("block mask must be boolean with the pose/block shape")
        return jax.device_put(value, self.device)

    def set_chi_sample_budget(self, expanded_limit, limit):
        self.chi_sample_budget = chi_sample_budget(expanded_limit, limit)

    def restrict_to_repacking(self, block_mask=None):
        restriction = self.restrict_to_repacking_masks
        if block_mask is not None:
            restriction = restriction | ~self._mask(block_mask)[..., None]
        self.per_block_is_block_type_allowed &= restriction

    def disable_packing_by_block_mask(self, block_mask):
        self.per_block_is_block_type_allowed &= ~self._mask(block_mask)[..., None]

    def add_conformer_sampler(self, sampler):
        self.add_conformer_sampler_by_block_mask(
            sampler, np.ones(self.is_real_block.shape, dtype=bool)
        )

    def add_conformer_sampler_by_block_mask(self, sampler, block_mask):
        mask = self._mask(block_mask)
        index = self.conformer_sampler_index.get(id(sampler))
        if index is None:
            self.conformer_sampler_index[id(sampler)] = len(self.conformer_samplers)
            self.conformer_samplers.append(sampler)
            self.per_block_conformer_sampler_allowed = jnp.concatenate(
                (self.per_block_conformer_sampler_allowed, mask[..., None]), axis=-1
            )
        else:
            self.per_block_conformer_sampler_allowed = (
                self.per_block_conformer_sampler_allowed.at[..., index].set(
                    self.per_block_conformer_sampler_allowed[..., index] | mask
                )
            )

    def disable_sampler_by_block_mask(self, sampler, block_mask):
        index = self.conformer_sampler_index[id(sampler)]
        self.per_block_conformer_sampler_allowed = (
            self.per_block_conformer_sampler_allowed.at[..., index].set(
                self.per_block_conformer_sampler_allowed[..., index]
                & ~self._mask(block_mask)
            )
        )

    def or_expand_chi(self, chi_ind):
        self.or_expand_chi_to(chi_ind, 1)

    def or_expand_chi_to(self, chi_ind, sample_level):
        if not 0 <= chi_ind < 4 or not 0 <= sample_level <= 1:
            raise ValueError("native sampling supports chi indices 0-3 and levels 0-1")
        self.per_block_chi_expansion = self.per_block_chi_expansion.at[
            ..., chi_ind
        ].max(sample_level)


class SetPackerTask:
    """Snapshot of task masks with canonical concrete row identities."""

    @classmethod
    def from_packer_task(cls, task):
        result = cls()
        result.__dict__.update(task.__dict__)
        result.conformer_samplers = list(task.conformer_samplers)
        result.conformer_sampler_index = dict(task.conformer_sampler_index)
        rows = prepare_packing_task_rows(
            np.asarray(task.is_real_block),
            np.asarray(task.per_block_considered_block_types),
            np.asarray(task.per_block_is_block_type_allowed),
        )
        for field in fields(rows):
            value = getattr(rows, field.name)
            # Device indices use the native int32 ABI, independent of x64 mode.
            if value.dtype.kind in "iu":
                if value.size and np.max(value) >= 1 << 31:
                    raise OverflowError("packing row indices exceed int32")
                value = value.astype(np.int32)
            setattr(result, field.name, jax.device_put(value, task.device))
        return result
