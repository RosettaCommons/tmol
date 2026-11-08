"""Eager task masks and device storage for common packing preparation."""

from dataclasses import fields

import jax
import jax.numpy as jnp
import numpy as np

from tmol_core.pack.task import chi_sample_budget, prepare_packing_task_rows
from .palette import PackerPalette


class PackerTask:
    """Select residue choices and samplers for a pose.

    Task construction and mutation are eager host operations. Palette defaults
    provide fallback coverage; additional samplers may be registered explicitly.
    Concrete tasks feed public conformer construction and native packing.
    """

    def __init__(self, systems, palette=None):
        palette = palette or PackerPalette()
        self.pose_stack = systems
        self.pbt = systems.packed_block_types
        self.device = systems.coords.device
        self._coordinate_shape = systems.coords.shape
        self._topology = (
            systems.block_type_ind,
            systems.block_coord_offset,
            systems.inter_residue_connections,
        )
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
        self.conformer_samplers = list(
            {id(s): s for s in palette.default_conformer_samplers()}.values()
        )
        self.conformer_sampler_index = {
            id(s): i for i, s in enumerate(self.conformer_samplers)
        }
        with jax.default_device(self.device):
            self.per_block_is_block_type_allowed = jnp.ones_like(
                self.per_block_considered_block_types, dtype=bool
            )
            self.per_block_chi_expansion = jnp.zeros(
                self.per_block_considered_block_types.shape + (4,), dtype=jnp.int32
            )
            self.per_block_conformer_sampler_allowed = jnp.ones(
                self.is_real_block.shape + (len(self.conformer_samplers),), dtype=bool
            )
        self.chi_sample_budget = None
        self.bump_check = False

    def _mask(self, value):
        value = np.asarray(value)
        if value.shape != self.is_real_block.shape or value.dtype != np.bool_:
            raise ValueError("block mask must be boolean with the pose/block shape")
        return jax.device_put(value, self.device)

    def _validate_pose(self, pose):
        if (
            pose.packed_block_types is not self.pbt
            or pose.coords.shape != self._coordinate_shape
            or pose.coords.device != self.device
            or any(
                expected is not actual and not np.array_equal(expected, actual)
                for expected, actual in zip(
                    self._topology,
                    (
                        pose.block_type_ind,
                        pose.block_coord_offset,
                        pose.inter_residue_connections,
                    ),
                )
            )
        ):
            raise ValueError(
                "sampling task belongs to a different pose layout or device"
            )

    def set_chi_sample_budget(self, expanded_limit, limit):
        self.chi_sample_budget = chi_sample_budget(expanded_limit, limit)

    def or_bump_check(self, setting=True):
        self.bump_check |= bool(setting)

    def restrict_to_repacking(self, block_mask=None):
        restriction = self.restrict_to_repacking_masks
        if block_mask is not None:
            restriction = restriction | ~self._mask(block_mask)[..., None]
        self.per_block_is_block_type_allowed &= restriction

    def restrict_absent_name3s(self, name3s, block_mask=None):
        """Restrict identities monotonically, preserving unselected blocks.

        Like Torch, an explicit mask rejects a restriction that empties a
        previously allowed selected block. A global restriction may empty rows.
        """
        names = frozenset(name3s)
        mask = self._mask(block_mask) if block_mask is not None else None
        choices = self.per_block_considered_block_types
        matches = jax.device_put(
            np.asarray([bt.name3 in names for bt in self.pbt.active_block_types]),
            self.device,
        )
        restriction = (choices >= 0) & matches[jnp.maximum(choices, 0)]
        if mask is not None:
            restriction |= ~mask[..., None]
        previous = self.per_block_is_block_type_allowed
        allowed = previous & restriction
        if mask is not None:
            emptied = np.asarray(mask & previous.any(-1) & ~allowed.any(-1))
            if emptied.any():
                pose, block = map(int, np.argwhere(emptied)[0])
                original = self.pbt.active_block_types[
                    int(self.per_block_orig_block_type[pose, block])
                ]
                raise ValueError(
                    f"pose {pose} block {block} ({original.name}): "
                    "no allowed block type "
                    f"has a name3 in {sorted(names)}"
                )
        self.per_block_is_block_type_allowed = allowed

    def disable_packing_by_block_mask(self, block_mask):
        self.per_block_is_block_type_allowed &= ~self._mask(block_mask)[..., None]

    def add_conformer_sampler(self, sampler):
        expand = getattr(sampler, "expand_into_task", None)
        if expand is not None:
            expand(self)
            return
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

    _validate_pose = PackerTask._validate_pose

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
        prepared = {}
        for field in fields(rows):
            value = getattr(rows, field.name)
            # Device indices use the native int32 ABI, independent of x64 mode.
            if value.dtype.kind in "iu":
                if value.size and np.max(value) >= 1 << 31:
                    raise OverflowError("packing row indices exceed int32")
                value = value.astype(np.int32)
            prepared[field.name] = value
        result.__dict__.update(jax.device_put(prepared, task.device))
        return result
