"""JAX gathers for the common chemical packing palette."""

import jax
import jax.numpy as jnp
import numpy as np

from tmol_core.pack.palette import prepare_packer_palette


class PackerPalette:
    """Inspect chemically compatible choices from the shared TMol palette.

    These choices feed public sampling, conformer construction and packing.
    Unsupported task operations raise explicitly.
    """

    def __init__(self, noncanonical_design=False):
        self.noncanonical_design = bool(noncanonical_design)
        self._prepared = None

    def default_conformer_samplers(self):
        from .current import FallbackSampler

        return [FallbackSampler()]

    def _data(self, pbt, orig):
        indices = np.asarray(orig)
        if indices.ndim != 2 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(
                "original block types must be a two-dimensional integer array"
            )
        if np.any(indices < -1) or np.any(indices >= pbt.n_types):
            raise ValueError("original block type index is outside the palette")
        owners = (pbt.chem_db, pbt.active_block_types, self.noncanonical_design)
        if self._prepared is None or not all(
            x is y for x, y in zip(owners, self._prepared[0])
        ):
            self._prepared = (
                owners,
                prepare_packer_palette(*owners[:2], noncanonical_design=owners[2]),
            )
        return self._prepared[1], jax.device_put(indices, orig.device), orig.device

    def block_types_from_original(self, pbt, orig):
        data, indices, device = self._data(pbt, orig)
        with jax.default_device(device):
            real, safe = indices >= 0, jnp.maximum(indices, 0)
            counts = jnp.asarray(data.n_allowed, dtype=jnp.int32)
            choices = jnp.asarray(data.allowed_types, dtype=jnp.int32)
            original = jnp.asarray(data.is_original)
            return (
                jnp.where(real, counts[safe], 0),
                jnp.where(real[..., None], choices[safe], -1),
                real[..., None] & original[safe],
            )

    def create_restrict_to_repacking_mask(self, pbt, orig):
        data, indices, device = self._data(pbt, orig)
        with jax.default_device(device):
            return (indices >= 0)[..., None] & jnp.asarray(data.repack_mask)[
                jnp.maximum(indices, 0)
            ]
