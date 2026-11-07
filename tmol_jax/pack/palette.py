"""JAX gathers for the common chemical packing palette."""

import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import replace

from tmol_core.cache import AnnotationKey, PreparationCache
from tmol_core.pack.palette import prepare_packer_palette


class PackerPalette:
    """Inspect chemically compatible choices from the shared TMol palette.

    These choices feed public sampling, conformer construction and packing.
    Unsupported task operations raise explicitly.
    """

    def __init__(self, noncanonical_design=False):
        self.noncanonical_design = bool(noncanonical_design)

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
        if not hasattr(pbt, "_packer_palette_cache"):
            pbt._packer_palette_cache = PreparationCache()
        key = AnnotationKey.from_sources(
            pbt.chem_db,
            *pbt.active_block_types,
            settings=(self.noncanonical_design, orig.device),
        )

        def prepare():
            host = prepare_packer_palette(
                pbt.chem_db, pbt.active_block_types, self.noncanonical_design
            )
            return replace(
                host,
                **{
                    name: jax.device_put(
                        np.asarray(getattr(host, name), dtype=dtype), orig.device
                    )
                    for name, dtype in (
                        ("n_allowed", np.int32),
                        ("allowed_types", np.int32),
                        ("is_original", bool),
                        ("repack_mask", bool),
                    )
                },
            )

        data = pbt._packer_palette_cache.get_or_create("choices", key, prepare)
        return data, orig, orig.device

    def block_types_from_original(self, pbt, orig):
        data, indices, device = self._data(pbt, orig)
        with jax.default_device(device):
            real, safe = indices >= 0, jnp.maximum(indices, 0)
            return (
                jnp.where(real, data.n_allowed[safe], 0),
                jnp.where(real[..., None], data.allowed_types[safe], -1),
                real[..., None] & data.is_original[safe],
            )

    def create_restrict_to_repacking_mask(self, pbt, orig):
        data, indices, device = self._data(pbt, orig)
        with jax.default_device(device):
            return (indices >= 0)[..., None] & data.repack_mask[jnp.maximum(indices, 0)]
