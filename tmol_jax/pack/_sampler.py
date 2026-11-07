"""Framework storage for common-backed residue sampling policies."""

import jax
import numpy as np


class SamplerSupport:
    def defines_rotamers_for_bts(self, pbt, bt_inds):
        """Query prepared residue support with the input index shape.

        Like task preparation, this query is eager. Chemistry comes from the
        sampler's existing residue policy and its parameter snapshot.
        """
        indices = np.asarray(bt_inds)
        size = len(pbt.active_block_types)
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("block type indices must be integers")
        if np.any((indices < -size) | (indices >= size)):
            raise IndexError("block type index is outside the palette")
        supported = np.asarray(
            [self.defines_rotamers_for_rt(rt) for rt in pbt.active_block_types],
            dtype=bool,
        )
        device = pbt.n_atoms.device
        return jax.device_put(supported, device)[jax.device_put(indices, device)]
