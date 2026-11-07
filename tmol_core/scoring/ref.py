"""Reference energies resolved from one shared chemistry record and weight map."""

import numpy as np

from tmol_core.chemical import _l_base_name


def reference_weight(block_type, weights):
    """Prefer an explicit residue weight, then its mirrored L residue, then zero."""
    value = weights.get(block_type.base_name)
    return weights.get(_l_base_name(block_type), 0.0) if value is None else value


def reference_energies(block_type_indices, values):
    """Prepare constant block and pose energies from host topology and weights.

    ``values`` contains one floating weight per packed type; negative indices
    denote padding. Accumulate stored weights in float64 before rounding once
    to their original dtype, so frontend/device float32 reduction trees do not
    change a pose's reference energy. Coordinate derivatives remain zero.
    """
    indices = np.asarray(block_type_indices)
    values = np.asarray(values)
    per_block = np.zeros(indices.shape, dtype=values.dtype)
    real = indices >= 0
    per_block[real] = values[indices[real]]
    per_pose = per_block.sum(axis=1, dtype=np.float64).astype(values.dtype)
    return per_block, per_pose
