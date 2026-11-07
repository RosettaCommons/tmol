"""Device execution of the shared producer-declared conformer correspondence."""

from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from tmol_core.pack.correlation import collapse_group_rotamers, expand_group_assignments


@jax.jit
def _compatible(indices, group, conformer):
    a, b = indices[1:]
    return (group[a] < 0) | (group[a] != group[b]) | (conformer[a] == conformer[b])


@partial(jax.jit, static_argnames=("size",))
def _remap(values, indices, keep, mapping, blocks, *, size):
    selected = jnp.nonzero(keep, size=size)[0]
    a, b = mapping[indices[1:, selected]]
    swap = blocks[a] > blocks[b]
    return values[selected], jnp.stack(
        (indices[0, selected], jnp.where(swap, b, a), jnp.where(swap, a, b))
    )


def collapse_for_graph(rotamers, values, indices):
    fields = (
        "n_rots_for_pose",
        "rot_offset_for_pose",
        "n_rots_for_block",
        "rot_offset_for_block",
        "pose_for_rot",
        "block_type_ind_for_rot",
        "block_ind_for_rot",
    )
    host = SimpleNamespace(
        **{name: np.asarray(getattr(rotamers, name)) for name in fields}
    )
    result = collapse_group_rotamers(host, rotamers.correlated_groups)
    if result is None:
        return rotamers, values, indices, None
    collapse, group, conformer = result
    arrays = jax.device_put(
        tuple(
            np.asarray(a, np.int32)
            for a in (
                group,
                conformer,
                collapse.rot_map,
                collapse.compact_block_ind_for_rot,
            )
        ),
        rotamers.device,
    )
    keep = _compatible(indices, *arrays[:2])
    size = int(np.asarray(jnp.sum(keep)))
    values, indices = _remap(values, indices, keep, *arrays[2:], size=size)
    compact = jax.device_put(
        {
            name: np.asarray(getattr(collapse, "compact_" + name), np.int32)
            for name in fields
        },
        rotamers.device,
    )
    layout = SimpleNamespace(**compact, device=rotamers.device)
    return layout, values, indices, collapse


def expand_selected(selected, collapse, device):
    """Map compact choices to original identities before native pose layout."""
    if collapse is None:
        return selected
    original = np.asarray(selected).copy().view(np.int64).reshape(selected.shape[:-1])
    real = original >= 0
    original[real] = collapse.compact_to_orig[original[real]]
    original = expand_group_assignments(original, collapse)
    return jax.device_put(original.view(np.uint8).reshape(*original.shape, 8), device)
