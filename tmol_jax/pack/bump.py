"""Native bump bounds with two explicit allocation barriers and fresh scratch."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ._ffi import stage
from .graph import _integer, _wide, prefix_offsets_and_total
from .dispatch import _narrow_prefix


@partial(jax.jit, static_argnames=("operation", "outputs", "extent", "chunk_size"))
def _stage(arguments, *, operation, outputs, extent=0, chunk_size=16):
    return stage(
        arguments,
        "tmol_graph_bounds",
        outputs,
        operation,
        extent,
        chunk_size,
        aliases={7: 4, 9: 5, 10: 1} if operation == 2 else None,
    )


def bump_bounds(rotamers, values, indices, *, chunk_size=16):
    """Return shared native per-rotamer and per-block lower bounds.

    Sparse indices must be canonical upper-triangle entries with diagonal
    one-body entries. Duplicates accumulate; no independence is presumed.
    Cardinality and storage allocation are eager, while all scientific bounds
    and scans execute on the input device.
    """
    p, b = rotamers.n_rots_for_block.shape
    r = rotamers.pose_for_rot.size
    arguments = (
        rotamers.n_rots_for_block,
        rotamers.rot_offset_for_block,
        rotamers.pose_for_rot,
        rotamers.block_ind_for_rot,
        indices,
        values,
    )
    with jax.default_device(values.device):
        _, adjacency, counts = _stage(
            arguments,
            operation=0,
            outputs=(_integer(p, b), _integer(p, b, b), _wide(p, b, b)),
            chunk_size=chunk_size,
        )
        pair_offsets, total = prefix_offsets_and_total(counts.reshape(-1, 8))
        chunks = int(np.asarray(total).copy().view(np.int64)[0])
        if chunks > np.iinfo(np.int32).max:
            raise OverflowError("bump chunk-pair dispatch exceeds int32")
        pair_offsets = pair_offsets.reshape(p, b, b, 8)
        sizes = _stage(
            (*arguments, pair_offsets),
            operation=1,
            outputs=_wide(chunks),
            extent=chunks,
            chunk_size=chunk_size,
        )
        chunk_offsets, total = prefix_offsets_and_total(sizes)
        energies = int(np.asarray(total).copy().view(np.int64)[0])
        shape = jax.ShapeDtypeStruct
        results = _stage(
            (
                *arguments,
                adjacency,
                pair_offsets,
                sizes,
                chunk_offsets,
                jnp.full((p, b), 1234, values.dtype),
            ),
            operation=2,
            extent=energies,
            chunk_size=chunk_size,
            outputs=(
                shape((r,), values.dtype),
                shape((p, b), values.dtype),
                shape((r,), values.dtype),
                shape((energies,), values.dtype),
                _wide(p, b, b),
                _wide(chunks),
            ),
        )
        return results[:2]


@jax.jit
def retained_entry_mapping(
    rotamers, fixed, indices, block_of, original_offsets, original_counts
):
    """Compact surviving entry identities while preserving original assignments.

    Native retention already chose the surviving molten and fixed conformers.
    This is only a layout conversion for the streaming graph's contiguous rows.
    """
    inverse = _narrow_prefix(rotamers)
    fixed = _narrow_prefix(fixed.reshape(-1, 8)).reshape(fixed.shape[:-1])
    n = block_of.size
    keep = jnp.zeros(n, jnp.bool_).at[inverse].set(True)
    keep = keep.at[jnp.where(fixed >= 0, fixed, n)].set(True, mode="drop")
    prefix = jnp.concatenate(
        (jnp.zeros(1, jnp.int32), jnp.cumsum(keep, dtype=jnp.int32))
    )
    offsets = prefix[original_offsets]
    counts = prefix[original_offsets + original_counts] - offsets
    entry_keep = keep[indices[1]] & keep[indices[2]]
    new_fixed = jnp.where(fixed >= 0, prefix[jnp.maximum(fixed, 0)], -1)
    return keep, entry_keep, prefix[:-1], offsets, counts, new_fixed


@partial(jax.jit, static_argnames=("rotamers", "entries"))
def compact_entries(
    values, indices, keep, entry_keep, mapping, block_of, *, rotamers, entries
):
    selected = jnp.nonzero(entry_keep, size=entries)[0]
    source = jnp.nonzero(keep, size=rotamers)[0]
    return (
        values[selected],
        jnp.stack(
            (
                indices[0, selected],
                mapping[indices[1, selected]],
                mapping[indices[2, selected]],
            )
        ),
        block_of[source],
    )
