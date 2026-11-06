"""Device execution for shared conformer count and ordering plans."""

import jax
import jax.numpy as jnp
import numpy as np

from tmol_core.pack.conformer_merge import (
    prepare_conformer_merge_counts,
    conformer_rows_are_ordered,
    conformer_destinations,
)


@jax.jit
def _merge_rows(indices, counts, offsets, ends):
    total = sum(row.shape[0] for row in indices)
    sampler = jnp.empty(total, dtype=jnp.int32)
    considered = jnp.empty(total, dtype=jnp.int32)
    choices = jnp.arange(counts.shape[1], dtype=jnp.int32)
    valid, destinations = jnp.asarray(True), []
    for i, rows in enumerate(indices):
        valid &= conformer_rows_are_ordered(rows, counts[i], ends[i], choices)
        destination = conformer_destinations(
            offsets[i], rows, jnp.arange(rows.shape[0], dtype=jnp.int32)
        )
        sampler = sampler.at[destination].set(i)
        considered = considered.at[destination].set(rows.astype(jnp.int32))
        destinations.append(destination)
    return valid, sampler, considered, tuple(destinations)


def merge_conformer_samples(samples):
    """Merge eager sample rows while keeping per-conformer arrays on device."""
    if not samples:
        raise ValueError("At least one conformer sampler is required")
    device = samples[0][0].device
    if any(a.device != device for sample in samples for a in sample[:2]):
        raise ValueError("Conformer samples must share a device")
    if any(
        a.dtype not in (jnp.int32, jnp.int64) for sample in samples for a in sample[:2]
    ):
        raise ValueError("Conformer sample counts and indices must be integers")
    counts = np.stack([np.asarray(sample[0]) for sample in samples])
    plan = prepare_conformer_merge_counts(
        counts, [sample[1].size for sample in samples]
    )

    def put(value):
        return jax.device_put(np.asarray(value, dtype=np.int32), device)

    valid, sampler, considered, destinations = _merge_rows(
        tuple(sample[1] for sample in samples),
        put(counts),
        put(plan.offsets),
        put(plan.source_ends),
    )
    if not bool(valid):
        raise ValueError("Conformer source rows do not match ordered sample counts")
    return (
        put(plan.counts),
        sampler,
        considered,
        [sampler == i for i in range(len(samples))],
        list(destinations),
    )
