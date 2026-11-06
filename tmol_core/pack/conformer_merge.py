"""Eager metadata plans for combining conformer samplers and enforcing budgets."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConformerMerge:
    counts: np.ndarray
    sampler: np.ndarray
    considered: np.ndarray
    destinations: tuple[np.ndarray, ...]


def prepare_conformer_merge(samples):
    """Preserve considered-type, sampler, then original source-row ordering.

    Each pair contains host counts and ordered considered-choice indices from
    one sampler. These are discrete topology metadata, not coordinates or DOFs.
    Validate every row before allocating or populating the merged plan.
    """
    if not samples:
        raise ValueError("At least one conformer sampler is required")
    samples = [(np.asarray(counts), np.asarray(indices)) for counts, indices in samples]
    if any(
        a.dtype not in (np.dtype("int32"), np.dtype("int64"))
        for sample in samples
        for a in sample
    ):
        raise ValueError("Conformer sample counts and indices must be integers")
    n_choices = samples[0][0].size
    if any(c.shape != (n_choices,) or i.ndim != 1 for c, i in samples):
        raise ValueError("Conformer sample counts and source rows must be vectors")
    n_conformers = sum(indices.size for _, indices in samples)
    if max(n_conformers, n_choices) > np.iinfo(np.int32).max:
        raise ValueError("Conformer sample counts exceed native 32-bit indexing")
    counts = np.stack([sample[0] for sample in samples]).astype(np.int64)
    for row, (_, indices) in zip(counts, samples):
        if np.any(row < 0) or np.any(row > indices.size) or row.sum() != indices.size:
            raise ValueError("Conformer sample counts do not match the source rows")
    source_ends = counts.cumsum(axis=1)
    expected_types = np.arange(n_choices)
    for row, ends, (_, indices) in zip(counts, source_ends, samples):
        if not indices.size:
            continue
        starts = np.clip(ends - row, 0, indices.size - 1)
        last = np.clip(ends - 1, 0, indices.size - 1)
        boundaries = (indices[starts] == expected_types) & (
            indices[last] == expected_types
        )
        if not np.all((row == 0) | boundaries) or not np.all(
            indices[1:] >= indices[:-1]
        ):
            raise ValueError("Conformer source rows do not match ordered sample counts")
    merged_counts = counts.sum(axis=0)
    merged_offsets = merged_counts.cumsum() - merged_counts
    offsets = merged_offsets + counts.cumsum(axis=0) - source_ends
    sampler = np.empty(n_conformers, dtype=np.int64)
    considered = np.empty(n_conformers, dtype=np.result_type(*[s[1] for s in samples]))
    destinations = []
    for i, (_, indices) in enumerate(samples):
        destination = offsets[i, indices] + np.arange(indices.size, dtype=np.int64)
        sampler[destination] = i
        considered[destination] = indices
        destinations.append(destination)
    return ConformerMerge(
        merged_counts.astype(np.result_type(*[s[0] for s in samples]), copy=False),
        sampler,
        considered,
        tuple(destinations),
    )


def check_task_sample_budget(shape, pose_of, block_of, counts, budget):
    """Bound combined choices/samplers at each physical residue before merging."""
    if budget is None or not shape[0] * shape[1]:
        return
    counts = np.asarray(counts, dtype=np.int64)
    capacity = np.iinfo(np.int32).max
    if np.any(counts < 0) or np.any(counts > capacity):
        raise ValueError(
            "Sampling count exceeds index capacity or contains a negative count"
        )
    per_block = np.zeros(shape[0] * shape[1], dtype=np.int64)
    np.add.at(
        per_block,
        np.asarray(pose_of) * shape[1] + np.asarray(block_of),
        counts.sum(axis=0),
    )
    index = int(per_block.argmax())
    largest, maximum = int(per_block[index]), max(budget)
    if largest > maximum:
        raise ValueError(
            f"Sampling budget {maximum} cannot fit {largest} combined rotamers "
            f"at pose {index // shape[1]} block {index % shape[1]} "
            "across samplers and allowed residue types"
        )
