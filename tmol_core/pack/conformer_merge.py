"""Eager metadata plans for combining conformer samplers and enforcing budgets."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConformerMerge:
    counts: np.ndarray
    sampler: np.ndarray
    considered: np.ndarray
    destinations: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class ConformerMergeCounts:
    counts: np.ndarray
    intervals: np.ndarray
    offsets: np.ndarray
    total: int


def prepare_conformer_merge_counts(counts, sizes):
    """Plan small per-choice prefixes without transferring per-conformer rows."""
    counts = np.asarray(counts)
    if counts.dtype not in (np.dtype("int32"), np.dtype("int64")):
        raise ValueError("Conformer sample counts and indices must be integers")
    if counts.ndim != 2 or counts.shape[0] != len(sizes):
        raise ValueError("Conformer sample counts must be equal-length vectors")
    total = sum(sizes)
    if max(total, counts.shape[1]) > np.iinfo(np.int32).max:
        raise ValueError("Conformer sample counts exceed native 32-bit indexing")
    wide = counts.astype(np.int64, copy=False)
    for row, size in zip(wide, sizes):
        if np.any(row < 0) or np.any(row > size) or row.sum() != size:
            raise ValueError("Conformer sample counts do not match the source rows")
    ends = wide.cumsum(axis=1)
    intervals = np.stack((ends - wide, ends - 1))
    np.clip(
        intervals, 0, np.maximum(np.asarray(sizes) - 1, 0)[None, :, None], out=intervals
    )
    merged = wide.sum(axis=0)
    offsets = merged.cumsum() - merged + wide.cumsum(axis=0) - ends
    return ConformerMergeCounts(merged.astype(counts.dtype), intervals, offsets, total)


def conformer_rows_are_ordered(indices, starts, ends, occupied, choices):
    """Validate ordered source intervals on arrays owned by the executor."""
    if not indices.shape[0]:
        return (indices >= 0).all()
    boundaries = (indices[starts] == choices) & (indices[ends] == choices)
    return ((~occupied) | boundaries).all() & (indices[1:] >= indices[:-1]).all()


def conformer_destinations(offsets, choices, source_rows):
    """Map each source row into the common considered-type/sampler ordering."""
    return offsets[choices] + source_rows


def prepare_conformer_merge(samples):
    """Assemble a complete host plan for host-resident conformer metadata."""
    if not samples:
        raise ValueError("At least one conformer sampler is required")
    samples = [(np.asarray(c), np.asarray(i)) for c, i in samples]
    if any(
        i.dtype not in (np.dtype("int32"), np.dtype("int64")) or i.ndim != 1
        for _, i in samples
    ):
        raise ValueError("Conformer sample counts and indices must be integer vectors")
    counts = np.stack([c for c, _ in samples])
    plan = prepare_conformer_merge_counts(counts, [i.size for _, i in samples])
    choices = np.arange(counts.shape[1])
    for i, (_, indices) in enumerate(samples):
        if not conformer_rows_are_ordered(
            indices, *plan.intervals[:, i], counts[i] > 0, choices
        ):
            raise ValueError("Conformer source rows do not match ordered sample counts")
    sampler = np.empty(plan.total, dtype=np.int64)
    considered = np.empty(plan.total, dtype=np.result_type(*[s[1] for s in samples]))
    destinations = []
    for i, (_, indices) in enumerate(samples):
        destination = conformer_destinations(
            plan.offsets[i], indices, np.arange(indices.size, dtype=np.int64)
        )
        sampler[destination] = i
        considered[destination] = indices
        destinations.append(destination)
    return ConformerMerge(plan.counts, sampler, considered, tuple(destinations))


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


def conformer_origin_counts(shape, pose_of, block_of, counts):
    """Aggregate per-choice counts into physical residue and pose prefixes."""
    block_counts = np.zeros(shape[0] * shape[1], dtype=np.int64)
    np.add.at(
        block_counts, np.asarray(pose_of) * shape[1] + np.asarray(block_of), counts
    )
    block_counts = block_counts.reshape(shape)
    pose_counts = block_counts.sum(axis=1)
    return (
        pose_counts,
        pose_counts.cumsum() - pose_counts,
        block_counts,
        (block_counts.ravel().cumsum() - block_counts.ravel()).reshape(shape),
    )


def prepare_conformer_origins(shape, pose_of, block_of, considered):
    """Index host-resident conformers by physical pose and residue position."""
    considered = np.asarray(considered)
    poses = np.asarray(pose_of, dtype=np.int64)[considered]
    blocks = np.asarray(block_of, dtype=np.int32)[considered]
    counts = np.bincount(considered, minlength=len(pose_of))
    return (*conformer_origin_counts(shape, pose_of, block_of, counts), poses, blocks)
