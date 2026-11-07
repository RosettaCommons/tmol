"""Shared fragment-column identity and bounded reduction planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from tmol_core.io.split_block_mapping import SplitBlockEntry, SplitBlockMapping


class _LegacyFragmentRecord(Protocol):
    """Fields consumed from the compatibility ligand mapping."""

    pose_index: int
    block_index: int
    pose_residue_label: int


class _LegacyFragmentMapping(Protocol):
    """Structural type for the compatibility fragmented-ligand mapping."""

    blocks: tuple[_LegacyFragmentRecord, ...]


FragmentRecord = SplitBlockEntry | _LegacyFragmentRecord


@dataclass(frozen=True)
class FragmentInteractionScores:
    """Per-fragment interactions with an explicitly selected partner."""

    scores: object
    mapping: tuple[FragmentRecord, ...]


@dataclass(frozen=True)
class _NormalizedFragmentRecord:
    """Common indexing fields for current and compatibility mappings."""

    pose_index: int
    block_index: int
    column_key: tuple[int, ...]
    source: FragmentRecord


@dataclass(frozen=True)
class FragmentMappingPlan:
    fragment_records: tuple[FragmentRecord, ...]
    record_linear_indices: tuple[int, ...]
    fragment_block_indices: tuple[int, ...]


def fragment_mapping_plan(mapping, n_poses, max_n_blocks):
    """Validate mature current/legacy column identity across all poses."""
    if isinstance(mapping, SplitBlockMapping):
        records = tuple(
            _NormalizedFragmentRecord(
                pose_index=entry.pose_ind,
                block_index=entry.block_ind,
                column_key=(entry.block_ind,),
                source=entry,
            )
            for entry in mapping.entries
        )
        require_pose_zero_record = True
    else:
        records = tuple(
            _NormalizedFragmentRecord(
                pose_index=record.pose_index,
                block_index=record.block_index,
                column_key=(record.pose_residue_label, record.block_index),
                source=record,
            )
            for record in mapping.blocks
        )
        require_pose_zero_record = False

    records_by_pose: list[list[_NormalizedFragmentRecord]] = [
        [] for _ in range(n_poses)
    ]
    for record in records:
        if not (
            0 <= record.pose_index < n_poses and 0 <= record.block_index < max_n_blocks
        ):
            raise ValueError(
                "Fragment mapping entry is outside the pose stack: "
                f"pose {record.pose_index}, block {record.block_index}"
            )
        records_by_pose[record.pose_index].append(record)

    pose_zero_records = tuple(
        sorted(records_by_pose[0], key=lambda record: record.block_index)
    )
    if require_pose_zero_record and not pose_zero_records:
        raise ValueError("split_block_mapping has no entries for pose 0")

    expected_columns = {record.column_key for record in pose_zero_records}
    for pose_records in records_by_pose:
        pose_columns = {record.column_key for record in pose_records}
        if len(pose_columns) != len(pose_records):
            raise ValueError("Fragment mapping contains duplicate columns in one pose")
        if pose_columns != expected_columns:
            raise ValueError(
                "Fragment mappings must use identical block topology in every pose"
            )

    return FragmentMappingPlan(
        tuple(record.source for record in pose_zero_records),
        tuple(
            record.pose_index * max_n_blocks + record.block_index for record in records
        ),
        tuple(record.block_index for record in pose_zero_records),
    )


def fragments_per_chunk(n_terms, n_poses, n_blocks, itemsize, max_workspace_bytes):
    """Bound the two gathered score tensors using the mature workspace policy."""
    return max(1, max_workspace_bytes // (2 * n_terms * n_poses * n_blocks * itemsize))
