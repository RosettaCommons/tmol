"""Shared correspondence and ownership validation for joint conformers."""

import attr
import numpy


@attr.s(auto_attribs=True, slots=True, frozen=True)
class CorrelatedBlockGroup:
    """Blocks whose rotamer k was emitted as the same joint conformer."""

    pose: int
    blocks: tuple[int, ...]

    @property
    def anchor(self):
        return self.blocks[0]


def correlation_indices(shape, groups, counts=None):
    out = numpy.full(tuple(shape), -1, dtype=numpy.int32)
    if groups:
        if counts is None:
            raise ValueError("Correlated group validation requires block counts")
        claimed = set()
        for gi, group in enumerate(groups):
            sampled = list(group.blocks)
            if (
                not 0 <= group.pose < counts.shape[0]
                or any(not 0 <= block < counts.shape[1] for block in sampled)
                or len(set(sampled)) != len(sampled)
            ):
                raise ValueError("Invalid correlated block indices")
            members = {(group.pose, block) for block in sampled}
            if claimed & members:
                raise ValueError(
                    "A rotamer block belongs to multiple correlated groups"
                )
            claimed.update(members)
            if len(sampled) < 2:
                continue
            if counts[group.pose, sampled[0]] < 1 or not numpy.all(
                counts[group.pose, sampled] == counts[group.pose, sampled[0]]
            ):
                raise ValueError(
                    f"group at anchor {group.anchor} in pose {group.pose} has members with differing rotamer counts; they cannot be in lockstep"
                )
            if counts[group.pose, sampled[0]] == 1:
                continue
            out[group.pose, sampled] = gi
    return out


def correlated_groups_for_samples(pose_of, block_of, counts, declarations):
    """Validate producer correspondence and return nonoverlapping block groups."""
    total = {}
    for gbt, count in enumerate(counts.sum(axis=0)):
        key = int(pose_of[gbt]), int(block_of[gbt])
        total[key] = total.get(key, 0) + int(count)
    groups, claimed = [], set()
    for sampler, gbts in declarations:
        if len(set(gbts)) != len(gbts) or any(g < 0 or g >= len(pose_of) for g in gbts):
            raise ValueError("Invalid correlated considered-block indices")
        poses = {int(pose_of[g]) for g in gbts}
        blocks = tuple(int(block_of[g]) for g in gbts)
        if len(poses) != 1 or len(set(blocks)) != len(blocks):
            raise ValueError(
                "A correlated group must contain distinct blocks in one pose"
            )
        pose = poses.pop()
        expected = int(counts[sampler, gbts[0]])
        if expected < 1 or any(int(counts[sampler, g]) != expected for g in gbts):
            raise ValueError("Correlated members have differing rotamer counts")
        members = {(pose, block) for block in blocks}
        if members & claimed or any(total[key] != expected for key in members):
            raise ValueError(
                "Independent or overlapping samplers add rotamers to a correlated group"
            )
        claimed.update(members)
        if len(blocks) > 1:
            groups.append(CorrelatedBlockGroup(pose, blocks))
    return tuple(groups)


@attr.s(auto_attribs=True, frozen=True, slots=True)
class GroupCollapse:
    """How a group's members were folded into one choice for the packer.

    The packer is handed a smaller rotamer set than was built: a member's
    rotamers past the first are gone, and every reference to them has been sent
    to the representative's corresponding rotamer, so the packer chooses once
    for the whole group and cannot pick conformer 5 of one sugar against
    conformer 200 of the next. The tensors named ``compact_`` describe that
    smaller set and are what the interaction graph is built from;
    ``rot_offset_for_block`` and ``members`` stay in the original indexing,
    which is what the rotamer set's coordinates are addressed by.
    """

    # original rotamer index -> its index in the compacted set, with a member's
    #    rotamers pointing at the representative's
    rot_map: numpy.ndarray
    # compacted rotamer index -> the original rotamer it stands for
    compact_to_orig: numpy.ndarray
    compact_n_rots_for_block: numpy.ndarray
    compact_rot_offset_for_block: numpy.ndarray
    compact_block_ind_for_rot: numpy.ndarray
    compact_pose_for_rot: numpy.ndarray
    compact_block_type_ind_for_rot: numpy.ndarray
    compact_n_rots_for_pose: numpy.ndarray
    compact_rot_offset_for_pose: numpy.ndarray
    # original indexing, for writing the chosen conformer back out
    rot_offset_for_block: numpy.ndarray
    n_rots_for_block: numpy.ndarray
    # (pose, representative block, [(member block, that member's rot offset)])
    members: tuple


def collapse_group_rotamers(layout, groups):
    """Fold each group's members into a single block for the packer.

    Returns the replacement bookkeeping, or None when there is nothing to fold.
    """
    if not groups:
        return None

    correlation_indices(layout.n_rots_for_block.shape, groups, layout.n_rots_for_block)
    n_rots = layout.block_ind_for_rot.shape[0]
    rot_map = numpy.arange(n_rots, dtype=numpy.int64)
    keep = numpy.ones((n_rots,), dtype=numpy.bool_)
    n_rots_for_block = layout.n_rots_for_block.copy()
    rot_offset = layout.rot_offset_for_block
    group_of_rot = numpy.full((n_rots,), -1, dtype=numpy.int64)
    conformer_of_rot = numpy.full((n_rots,), -1, dtype=numpy.int64)

    members = []
    for gi, group in enumerate(groups):
        counts = [int(n_rots_for_block[group.pose, b]) for b in group.blocks]
        sampled = [b for b, c in zip(group.blocks, counts) if c > 1]
        if len(sampled) < 2:
            continue  # nothing to correlate
        n_conf = int(n_rots_for_block[group.pose, sampled[0]])
        if any(int(n_rots_for_block[group.pose, b]) != n_conf for b in sampled):
            raise ValueError(
                f"group at anchor {group.anchor} has members with differing "
                "rotamer counts; they cannot be in lockstep"
            )

        rep = sampled[0]
        rep_off = int(rot_offset[group.pose, rep])
        entry = []
        for b in sampled:
            off = int(rot_offset[group.pose, b])
            entry.append((b, off))
            idx = numpy.arange(off, off + n_conf, dtype=numpy.int64)
            group_of_rot[idx] = gi
            conformer_of_rot[idx] = numpy.arange(n_conf, dtype=numpy.int64)
            if b == rep:
                continue
            rot_map[idx] = numpy.arange(rep_off, rep_off + n_conf, dtype=numpy.int64)
            # One rotamer, not none: a real block with no rotamers is neither
            #    molten nor background, and the packer's bookkeeping needs it to
            #    be one or the other. The one it keeps carries no energy -- every
            #    reference to this member was remapped onto the representative --
            #    and its coordinates are overwritten once the group's conformer
            #    is chosen, so it is a placeholder and nothing more.
            keep[idx[1:]] = False
            n_rots_for_block[group.pose, b] = 1
        members.append((group.pose, rep, tuple(entry)))

    if not members:
        return None

    # Counts and offsets have to keep describing a partition of the rotamer
    #    range: the interaction graph recovers a block's rotamer count by
    #    differencing consecutive blocks' offsets, so a rotamer left sitting in
    #    a gap is silently charged to whichever block precedes it.
    # Include the exclusive end: trailing zero-rotamer blocks and empty poses
    # can have an offset equal to n_rots.
    orig_to_compact = numpy.concatenate(
        (numpy.zeros(1, dtype=numpy.int64), keep.cumsum(0))
    )
    compact_to_orig = numpy.flatnonzero(keep)

    compact_rot_offset_for_block = numpy.where(
        rot_offset >= 0,
        orig_to_compact[rot_offset.clip(min=0).astype(numpy.int64)],
        rot_offset,
    )
    compact_rot_offset_for_pose = orig_to_compact[
        layout.rot_offset_for_pose.astype(numpy.int64)
    ]
    n_rots_for_pose = numpy.zeros_like(layout.n_rots_for_pose)
    n_poses = n_rots_for_pose.shape[0]
    for pose in range(n_poses):
        start = int(layout.rot_offset_for_pose[pose])
        stop = start + int(layout.n_rots_for_pose[pose])
        n_rots_for_pose[pose] = int(keep[start:stop].sum())

    return (
        GroupCollapse(
            rot_map=orig_to_compact[rot_map],
            compact_to_orig=compact_to_orig,
            compact_n_rots_for_block=n_rots_for_block,
            compact_rot_offset_for_block=compact_rot_offset_for_block.astype(
                rot_offset.dtype
            ),
            compact_block_ind_for_rot=layout.block_ind_for_rot[compact_to_orig],
            compact_pose_for_rot=layout.pose_for_rot[compact_to_orig],
            compact_block_type_ind_for_rot=layout.block_type_ind_for_rot[
                compact_to_orig
            ],
            compact_n_rots_for_pose=n_rots_for_pose,
            compact_rot_offset_for_pose=compact_rot_offset_for_pose.astype(
                layout.rot_offset_for_pose.dtype
            ),
            rot_offset_for_block=rot_offset,
            n_rots_for_block=n_rots_for_block,
            members=tuple(members),
        ),
        group_of_rot,
        conformer_of_rot,
    )


def expand_group_assignments(assignment, collapse):
    """Expand ORIGINAL source indices to all joint-conformer members.

    Map retained compact assignments through compact_to_orig before calling.
    The result selects original type, atom layout and coordinates together;
    pass it to pose imposition rather than patching coordinates afterward.
    Negative assignments outside groups (padding) are preserved.
    """
    result = numpy.array(assignment, copy=True)
    if result.shape != collapse.rot_offset_for_block.shape:
        raise ValueError("Assignment and original block layout shapes differ")
    for pose, rep, entry in collapse.members:
        chosen = int(result[pose, rep]) - int(collapse.rot_offset_for_block[pose, rep])
        if not 0 <= chosen < int(collapse.n_rots_for_block[pose, rep]):
            raise ValueError(
                f"Group representative block {rep} of pose {pose} has an invalid original rotamer assignment"
            )
        for block, offset in entry:
            result[pose, block] = offset + chosen
    return result
