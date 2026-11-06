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
