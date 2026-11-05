"""Bounding the rotamers a set of sampled chi enumerates.

The budget is applied when rotamers are generated, not when a residue type is
built: a block's chi may be enumerated on their own or jointly with those of the
blocks conjugated to it, and which of those it is depends on the pose. Sampled
chi are therefore stored complete and frozen here.

Freezing runs tip-first, ordered by each chi's depth in the kinforest that will
build it. For a single block that is the residue's own tree, so the ordering is
the familiar one -- a chi near the backbone swings the whole sidechain where one
at the tip moves a couple of atoms. For a group of conjugated blocks it is the
group's tree, so the outermost sugar's outermost chi freezes first. The two are
the same rule; only the tree differs.
"""

import attr
from copy import copy
from tmol_core.pack.chi import (  # noqa: F401
    n_conformers as _n_conformers,
    apply_chi_sample_budget as _apply_chi_sample_budget,
    budgeted_chi_samples as _budgeted_chi_samples,
    chi_depths as _chi_depths,
)


def _sampler_for_task(sampler, task):
    """Apply an explicit task budget to a private sampler/task view."""
    budget = getattr(task, "chi_sample_budget", None)
    if budget is None or budget == (
        sampler.chi_sample_expanded_limit,
        sampler.chi_sample_limit,
    ):
        return sampler, task
    configured = attr.evolve(
        sampler, chi_sample_expanded_limit=budget[0], chi_sample_limit=budget[1]
    )
    private_task = copy(task)
    private_task.conformer_sampler_index = dict(task.conformer_sampler_index)
    private_task.conformer_sampler_index[id(configured)] = task.conformer_sampler_index[
        id(sampler)
    ]
    return configured, private_task


def _checked_sample_count(counts, expanded_limit, limit):
    """Check index capacity and budget before narrowing counts or allocating rows."""
    import torch

    if counts.numel() == 0:
        return 0
    capacity = torch.iinfo(torch.int32).max
    if counts.numel() > capacity:
        raise ValueError(f"Sampling count exceeds index capacity {capacity}")
    smallest, largest = torch.aminmax(counts)
    total, smallest, largest = torch.stack(
        (counts.sum(dtype=torch.int64), smallest, largest)
    ).tolist()
    # Check each count before trusting the sum: invalid int64 inputs can wrap
    # even an int64 reduction. Valid counts and lengths cannot overflow it.
    if smallest < 0 or largest > capacity or total > capacity:
        raise ValueError(
            f"Sampling count exceeds index capacity {capacity} "
            "or contains a negative count"
        )
    maximum = max(expanded_limit, limit)
    if largest > maximum:
        raise ValueError(
            f"Sampling budget {maximum} cannot fit the required library states"
        )
    return total


def _check_task_sample_budget(task, samples):
    """Bound each physical residue's combined sampler/type rows before merging."""
    budget = getattr(task, "chi_sample_budget", None)
    if budget is None:
        return
    import torch

    n_poses, n_blocks = task.per_block_orig_block_type.shape
    if n_poses * n_blocks == 0:
        return
    counts = torch.stack([sample[0].to(torch.int64) for sample in samples]).sum(dim=0)
    per_block = torch.zeros(n_poses * n_blocks, dtype=torch.int64, device=counts.device)
    per_block.index_add_(0, task.cons_bt_pose * n_blocks + task.cons_bt_block, counts)
    largest, index = per_block.max(dim=0)
    largest, index = torch.stack((largest, index)).tolist()
    maximum = max(budget)
    if largest > maximum:
        raise ValueError(
            f"Sampling budget {maximum} cannot fit {largest} combined rotamers "
            f"at pose {index // n_blocks} block {index % n_blocks} "
            "across samplers and allowed residue types"
        )
