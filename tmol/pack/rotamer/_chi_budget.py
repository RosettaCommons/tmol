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
    checked_sample_count,
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

    n_counts = counts.numel()
    if n_counts == 0 or n_counts > torch.iinfo(torch.int32).max:
        # Avoid reducing empty/oversized tensors; the core owns error policy.
        return checked_sample_count(n_counts, 0, 0, 0, expanded_limit, limit)
    smallest, largest = torch.aminmax(counts)
    total, smallest, largest = torch.stack(
        (counts.sum(dtype=torch.int64), smallest, largest)
    ).tolist()
    return checked_sample_count(
        n_counts, total, smallest, largest, expanded_limit, limit
    )


def _check_task_sample_budget(task, samples):
    """Bound combined sampler/type rows through the common budget policy."""
    from tmol_core.pack.conformer_merge import (
        check_task_sample_budget,
        validate_task_sample_budget,
    )
    import numpy
    import torch

    budget = getattr(task, "chi_sample_budget", None)
    shape = task.per_block_orig_block_type.shape
    if budget is None or not all(shape):
        return
    if samples[0][0].device.type == "cpu":
        check_task_sample_budget(
            shape,
            task.cons_bt_pose.numpy(),
            task.cons_bt_block.numpy(),
            numpy.stack([sample[0].numpy() for sample in samples]),
            budget,
        )
        return
    counts = torch.stack([sample[0].long() for sample in samples]).sum(0)
    per_block = torch.zeros(
        shape[0] * shape[1], dtype=torch.int64, device=counts.device
    )
    per_block.index_add_(0, task.cons_bt_pose * shape[1] + task.cons_bt_block, counts)
    largest, index = per_block.max(0)
    largest, index = torch.stack((largest, index)).tolist()
    validate_task_sample_budget(largest, index, shape, budget)
