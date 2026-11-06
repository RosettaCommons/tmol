"""Which blocks pack together because a covalent bond ties them.

An amino acid with something bonded to its sidechain -- a glycan, a ligand --
cannot be packed a residue at a time: moving the sidechain moves what hangs off
it, and sampling the two independently breaks the bond that joins them. The
group is the anchor residue plus everything reachable from it through
conjugation connections, and it is packed as one unit.

The anchor supplies its own chi from its rotamer library; the attached blocks
supply theirs from their chi_samples. A free ligand, or a sugar that is not
bonded to a sidechain at all, forms no group and is left alone.
"""

import attr
from tmol_core.pack.correlation import GroupCollapse, expand_group_assignments
from collections import OrderedDict
import numpy
import torch

from tmol.pose._conjugated_groups import (  # noqa: F401
    ConjugatedGroup,
    _find_conjugated_groups,
    _lockstep_group_for_block,
)


from tmol_core.pack.conjugated import (  # noqa: F401 legacy import paths
    GroupSamplingTopology,  # noqa: F401 legacy import path
    GroupSamplingPlan,  # noqa: F401 legacy import path
    group_atom_context,
    group_sampling_topology,
    sampled_group_chi,
    resolve_group_atom as _resolve_group_atom,  # noqa: F401 legacy import path
)


def _group_atom_context(group, pose_stack):
    pbt = pose_stack.packed_block_types
    indices = pose_stack.block_type_ind[group.pose, list(group.blocks)].tolist()
    return group_atom_context(group, [pbt.active_block_types[i] for i in indices])


def _group_sampling_topology(group, pose_stack, fixed_owners=()):
    """Independent axes that preserve cycles and fixed external attachments.

    Cycle edges cannot turn independently. A bridge can turn only if its
    moving subtree contains no fixed atom. External connection atoms and their
    neighbors stay fixed to preserve the boundary's bonds and bond angles.
    Polymer backbones with external attachments stay fixed; a freely attached
    polymer member can move with the group. Every atom in a task-frozen member
    stays fixed; its owner index is part of the cached constraint identity.
    """
    pbt = pose_stack.packed_block_types
    indices = tuple(pose_stack.block_type_ind[group.pose, list(group.blocks)].tolist())
    key = (
        indices,
        group.links,
        tuple((a, c) for a, c, _b, _bc in group.external_links),
        tuple(fixed_owners),
    )
    cache = getattr(pbt, "conjugated_sampling_topology_cache", None)
    if cache is None:
        cache = OrderedDict()
        pbt.conjugated_sampling_topology_cache = cache
    if key not in cache:
        cache[key] = _group_sampling_topology_impl(group, pose_stack, fixed_owners)
        if len(cache) > 32:
            cache.popitem(last=False)
    cache.move_to_end(key)
    return cache[key]


def _group_sampling_topology_impl(group, pose_stack, fixed_owners=()):
    return group_sampling_topology(
        group,
        pose_stack.packed_block_types.active_block_types,
        pose_stack.block_type_ind[group.pose, list(group.blocks)].tolist(),
        fixed_owners,
    )


def _group_sampled_chi(
    group,
    pose_stack,
    expanded_limit,
    limit,
    library_size=1,
    reserve_current=False,
    topology=None,
    exclude_axes=(),
    include_anchor=False,
    sampled_members=None,
):
    topology = topology or _group_sampling_topology(group, pose_stack)
    return sampled_group_chi(
        group,
        topology,
        expanded_limit,
        limit,
        library_size,
        reserve_current,
        exclude_axes,
        include_anchor,
        sampled_members,
    )


def _add_conjugated_group_sampler(task, pose_stack, sampler=None, exclude=()):
    """Attach the group sampler and keep independent samplers off its members.

    A group is sampled as one unit. When a rotamer library is available the
    anchor is sampled along with it, its library chi multiplying the tree's
    conformers; otherwise the anchor keeps the conformation it came in with.
    Either way no other sampler may move it on its own, since that would carry
    the bonded residue along in the structure but not in the rotamer.
    """
    from tmol.pack.rotamer._conjugated_chi_sampler import ConjugatedChiSampler
    from tmol.pack.rotamer._fallback_sampler import FallbackSampler
    from tmol.pack.rotamer._include_current_sampler import IncludeCurrentSampler

    groups = _find_conjugated_groups(pose_stack)
    if not groups:
        return None

    if sampler is None:
        library = next(
            (
                s
                for s in task.conformer_samplers
                if type(s).__name__ == "DunbrackChiSampler"
            ),
            None,
        )
        sampler = ConjugatedChiSampler(library_sampler=library)
    with_anchor = sampler.library_sampler is not None

    sampled = numpy.zeros((pose_stack.n_poses, pose_stack.max_n_blocks), dtype=bool)
    members = numpy.zeros_like(sampled)
    for group in groups:
        members[group.pose, list(group.blocks)] = True
        for block in group.blocks if with_anchor else group.blocks[1:]:
            sampled[group.pose, block] = True

    def _mask(arr):
        return torch.tensor(arr, dtype=torch.bool, device=pose_stack.device)

    task.add_conformer_sampler_by_block_mask(sampler, _mask(sampled))

    # a sampler that hands back the pose's own conformation is only harmless on
    #    an anchor the group does not sample: once the group samples it, that
    #    extra rotamer would put the anchor out of step with its members
    for other in list(task.conformer_samplers):
        if other is sampler or other in exclude:
            continue
        mask = (
            sampled
            if isinstance(other, (FallbackSampler, IncludeCurrentSampler))
            else members
        )
        task.disable_sampler_by_block_mask(other, _mask(mask))
    return sampler


def _collapse_group_rotamers(pose_stack, rotamer_set, groups):
    if not groups:
        return None
    from types import SimpleNamespace
    from tmol_core.pack.correlation import collapse_group_rotamers

    fields = (
        "block_ind_for_rot",
        "n_rots_for_block",
        "rot_offset_for_block",
        "rot_offset_for_pose",
        "n_rots_for_pose",
        "pose_for_rot",
        "block_type_ind_for_rot",
    )
    layout = SimpleNamespace(
        **{name: getattr(rotamer_set, name).detach().cpu().numpy() for name in fields}
    )
    result = collapse_group_rotamers(layout, groups)
    if result is None:
        return None
    collapse, group, conformer = result
    device = rotamer_set.coords.device
    return (
        attr.evolve(
            collapse,
            **{
                field.name: torch.as_tensor(
                    getattr(collapse, field.name), device=device
                )
                for field in attr.fields(GroupCollapse)
                if field.name != "members"
            },
        ),
        torch.as_tensor(group, device=device),
        torch.as_tensor(conformer, device=device),
    )


def _collapse_group_energies(energies, collapse, group_of_rot, conformer_of_rot):
    """Remap a group's rotamer pair energies onto its representative.

    A pair that puts two members in DIFFERENT conformers describes a structure
    the group cannot adopt, so it is dropped rather than folded in; every other
    pair is summed onto the representative by coalescing. The summing matters:
    the interaction graph assigns a one-body energy rather than accumulating
    it, so the total has to be in one entry by the time it gets there.

    The returned tensor is indexed by the compacted rotamer numbering.
    """
    energies = energies.coalesce()
    idx = energies.indices()
    vals = energies.values()

    r1, r2 = idx[1].to(torch.int64), idx[2].to(torch.int64)
    g1, g2 = group_of_rot[r1], group_of_rot[r2]
    inconsistent = (
        (g1 >= 0) & (g1 == g2) & (conformer_of_rot[r1] != conformer_of_rot[r2])
    )
    keep = torch.logical_not(inconsistent)

    new_r1 = collapse.rot_map[r1[keep]]
    new_r2 = collapse.rot_map[r2[keep]]

    # Folding members onto a representative moves rotamers to a lower block, so
    #    a pair that was in order may no longer be. The interaction graph
    #    requires block1 < block2 and DROPS anything else without complaint, so
    #    put each pair back in order; the energy is symmetric either way.
    b1 = collapse.compact_block_ind_for_rot[new_r1]
    b2 = collapse.compact_block_ind_for_rot[new_r2]
    swap = b1 > b2
    ordered_r1 = torch.where(swap, new_r2, new_r1)
    ordered_r2 = torch.where(swap, new_r1, new_r2)

    n_compact = collapse.compact_to_orig.shape[0]
    new_idx = torch.stack([idx[0][keep], ordered_r1, ordered_r2])
    return torch.sparse_coo_tensor(
        new_idx, vals[keep], size=(energies.shape[0], n_compact, n_compact)
    ).coalesce()


def _write_group_members(new_pose_stack, rotamer_set, collapse, assignment):
    """Give every member of a group the conformer chosen for its representative.

    The packer decided once per group, on the representative's block; the other
    members were folded onto it and never assigned. ``assignment`` gives the
    rotamer each block ended up with, so the conformer index is read from the
    representative rather than guessed at.
    """
    coords = new_pose_stack.coords.clone()
    pbt = new_pose_stack.packed_block_types

    host_collapse = attr.evolve(
        collapse,
        rot_offset_for_block=collapse.rot_offset_for_block.cpu().numpy(),
        n_rots_for_block=collapse.n_rots_for_block.cpu().numpy(),
    )
    sources = expand_group_assignments(assignment.cpu().numpy(), host_collapse)
    for pose, rep, entry in collapse.members:
        for block, off in entry:
            if block == rep:
                continue
            n = int(pbt.n_atoms[int(new_pose_stack.block_type_ind[pose, block])])
            start = int(new_pose_stack.block_coord_offset[pose, block])
            src = int(rotamer_set.coord_offset_for_rot[int(sources[pose, block])])
            coords[pose, start : start + n] = rotamer_set.coords[src : src + n]

    return attr.evolve(new_pose_stack, coords=coords)
