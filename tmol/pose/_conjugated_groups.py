"""Which blocks pack together because a covalent bond ties them.

An amino acid with something bonded to its sidechain -- a glycan, a ligand --
cannot be packed a residue at a time: moving the sidechain moves what hangs off
it, and sampling the two independently breaks the bond that joins them. The
group is the anchor residue plus everything reachable from it through
conjugation connections, and it is packed as one unit.

This lives beside PoseStack rather than in the packer because scoring needs it
too: a group's members are sampled in lockstep, so the rotamer-pair enumerator
has to know which blocks move together in order not to enumerate combinations
the group cannot adopt.
"""

from typing import List
import numpy
import torch
from tmol.pose._pose_stack import PoseStack
from tmol.pose._packed_block_types import (
    _annotate_packed_block_types_w_conjugation_conns,
)
from tmol_core.pack.conjugated import ConjugatedGroup, find_conjugated_groups


def _find_conjugated_groups(pose_stack: PoseStack) -> List[ConjugatedGroup]:
    pbt = pose_stack.packed_block_types
    _annotate_packed_block_types_w_conjugation_conns(pbt)
    if not bool(pbt.conjugation_conn.any()):
        return []
    return find_conjugated_groups(
        pose_stack.block_type_ind.cpu().numpy(),
        pose_stack.inter_residue_connections.cpu().numpy(),
        pbt.conjugation_conn.cpu().numpy(),
        pbt.n_conn.cpu().numpy(),
        numpy.asarray(
            [bt.properties.polymer.is_polymer for bt in pbt.active_block_types]
        ),
    )


def _lockstep_group_for_block(pose_stack, rotamer_set) -> torch.Tensor:
    """Per-block id shared by the blocks a group samples in lockstep.

    Rotamer k of every block carrying the same id is group conformer k, so a
    pair of these blocks only ever coexists at matching rotamer indices. Blocks
    outside such a group get -1. Only blocks the packer actually samples are
    marked, and only where the group has more than one of them; a group with a
    single sampled block has nothing to stay in step with.
    """
    if hasattr(rotamer_set, "group_for_block"):
        return rotamer_set.group_for_block
    from tmol.pack.rotamer._rotamer_set import _correlation_indices

    return _correlation_indices(
        rotamer_set.n_rots_for_block, getattr(rotamer_set, "correlated_groups", ())
    )
