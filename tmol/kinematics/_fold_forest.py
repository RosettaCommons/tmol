"""Torch pose adapter for the shared chemical fold-forest implementation."""

from tmol.pose._pose_stack import PoseStack
from tmol.pose._packed_block_types import _annotate_packed_block_types_w_kinematic_conns
from tmol_core.kinematics.fold_forest import (
    EdgeType as EdgeType,
    FoldForest as _FoldForest,
    _build_pose_fold_forest as _build_pose_fold_forest,
    _chemical_spanning_forest as _chemical_spanning_forest,
)

__all__ = [
    "EdgeType",
    "FoldForest",
    "_build_pose_fold_forest",
    "_chemical_spanning_forest",
]


class FoldForest(_FoldForest):
    @classmethod
    def reasonable_fold_forest(cls, pose_stack: PoseStack):
        """Create a fold forest for each pose using only backbone (up/down)
        polymer connectivity.

        Each biological chain (same chain_id) is rooted with a single
        root-jump to its first residue.  Polymer gaps within that chain
        (chain breaks) become ordinary jump edges connecting the last
        residue before the gap to the first residue after it.  Gaps
        between different biological chains produce separate root-jumps.
        Cyclic polymers (C→N cyclisation) are broken at the bond entering
        the lowest-index residue; that bond is dropped to keep the forest
        a valid tree.  Every remaining kinematic connection -- a glycan on
        a serine, a ligand on a lysine -- becomes a chemical edge, so torsions across it propagate downstream; a chemical bond
        that would close a cycle is the one dropped.
        """
        pbt = pose_stack.packed_block_types
        _annotate_packed_block_types_w_kinematic_conns(pbt)
        return cls.from_connectivity(
            pose_stack.block_type_ind.cpu().numpy(),
            pose_stack.inter_residue_connections.cpu().numpy(),
            pose_stack.chain_id.cpu().numpy(),
            pbt.up_conn_inds.cpu().numpy(),
            pbt.down_conn_inds.cpu().numpy(),
            pbt.n_conn.cpu().numpy(),
            pbt.kinematic_conn.cpu().numpy(),
        )
