"""JAX pose metadata adapter for the common chemical fold-forest constructor."""

import numpy as np

from tmol_core.chemical.packed import PackedBlockTypesData
from tmol_core.kinematics.fold_forest import FoldForest as _FoldForest


class FoldForest(_FoldForest):
    @classmethod
    def reasonable_fold_forest(cls, pose_stack, *, chain_id=None):
        """Build fold edges from explicit chemistry and biological chain labels.

        PDB metadata supplies chain labels by default. For a manually prepared
        pose without those labels, pass ``chain_id`` with shape [poses,blocks].
        Chemical cycles, kinematic connections and non-kinematic closures use
        the same spanning policy as Torch. Atom-level forest construction is a
        separate operation.
        """
        block_types = np.asarray(pose_stack.block_type_ind)
        if chain_id is None:
            info = pose_stack.pdb_info
            if info is None or len(info.chain_id) != len(block_types):
                raise ValueError(
                    "chain_id is required when pose chain metadata is absent"
                )
            chain_id = np.full(block_types.shape, None, dtype=object)
            for pose, labels in enumerate(info.chain_id):
                real = np.flatnonzero(block_types[pose] >= 0)
                needed = int(real[-1]) + 1 if len(real) else 0
                if len(labels) < needed or len(labels) > block_types.shape[1]:
                    raise ValueError(
                        "pose chain metadata does not match its block layout"
                    )
                chain_id[pose, : len(labels)] = labels
        else:
            chain_id = np.asarray(chain_id)
            if chain_id.shape != block_types.shape:
                raise ValueError("chain_id must have shape (poses, blocks)")
        if any(label is None for label in chain_id[block_types >= 0]):
            raise ValueError("chain_id must identify every real block")
        pbt = pose_stack.packed_block_types
        down, up = PackedBlockTypesData.join_polymeric_connections(
            pbt.active_block_types
        )
        flags = PackedBlockTypesData.join_kinematic_connections(
            pbt.active_block_types, pbt.max_n_conn
        )
        return cls.from_connectivity(
            block_types,
            np.asarray(pose_stack.inter_residue_connections),
            chain_id,
            up,
            down,
            np.asarray(pbt.n_conn),
            flags,
        )
