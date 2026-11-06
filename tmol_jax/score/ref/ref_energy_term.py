"""Coordinate-independent reference energies from shared chemical identity."""

import jax
import numpy as np

from tmol_core.scoring.ref import reference_weight
from tmol_jax.score.energy_term import EnergyTerm
from tmol_jax.score.score_types import ScoreType


class RefEnergyTerm(EnergyTerm):
    def set_options(self, options):
        self.weights_override = options.get("ref_weights")

    def score_types_provided(self):
        return [ScoreType.ref]

    def block_type_values(self, pbt):
        """Resolve the same chemical identities for whole-pose and rotamer scores."""
        weights = (
            getattr(self, "weights_override", None) or self.param_db.scoring.ref.weights
        )
        return np.asarray(
            [reference_weight(block, weights) for block in pbt.active_block_types],
            dtype=np.float32,
        )

    def setup_for_pose_stack(self, pose_stack, pbt=None):
        pbt = pose_stack.packed_block_types if pbt is None else pbt
        values = self.block_type_values(pbt)
        indices = np.asarray(pose_stack.block_type_ind)
        per_block = np.where(indices >= 0, values[np.maximum(indices, 0)], 0)
        return jax.device_put(per_block.sum(axis=1), device=pose_stack.coords.device)

    def __call__(self, pose_stack, term_state, coords_override=None):
        coords = pose_stack.coords if coords_override is None else coords_override
        return {ScoreType.ref: term_state.astype(coords.dtype)}


__all__ = ["RefEnergyTerm"]
