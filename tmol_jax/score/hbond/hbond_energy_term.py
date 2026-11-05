"""Hydrogen bonds through shared chemistry, native bases and pair physics."""

from tmol_jax.score.energy_term import EnergyTerm
from tmol_jax.score.hbond.native import prepare_hbond, hbond_energy
from tmol_jax.score.score_types import ScoreType


class HBondEnergyTerm(EnergyTerm):
    def score_types_provided(self):
        return [ScoreType.hbond]

    def setup_for_pose_stack(self, pose_stack, pbt=None):
        return prepare_hbond(pose_stack, self.param_db)

    def __call__(self, pose_stack, term_state, coords_override=None):
        coords = pose_stack.coords if coords_override is None else coords_override
        return {ScoreType.hbond: hbond_energy(coords, term_state)}
