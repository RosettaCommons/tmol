"""Hydrogen bonds through shared chemistry, native bases and pair physics."""

from tmol_jax.score.energy_term import EnergyTerm
from tmol_jax.score.hbond.native import prepare_hbond, hbond_energy
from tmol_jax.score.score_types import ScoreType


class HBondEnergyTerm(EnergyTerm):
    @staticmethod
    def score_types_provided():
        return [ScoreType.hbond]

    def setup_for_pose_stack(self, pose_stack, pbt=None):
        return prepare_hbond(pose_stack, self.param_db)

    @classmethod
    def evaluate(cls, term_state, coords):
        return {ScoreType.hbond: hbond_energy(coords, term_state)}
