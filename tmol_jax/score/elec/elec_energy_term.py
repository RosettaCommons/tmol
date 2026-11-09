"""Thin JAX execution of the shared native elec term."""

from tmol_jax.score.energy_term import EnergyTerm
from tmol_jax.score.score_types import ScoreType


class ElecEnergyTerm(EnergyTerm):
    @staticmethod
    def score_types_provided():
        return [ScoreType.fa_elec]

    def setup_for_pose_stack(self, pose_stack, pbt=None):
        from tmol_jax.score.nonbonded import prepare_independent_nonbonded

        return prepare_independent_nonbonded(
            pose_stack, self.param_db, electrostatic=True
        )

    @classmethod
    def evaluate(cls, term_state, coords):
        from tmol_jax.score.nonbonded import independent_nonbonded_energy

        values = independent_nonbonded_energy(coords, term_state)
        return {ScoreType.fa_elec: values}
