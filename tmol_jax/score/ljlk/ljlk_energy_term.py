"""Thin JAX execution of the shared native ljlk term."""

from tmol_jax.score.energy_term import EnergyTerm
from tmol_jax.score.score_types import ScoreType


class LJLKEnergyTerm(EnergyTerm):
    def set_options(self, options):
        self.soft_repulsive = options.get("soft_rep", False)

    @staticmethod
    def score_types_provided():
        return [ScoreType.fa_ljatr, ScoreType.fa_ljrep, ScoreType.fa_lk]

    def setup_for_pose_stack(self, pose_stack, pbt=None):
        from tmol_jax.score.nonbonded import prepare_independent_nonbonded

        return prepare_independent_nonbonded(
            pose_stack,
            self.param_db,
            electrostatic=False,
            soft_repulsive=getattr(self, "soft_repulsive", False),
        )

    @classmethod
    def evaluate(cls, term_state, coords):
        from tmol_jax.score.nonbonded import independent_nonbonded_energy

        values = independent_nonbonded_energy(coords, term_state)
        return dict(zip(cls.score_types_provided(), values))
