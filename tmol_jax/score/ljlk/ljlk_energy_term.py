"""Thin JAX execution of the shared native ljlk term."""

from tmol_jax.score.energy_term import EnergyTerm
from tmol_jax.score.score_types import ScoreType


class LJLKEnergyTerm(EnergyTerm):
    def score_types_provided(self):
        return [ScoreType.fa_ljatr, ScoreType.fa_ljrep, ScoreType.fa_lk]

    def setup_for_pose_stack(self, pose_stack, pbt=None):
        from tmol_jax.score.nonbonded import prepare_independent_nonbonded

        return prepare_independent_nonbonded(
            pose_stack, self.param_db, electrostatic=False
        )

    def __call__(self, pose_stack, term_state, coords_override=None):
        from tmol_jax.score.nonbonded import independent_nonbonded_energy

        coords = pose_stack.coords if coords_override is None else coords_override
        values = independent_nonbonded_energy(coords, term_state)
        return dict(zip(self.score_types_provided(), values))
