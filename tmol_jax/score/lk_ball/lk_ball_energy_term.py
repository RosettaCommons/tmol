"""LK-ball scoring through shared host preparation and native water physics."""

from tmol_jax.score.energy_term import EnergyTerm
from tmol_jax.score.lk_ball.native import prepare_lk_ball, lk_ball_energy
from tmol_jax.score.score_types import ScoreType


class LKBallEnergyTerm(EnergyTerm):
    @staticmethod
    def score_types_provided():
        return [
            ScoreType.lk_ball_iso,
            ScoreType.lk_ball,
            ScoreType.lk_bridge,
            ScoreType.lk_bridge_uncpl,
        ]

    def setup_for_pose_stack(self, pose_stack, pbt=None):
        return self._prepared_pose(
            pose_stack, lambda: prepare_lk_ball(pose_stack, self.param_db)
        )

    @classmethod
    def evaluate(cls, term_state, coords):
        return dict(zip(cls.score_types_provided(), lk_ball_energy(coords, term_state)))

    @staticmethod
    def weighted_energy(term_state, coords, weights):
        """Compose the four channel derivatives in one common reverse pass."""
        return lk_ball_energy(coords, term_state, weights)
