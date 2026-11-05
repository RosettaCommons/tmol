"""LK-ball scoring through shared host preparation and native water physics."""

from tmol_jax.score.energy_term import EnergyTerm
from tmol_jax.score.lk_ball.native import prepare_lk_ball, lk_ball_energy
from tmol_jax.score.score_types import ScoreType


class LKBallEnergyTerm(EnergyTerm):
    def score_types_provided(self):
        return [
            ScoreType.lk_ball_iso,
            ScoreType.lk_ball,
            ScoreType.lk_bridge,
            ScoreType.lk_bridge_uncpl,
        ]

    def setup_for_pose_stack(self, pose_stack, pbt=None):
        return prepare_lk_ball(pose_stack, self.param_db)

    def __call__(self, pose_stack, term_state, coords_override=None):
        coords = pose_stack.coords if coords_override is None else coords_override
        return dict(
            zip(self.score_types_provided(), lk_ball_energy(coords, term_state))
        )
