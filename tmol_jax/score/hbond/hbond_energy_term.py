"""Hydrogen bonds through shared chemistry, native bases and pair physics."""

from tmol_jax.score.energy_term import EnergyTerm
from tmol_jax.score._native import register
from tmol_jax.score.hbond.native import prepare_hbond, hbond_energy
from tmol_jax.score.score_types import ScoreType


class HBondEnergyTerm(EnergyTerm):
    @staticmethod
    def score_types_provided():
        return [ScoreType.hbond]

    def setup_for_pose_stack(self, pose_stack, pbt=None):
        parameters = self._prepared_pose(
            pose_stack, lambda: prepare_hbond(pose_stack, self.param_db)
        )
        precise = pose_stack.coords.device.platform == "gpu"
        if precise:
            register("tmol_hbond_precise")
        return parameters, precise

    @classmethod
    def evaluate(cls, term_state, coords):
        parameters, precise = term_state
        return {ScoreType.hbond: hbond_energy(coords, parameters, precise=precise)}
