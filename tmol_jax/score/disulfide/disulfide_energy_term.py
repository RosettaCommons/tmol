"""Disulfide topology and stereochemistry from the shared scientific core."""

import jax

from tmol_core.scoring.disulfide import (
    pack_disulfide_connections,
    native_disulfide_parameters,
)
from tmol_jax.score._native import (
    atom_downstream_of_conn,
    coordinate_energy,
    pose_arguments,
    register,
)
from tmol_jax.score.energy_term import EnergyTerm, prepared_palette
from tmol_jax.score.score_types import ScoreType


class DisulfideEnergyTerm(EnergyTerm):
    @staticmethod
    def score_types_provided():
        return [ScoreType.disulfide]

    def setup_for_pose_stack(self, pose_stack, pbt=None):
        pbt = pose_stack.packed_block_types if pbt is None else pbt
        device = pose_stack.coords.device

        def prepare():
            connections, mirrored = pack_disulfide_connections(
                pbt.active_block_types, max_n_conn=pbt.max_n_conn
            )
            return jax.device_put(
                (
                    connections,
                    atom_downstream_of_conn(pbt),
                    native_disulfide_parameters(
                        self.param_db.scoring.disulfide.global_parameters, mirrored
                    ),
                ),
                device=device,
            )

        chemical = prepared_palette(
            pbt, self.param_db, "disulfide", prepare, device=device
        )
        register("tmol_disulfide")
        return (
            jax.device_put(tuple(pose_arguments(pose_stack)), device=device) + chemical
        )

    @classmethod
    def evaluate(cls, term_state, coords):
        return {
            ScoreType.disulfide: coordinate_energy(
                "tmol_disulfide", coords, term_state, always_gradient=True
            )
        }
