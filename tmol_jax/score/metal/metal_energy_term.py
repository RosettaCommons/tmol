"""Thin coordinate-runtime adapter for canonical metal coordination scoring."""

import jax
import jax.numpy as jnp
import numpy as np

from tmol_core.scoring.metal import MetalCoordinationPreparation, validate_donors
from tmol_jax.score._native import coordinate_energy, pose_arguments, register
from tmol_jax.score.energy_term import EnergyTerm
from tmol_jax.score.score_types import ScoreType


class MetalCoordinationEnergyTerm(EnergyTerm):
    def score_types_provided(self):
        return [ScoreType.metal_coordination]

    def setup_for_pose_stack(self, pose_stack, pbt=None, *, include_empty=False):
        pbt = pose_stack.packed_block_types if pbt is None else pbt
        types = pbt.active_block_types
        indices = np.asarray(pose_stack.block_type_ind)
        used = np.unique(indices[indices >= 0])
        if not include_empty and all(not types[i].metal_sites for i in used):
            return ()
        prep = MetalCoordinationPreparation(self.param_db.scoring.metal_coordination)
        host = prep.prepare_packed(
            types, pbt.chem_db.atom_types, max_n_conn=pbt.max_n_conn
        )
        validate_donors(
            indices,
            np.asarray(pose_stack.inter_residue_connections),
            host["metal_coordination_conn_virt"],
            host["metal_coordination_conn_key"],
        )
        register("tmol_metal_coordination")
        return tuple(
            jax.device_put(a, device=pose_stack.coords.device)
            for a in (
                *pose_arguments(pose_stack),
                pbt.conn_atoms,
                *(
                    host["metal_coordination_" + name]
                    for name in (
                        "conn_metal",
                        "conn_virt",
                        "conn_key",
                        "site_params",
                        "fan_atoms",
                        "fan_params",
                        "bridge_internal_metal",
                        "bridge_internal_d0",
                        "bridge_params",
                    )
                ),
            )
        )

    def __call__(self, pose_stack, term_state, coords_override=None):
        coords = pose_stack.coords if coords_override is None else coords_override
        values = (
            coordinate_energy("tmol_metal_coordination", coords, term_state)
            if term_state
            else jnp.zeros(coords.shape[0], coords.dtype)
        )
        return {ScoreType.metal_coordination: values}
