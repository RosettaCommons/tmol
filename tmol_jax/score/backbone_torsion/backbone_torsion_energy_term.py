"""Backbone scoring through shared host preparation and native physics."""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from tmol_core.scoring.backbone_torsion import (
    BackboneTorsionPreparation,
    prepare_backbone_tables,
)
from tmol_jax.score._native import (
    atom_downstream_of_conn,
    coordinate_energy,
    fit_bspline_coefficients,
    pose_arguments,
    register,
)
from tmol_jax.score.energy_term import EnergyTerm
from tmol_jax.score.score_types import ScoreType


class BackboneTorsionTermState(NamedTuple):
    parameters: tuple


class BackboneTorsionEnergyTerm(EnergyTerm):
    def score_types_provided(self):
        return [ScoreType.rama, ScoreType.omega]

    def setup_for_pose_stack(self, pose_stack, pbt=None):
        pbt = pose_stack.packed_block_types if pbt is None else pbt
        db = self.param_db.scoring
        rama, omega = db.rama, db.omega_bbdep
        prep = self._prepared(
            "resolver",
            lambda: BackboneTorsionPreparation(
                rama.rama_lookup,
                omega.bbdep_omega_lookup,
                [row.table_id for row in rama.rama_tables],
                [row.table_id for row in omega.bbdep_omega_tables],
            ),
        )
        host = self._prepared(
            "packed",
            lambda: prep.prepare_packed(
                [prep.prepare_block(bt) for bt in pbt.active_block_types]
            ),
            *pbt.active_block_types,
        )
        rt, ot = self._prepared(
            "tables",
            lambda: prepare_backbone_tables(
                rama.rama_tables,
                omega.bbdep_omega_tables,
                fit_bspline_coefficients,
            ),
            settings=(fit_bspline_coefficients,),
        )
        register("tmol_backbone_torsion")
        return BackboneTorsionTermState(
            tuple(
                jnp.asarray(a, device=pose_stack.coords.device)
                for a in (
                    *pose_arguments(pose_stack),
                    atom_downstream_of_conn(pbt),
                    *(
                        host[name]
                        for name in (
                            "bt_rama_table",
                            "bt_omega_table",
                            "bt_lower_conn_ind",
                            "bt_upper_conn_ind",
                            "bt_is_pro",
                            "bt_backbone_torsion_atoms",
                        )
                    ),
                    rt.tables,
                    np.concatenate((rt.bbstarts, rt.bbsteps), axis=1),
                    ot.tables,
                    np.concatenate((ot.bbstarts, ot.bbsteps), axis=1),
                )
            )
        )

    def __call__(self, pose_stack, term_state, coords_override=None):
        coords = pose_stack.coords if coords_override is None else coords_override
        values = coordinate_energy(
            "tmol_backbone_torsion", coords, term_state.parameters, channels=2
        )
        return dict(zip(self.score_types_provided(), values))
