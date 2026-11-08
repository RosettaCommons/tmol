"""Backbone scoring through shared host preparation and native physics."""

from typing import NamedTuple

import jax
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
    workspace_sizes,
)
from tmol_jax.score.energy_term import EnergyTerm, prepared_palette
from tmol_jax.score.score_types import ScoreType


class BackboneTorsionTermState(NamedTuple):
    parameters: tuple
    precise_gradient: bool = False


jax.tree_util.register_pytree_node(
    BackboneTorsionTermState,
    lambda state: ((state.parameters,), state.precise_gradient),
    lambda precise, children: BackboneTorsionTermState(children[0], precise),
)


class BackboneTorsionEnergyTerm(EnergyTerm):
    @staticmethod
    def score_types_provided():
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
            *pbt._scoring_sources,
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
        device = pose_stack.coords.device
        chemical_arrays = prepared_palette(
            pbt,
            self.param_db,
            "backbone",
            lambda: jax.device_put(
                (
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
                ),
                device=device,
            ),
            device=device,
        )
        return BackboneTorsionTermState(
            jax.device_put(tuple(pose_arguments(pose_stack)), device=device)
            + chemical_arrays,
            device.platform == "gpu",
        )

    @classmethod
    def evaluate(cls, term_state, coords):
        target = "tmol_backbone_torsion"
        dimensions = (coords.shape[0],)
        if term_state.precise_gradient and coords.dtype == jnp.float32:
            target += "_precise"
            dimensions += (coords.shape[0] * coords.shape[1],)
            register(target)
        values = coordinate_energy(
            target,
            coords,
            term_state.parameters,
            channels=2,
            workspace_bytes=workspace_sizes(
                target, dimensions, coords.dtype.itemsize * 8
            ),
        )
        return dict(zip(cls.score_types_provided(), values))
