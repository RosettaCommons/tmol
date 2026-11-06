"""Dunbrack execution using common library/topology preparation and physics."""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from tmol_core.scoring.dunbrack import prepare_dunbrack_tables
from tmol_core.scoring.dunbrack_topology import (
    pack_dunbrack_blocks,
    prepare_dunbrack_block,
)
from tmol_jax.score._native import (
    atom_downstream_of_conn,
    coordinate_energy,
    fit_bspline_coefficients,
    pose_arguments,
    register,
    workspace_sizes,
)
from tmol_jax.score.energy_term import EnergyTerm
from tmol_jax.score.score_types import ScoreType


class DunbrackTermState(NamedTuple):
    parameters: tuple


class DunbrackEnergyTerm(EnergyTerm):
    def score_types_provided(self):
        return [
            ScoreType.dunbrack_rot,
            ScoreType.dunbrack_rotdev,
            ScoreType.dunbrack_semirot,
        ]

    def setup_for_pose_stack(self, pose_stack, pbt=None):
        pbt = pose_stack.packed_block_types if pbt is None else pbt
        parameters = self._prepared(
            "device_parameters",
            lambda: self._packed_parameters(pbt),
            *pbt.active_block_types,
            settings=(pbt.max_n_atoms, pbt.max_n_conn, pbt.n_atoms.device),
        )
        if not parameters:
            return DunbrackTermState(())
        return DunbrackTermState(
            tuple(
                jnp.asarray(a, device=pose_stack.coords.device)
                for a in pose_arguments(pose_stack)
            )
            + parameters
        )

    def _packed_parameters(self, pbt):
        tables = self._prepared(
            "tables",
            lambda: prepare_dunbrack_tables(
                self.param_db.scoring.dunbrack,
                lambda tables: [fit_bspline_coefficients(t) for t in tables],
            ),
        )
        lookup = (
            tables.all_table_indices,
            tables.rotameric_table_indices,
            tables.semirotameric_table_indices,
        )
        blocks = pack_dunbrack_blocks(
            [
                prepare_dunbrack_block(bt, lookup, tables.scoring_db_aux)
                for bt in pbt.active_block_types
            ]
        )
        if not np.any(blocks["rotamer_table_set"] >= 0):
            return ()
        register("tmol_dunbrack")
        arrays = (
            atom_downstream_of_conn(pbt),
            *tables.scoring_db.values(),
            *blocks.values(),
        )
        parameters = []
        for array in arrays:
            array = np.asarray(array)
            if np.issubdtype(array.dtype, np.integer):
                if array.size and (
                    array.min() < np.iinfo(np.int32).min
                    or array.max() > np.iinfo(np.int32).max
                ):
                    raise ValueError("native Dunbrack indices exceed int32")
                array = array.astype(np.int32)
            parameters.append(jnp.asarray(array, device=pbt.n_atoms.device))
        return tuple(parameters)

    def __call__(self, pose_stack, term_state, coords_override=None):
        coords = pose_stack.coords if coords_override is None else coords_override
        parameters = term_state.parameters
        if parameters:
            dimensions = (parameters[0].size, parameters[24].shape[1])
            values = coordinate_energy(
                "tmol_dunbrack",
                coords,
                parameters,
                channels=3,
                workspace_bytes=workspace_sizes(
                    "tmol_dunbrack", dimensions, coords.dtype.itemsize * 8
                ),
            )
        else:
            values = jnp.zeros((3, coords.shape[0]), coords.dtype)
        return dict(zip(self.score_types_provided(), values))
