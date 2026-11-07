"""Dunbrack execution using common library/topology preparation and physics."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tmol_core.scoring.dunbrack_topology import (
    pack_dunbrack_blocks,
    prepare_dunbrack_block,
)
from tmol_jax.score._native import (
    atom_downstream_of_conn,
    coordinate_energy,
    pose_arguments,
    register,
    workspace_sizes,
)
from tmol_jax.score.energy_term import EnergyTerm
from tmol_jax.score.score_types import ScoreType
from .parameters import prepare_dunbrack_parameters


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
            lambda: self._packed_parameters(pbt, pose_stack.coords.dtype),
            *pbt.active_block_types,
            settings=(
                pbt.max_n_atoms,
                pbt.max_n_conn,
                pbt.n_atoms.device,
                pose_stack.coords.dtype,
            ),
        )
        if not parameters:
            return DunbrackTermState(())
        return DunbrackTermState(
            jax.device_put(
                tuple(pose_arguments(pose_stack)),
                device=pose_stack.coords.device,
            )
            + parameters
        )

    def _packed_parameters(self, pbt, coordinate_dtype):
        tables = self._prepared(
            "tables",
            lambda: prepare_dunbrack_parameters(self.param_db, pbt),
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
            elif np.issubdtype(array.dtype, np.floating):
                # Native tables share coordinate precision. Convert once during
                # preparation instead of every objective evaluation. The common
                # fitted coefficients retain their original precision.
                array = array.astype(coordinate_dtype, copy=False)
            parameters.append(jax.device_put(array, device=pbt.n_atoms.device))
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
