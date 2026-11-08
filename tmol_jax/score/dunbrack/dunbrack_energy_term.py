"""Dunbrack execution using common library/topology preparation and physics."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tmol_core.scoring.dunbrack import prepare_dunbrack_lookup
from tmol_core.scoring.dunbrack_topology import (
    dunbrack_scoring_table_indices,
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
from .parameters import (
    device_dunbrack_tables,
    native_parameters,
    prepare_dunbrack_parameters,
)


class DunbrackTermState(NamedTuple):
    parameters: tuple


class DunbrackEnergyTerm(EnergyTerm):
    @staticmethod
    def score_types_provided():
        return [
            ScoreType.dunbrack_rot,
            ScoreType.dunbrack_rotdev,
            ScoreType.dunbrack_semirot,
        ]

    def setup_for_pose_stack(self, pose_stack, pbt=None, *, additional_block_types=()):
        pbt = pose_stack.packed_block_types if pbt is None else pbt
        libraries = self._prepared(
            "scoring_libraries",
            lambda: self._scoring_libraries(pbt),
            *pbt._scoring_sources,
        )
        # Keep the complete palette and its original IDs. Native table storage
        # needs only the actual pose and sampled/design candidates.
        selected = set()
        for types in (pose_stack.block_type_ind, *additional_block_types):
            indices = np.asarray(types)
            selected.update(map(int, libraries[np.unique(indices[indices >= 0])]))
        selected.discard(-1)
        table_indices = tuple(sorted(selected))
        if not table_indices:
            return DunbrackTermState(())
        parameters = self._prepared(
            "device_parameters",
            lambda: self._packed_parameters(
                pbt, pose_stack.coords.dtype, table_indices
            ),
            *pbt._scoring_sources,
            settings=(
                pbt.max_n_atoms,
                pbt.max_n_conn,
                pbt.n_atoms.device,
                pose_stack.coords.dtype,
                table_indices,
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

    def _scoring_libraries(self, pbt):
        lookup = prepare_dunbrack_lookup(self.param_db.scoring.dunbrack)
        return np.array(
            [
                dunbrack_scoring_table_indices(bt, lookup)[0]
                for bt in pbt.active_block_types
            ],
            dtype=np.int32,
        )

    def _packed_parameters(self, pbt, coordinate_dtype, table_indices):
        tables = prepare_dunbrack_parameters(self.param_db, table_indices=table_indices)
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
        device = pbt.n_atoms.device
        # Only library tables are shared across poses. Atom connections and
        # residue-specific topology are prepared independently for each palette.
        downstream = native_parameters(
            (atom_downstream_of_conn(pbt),), coordinate_dtype, device
        )
        return (
            downstream
            + device_dunbrack_tables(tables, coordinate_dtype, device)
            + native_parameters(blocks.values(), coordinate_dtype, device)
        )

    @classmethod
    def evaluate(cls, term_state, coords):
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
            values = jnp.zeros_like(coords, shape=(3, coords.shape[0]))
        return dict(zip(cls.score_types_provided(), values))
