"""Cartesian bonded scoring from the shared preparation and native kernels."""

import jax
import numpy as np

from tmol_core.scoring.cartbonded import CartBondedPreparation
from tmol_jax.score._native import (
    bonded_pose_arguments,
    coordinate_energy,
    register,
    workspace_sizes,
)
from tmol_jax.score.energy_term import EnergyTerm
from tmol_jax.score.score_types import ScoreType


class CartBondedEnergyTerm(EnergyTerm):
    @staticmethod
    def score_types_provided():
        return [
            ScoreType.cart_lengths,
            ScoreType.cart_angles,
            ScoreType.cart_torsions,
            ScoreType.cart_impropers,
            ScoreType.cart_hxltorsions,
        ]

    def setup_for_pose_stack(self, pose_stack, pbt=None):
        pbt = pose_stack.packed_block_types if pbt is None else pbt
        prep = self._prepared(
            "resolver",
            lambda: CartBondedPreparation(
                self.param_db.scoring.cartbonded,
                self.param_db.scoring.genbonded.rosetta_typed,
            ),
        )
        types = pbt.active_block_types
        host = self._prepared("packed", lambda: prep.prepare_packed(types), *types)
        register("tmol_cartbonded")
        precise = pose_stack.coords.device.platform == "gpu"
        if precise:
            register("tmol_cartbonded_precise")
        parameters = jax.device_put(
            (
                *bonded_pose_arguments(pose_stack, pbt),
                host.atom_unique_ids,
                host.atom_wildcard_ids,
                host.atom_is_rosetta,
                np.asarray([bt.is_ligand_fragment for bt in types], dtype=np.int32),
                host.atom_cross_ids,
                host.connection_hash_keys,
                host.connection_spans,
                host.connection_paths,
                host.cartbonded_params_hash_keys,
                host.cartbonded_params_hash_values,
                host.cartbonded_subgraphs,
                host.cartbonded_subgraph_offsets,
                host.cartbonded_subgraph_type_counts,
                host.cartbonded_subgraph_type_offsets,
                host.cartbonded_subgraph_param_indices,
            ),
            device=pose_stack.coords.device,
        )

        return parameters, precise

    @classmethod
    def evaluate(cls, term_state, coords):
        parameters, precise = term_state
        target, sizes = "tmol_cartbonded", None
        if precise and coords.dtype == np.float32:
            target += "_precise"
            sizes = workspace_sizes(target, coords.shape[:2], 32)
        values = coordinate_energy(
            target, coords, parameters, channels=5, workspace_bytes=sizes
        )
        return dict(zip(cls.score_types_provided(), values))
