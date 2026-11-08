"""Generic bonded torsions from shared preparation and whole-pose kernels."""

import jax
import jax.numpy as jnp
import numpy as np

from tmol_core.scoring.genbonded import GenBondedPreparation
from tmol_jax.score._native import coordinate_energy, register, bonded_pose_arguments
from tmol_jax.score.energy_term import EnergyTerm, prepared_palette
from tmol_jax.score.score_types import ScoreType


class GenBondedEnergyTerm(EnergyTerm):
    @staticmethod
    def score_types_provided():
        return [ScoreType.gen_torsions]

    def setup_for_pose_stack(self, pose_stack, pbt=None, *, include_empty=False):
        pbt = pose_stack.packed_block_types if pbt is None else pbt
        prep = self._prepared(
            "resolver",
            lambda: GenBondedPreparation(
                self.param_db.chemical, self.param_db.scoring.genbonded
            ),
        )
        types = pbt.active_block_types
        blocks = self._prepared(
            "blocks",
            lambda: [prep.prepare_block(bt) for bt in types],
            *pbt._scoring_sources,
        )
        # Decide interaction ownership from the common parameters, not names.
        if not include_empty and all(
            atom.atom_type in prep.gen_database.rosetta_typed
            for bt in types
            for atom in bt.atoms
        ):
            return ()
        register("tmol_genbonded")
        packed = list(
            self._prepared(
                "packed",
                lambda: prep.prepare_packed(types, blocks),
                *pbt._scoring_sources,
            )
        )
        ports = pbt.max_n_conn
        # JAX execution pads connection axes to two, native preparation to one.
        for i in (5, 6):
            if packed[i].shape[1] < ports:
                padding = [(0, 0)] * packed[i].ndim
                padding[1] = (0, ports - packed[i].shape[1])
                packed[i] = np.pad(packed[i], padding)
        device = pose_stack.coords.device
        chemical_arrays = prepared_palette(
            pbt,
            self.param_db,
            "genbonded",
            lambda: jax.device_put(
                (*packed, *self._prepared("inter_tables", prep.inter_tables)),
                device=device,
            ),
            device=device,
        )
        return (
            jax.device_put(bonded_pose_arguments(pose_stack, pbt), device=device)
            + chemical_arrays
        )

    @classmethod
    def evaluate(cls, term_state, coords):
        if not term_state:
            value = jnp.zeros(coords.shape[0], coords.dtype)
        else:
            value = coordinate_energy("tmol_genbonded", coords, term_state)
        return {ScoreType.gen_torsions: value}
