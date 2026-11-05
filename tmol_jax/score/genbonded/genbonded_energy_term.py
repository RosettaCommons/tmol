"""Generic bonded torsions from shared preparation and whole-pose kernels."""

import jax.numpy as jnp
import numpy as np

from tmol_core.scoring.genbonded import GenBondedPreparation
from tmol_jax.score._native import coordinate_energy, register, bonded_pose_arguments
from tmol_jax.score.energy_term import EnergyTerm
from tmol_jax.score.score_types import ScoreType


class GenBondedEnergyTerm(EnergyTerm):
    def score_types_provided(self):
        return [ScoreType.gen_torsions]

    def setup_for_pose_stack(self, pose_stack, pbt=None):
        pbt = pose_stack.packed_block_types if pbt is None else pbt
        prep = GenBondedPreparation(
            self.param_db.chemical, self.param_db.scoring.genbonded
        )
        types = pbt.active_block_types
        blocks = [prep.prepare_block(bt) for bt in types]
        # Decide interaction ownership from the common parameters, not names.
        if all(
            atom.atom_type in prep.gen_database.rosetta_typed
            for bt in types
            for atom in bt.atoms
        ):
            return ()
        register("tmol_genbonded")
        packed = list(prep.prepare_packed(types, blocks))
        ports = pbt.max_n_conn
        # JAX execution pads connection axes to two, native preparation to one.
        for i in (5, 6):
            if packed[i].shape[1] < ports:
                padding = [(0, 0)] * packed[i].ndim
                padding[1] = (0, ports - packed[i].shape[1])
                packed[i] = np.pad(packed[i], padding)
        return tuple(
            jnp.asarray(a)
            for a in (
                *bonded_pose_arguments(pose_stack, pbt),
                *packed,
                *prep.inter_tables(),
            )
        )

    def __call__(self, pose_stack, term_state, coords_override=None):
        coords = pose_stack.coords if coords_override is None else coords_override
        if not term_state:
            value = jnp.zeros(coords.shape[0], coords.dtype)
        else:
            value = coordinate_energy("tmol_genbonded", coords, term_state)
        return {ScoreType.gen_torsions: value}
