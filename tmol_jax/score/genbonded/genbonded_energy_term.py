"""Generic bonded torsions from shared preparation and whole-pose kernels."""

import jax.numpy as jnp
import numpy as np

from tmol_core.chemical.constants import MAX_PATHS_FROM_CONNECTION
from tmol_core.scoring.genbonded import GenBondedPreparation
from tmol_jax.score._native import coordinate_energy, register
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
        paths = np.full((len(types), ports, MAX_PATHS_FROM_CONNECTION, 3), -1, np.int32)
        for i, bt in enumerate(types):
            paths[i, : bt.n_conn] = bt.atom_paths_from_conn
        # JAX execution pads connection axes to two, native preparation to one.
        for i in (5, 6):
            if packed[i].shape[1] < ports:
                padding = [(0, 0)] * packed[i].ndim
                padding[1] = (0, ports - packed[i].shape[1])
                packed[i] = np.pad(packed[i], padding)
        bti = np.asarray(pose_stack.block_type_ind)
        poses, blocks = bti.shape
        first_rot = np.arange(poses * blocks, dtype=np.int32).reshape(poses, blocks)
        first_rot = np.where(bti >= 0, first_rot, -1)
        offsets = (
            np.asarray(pose_stack.block_coord_offset)
            + np.arange(poses, dtype=np.int32)[:, None] * pose_stack.coords.shape[1]
        )
        return tuple(
            jnp.asarray(a)
            for a in (
                offsets.reshape(-1),
                first_rot,
                bti,
                pose_stack.inter_residue_connections,
                paths,
                *packed,
                *prep.inter_tables(),
            )
        )

    def __call__(self, pose_stack, term_state, coords_override=None):
        coords = pose_stack.coords if coords_override is None else coords_override
        if not term_state:
            value = jnp.zeros(coords.shape[0], coords.dtype)
        else:
            arrays = tuple(
                a.astype(coords.dtype) if jnp.issubdtype(a.dtype, jnp.floating) else a
                for a in term_state
            )
            value = coordinate_energy("tmol_genbonded", coords, arrays)
        return {ScoreType.gen_torsions: value}
