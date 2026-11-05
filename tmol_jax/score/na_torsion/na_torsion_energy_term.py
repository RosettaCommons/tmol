"""Thin runtime adapter for common nucleic-acid host and native science."""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from tmol_core.scoring.na_torsion import _block_type_params
from tmol_core.scoring.na_torsion_params import prepare_na_torsion_parameters
from tmol_core.scoring.na_torsion_topology import prepare_na_torsion_topology
from tmol_jax.score._native import atom_downstream_of_conn, coordinate_energy, register
from tmol_jax.score.energy_term import EnergyTerm
from tmol_jax.score.score_types import ScoreType


class NaTorsionTermState(NamedTuple):
    parameters: tuple
    attributes: dict


class NaTorsionEnergyTerm(EnergyTerm):
    def score_types_provided(self):
        return [ScoreType.na_torsion, ScoreType.na_torsion_well]

    def setup_for_pose_stack(self, pose_stack, pbt=None):
        pbt = pose_stack.packed_block_types if pbt is None else pbt
        elements = {at.name: at.element for at in pbt.chem_db.atom_types}
        blocks = [_block_type_params(bt, elements) for bt in pbt.active_block_types]
        bti = np.asarray(pose_stack.block_type_ind)
        base = np.asarray([row["base"] for row in blocks], np.int32)
        if not np.any(base[bti[bti >= 0]] >= 0):
            return NaTorsionTermState((), {})
        if max(bti.size, np.prod(pose_stack.coords.shape[:2])) > np.iinfo(np.int32).max:
            raise ValueError("native NA topology exceeds int32 indexing")
        topology = prepare_na_torsion_topology(
            bti,
            base,
            np.stack([row["uaids"] for row in blocks]),
            np.stack([row["ring"] for row in blocks]),
            np.asarray([row["down"] for row in blocks], np.int32),
            atom_downstream_of_conn(pbt),
            np.asarray(pose_stack.block_coord_offset),
            np.asarray(pose_stack.inter_residue_connections),
            pose_stack.coords.shape[1],
        )
        params = prepare_na_torsion_parameters(self.param_db.scoring.na_torsion)
        arrays = (
            *topology,
            *(
                params[name]
                for name in (
                    "backbone_means",
                    "backbone_sdev",
                    "sugar_means",
                    "chi_means",
                    "sdev_sugar",
                    "sdev_chi",
                    "well_pucker",
                    "well_alpha_gamma",
                    "well_bibii_pucker",
                    "well_alphanext_bibii",
                    "well_chi_syn",
                    "is_north",
                    "weight_bb",
                    "weight_chi",
                    "weight_sugar",
                )
            ),
        )
        register("tmol_na_torsion")
        return NaTorsionTermState(
            tuple(
                jnp.asarray(a.astype(np.int32) if a.dtype.kind == "i" else a)
                for a in arrays
            ),
            {
                name: np.float64(params[name])
                for name in ("pucker_temperature", "bin_blend_sdev")
            },
        )

    def __call__(self, pose_stack, term_state, coords_override=None):
        coords = pose_stack.coords if coords_override is None else coords_override
        values = (
            coordinate_energy(
                "tmol_na_torsion",
                coords,
                term_state.parameters,
                channels=2,
                attributes=term_state.attributes,
            )
            if term_state.parameters
            else jnp.zeros((2, coords.shape[0]), coords.dtype)
        )
        return dict(zip(self.score_types_provided(), values))
