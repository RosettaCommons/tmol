"""Thin runtime adapter for common nucleic-acid host and native science."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tmol_core.scoring.na_torsion import _block_type_params
from tmol_core.scoring.na_torsion_params import prepare_na_torsion_parameters
from tmol_core.scoring.na_torsion_topology import prepare_na_torsion_topology
from tmol_jax.score._native import atom_downstream_of_conn, coordinate_energy, register
from tmol_jax.score.energy_term import EnergyTerm, prepared_palette
from tmol_jax.score.score_types import ScoreType


class NaTorsionTermState(NamedTuple):
    parameters: tuple
    attributes: dict


def na_block_parameters(pbt):
    def prepare():
        elements = {at.name: at.element for at in pbt.chem_db.atom_types}
        return [_block_type_params(bt, elements) for bt in pbt.active_block_types]

    return prepared_palette(pbt, pbt.chem_db, "na_blocks", prepare)


def na_parameter_state(database, device=None):
    params = prepare_na_torsion_parameters(database.scoring.na_torsion)
    return NaTorsionTermState(
        jax.device_put(
            tuple(
                (
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
                )
            ),
            device=device,
        ),
        {
            name: np.float64(params[name])
            for name in ("pucker_temperature", "bin_blend_sdev")
        },
    )


def na_packed_state(pbt, database, device):
    """Reuse chemical metadata and parameters; pose topology stays separate."""

    def prepare():
        blocks = na_block_parameters(pbt)
        metadata = jax.device_put(
            (
                np.asarray(atom_downstream_of_conn(pbt), np.int32),
                *(
                    np.asarray([row[name] for row in blocks], np.int32)
                    for name in ("base", "uaids", "ring", "down")
                ),
            ),
            device=device,
        )
        return metadata, na_parameter_state(database, device)

    return prepared_palette(pbt, database, "na_torsion", prepare, device=device)


class NaTorsionEnergyTerm(EnergyTerm):
    @staticmethod
    def score_types_provided():
        return [ScoreType.na_torsion, ScoreType.na_torsion_well]

    def setup_for_pose_stack(self, pose_stack, pbt=None):
        pbt = pose_stack.packed_block_types if pbt is None else pbt
        blocks = na_block_parameters(pbt)
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
        _, state = na_packed_state(pbt, self.param_db, pose_stack.coords.device)
        register("tmol_na_torsion")
        return NaTorsionTermState(
            jax.device_put(
                tuple(
                    (a.astype(np.int32) if a.dtype.kind == "i" else a for a in topology)
                ),
                device=pose_stack.coords.device,
            )
            + state.parameters,
            state.attributes,
        )

    @classmethod
    def evaluate(cls, term_state, coords):
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
        return dict(zip(cls.score_types_provided(), values))
