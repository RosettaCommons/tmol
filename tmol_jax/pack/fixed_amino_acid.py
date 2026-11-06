"""Ideal fixed-amino-acid conformers using the common chemical ownership policy."""

import jax
import jax.numpy as jnp
import numpy as np

from tmol_core.pack.chi import is_fixed_amino_acid, fixed_amino_acid_roots
from .current import IncludeCurrentSampler


class FixedAAChiSampler(IncludeCurrentSampler):
    """Build one ideal sidechain for amino acids with no non-proton chi.

    The common policy includes Gly, Ala, D variants and suitable custom types.
    Mainchain coordinates use the same native conformer construction as other
    chi samplers; glycine rebuilds both alpha hydrogens under that policy.
    """

    def defines_rotamers_for_rt(self, restype):
        return is_fixed_amino_acid(restype)

    def first_sc_atoms_for_rt(self, restype):
        return fixed_amino_acid_roots(restype)

    def sample_chi_for_poses(self, pose, task):
        task, rows, sampler = self._selection_inputs(pose, task)
        with jax.default_device(pose.device):
            builds = jnp.asarray(
                [
                    is_fixed_amino_acid(bt)
                    for bt in pose.packed_block_types.active_block_types
                ],
                jnp.bool_,
            )
            selected = (
                task.is_cons_bt_allowed
                & task.per_block_conformer_sampler_allowed[rows[:2]][:, sampler]
                & builds[task.cons_bt_block_type]
            )
            choices = jax.device_put(
                np.flatnonzero(np.asarray(selected)).astype(np.int32), pose.device
            )
            return (
                selected.astype(jnp.int32),
                choices,
                jnp.full((choices.size, 1), -1, jnp.int32),
                jnp.zeros((choices.size, 1), jnp.float32),
            )

    def create_samples_for_poses(self, pose, task):
        counts, choices, atoms, chi = self.sample_chi_for_poses(pose, task)
        return (
            counts,
            choices,
            {
                "chi_defining_atom_for_rotamer": atoms,
                "chi_for_rotamers": chi.astype(pose.coords.dtype),
            },
        )
