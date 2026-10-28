"""Disulfide energy term — pure-JAX port of :mod:`tmol.score.disulfide`.

Every disulfide bond in the pose topology (a connection between two cysteine
SG atoms) is scored with :func:`potentials.disulfide_pair_E`.

The term lives under a single :class:`~tmol_jax.score.score_types.ScoreType`
(``disulfide``) and follows the :class:`~tmol_jax.score.energy_term.EnergyTerm`
two-stage contract: :meth:`setup_for_pose_stack` precomputes static per-pose
atom-index tables (CPU / numpy work, free of tracers) and ``__call__``
is the jit-safe hot path.
"""

from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.score.disulfide.potentials import (
    _params_as_dict,
    disulfide_pair_E,
)
from tmol_jax.score.energy_term import EnergyTerm
from tmol_jax.score.score_types import ScoreType

# Generic-position placeholder geometry (CA_i, CB_i, SG_i, CA_j, CB_j, SG_j)
# substituted for invalid Cys-Cys pairs so the pair potential's angle /
# dihedral gradients are finite on masked-out slots. No coincident atoms,
# no collinear angle/dihedral triples, non-planar — the exact values are
# irrelevant because the energy of invalid pairs is masked to 0.
_CANONICAL_CYSTINE = jnp.asarray(
    [
        [0.0, 0.0, 0.0],  # CA_i
        [1.5, 0.0, 0.0],  # CB_i
        [2.5, 1.2, 0.3],  # SG_i
        [0.2, 4.0, 2.5],  # CA_j
        [1.6, 4.2, 1.8],  # CB_j
        [2.6, 3.2, 1.0],  # SG_j
    ]
)


class DisulfideEnergyTerm(EnergyTerm):
    """Rosetta beta_nov2016 disulfide term as a pure-JAX EnergyTerm."""

    def score_types_provided(self) -> list[ScoreType]:
        return [ScoreType.disulfide]

    def _cached_params(self) -> dict[str, Any]:
        """Cache the 26 disulfide global scalars converted to float32 jnp.

        They're pure-database constants; rebuilding the dict + ``jnp.asarray``
        wrappers every setup took ~0.5 ms on 1UBQ (where there are no Cys
        to score). One-time cached on the term instance.
        """
        cached = getattr(self, "_params_cache", None)
        if cached is not None:
            return cached
        disulfide_db = self.param_db.scoring.disulfide
        params = _params_as_dict(disulfide_db.global_parameters)
        params = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in params.items()}
        self._params_cache = params
        return params

    # ------------------------------------------------------------------ #
    # Setup                                                              #
    # ------------------------------------------------------------------ #
    def setup_for_pose_stack(self, pose_stack: Any, pbt: Any = None) -> dict[str, Any]:
        """Index the CA, CB and SG atoms of every disulfide-bonded cystine pair.

        Pairs come from the pose topology (a connection between two cysteines
        at their SG atoms), like TMol's, not from SG-SG proximity. The state
        holds ``atoms``: ``int32[n_poses, max_pairs, 6]`` flat atom indices
        ordered (CA_i, CB_i, SG_i, CA_j, CB_j, SG_j), ``valid`` padding masks
        and the float32 database ``params``.
        """
        if pbt is None:
            pbt = pose_stack.packed_block_types
        block_types = np.asarray(pose_stack.block_type_ind)
        offsets = np.asarray(pose_stack.block_coord_offset)
        connections = np.asarray(pose_stack.inter_residue_connections)
        conn_atoms = np.asarray(pbt.conn_atoms)

        def cys_atoms(block_type):
            a2i = pbt.active_block_types[block_type].atom_to_idx
            return tuple(a2i.get(name, -1) for name in ("CA", "CB", "SG"))

        pairs = []
        for pose in range(block_types.shape[0]):
            found = []
            for i, j_conn in zip(*np.nonzero(connections[pose, :, :, 0] >= 0)):
                j, conn = (int(x) for x in connections[pose, i, j_conn])
                bi, bj = block_types[pose, i], block_types[pose, j]
                if i >= j or "CYS" not in (pbt.active_block_types[bi].name3,):
                    continue
                ca_i, cb_i, sg_i = cys_atoms(bi)
                ca_j, cb_j, sg_j = cys_atoms(bj)
                if conn_atoms[bi, j_conn] != sg_i or conn_atoms[bj, conn] != sg_j:
                    continue
                if min(ca_i, cb_i, sg_i, ca_j, cb_j, sg_j) < 0:
                    continue
                found.append(
                    [
                        offsets[pose, i] + ca_i,
                        offsets[pose, i] + cb_i,
                        offsets[pose, i] + sg_i,
                        offsets[pose, j] + ca_j,
                        offsets[pose, j] + cb_j,
                        offsets[pose, j] + sg_j,
                    ]
                )
            pairs.append(found)
        width = max(1, max(len(found) for found in pairs))
        atoms = np.zeros((len(pairs), width, 6), dtype=np.int32)
        valid = np.zeros((len(pairs), width), dtype=bool)
        for pose, found in enumerate(pairs):
            if found:
                atoms[pose, : len(found)] = found
                valid[pose, : len(found)] = True
        return dict(
            atoms=jnp.asarray(atoms),
            valid=jnp.asarray(valid),
            params=self._cached_params(),
        )

    def __call__(
        self,
        pose_stack: Any,
        term_state: dict[str, Any],
        coords_override: Optional[jax.Array] = None,
    ) -> dict[ScoreType, jax.Array]:
        coords = pose_stack.coords if coords_override is None else coords_override
        atoms, valid = term_state["atoms"], term_state["valid"]
        index = atoms[..., None].repeat(3, axis=-1).reshape(atoms.shape[0], -1, 3)
        xyz = jnp.take_along_axis(coords, index, axis=1).reshape(*atoms.shape, 3)
        # Padded slots use a constant non-degenerate cystine so the masked
        # energies have finite (zero) gradients.
        xyz = jnp.where(
            valid[..., None, None], xyz, _CANONICAL_CYSTINE.astype(xyz.dtype)
        )
        energy = disulfide_pair_E(
            *(xyz[..., k, :] for k in range(6)), term_state["params"]
        )
        per_pose = jnp.sum(jnp.where(valid, energy, 0.0), axis=1)
        return {ScoreType.disulfide: per_pose.astype(jnp.float32)}


__all__ = ["DisulfideEnergyTerm"]
