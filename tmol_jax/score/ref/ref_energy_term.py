"""Sum database reference energies over real residues in each pose.

Parameters come from ref.yaml. This one-body term is independent of coordinates,
so its coordinate gradient is zero. JAX packing is not implemented."""

from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.score.energy_term import EnergyTerm
from tmol_jax.score.score_types import ScoreType


class RefEnergyTerm(EnergyTerm):
    """Per-residue-type constant reference energy.

    The score for a pose is ``sum_b ref_values[block_type_ind[p, b]]``
    over real (non-padding) blocks. The ``ref_values`` vector is built
    once per :class:`PackedBlockTypes` palette by looking up each
    block-type's ``base_name`` (the three-letter residue name) in
    ``param_db.scoring.ref.weights``; residues with no entry get
    ``0.0`` (matches upstream tmol behavior).
    """

    def __init__(self, param_db: Any):
        super().__init__(param_db=param_db)
        # Raw ``{residue_name: weight}`` mapping from ``ref.yaml``.
        # Stored by reference; used in ``setup_for_pose_stack``.
        self._ref_weights: dict[str, float] = dict(param_db.scoring.ref.weights)

    # Metadata.                                                          #

    def score_types_provided(self) -> list[ScoreType]:
        return [ScoreType.ref]

    # Setup.                                                             #

    def setup_for_pose_stack(
        self, pose_stack: Any, pbt: Any = None
    ) -> dict[str, jax.Array]:
        """Build the ``[n_restypes]`` ref-value vector for this palette.

        The order matches ``PackedBlockTypes.active_block_types``, so
        ``ref_values[block_type_ind[p, b]]`` is the weight for the
        residue type occupying slot ``b`` of pose ``p``.

        Returns a pytree so it survives ``jit`` / ``vmap`` flattening.
        """
        if pbt is None:
            pbt = pose_stack.packed_block_types

        # Per-type weights, gathered from the yaml mapping by base_name.
        # Missing entries default to 0.0 — matches upstream behavior.
        weights_np = np.zeros((pbt.n_types,), dtype=np.float32)
        for i, bt in enumerate(pbt.active_block_types):
            w = self._ref_weights.get(bt.base_name, 0.0)
            weights_np[i] = float(w)

        return {"ref_values": jnp.asarray(weights_np, dtype=jnp.float32)}

    # Evaluation (hot path).                                             #

    def __call__(
        self,
        pose_stack: Any,
        term_state: dict[str, jax.Array],
        coords_override: Optional[jax.Array] = None,
    ) -> dict[ScoreType, jax.Array]:
        """Sum per-block ref weights across each pose.

        The reference term is coordinate-independent; ``coords_override``
        is accepted only for EnergyTerm API compatibility with other
        terms that do depend on coords.
        """
        del coords_override  # not used — ref is coord-independent

        ref_values = term_state["ref_values"]  # float32[n_types]
        bti = pose_stack.block_type_ind  # int32[n_poses, max_n_blocks]

        # Gather weights, clamping padding (-1) indices to 0 so the
        # gather is safe; then ``where`` zeros the padding contributions
        # so they never reach the sum.
        real = bti >= 0
        safe_idx = jnp.where(real, bti, 0)
        per_block = ref_values[safe_idx]
        per_block = jnp.where(real, per_block, 0.0).astype(jnp.float32)

        per_pose = jnp.sum(per_block, axis=1).astype(jnp.float32)
        return {ScoreType.ref: per_pose}


__all__ = ["RefEnergyTerm"]
