"""Disulfide score term (pure JAX)."""

from __future__ import annotations

from tmol_jax.score.disulfide.disulfide_energy_term import DisulfideEnergyTerm
from tmol_jax.score.disulfide.potentials import disulfide_pair_E

__all__ = [
    "DisulfideEnergyTerm",
    "disulfide_pair_E",
]
