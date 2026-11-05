"""Prepared kinematic indices with JAX-owned device buffers."""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from tmol_core.kinematics import prepare_kinematic_forest
from ._native import inverse_coordinates


@dataclass(frozen=True, eq=False, init=False)
class KinematicForest:
    """Validate a forest once and initialize DOFs for changing coordinates.

    Indices and input coordinates use kinematic order, including virtual root
    zero. The shared core validates topology; this adapter owns device arrays.
    Capture the prepared forest in a jitted function, or jit ``forest.inverse``.
    Pose-to-forest construction and internal-coordinate minimization are not
    provided by this operation.
    """

    _indices: object = field(repr=False)
    _arrays: tuple = field(repr=False)

    def __init__(self, parent, frame_x, frame_y, frame_z, doftype, *, device=None):
        indices = prepare_kinematic_forest(parent, frame_x, frame_y, frame_z, doftype)
        arrays = tuple(
            jax.device_put(getattr(indices, name), device)
            for name in ("parent", "frame_x", "frame_y", "frame_z", "doftype")
        )
        object.__setattr__(self, "_indices", indices)
        object.__setattr__(self, "_arrays", arrays)

    @property
    def n_nodes(self):
        return self._indices.n_nodes

    @property
    def device(self):
        return self._arrays[0].device

    def inverse(self, coords):
        """Return float32/64 DOFs [nodes,9], without differentiating initialization."""
        return inverse_coordinates(jnp.asarray(coords), self._arrays)
