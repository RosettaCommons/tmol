"""JAX execution primitives for common constraint formulas."""

from functools import wraps

import jax.numpy as jnp

from tmol_core.scoring import constraints as common
from tmol_jax.score._native import coordinate_energy, register


def _norm(value):
    squared = jnp.sum(value * value, axis=-1)
    nonzero = squared > 0
    return jnp.where(nonzero, jnp.sqrt(jnp.where(nonzero, squared, 1)), 0)


def torsion(atoms):
    register("tmol_constraint_torsion")
    return coordinate_energy("tmol_constraint_torsion", atoms[:, :4], ())


_OPS = common.ConstraintOps(_norm, jnp.where, jnp.trunc, jnp.sign, torsion)


def _callback(function, atoms, parameters):
    @wraps(function)
    def score(xyz, params):
        return function(xyz, params, _OPS)

    score._tmol_constraint_atoms = atoms
    score._tmol_constraint_parameters = parameters
    return score


harmonic = _callback(common.harmonic, 2, 2)
harmonic_coordinate = _callback(common.harmonic_coordinate, 1, 5)
bounded = _callback(common.bounded, 2, 4)
circularharmonic = _callback(common.circularharmonic, 4, 3)
coordinate_flatbottom = _callback(common.coordinate_flatbottom, 1, 5)
