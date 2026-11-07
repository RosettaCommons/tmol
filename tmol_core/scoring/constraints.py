"""Shared vectorized constraint physics with five execution primitives.

These formulas remain in the frontend's differentiation graph. ``norm`` takes
vectors along the last dimension and uses the zero derivative at zero length;
``torsion`` evaluates TMol's canonical four-atom angle. The other primitives
are elementwise where, truncation, and sign. No framework is imported here.
"""

from dataclasses import dataclass
from collections.abc import Callable
import math


@dataclass(frozen=True)
class ConstraintOps:
    norm: Callable
    where: Callable
    trunc: Callable
    sign: Callable
    torsion: Callable


def harmonic(atoms, params, ops):
    """Distance restraint; parameters are target distance and standard deviation."""
    dist = ops.norm(atoms[:, 0] - atoms[:, 1])
    return ((dist - params[:, 0]) / params[:, 1]) ** 2


def harmonic_coordinate(atoms, params, ops):
    """Coordinate restraint; parameters are distance, target xyz, deviation."""
    dist = ops.norm(atoms[:, 0] - params[:, 1:4])
    return ((dist - params[:, 0]) / params[:, 4]) ** 2


def bounded(atoms, params, ops):
    """Bounded distance; parameters are lower, upper, deviation, tail switch.

    The legacy upper transition is upper + 0.5 * deviation, independent of the
    tail-switch parameter. Mask operands before arithmetic so inactive branches
    retain zero derivatives even when their unused deviation is zero.
    """
    lb, ub, sd, rswitch = (params[:, i] for i in range(4))
    dist = ops.norm(atoms[:, 0] - atoms[:, 1])
    lower = dist < lb
    upper = (ub < dist) & (dist <= ub + 0.5 * sd)
    tail = dist > ub + 0.5 * sd
    lower_value = (ops.where(lower, lb - dist, 0) / ops.where(lower, sd, 1)) ** 2
    upper_value = (ops.where(upper, dist - ub, 0) / ops.where(upper, sd, 1)) ** 2
    switch = ops.where(tail, rswitch, 0)
    tail_value = (
        2 * switch * (ops.where(tail, dist - ub, 0) / ops.where(tail, sd, 1))
        - switch**2
    )
    return ops.where(tail, tail_value, ops.where(upper, upper_value, lower_value))


def circularharmonic(atoms, params, ops):
    """Periodic torsion restraint; parameters are target radians, deviation, offset."""
    angle = ops.torsion(atoms)
    delta = angle - params[:, 0]
    turns = delta / (2 * math.pi)
    rounded = ops.trunc(turns + ops.sign(turns) * 0.5)
    # Preserve the established subtraction order and away-from-zero tie rule.
    nearest = angle - rounded * (2 * math.pi)
    z = (nearest - params[:, 0]) / params[:, 1]
    return z * z + params[:, 2]


def coordinate_flatbottom(atoms, params, ops):
    """Flat-bottom coordinate restraint; parameters are target xyz, k, tolerance."""
    dist = ops.norm(atoms[:, 0] - params[:, :3])
    delta = ops.where(dist > params[:, 4], dist - params[:, 4], 0)
    return 0.5 * params[:, 3] * delta**2
