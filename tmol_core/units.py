"""`pint <pint:index>`-based unit support functions."""

import math

from typing import Any, Union, Tuple, NewType

import pint
import cattr

ureg = pint.UnitRegistry()
u = ureg.parse_expression
_BOND_ANGLE_LIMIT = (0.0, math.pi)
_DIHEDRAL_ANGLE_LIMIT = (-math.pi, math.pi)


def _simple_angle(angle: str):
    fields = angle.split()
    if len(fields) != 2 or fields[1] not in ("deg", "rad"):
        return None
    try:
        value = float(fields[0])
    except ValueError:
        return None
    return math.radians(value) if fields[1] == "deg" else value


def _parse_angle(
    angle: Union[float, str], lim: Tuple[float, float] = (-2 * math.pi, 2 * math.pi)
) -> float:
    """Parse an angle via :doc:`pint <pint:index>` and convert to radians.

    Args:
        angle: Unit-qualified angle or float value in radians.
        lim: Raise ValueError if outside [min, max] range in radians.

    Returns:
        Angle in radians.

    """

    if isinstance(angle, str):
        val = _simple_angle(angle)
        if val is None:
            val = ureg.parse_expression(angle)
    else:
        val = angle

    if not isinstance(val, pint.util.Number):
        # May have parsed to a just a number or a
        # quantity with units, if so convert to radians
        val.to(ureg.rad)

    if lim:
        minv, maxv = lim
        if minv is not None and val < minv:
            raise ValueError(f"angle: {angle!r} outside of allowed range: {lim}")
        if maxv is not None and val > maxv:
            raise ValueError(f"angle: {angle!r} outside of allowed range: {lim}")

    return float(val)


def _parse_bond_angle(v: Union[float, str]) -> float:
    """Parse a bond angle on the range [0, pi) via pint."""
    return _parse_angle(v, lim=_BOND_ANGLE_LIMIT)


def _parse_dihedral_angle(v) -> float:
    """Parse a dihedral angle on the range [-pi, pi) via pint."""
    return _parse_angle(v, lim=_DIHEDRAL_ANGLE_LIMIT)


Angle = NewType("Angle", float)
cattr.register_structure_hook(Angle, lambda v, t: _parse_angle(v))

BondAngle = NewType("BondAngle", float)
cattr.register_structure_hook(BondAngle, lambda v, t: _parse_bond_angle(v))

DihedralAngle = NewType("DihedralAngle", float)
cattr.register_structure_hook(DihedralAngle, lambda v, t: _parse_dihedral_angle(v))


def parse_angle(v: Any) -> float:
    """Parse a number or unit-qualified string into radians."""
    if v is None:
        return v  # type: ignore[return-value]
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        parts = v.strip().split()
        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 2:
            val = float(parts[0])
            unit = parts[1].lower()
            if unit in ("rad", "radian", "radians"):
                return val
            if unit in ("deg", "degree", "degrees"):
                return val * math.pi / 180.0
            raise ValueError(f"Unknown angle unit: {unit!r} in {v!r}")
        raise ValueError(f"Cannot parse angle: {v!r}")
    raise TypeError(f"Cannot parse angle from type {type(v).__name__}: {v!r}")
