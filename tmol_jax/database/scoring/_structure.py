"""Use the shared attrs converter, preserving JAX's unrestricted angle parser."""

from typing import NewType
import cattr

from tmol_core.units import parse_angle

Angle = NewType("Angle", float)
cattr.register_structure_hook(Angle, lambda value, _: parse_angle(value))
structure = cattr.structure

__all__ = ["Angle", "structure"]
