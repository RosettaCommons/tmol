"""Compatibility names for the shared host chemistry algorithms."""

from tmol_core.chemical.ideal_coords import (
    _eye4 as eye4,
    _normalize as normalize,
    _frame_from_coords as frame_from_coords,
    _rot_x as rot_x,
    _rot_z as rot_z,
    _trans_z as trans_z,
    _build_coords_from_icoors as build_coords_from_icoors,
    _build_ideal_coords as build_ideal_coords,
)

__all__ = [
    "eye4",
    "normalize",
    "frame_from_coords",
    "rot_x",
    "rot_z",
    "trans_z",
    "build_coords_from_icoors",
    "build_ideal_coords",
]
