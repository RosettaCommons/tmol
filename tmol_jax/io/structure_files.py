"""Shared molecular file formats with JAX pose export."""

from tmol_core.io.structure_io import atom_array_from_file, file_from_atom_array
from .structure_export import atom_array_from_pose_stack

__all__ = ["atom_array_from_file", "file_from_atom_array", "file_from_pose_stack"]


def file_from_pose_stack(pose_stack, path=None, format=None, **kwargs):
    """Write a pose as PDB/CIF, or return text when no path is supplied.

    Export options are passed to ``atom_array_from_pose_stack``. File export is
    eager host I/O; use the pose's parameter context when reading custom chemistry.
    """
    return file_from_atom_array(
        atom_array_from_pose_stack(pose_stack, **kwargs), path, format
    )
