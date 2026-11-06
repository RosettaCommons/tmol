"""Structure readers/writers shared with core; Torch pose conversion."""

from pathlib import Path
from typing import Literal, Optional, Union

import biotite.structure as struc
import torch

from tmol_core.io import structure_io as _shared
from tmol_core.io.structure_io import (  # noqa: F401
    StructureFormat,
    StructureSource,
    READ_FORMATS,
    WRITE_FORMATS,
    atom_array_from_file,
    file_from_atom_array,
)
from tmol.io._pose_stack_from_biotite import (
    atom_array_from_pose_stack,
    pose_stack_from_atom_array,
)
from tmol.pose._pose_stack import PoseStack


def __getattr__(name):
    return getattr(_shared, name)


def pose_stack_from_file(
    source: StructureSource,
    device: torch.device,
    format: Optional[StructureFormat] = None,
    *,
    model: int = 1,
    assembly_id: Optional[str] = None,
    residue_start: Optional[int] = None,
    residue_end: Optional[int] = None,
    **kwargs,
):
    """Build a PoseStack from a PDB, CIF, MOL2 or SDF structure.

    Reads with atom_array_from_file and builds with pose_stack_from_atom_array,
    which takes the remaining keyword arguments. residue_start and residue_end
    select a half-open range of residues.
    """
    array = atom_array_from_file(source, format, model=model, assembly_id=assembly_id)
    if residue_start is not None or residue_end is not None:
        boundaries = struc.get_residue_starts(array, add_exclusive_stop=True)
        start, stop, _ = slice(residue_start, residue_end).indices(len(boundaries) - 1)
        array = array[boundaries[start] : boundaries[stop]]
    return pose_stack_from_atom_array(array, device, **kwargs)


def file_from_pose_stack(
    pose_stack: PoseStack,
    path: Optional[Union[str, Path]] = None,
    format: Optional[Literal["pdb", "cif"]] = None,
    **kwargs,
) -> Union[str, Path]:
    """Write a PoseStack as PDB or CIF; return the text when path is None.

    Converts with atom_array_from_pose_stack, which takes the remaining keyword
    arguments, and writes with file_from_atom_array.
    """
    return file_from_atom_array(
        atom_array_from_pose_stack(pose_stack, **kwargs), path, format
    )
