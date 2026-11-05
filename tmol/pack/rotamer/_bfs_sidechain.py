from typing import List

from tmol.types._functional import _validate_args
from tmol.chemical import RefinedResidueType
from tmol_core.pack.sidechain import (  # noqa: F401
    sidechain_atoms_from_parents as _bfs_sidechain_atoms_jit,  # noqa: F401
    sidechain_region,
)


@_validate_args
def _bfs_sidechain_atoms(restype: RefinedResidueType, sc_roots: List[int]):
    return sidechain_region(restype, sc_roots)[1]
