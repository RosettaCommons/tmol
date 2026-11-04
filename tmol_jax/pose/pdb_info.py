"""Host-only PDB labels, indexed along the same pose axis as PoseStack.

chain_id and res_name contain one list per pose, one label per real residue.
res_num is int32[poses, max_blocks], with -1 padding. atom_record_type contains
one ATOM/HETATM label per atom, nested by pose and residue. These labels are
metadata, not differentiable pytree leaves."""

from __future__ import annotations

from typing import List

import attrs
import numpy as np

DEFAULT_ATOM_OCCUPANCY = 1.0
DEFAULT_ATOM_B_FACTOR = 0.0


@attrs.define
class PDBInfo:
    """Plain-Python PDB metadata for a ``PoseStack``.

    See the module docstring for the exact layout; this is NOT a pytree.
    """

    chain_id: List[List[str]]
    res_num: np.ndarray  # shape (n_poses, max_n_blocks), dtype int32
    res_name: List[List[str]]
    atom_record_type: List[List[List[str]]]

    # ---------------------------------------------------------------
    # Shape helpers
    # ---------------------------------------------------------------
    @property
    def n_poses(self) -> int:
        return int(self.res_num.shape[0]) if self.res_num.ndim == 2 else 0

    @property
    def max_n_blocks(self) -> int:
        return int(self.res_num.shape[1]) if self.res_num.ndim == 2 else 0

    # ---------------------------------------------------------------
    # Slicing
    # ---------------------------------------------------------------
    def __getitem__(self, index) -> "PDBInfo":
        """Select poses with the same integer, slice or array indexing as a pose."""
        indices = np.atleast_1d(np.arange(self.n_poses)[index])
        if indices.ndim != 1:
            raise IndexError("pose selection must be one dimensional")
        return PDBInfo(
            chain_id=[list(self.chain_id[i]) for i in indices],
            res_num=self.res_num[indices].copy(),
            res_name=[list(self.res_name[i]) for i in indices],
            atom_record_type=[
                [list(atoms) for atoms in self.atom_record_type[i]] for i in indices
            ],
        )

    # ---------------------------------------------------------------
    # Factories
    # ---------------------------------------------------------------
    @classmethod
    def empty(cls, n_poses: int, max_n_blocks: int) -> "PDBInfo":
        """An all-zero/empty ``PDBInfo`` with no per-atom records yet.

        ``res_num`` is filled with ``-1`` (the "padding" sentinel). The
        nested string lists are created as empty inner lists ready to be
        filled in by higher-level construction code.
        """
        return cls(
            chain_id=[[] for _ in range(n_poses)],
            res_num=np.full((n_poses, max_n_blocks), -1, dtype=np.int32),
            res_name=[[] for _ in range(n_poses)],
            atom_record_type=[[] for _ in range(n_poses)],
        )


__all__ = [
    "PDBInfo",
    "DEFAULT_ATOM_OCCUPANCY",
    "DEFAULT_ATOM_B_FACTOR",
]
