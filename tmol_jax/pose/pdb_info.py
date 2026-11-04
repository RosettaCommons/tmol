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

    # Shape helpers
    @property
    def n_poses(self) -> int:
        for values in (
            self.res_num,
            self.chain_id,
            self.res_name,
            self.atom_record_type,
        ):
            if values is not None and len(values):
                return len(values)
        return 0

    @property
    def max_n_blocks(self) -> int:
        if self.res_num is not None and self.res_num.ndim == 2:
            return int(self.res_num.shape[1])
        return 0

    # Slicing
    def __getitem__(self, index) -> "PDBInfo":
        """Select poses with the same integer, slice or array indexing as a pose."""
        indices = np.atleast_1d(np.arange(self.n_poses)[index])
        if indices.ndim != 1:
            raise IndexError("pose selection must be one dimensional")
        return self.take(indices)

    def take(self, indices) -> "PDBInfo":
        """Select normalized pose indices, preserving missing metadata fields."""

        def rows(values):
            return None if values is None else [list(values[i]) for i in indices]

        records = self.atom_record_type
        return PDBInfo(
            chain_id=rows(self.chain_id),
            res_num=None if self.res_num is None else self.res_num[indices].copy(),
            res_name=rows(self.res_name),
            atom_record_type=(
                None
                if records is None
                else [[list(atoms) for atoms in records[i]] for i in indices]
            ),
        )

    # Factories
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
