"""Per-atom / per-residue PDB metadata for a PoseStack.

This is the JAX-port counterpart of ``tmol.pose.pdb_info.PDBInfo``. The
upstream PyTorch implementation stored everything in numpy arrays of
``object`` dtype, alongside integer residue labels and float occupancy /
B-factor arrays. In the JAX port we trim this down to the information
actually consumed by the PDB writer and keep it as plain Python lists of
strings (for the per-atom / per-chain / per-residue labels) and a single
numpy integer array for residue numbers.

Crucially this class is NOT a pytree. The data it holds has no impact on
any structural computation (energies, gradients, etc.) — it only exists to
preserve author-supplied labels across a round-trip to/from a PDB file. It
therefore lives outside the jitted/traced world as plain Python metadata
attached to a ``PoseStack`` object.

Layout
------
- ``chain_id``:           ``list[list[str]]``  length ``n_poses``; each
  inner list has length ``n_blocks_this_pose`` and holds the chain
  label (e.g. ``"A"``, ``"B"``) for each residue.
- ``res_num``:            ``numpy.ndarray`` of shape
  ``(n_poses, max_n_blocks)`` and dtype ``int32`` — the integer residue
  number for each residue (``-1`` for padding slots beyond a pose's
  real residue count).
- ``res_name``:           ``list[list[str]]``  length ``n_poses``; each
  inner list has length ``n_blocks_this_pose`` and holds the 3-letter
  residue name (e.g. ``"ALA"``, ``"HIS"``) for each residue.
- ``atom_record_type``:   ``list[list[list[str]]]``  length ``n_poses``;
  each pose holds a list per block, each block holds a list per atom,
  and each entry is one of ``"ATOM"`` / ``"HETATM"``.

Shape invariants
----------------
- ``len(chain_id) == len(res_name) == len(atom_record_type) == n_poses``.
- ``res_num.shape == (n_poses, max_n_blocks)``.
- For each pose ``p``:
    - ``len(chain_id[p]) == len(res_name[p]) == len(atom_record_type[p])``
      which equals the number of real residues in pose ``p``.
    - ``len(atom_record_type[p][b]) == n_atoms(block_type[p, b])``.

The constructor is permissive: it does not enforce those invariants so
that callers can build up a ``PDBInfo`` incrementally.
"""

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
