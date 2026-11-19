"""Dunbrack rotamers of the flexible receptor sidechains, expressed as genes.

Rosetta's GA never mutates a sidechain chi: the packer re-chooses whole
rotamers from a backbone-dependent library on every candidate of every
generation (``GridScorer::packer_loop``). This module supplies that library to
the docking gene, so a sidechain can be resampled or inherited as one rotamer
rather than one chi at a time.
"""

from dataclasses import dataclass

import torch

from tmol.dock.ligand_dofs import (
    LigandDockingDOFs,
    _batched_coord_dihedrals,
    _local_torsion_atoms,
    _sidechain_logical_torsions_from_coords,
    _wrap_angle,
)
from tmol.pack._packer_task import PackerPalette, PackerTask, SetPackerTask
from tmol.pack.rotamer._build_rotamers import build_rotamers
from tmol.pose._pose_stack import PoseStack


@dataclass(frozen=True)
class RotamerGeneLibrary:
    """Gene values of each flexible residue's rotamers.

    ``column_genes[c, r]`` is the gene value of sidechain column ``c`` under
    rotamer ``r`` of the residue that column belongs to. Columns are the
    trailing sidechain columns of a gene, in ``sidechain_torsions`` order.
    """

    column_genes: torch.Tensor  # [sidechain column, rotamer]
    column_group: torch.Tensor  # [sidechain column] -> residue group
    n_rotamers: torch.Tensor  # [residue group]

    @property
    def n_sidechain_columns(self) -> int:
        return int(self.column_genes.shape[0])

    @property
    def n_groups(self) -> int:
        return int(self.n_rotamers.shape[0])

    def _gather(self, rotamer_index: torch.Tensor) -> torch.Tensor:
        """Gene values of a per-candidate rotamer choice, ``[gene, column]``."""
        per_column = rotamer_index[:, self.column_group]
        return torch.gather(
            self.column_genes[None].expand(per_column.shape[0], -1, -1),
            2,
            per_column[:, :, None],
        ).squeeze(-1)

    def draw(
        self, n_genes: int, *, generator: torch.Generator, dtype: torch.dtype
    ) -> torch.Tensor:
        """Uniform rotamer gene values for every residue, ``[gene, column]``.

        Rosetta's packer draws uniformly from an already probability-filtered
        set rather than by rotamer probability (``GridScorer.cc:2777``).
        """
        device = self.column_genes.device
        uniform = torch.rand(
            (n_genes, self.n_groups), device=device, generator=generator
        )
        rotamer_index = (uniform * self.n_rotamers[None, :]).to(torch.int64)
        rotamer_index = torch.minimum(rotamer_index, self.n_rotamers[None, :] - 1)
        return self._gather(rotamer_index).to(dtype)

    def substitute(
        self,
        genes: torch.Tensor,
        group: torch.Tensor,
        rotamer_index: torch.Tensor,
    ) -> torch.Tensor:
        """Replace one residue's chis per candidate, leaving the rest alone."""
        columns = genes[:, -self.n_sidechain_columns :]
        chosen = self._gather(rotamer_index).to(genes.dtype)
        moved = self.column_group[None, :] == group[:, None]
        result = genes.clone()
        result[:, -self.n_sidechain_columns :] = torch.where(moved, chosen, columns)
        return result

    def random_moves(
        self,
        n_genes: int,
        *,
        generator: torch.Generator,
        weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One residue per candidate, and a rotamer for every residue.

        ``weights`` biases which residue each candidate proposes; without it
        every flexible residue is equally likely.
        """
        device = self.column_genes.device
        if weights is None:
            group = torch.randint(
                self.n_groups, (n_genes,), device=device, generator=generator
            )
        else:
            group = torch.multinomial(
                weights, 1, replacement=True, generator=generator
            ).squeeze(1)
        uniform = torch.rand(
            (n_genes, self.n_groups), device=device, generator=generator
        )
        rotamer_index = torch.minimum(
            (uniform * self.n_rotamers[None, :]).to(torch.int64),
            self.n_rotamers[None, :] - 1,
        )
        return group, rotamer_index


def _rotamer_chis(
    pose_stack: PoseStack,
    docking_dofs: LigandDockingDOFs,
) -> tuple[list[torch.Tensor], list[int]]:
    """Measure every built rotamer's docking chis, per flexible residue."""

    palette = PackerPalette()
    task = PackerTask(pose_stack, palette)
    task.restrict_to_repacking()
    mask = torch.zeros(
        (pose_stack.n_poses, pose_stack.max_n_blocks),
        dtype=torch.bool,
        device=pose_stack.device,
    )
    blocks = torch.as_tensor(
        list(docking_dofs.sidechain_blocks),
        dtype=torch.int64,
        device=pose_stack.device,
    )
    mask[:, blocks] = True
    task.disable_packing_by_block_mask(~mask)

    _, rotamer_set = build_rotamers(
        pose_stack,
        SetPackerTask.from_packer_task(task),
        pose_stack.packed_block_types.chem_db,
    )

    block_of_rotamer = rotamer_set.block_ind_for_rot.to(torch.int64)
    type_of_rotamer = rotamer_set.block_type_ind_for_rot.to(torch.int64)
    chis: dict[int, torch.Tensor] = {}
    counts: dict[int, int] = {}
    for block in docking_dofs.sidechain_blocks:
        # A rotamer of another block type (an alternate tautomer) indexes its
        # atoms differently, so the docking chis are not measurable on it.
        block_type_index = int(pose_stack.block_type_ind64[0, block])
        selected = torch.nonzero(
            (block_of_rotamer == block) & (type_of_rotamer == block_type_index),
            as_tuple=False,
        ).flatten()
        offsets = rotamer_set.coord_offset_for_rot[selected].to(torch.int64)
        block_type = pose_stack.block_type(0, block)
        atoms = torch.tensor(
            [
                _local_torsion_atoms(block_type, mapping.torsion_index)
                for mapping in docking_dofs.sidechain_torsions
                if mapping.block == block
            ],
            dtype=torch.int64,
            device=pose_stack.device,
        )
        indices = offsets[:, None, None] + atoms[None, :, :]
        chis[block] = _batched_coord_dihedrals(rotamer_set.coords, indices)
        counts[block] = int(selected.numel())
    return chis, counts


def build_rotamer_gene_library(
    pose_stack: PoseStack,
    docking_dofs: LigandDockingDOFs,
    base_genes: torch.Tensor,
) -> RotamerGeneLibrary | None:
    """Build the rotamer gene table for one reference pose.

    ``base_genes`` are that pose's sidechain gene columns; a rotamer's gene is
    the base gene displaced by the chi change it makes, the same delta rule
    the inverse-kinematics path uses.
    """

    mappings = docking_dofs.sidechain_torsions
    if not mappings or pose_stack.n_poses != 1:
        return None

    chis, counts = _rotamer_chis(pose_stack, docking_dofs)
    if not counts or min(counts.values()) == 0:
        return None

    base_chis = _sidechain_logical_torsions_from_coords(pose_stack, docking_dofs)[0]
    device = pose_stack.device
    max_rotamers = max(counts.values())
    group_of_block = {block: i for i, block in enumerate(docking_dofs.sidechain_blocks)}

    column_genes = torch.empty(
        (len(mappings), max_rotamers), dtype=base_genes.dtype, device=device
    )
    column_group = torch.empty((len(mappings),), dtype=torch.int64, device=device)
    within_block: dict[int, int] = {}
    for column, mapping in enumerate(mappings):
        index = within_block.get(mapping.block, 0)
        within_block[mapping.block] = index + 1
        rotamer_chis = chis[mapping.block][:, index].to(base_genes.dtype)
        genes = _wrap_angle(
            base_genes[column] + _wrap_angle(rotamer_chis - base_chis[column])
        )
        padded = genes[0].expand(max_rotamers).clone()
        padded[: genes.shape[0]] = genes
        column_genes[column] = padded
        column_group[column] = group_of_block[mapping.block]
    return RotamerGeneLibrary(
        column_genes=column_genes,
        column_group=column_group,
        n_rotamers=torch.tensor(
            [counts[block] for block in docking_dofs.sidechain_blocks],
            dtype=torch.int64,
            device=device,
        ),
    )
