from typing import Sequence

import attr
import torch

from tmol.chemical._constants import MAX_SIG_BOND_SEPARATION
from tmol.types._torch import Tensor


@attr.s(auto_attribs=True, frozen=True)
class InterBlockBondsep:
    """Connection bond separations below the cap of nearby block pairs; each row
    lists its ``block2`` slots ascending, then empty slots (block -1, the cap)."""

    near_blocks: Tensor[torch.int32][:, :, :, 2]  # pose, block1, slot, (block2, min)
    bondsep: Tensor[torch.int8][:, :, :, :, :]  # pose, block1, slot, conn1, conn2

    @property
    def n_poses(self) -> int:
        return self.near_blocks.shape[0]

    @property
    def max_n_blocks(self) -> int:
        return self.near_blocks.shape[1]

    @property
    def n_slots(self) -> int:
        return self.near_blocks.shape[2]

    @property
    def max_n_conn(self) -> int:
        return self.bondsep.shape[3]

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        """Shape of the equivalent dense table."""
        n, b, m = self.n_poses, self.max_n_blocks, self.max_n_conn
        return (n, b, b, m, m)

    @property
    def device(self) -> torch.device:
        return self.bondsep.device

    @classmethod
    def empty(
        cls,
        n_poses: int,
        max_n_blocks: int,
        max_n_conn: int,
        device: torch.device,
        n_slots: int,
    ) -> "InterBlockBondsep":
        """Separations for blocks that are all at least the cap apart."""
        near_blocks = torch.full(
            (n_poses, max_n_blocks, n_slots, 2),
            MAX_SIG_BOND_SEPARATION,
            dtype=torch.int32,
            device=device,
        )
        near_blocks[..., 0] = -1
        bondsep = torch.full(
            (n_poses, max_n_blocks, n_slots, max_n_conn, max_n_conn),
            MAX_SIG_BOND_SEPARATION,
            dtype=torch.int8,
            device=device,
        )
        return cls(near_blocks=near_blocks, bondsep=bondsep)

    @classmethod
    def from_bonded_graph(
        cls,
        counts: Tensor[torch.int32][:, :],
        intra_separation: Tensor[torch.int32][:, :, :, :],
        connections: Tensor[torch.int64][:, :, :, 2],
    ) -> "InterBlockBondsep":
        """Shortest paths below the cap between ``[pose, block, conn]`` connections, one
        bond per ``connections`` partner (-1: none) plus ``intra_separation`` in a block.
        """
        from tmol_core.pose.bond_separation import bondseparation_from_graph

        prepared = bondseparation_from_graph(
            counts.numpy(force=True),
            intra_separation.numpy(force=True),
            connections.numpy(force=True),
        )

        def tensor(array):
            value = torch.from_numpy(array)
            if not array.size:
                return torch.empty(
                    array.shape, dtype=value.dtype, device=connections.device
                )
            return value.to(connections.device)

        return cls(
            near_blocks=tensor(prepared.near_blocks), bondsep=tensor(prepared.bondsep)
        )

    @classmethod
    def concatenate(
        cls,
        parts: Sequence["InterBlockBondsep"],
        max_n_blocks: int,
        max_n_conn: int,
    ) -> "InterBlockBondsep":
        """Stack the poses of several tables, padding blocks and connections."""
        result = cls.empty(
            sum(part.n_poses for part in parts),
            max_n_blocks,
            max_n_conn,
            parts[0].device,
            max(part.n_slots for part in parts),
        )
        offset = 0
        for part in parts:
            n, b, k, m = part.n_poses, part.max_n_blocks, part.n_slots, part.max_n_conn
            result.near_blocks[offset : offset + n, :b, :k] = part.near_blocks
            result.bondsep[offset : offset + n, :b, :k, :m, :m] = part.bondsep
            offset += n
        return result

    def to_dense(self) -> Tensor[torch.int8][:, :, :, :, :]:
        """Materialize the ``[pose, block1, block2, conn1, conn2]`` table."""
        dense = torch.full(
            self.shape, MAX_SIG_BOND_SEPARATION, dtype=torch.int8, device=self.device
        )
        pose, block1, slot = torch.nonzero(self.near_blocks[..., 0] >= 0, as_tuple=True)
        block2 = self.near_blocks[pose, block1, slot, 0].to(torch.int64)
        dense[pose, block1, block2] = self.bondsep[pose, block1, slot]
        return dense

    def select_poses(self, start: int, stop: int) -> "InterBlockBondsep":
        """View of poses ``start`` to ``stop``."""
        return InterBlockBondsep(
            near_blocks=self.near_blocks[start:stop],
            bondsep=self.bondsep[start:stop],
        )

    def clone(self) -> "InterBlockBondsep":
        return InterBlockBondsep(
            near_blocks=self.near_blocks.clone(), bondsep=self.bondsep.clone()
        )

    def to(self, device: torch.device) -> "InterBlockBondsep":
        return InterBlockBondsep(
            near_blocks=self.near_blocks.to(device), bondsep=self.bondsep.to(device)
        )
