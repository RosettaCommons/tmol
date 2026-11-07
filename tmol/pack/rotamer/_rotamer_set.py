import attr
import torch
from tmol.types._attrs import ValidateAttrs
from tmol.types._torch import Tensor


from tmol_core.pack.correlation import (  # noqa: F401
    CorrelatedBlockGroup,
    correlation_indices,
)


def _correlation_indices(n_rots_for_block, groups):
    counts = n_rots_for_block.cpu().numpy() if groups else None
    return torch.tensor(
        correlation_indices(n_rots_for_block.shape, groups, counts),
        device=n_rots_for_block.device,
    )


@attr.s(auto_attribs=True, slots=True, frozen=True)
class RotamerSet(ValidateAttrs):
    """Packed coordinates and pose/block indexing for generated rotamers."""

    n_rots_for_pose: Tensor[torch.int64][:]
    rot_offset_for_pose: Tensor[torch.int64][:]
    n_rots_for_block: Tensor[torch.int64][:, :]
    rot_offset_for_block: Tensor[torch.int64][:, :]
    pose_for_rot: Tensor[torch.int64][:]
    block_type_ind_for_rot: Tensor[torch.int64][:]
    block_ind_for_rot: Tensor[torch.int32][:]
    coord_offset_for_rot: Tensor[torch.int32][:]
    coords: Tensor[torch.float32][:, 3]

    # Chemical connectivity alone does not imply correlated sampling: OptH
    # samples independent protons on connected blocks. Producers must declare
    # correspondence explicitly, and build_rotamers validates merged ownership.
    correlated_groups: tuple[CorrelatedBlockGroup, ...] = ()
    group_for_block: Tensor[torch.int32][:, :] = attr.ib()

    @group_for_block.default
    def _group_for_block(self):
        return _correlation_indices(self.n_rots_for_block, self.correlated_groups)

    first_rot_block_type: Tensor[torch.int64][:, :] = attr.ib()

    @first_rot_block_type.default
    def _block_type_for_first_rot_for_block(self):
        block_type_for_first_rot_for_block = torch.full_like(
            self.rot_offset_for_block, -1
        )
        does_block_type_have_rots = self.n_rots_for_block != 0
        block_type_for_first_rot_for_block[does_block_type_have_rots] = (
            self.block_type_ind_for_rot[
                self.rot_offset_for_block[does_block_type_have_rots]
            ]
        )
        return block_type_for_first_rot_for_block

    max_n_rots_per_pose: int = attr.ib()

    @max_n_rots_per_pose.default
    def _max_n_rots_per_pose(self):
        return int(torch.max(self.n_rots_for_pose).cpu().item())

    pose_ind_for_atom: Tensor[torch.int64][:] = attr.ib()

    @pose_ind_for_atom.default
    def _pose_ind_for_atom(self):
        n_atoms = self.coords.shape[0]
        pifa = torch.zeros((n_atoms,), dtype=torch.int64, device=self.coords.device)
        # mark the first atom for the first rotamer in each pose after pose 0
        pifa[self.coord_offset_for_rot[self.rot_offset_for_pose[1:]]] = 1
        return torch.cumsum(pifa, dim=0)

    @property
    def n_rotamers_total(self):
        return self.block_ind_for_rot.shape[0]

    def to_atom_array(self, pbt):
        from tmol_core.pack.rotamer_set import atom_array_from_rotamers

        return atom_array_from_rotamers(
            pbt.active_block_types,
            self.coords.cpu().numpy(),
            self.block_type_ind_for_rot.cpu().numpy(),
            self.block_ind_for_rot.cpu().numpy(),
            self.coord_offset_for_rot.cpu().numpy(),
        )
