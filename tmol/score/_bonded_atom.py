import attr


import torch
from tmol_core.chemical.bonds import index_bonds, directed_bonds

from tmol.types._torch import Tensor


@attr.s(auto_attribs=True, frozen=True, slots=True)
class IndexedBonds:
    """Sorted atom bonds with per-atom spans for efficient neighborhood lookup."""

    # bonds = [ nstacks x nbonds x 2 ]
    #   with 2 = (atom1ind atom2ind)
    # bond_spans = [ nstacks x max-natoms x 2 ]
    #   with 2 = (bond start ind, bond end ind + 1)
    bonds: Tensor[int][:, :, 2]
    bond_spans: Tensor[int][:, :, 2]

    @classmethod
    def from_bonds(cls, src_bonds, minlength=None):
        return cls(
            **{
                name: torch.from_numpy(value)
                for name, value in index_bonds(src_bonds, minlength).items()
            }
        )

    @classmethod
    def to_directed(cls, src_bonds):
        """Convert a potentially-undirected bond-table into dense, directed bonds.
        The input "bonds" tensor is a two dimensional array of nbonds x 3,
        where the 2nd dimension holds [stack index, atom 1 index, atom 2 index].

        Eg. Converts
        [[0, 0, 1], [0, 0, 2]]
        into
        [[0, 0, 1], [0, 1, 0], [0, 0, 2], [0, 2, 0]]
        """

        return directed_bonds(src_bonds)
