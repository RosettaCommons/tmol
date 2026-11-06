from typing import Optional

import copy
import attr
import numpy

DEFAULT_ATOM_OCCUPANCY = 1.0
# fpd: B factors of 0 are badly behaved in crystal refinement/MR/etc., 30 is a very reasonable "average"
DEFAULT_ATOM_B_FACTOR = 30.0


@attr.s(auto_attribs=True)
class PDBInfo:
    """Holds other information about a structure as it's read in from a file.

    The data held in this class has no impact on structure calculations, e.g.
    the energy of a conformation, but it is useful for preserving information
    about input structures for later output. If the information starts to
    diverge from the actual conformation held in a PoseStack, e.g. if residues
    are added or deleted, then it is the responsibility of the code that makes
    those changes to also update the PDBInfo object accordingly.

    Datamembers:

    residue_labels: numpy array of strings giving residue ids for each residue.
        shape: [n_poses x max_n_residues]
    residue_insertion_codes: numpy array of strings giving insertion codes
        for each residue.
        shape: [n_poses x max_n_residues]
    chain_labels: numpy array of strings giving chain labels for each residue.
        shape: [n_poses x max_n_residues]
    atom_occupancy: numpy array of floats giving occupancy for each atom.
        shape: [n_poses x max_n_atoms_per_pose]
    atom_b_factor: numpy array of floats giving B-factors for each atom.
        shape: [n_poses x max_n_atoms_per_pose]
    """

    residue_labels: numpy.ndarray
    residue_insertion_codes: numpy.ndarray
    chain_labels: numpy.ndarray
    atom_occupancy: numpy.ndarray
    atom_b_factor: numpy.ndarray
    # per residue: where a metal split out of a component sat, else None
    metal_origins: Optional[numpy.ndarray] = None
    # per residue: the input's RESIDUE_ANNOTATIONS as one structured array
    residue_annotations: Optional[numpy.ndarray] = None

    def split(self, index) -> "PDBInfo":
        """Split out a single pose's worth of PDBInfo from a batch."""
        return self.sliced(index, index + 1)

    def sliced(self, start, stop) -> "PDBInfo":
        """A copy of poses ``start:stop``; absent optional fields stay None."""
        return type(self)(
            **{
                name: None if value is None else value[start:stop].copy()
                for name, value in attr.asdict(self, recurse=False).items()
            }
        )

    def take(self, indices) -> "PDBInfo":
        """Copy selected poses, including optional structured annotations."""
        indices = numpy.asarray(indices, dtype=numpy.int64)
        if indices.ndim != 1:
            raise IndexError("pose selection must be one dimensional")
        return type(self)(
            **{
                name: (
                    None
                    if value is None
                    else (
                        copy.deepcopy(value[indices])
                        if isinstance(value, numpy.ndarray)
                        else [copy.deepcopy(value[i]) for i in indices]
                    )
                )
                for name, value in attr.asdict(self, recurse=False).items()
            }
        )
