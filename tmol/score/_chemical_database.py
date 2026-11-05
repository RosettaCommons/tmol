import functools

import attr

import pandas

import torch
import numpy

from tmol.database.chemical import ChemicalDatabase
from tmol.utility import resolve_device

from tmol.types._torch import Tensor
from tmol.types._tensor import TensorGroup
from tmol.types._array import NDArray
from tmol.types._attrs import ValidateAttrs

from tmol_core.scoring.atom_types import (
    AcceptorHybridization,
    HYBRIDIZATION_NAMES,
    atom_type_parameters,
    block_type_indices,
)

# Retain the historical pandas lookup on the public enum.
AcceptorHybridization._index = pandas.Index(HYBRIDIZATION_NAMES)


@attr.s(auto_attribs=True, frozen=True, slots=True)
class AtomTypeParams(TensorGroup, ValidateAttrs):
    """Packed donor, acceptor, hydrogen, and hybridization atom-type flags."""

    is_acceptor: Tensor[bool][...]
    acceptor_hybridization: Tensor[torch.int32][...]

    is_donor: Tensor[bool][...]

    is_hydrogen: Tensor[bool][...]
    is_hydroxyl: Tensor[bool][...]
    is_polarh: Tensor[bool][...]


@attr.s(frozen=True, slots=True, auto_attribs=True)
class AtomTypeParamResolver(ValidateAttrs):
    """Container for global/type/pair parameters, indexed by atom type name.

    Param resolver stores pair parameters for a collection of atom types, using
    a pandas Index to map from string atom type to a resolver-specific integer type
    index.
    """

    # str->int index from atom type name to index within parameter tensors
    index: pandas.Index

    # shape [n] per-type parameters, source params used to calculate pair parameters
    params: AtomTypeParams

    device: torch.device

    def block_type_indices(self, block_type):
        """Resolve real atoms, keeping the low-level padding sentinel separate."""
        return block_type_indices(self.index, block_type)

    def type_idx(self, atom_types: NDArray[object][...]) -> Tensor[torch.int64][...]:
        """Convert array of atom type names to parameter indices.

        pandas.Index.get_indexer only operates on 1-d input arrays. Coerces
        higher-dimensional arrays, as may be produced via broadcasting, into
        lower-dimensional views to resolver parameter indices.
        """
        if not isinstance(atom_types, numpy.ndarray):
            atom_types = numpy.array(atom_types, dtype=object)

        return torch.from_numpy(
            self.index.get_indexer(atom_types.ravel()).reshape(atom_types.shape)
        ).to(self.device)

    @classmethod
    def from_database(cls, chemical_database: ChemicalDatabase, device: torch.device):
        """Initialize param resolver for all atom types in database."""
        return cls._from_atom_types(
            chemical_database.atom_types, resolve_device(device)
        )

    @classmethod
    @functools.lru_cache(maxsize=32)
    def _from_atom_types(cls, atom_types, device):
        """Initialize and cache parameters for an immutable atom-type tuple."""

        names, values = atom_type_parameters(atom_types)
        return cls(
            index=pandas.Index(names),
            params=AtomTypeParams(
                **{
                    name: torch.from_numpy(value).to(device)
                    for name, value in values.items()
                }
            ),
            device=device,
        )
