import numpy
import torch

from ._annotation_cache import AnnotationKey, _cached_annotation, _store_annotation

from tmol.database import ParameterDatabase
from tmol_core.chemical.bonds import pack_bond_topology
from tmol.chemical import RefinedResidueType
from tmol.pose._packed_block_types import PackedBlockTypes
from tmol.score._energy_term import EnergyTerm
from tmol.score._bonded_atom import IndexedBonds


# @attr.s(auto_attribs=True)
class BondDependentTerm(EnergyTerm):
    device: torch.device

    def __init__(self, param_db: ParameterDatabase, device: torch.device, **kwargs):
        super(BondDependentTerm, self).__init__(param_db=param_db, device=device)
        self.device = device
        self._bond_block_key = AnnotationKey.from_sources()
        self._bond_packed_key = AnnotationKey.from_sources(settings=(self.device,))

    def setup_block_type(self, block_type: RefinedResidueType):
        super(BondDependentTerm, self).setup_block_type(block_type)
        cached = _cached_annotation(
            block_type, "_bond_dependent_annotation", self._bond_block_key
        )
        if cached is not None:
            return cached

        bonds = numpy.zeros((block_type.bond_indices.shape[0], 3), dtype=numpy.int32)
        bonds[:, 1:] = block_type.bond_indices.astype(numpy.int32)
        ib = IndexedBonds.from_bonds(bonds, minlength=block_type.n_atoms)
        setattr(block_type, "intrares_indexed_bonds", ib)
        return _store_annotation(
            block_type,
            "_bond_dependent_annotation",
            self._bond_block_key,
            ib,
            fields=("intrares_indexed_bonds",),
        )

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(BondDependentTerm, self).setup_packed_block_types(packed_block_types)

        cached = _cached_annotation(
            packed_block_types,
            "_bond_dependent_annotation",
            self._bond_packed_key,
        )
        if cached is not None:
            return cached

        arrays = pack_bond_topology(
            packed_block_types.active_block_types, packed_block_types.max_n_atoms
        )
        for name, array in arrays.items():
            # Empty NumPy arrays can have zero strides; keep native views valid.
            value = (
                torch.as_tensor(array, device=self.device)
                if array.size
                else torch.empty(array.shape, dtype=torch.int32, device=self.device)
            )
            setattr(packed_block_types, name, value)
        return _store_annotation(
            packed_block_types,
            "_bond_dependent_annotation",
            self._bond_packed_key,
            (
                packed_block_types.bond_separation,
                packed_block_types.n_all_bonds,
                packed_block_types.all_bonds,
                packed_block_types.atom_all_bond_ranges,
            ),
            fields=(
                "bond_separation",
                "n_all_bonds",
                "all_bonds",
                "atom_all_bond_ranges",
            ),
        )
