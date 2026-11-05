import attr
import numpy
import torch
import pandas

from tmol_core.scoring.hbond import (  # noqa: F401 - legacy private imports
    HBondBlockTypeParams,  # noqa: F401 - legacy import
    HBondChemicalData,
    prepare_hbond_block,
    pack_hbond_blocks,
    attached_H_for_don as _attached_H_for_don,  # noqa: F401
)

from .._annotation_cache import AnnotationKey, _cached_annotation, _store_annotation

from tmol.database import ParameterDatabase
from tmol.database.scoring import HBondDatabase

from tmol.chemical import RefinedResidueType
from tmol.pose._packed_block_types import PackedBlockTypes

from tmol.score.hbond._params import HBondParamResolver
from tmol.score._chemical_database import AtomTypeParamResolver
from tmol.score._bond_dependent_term import BondDependentTerm
from tmol.types._attrs import ValidateAttrs
from tmol.types._torch import Tensor


def _map_hbond_types(mapper, column, type_index, atom_types):
    try:
        names = mapper.reindex(atom_types)[column]
        names = names.where(pandas.notnull(names), None).to_numpy()
    except KeyError:
        names = numpy.full(len(atom_types), None, dtype=object)
    return type_index.get_indexer(names)


@attr.s(auto_attribs=True, frozen=True, slots=True)
class HBondPackedBlockTypesParams(ValidateAttrs):
    tile_n_donH: Tensor[torch.int32][:, :]
    tile_n_don_hvy: Tensor[torch.int32][:, :]
    tile_n_acc: Tensor[torch.int32][:, :]
    tile_donH_inds: Tensor[torch.int32][:, :, :]
    tile_donH_hvy_inds: Tensor[torch.int32][:, :, :]
    tile_don_hvy_inds: Tensor[torch.int32][:, :, :]
    tile_which_donH_of_donH_hvy: Tensor[torch.int32][:, :, :]
    tile_acc_inds: Tensor[torch.int32][:, :, :]
    tile_donorH_type: Tensor[torch.int32][:, :, :]
    tile_acceptor_type: Tensor[torch.int32][:, :, :]
    tile_acceptor_hybridization: Tensor[torch.int32][:, :, :]
    tile_acceptor_n_attached_H: Tensor[torch.int32][:, :, :]
    is_hydrogen: Tensor[torch.int32][:, :]


class HBondDependentTerm(BondDependentTerm):
    atom_type_resolver: AtomTypeParamResolver
    hbond_database: HBondDatabase
    hbond_resolver: HBondParamResolver
    device: torch.device
    tile_size = 32

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        super(HBondDependentTerm, self).__init__(param_db=param_db, device=device)
        self.atom_type_resolver = AtomTypeParamResolver.from_database(
            param_db.chemical, torch.device("cpu")
        )
        self.hbond_database = param_db.scoring.hbond
        self.hbond_resolver = HBondParamResolver.from_database(
            param_db.chemical, self.hbond_database, device
        )
        self._hbond_chemistry = HBondChemicalData.from_database(
            param_db.chemical, self.hbond_database
        )
        self._hbond_atom_type_index = self.atom_type_resolver.index
        self._hbond_hybridization = self._hbond_chemistry.hybridization
        self._hbond_is_hydrogen = self._hbond_chemistry.is_hydrogen
        self._hbond_acceptor_types = self._hbond_chemistry.acceptor_types
        self._hbond_donor_types = self._hbond_chemistry.donor_types
        self._hbond_block_key = AnnotationKey.from_sources(
            param_db.chemical, param_db.scoring.hbond, settings=(self.tile_size,)
        )
        self._hbond_packed_key = AnnotationKey.from_sources(
            param_db.chemical,
            param_db.scoring.hbond,
            settings=(
                self.hbond_resolver.pair_params.donor_weight.device,
                self.tile_size,
            ),
        )

    def setup_block_type(self, block_type: RefinedResidueType):
        super(HBondDependentTerm, self).setup_block_type(block_type)

        cached = _cached_annotation(
            block_type, "_hbond_annotation", self._hbond_block_key
        )
        if cached is not None:
            return cached

        bonds = block_type.intrares_indexed_bonds
        hbbt_params = prepare_hbond_block(
            block_type,
            self._hbond_chemistry,
            bonds.bonds.cpu().numpy()[0],
            bonds.bond_spans.cpu().numpy()[0],
            self.tile_size,
        )
        setattr(block_type, "hbbt_params", hbbt_params)
        return _store_annotation(
            block_type,
            "_hbond_annotation",
            self._hbond_block_key,
            hbbt_params,
            fields=("hbbt_params",),
        )

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(HBondDependentTerm, self).setup_packed_block_types(packed_block_types)

        cached = _cached_annotation(
            packed_block_types, "_hbond_annotation", self._hbond_packed_key
        )
        if cached is not None:
            return cached

        pbt = packed_block_types
        blocks = [
            HBondDependentTerm.setup_block_type(self, bt)
            for bt in pbt.active_block_types
        ]
        arrays = pack_hbond_blocks(blocks, pbt.max_n_atoms, self.tile_size)
        params = HBondPackedBlockTypesParams(
            **{
                name: torch.tensor(array, dtype=torch.int32, device=self.device)
                for name, array in arrays.items()
            }
        )

        setattr(packed_block_types, "hbpbt_params", params)
        return _store_annotation(
            packed_block_types,
            "_hbond_annotation",
            self._hbond_packed_key,
            params,
            fields=("hbpbt_params",),
            bindings=tuple(
                (block_type, "hbbt_params")
                for block_type in packed_block_types.active_block_types
            ),
        )
