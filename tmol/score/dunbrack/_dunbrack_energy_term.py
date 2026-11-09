import attr

import torch

from .._energy_term import EnergyTerm
from .._annotation_cache import AnnotationKey, _cached_annotation, _store_annotation

from tmol.database import ParameterDatabase
from tmol.score.dunbrack._params import (
    DunbrackParamResolver,
    ScoringDunbrackDatabaseView,
)
from tmol.chemical import RefinedResidueType
from tmol.pose._packed_block_types import PackedBlockTypes
from tmol.pose._pose_stack import PoseStack
from tmol_core.scoring.dunbrack_topology import (  # noqa: F401
    DunbrackBlockAttrs,  # noqa: F401 -- compatibility import
    _empty_dunbrack_attrs,  # noqa: F401 -- compatibility import
    prepare_dunbrack_block,
    dunbrack_torsion,
    pack_dunbrack_blocks,
    pack_dunbrack_field,
)


def _device_block_data(packed, device):
    if packed.size == 0:
        return torch.empty(packed.shape, dtype=torch.int32, device=device)
    return torch.from_numpy(packed).to(device)


class DunbrackEnergyTerm(EnergyTerm):
    device: torch.device  # = attr.ib()

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        super(DunbrackEnergyTerm, self).__init__(param_db=param_db, device=device)

        self.global_params = DunbrackParamResolver.from_database(
            param_db.scoring.dun, device
        )
        self.dunbrack_db = [
            getattr(self.global_params.scoring_db, field.name)
            for field in attr.fields(ScoringDunbrackDatabaseView)
        ]
        self.device = device
        self._annotation_key = AnnotationKey.from_sources(self.global_params)
        # Small lookup metadata are host-side; do not read device scalars per RT.
        self._table_indices = []
        for name in (
            "all_table_indices",
            "rotameric_table_indices",
            "semirotameric_table_indices",
        ):
            table = getattr(self.global_params, name)
            if not table.index.is_unique:
                raise ValueError("Dunbrack residue lookup names must be unique")
            self._table_indices.append(dict(table["dun_table_name"].items()))
        self._aux = {
            field.name: getattr(self.global_params.scoring_db_aux, field.name)
            .cpu()
            .tolist()
            for field in attr.fields(type(self.global_params.scoring_db_aux))
        }

    @classmethod
    def class_name(cls):
        return "Dunbrack"

    @classmethod
    def score_types(cls):
        import tmol.score.terms._dunbrack_creator

        return tmol.score.terms._dunbrack_creator.DunbrackTermCreator.score_types()

    def n_bodies(self):
        return 1

    def setup_block_type(self, block_type: RefinedResidueType):
        super(DunbrackEnergyTerm, self).setup_block_type(block_type)

        cached = _cached_annotation(
            block_type, "_dunbrack_annotation", self._annotation_key
        )
        if cached is None:
            cached = _store_annotation(
                block_type,
                "_dunbrack_annotation",
                self._annotation_key,
                self._create_block_attrs(block_type),
            )
        block_type.dunbrack_attrs = cached
        return cached

    def _create_block_attrs(self, block_type):
        return prepare_dunbrack_block(block_type, self._table_indices, self._aux)

    def get_torsion(self, name, block_type):
        return dunbrack_torsion(name, block_type)

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(DunbrackEnergyTerm, self).setup_packed_block_types(packed_block_types)

        cached = _cached_annotation(
            packed_block_types, "_dunbrack_packed_annotation", self._annotation_key
        )
        if cached is None:
            for block_type in packed_block_types.active_block_types:
                self.setup_block_type(block_type)
            prepared = pack_dunbrack_blocks(
                [bt.dunbrack_attrs for bt in packed_block_types.active_block_types]
            )
            cached = [
                _device_block_data(value, self.device) for value in prepared.values()
            ]
            _store_annotation(
                packed_block_types,
                "_dunbrack_packed_annotation",
                self._annotation_key,
                cached,
                bindings=tuple(
                    (block_type, "dunbrack_attrs")
                    for block_type in packed_block_types.active_block_types
                ),
            )
        packed_block_types.dunbrack_packed_block_data = cached
        return cached

    def pack_data_keyed_on_block_type(
        self, active_block_types, field_getter, device, default_fill=-1
    ):
        packed = pack_dunbrack_field(
            [field_getter(bt) for bt in active_block_types], default_fill
        )
        return _device_block_data(packed, device)

    def setup_poses(self, poses: PoseStack):
        super(DunbrackEnergyTerm, self).setup_poses(poses)

    def get_pose_score_term_function(self):
        from tmol.score.dunbrack.potentials._compiled import _dunbrack_pose_scores

        return _dunbrack_pose_scores

    score_only_in_no_grad = True

    def get_rotamer_score_term_function(self):
        from tmol.score.dunbrack.potentials._compiled import _dunbrack_rotamer_scores

        return _dunbrack_rotamer_scores

    def get_score_term_attributes(self, pose_stack):
        pbt = pose_stack.packed_block_types
        packed_data = self.setup_packed_block_types(pbt)

        return [
            pose_stack.inter_residue_connections,
            pbt.atom_downstream_of_conn,
            *self.dunbrack_db,
            *packed_data,
        ]
