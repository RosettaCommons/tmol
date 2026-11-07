import torch

import attr

from tmol.types._torch import Tensor
from tmol.score._energy_term import EnergyTerm
from .._annotation_cache import AnnotationKey, _cached_annotation, _store_annotation
from ._params import BackboneTorsionParamResolver
from tmol_core.scoring.backbone_torsion import (  # noqa: F401
    BackboneTorsionBlockTypeParams,
    invert_flags,
)
from tmol.database import ParameterDatabase

from tmol.chemical import RefinedResidueType
from tmol.pose._packed_block_types import PackedBlockTypes
from tmol.pose._pose_stack import PoseStack


@attr.s(frozen=True, slots=True, auto_attribs=True)
class BackboneTorsionPackedBlockTypesParams:
    bt_rama_table: Tensor[torch.int32][:, 2]  # non-prepro/prepro
    bt_omega_table: Tensor[torch.int32][:, 2]  # non-prepro/prepro
    bt_lower_conn_ind: Tensor[torch.int32][:]
    bt_upper_conn_ind: Tensor[torch.int32][:]
    bt_is_pro: Tensor[torch.int32][:]
    bt_backbone_torsion_atoms: Tensor[torch.int32][:, 12, 3]


def _invert_flag(found, block_type_name):
    """Whether each matched lookup row asks for mirrored torsions.

    A mirror negates both backbone torsions together, so a row inverting only
    one describes a potential this term cannot express.
    """
    return invert_flags(
        found["invert_phi"].values, found["invert_psi"].values, block_type_name
    )


class BackboneTorsionEnergyTerm(EnergyTerm):
    device: torch.device
    param_resolver: BackboneTorsionParamResolver
    rama_tables: Tensor[torch.float32][:, :, :]
    rama_table_params: Tensor[torch.float32][:, :, :]
    omega_tables: Tensor[torch.float32][:, :, :, :]
    omega_table_params: Tensor[torch.float32][:, :, :]

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        super(BackboneTorsionEnergyTerm, self).__init__(
            param_db=param_db, device=device
        )
        self.device = device

        self.param_resolver = BackboneTorsionParamResolver.from_database(
            param_db.scoring.rama, param_db.scoring.omega_bbdep, device
        )
        self._block_annotation_key = AnnotationKey.from_sources(
            param_db.scoring.rama, param_db.scoring.omega_bbdep
        )
        self._packed_annotation_key = AnnotationKey.from_sources(
            param_db.scoring.rama,
            param_db.scoring.omega_bbdep,
            settings=(self.device,),
        )

        self.rama_tables = self.param_resolver.rama_params.tables
        self.rama_table_params = torch.cat(
            (
                self.param_resolver.rama_params.bbstarts.to(torch.float32),
                self.param_resolver.rama_params.bbsteps.to(torch.float32),
            ),
            dim=1,
        )
        self.omega_tables = self.param_resolver.omega_params.tables
        self.omega_table_params = torch.cat(
            (
                self.param_resolver.omega_params.bbstarts.to(torch.float32),
                self.param_resolver.omega_params.bbsteps.to(torch.float32),
            ),
            dim=1,
        )

    @classmethod
    def class_name(cls):
        return "BackboneTorsion"

    @classmethod
    def score_types(cls):
        import tmol.score.terms._backbone_torsion_creator

        return (
            tmol.score.terms._backbone_torsion_creator.BackboneTorsionTermCreator.score_types()
        )

    def n_bodies(self):
        return 2

    def setup_block_type(self, block_type: RefinedResidueType):
        super(BackboneTorsionEnergyTerm, self).setup_block_type(block_type)
        cached = _cached_annotation(
            block_type, "_backbone_torsion_annotation", self._block_annotation_key
        )
        if cached is not None:
            return cached

        bt_bbtors_params = self.param_resolver.host_preparation.prepare_block(
            block_type
        )
        setattr(block_type, "backbone_torsion_params", bt_bbtors_params)
        return _store_annotation(
            block_type,
            "_backbone_torsion_annotation",
            self._block_annotation_key,
            bt_bbtors_params,
            fields=("backbone_torsion_params",),
        )

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(BackboneTorsionEnergyTerm, self).setup_packed_block_types(
            packed_block_types
        )
        cached = _cached_annotation(
            packed_block_types,
            "_backbone_torsion_annotation",
            self._packed_annotation_key,
        )
        if cached is not None:
            return cached
        block_params = [
            self.setup_block_type(bt) for bt in packed_block_types.active_block_types
        ]
        host = self.param_resolver.host_preparation.prepare_packed(block_params)
        backbone_torsion_params = BackboneTorsionPackedBlockTypesParams(
            **{
                name: torch.tensor(value, device=packed_block_types.device)
                for name, value in host.items()
            }
        )
        setattr(packed_block_types, "backbone_torsion_params", backbone_torsion_params)
        return _store_annotation(
            packed_block_types,
            "_backbone_torsion_annotation",
            self._packed_annotation_key,
            backbone_torsion_params,
            fields=("backbone_torsion_params",),
            bindings=tuple(
                (block_type, "backbone_torsion_params")
                for block_type in packed_block_types.active_block_types
            ),
        )

    def setup_poses(self, pose_stack: PoseStack):
        super(BackboneTorsionEnergyTerm, self).setup_poses(pose_stack)

    def get_pose_score_term_function(self):
        from tmol.score.backbone_torsion.potentials._compiled import (
            _backbone_torsion_pose_score,
        )

        return _backbone_torsion_pose_score

    def get_packing_rotamer_score_term_function(self):
        from tmol.score.backbone_torsion.potentials._compiled import (
            _iter_packing_rotamer_scores,
        )

        return _iter_packing_rotamer_scores

    def get_rotamer_score_term_function(self):
        from tmol.score.backbone_torsion.potentials._compiled import (
            _backbone_torsion_rotamer_score,
        )

        return _backbone_torsion_rotamer_score

    def get_score_term_attributes(self, pose_stack):
        pbt = pose_stack.packed_block_types
        return [
            pose_stack.block_type_ind,
            pose_stack.inter_residue_connections,
            pbt.atom_downstream_of_conn,
            pbt.backbone_torsion_params.bt_rama_table,
            pbt.backbone_torsion_params.bt_omega_table,
            pbt.backbone_torsion_params.bt_lower_conn_ind,
            pbt.backbone_torsion_params.bt_upper_conn_ind,
            pbt.backbone_torsion_params.bt_is_pro,
            pbt.backbone_torsion_params.bt_backbone_torsion_atoms,
            self.rama_tables,
            self.rama_table_params,
            self.omega_tables,
            self.omega_table_params,
        ]
