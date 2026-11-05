import torch

from .._energy_term import EnergyTerm
from .._annotation_cache import AnnotationKey, _cached_annotation, _store_annotation

from tmol.database import ParameterDatabase
from tmol.score.disulfide._params import DisulfideGlobalParams
from tmol.chemical import RefinedResidueType
from tmol.pose._packed_block_types import PackedBlockTypes
from tmol.pose._pose_stack import PoseStack


from tmol_core.scoring.disulfide import (
    DisulfideParameter as _DisulfideParameter,
    MIRRORED_PARAMETER_SIGNS,
    disulfide_connections,
    pack_disulfide_connections,
    disulfide_parameter_signs,
)

_MIRRORED_PARAMETER_SIGNS = torch.from_numpy(MIRRORED_PARAMETER_SIGNS)


class DisulfideEnergyTerm(EnergyTerm):
    device: torch.device  # = attr.ib()

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        super(DisulfideEnergyTerm, self).__init__(param_db=param_db, device=device)

        self.global_params = DisulfideGlobalParams.from_database(
            param_db.scoring.disulfide, device
        )
        self.device = device
        self._block_annotation_key = AnnotationKey.from_sources()
        self._packed_annotation_key = AnnotationKey.from_sources(
            settings=(self.device,)
        )

    @classmethod
    def class_name(cls):
        return "Disulfide"

    @classmethod
    def score_types(cls):
        import tmol.score.terms._disulfide_creator

        return tmol.score.terms._disulfide_creator.DisulfideTermCreator.score_types()

    def n_bodies(self):
        return 2

    def setup_block_type(self, block_type: RefinedResidueType):
        super(DisulfideEnergyTerm, self).setup_block_type(block_type)

        cached = _cached_annotation(
            block_type, "_disulfide_annotation", self._block_annotation_key
        )
        if cached is not None:
            return cached

        connections = disulfide_connections(block_type)

        setattr(block_type, "disulfide_connections", connections)
        return _store_annotation(
            block_type,
            "_disulfide_annotation",
            self._block_annotation_key,
            connections,
            fields=("disulfide_connections",),
        )

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(DisulfideEnergyTerm, self).setup_packed_block_types(packed_block_types)

        cached = _cached_annotation(
            packed_block_types,
            "_disulfide_annotation",
            self._packed_annotation_key,
        )
        if cached is not None:
            return cached

        block_connections = [
            self.setup_block_type(bt) for bt in packed_block_types.active_block_types
        ]
        connections, mirrored = pack_disulfide_connections(
            packed_block_types.active_block_types,
            packed_block_types.max_n_conn,
            block_connections,
        )
        disulfide_conns = torch.tensor(connections, device=self.device)
        is_mirrored = torch.tensor(mirrored, device=self.device)
        setattr(packed_block_types, "disulfide_conns", disulfide_conns)
        setattr(packed_block_types, "disulfide_bt_is_mirrored", is_mirrored)
        packed_block_types._disulfide_parameter_signs = torch.tensor(
            disulfide_parameter_signs(mirrored), device=self.device
        )
        return _store_annotation(
            packed_block_types,
            "_disulfide_annotation",
            self._packed_annotation_key,
            (disulfide_conns, is_mirrored),
            fields=(
                "disulfide_conns",
                "disulfide_bt_is_mirrored",
                "_disulfide_parameter_signs",
            ),
            bindings=tuple(
                (block_type, "disulfide_connections")
                for block_type in packed_block_types.active_block_types
            ),
        )

    def setup_poses(self, poses: PoseStack):
        super(DisulfideEnergyTerm, self).setup_poses(poses)

    @property
    def score_only_in_no_grad(self):
        # Tiny CPU disulfide calls do not amortize the per-call mode check.
        return self.device.type == "cuda"

    def get_pose_score_term_function(self):
        from tmol.score.disulfide.potentials._compiled import _disulfide_pose_scores

        return _disulfide_pose_scores

    def get_rotamer_score_term_function(self):
        from tmol.score.disulfide.potentials._compiled import _disulfide_rotamer_scores

        return _disulfide_rotamer_scores

    def pose_score_term_is_invariant_zero(self, pose_stack):
        """Disulfide scoring is zero unless the pose has a connected dslf edge."""
        pbt = pose_stack.packed_block_types
        block_types = pose_stack.block_type_ind64
        real_blocks = block_types >= 0
        dslf_connections = pbt.disulfide_conns[block_types.clamp_min(0)]
        connected = pose_stack.inter_residue_connections[..., 0] >= 0
        return not bool(
            (dslf_connections & connected & real_blocks.unsqueeze(-1)).any()
        )

    def get_score_term_attributes(self, pose_stack):
        # Column order and chirality signs are common chemistry. Stack and
        # multiply the live framework tensors without a device-to-host copy.
        values = torch.stack(
            [
                (
                    torch.ones_like(self.global_params.shift)
                    if p.attribute is None
                    else getattr(self.global_params, p.attribute)
                ).to(torch.float)
                for p in _DisulfideParameter
            ],
            dim=1,
        )
        pbt = pose_stack.packed_block_types
        per_block_type = values * pbt._disulfide_parameter_signs
        return [
            pose_stack.block_type_ind,
            pose_stack.inter_residue_connections,
            pbt.disulfide_conns,
            pbt.atom_downstream_of_conn,
            per_block_type,
        ]
