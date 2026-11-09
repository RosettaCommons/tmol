"""Metal coordination restraints."""

import torch

from tmol.database import ParameterDatabase
from tmol.pose._packed_block_types import PackedBlockTypes
from tmol.pose._pose_stack import PoseStack

from .._annotation_cache import AnnotationKey, _cached_annotation, _store_annotation
from .._energy_term import EnergyTerm

from tmol_core.scoring.metal import (  # noqa: F401
    MetalCoordinationPreparation,
    validate_donors,
    NOT_A_SITE,
    NO_VIRTUAL,
    MAX_INTERNAL_METALS,
)


class MetalCoordinationEnergyTerm(EnergyTerm, MetalCoordinationPreparation):
    """Per filled site, harmonics on donor distance and off-ray displacement plus a
    well depth; site fans held rigid; metals bridged across residues kept apart.
    """

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        super().__init__(param_db=param_db, device=device)
        self.device = device
        MetalCoordinationPreparation.__init__(self, param_db.scoring.metal_coordination)
        self._packed_annotation_key = AnnotationKey.from_sources(
            self.params, settings=(self.device,)
        )

    @classmethod
    def class_name(cls):
        return "MetalCoordination"

    @classmethod
    def score_types(cls):
        import tmol.score.terms._metal_coordination_creator

        return (
            tmol.score.terms._metal_coordination_creator.MetalCoordinationTermCreator.score_types()
        )

    def n_bodies(self):
        return 2

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super().setup_packed_block_types(packed_block_types)
        cached = _cached_annotation(
            packed_block_types,
            "_metal_coordination_annotation",
            self._packed_annotation_key,
        )
        if cached is not None:
            return cached

        pbt = packed_block_types
        host = self.prepare_packed(
            pbt.active_block_types, pbt.chem_db.atom_types, pbt.max_n_conn
        )
        fields = {
            name: torch.from_numpy(value).to(self.device)
            for name, value in host.items()
        }
        fields["metal_coordination_host"] = host
        for name, value in fields.items():
            setattr(pbt, name, value)
        return _store_annotation(
            pbt,
            "_metal_coordination_annotation",
            self._packed_annotation_key,
            tuple(fields[name] for name in host),
            fields=tuple(fields),
        )

    def pose_score_term_is_invariant_zero(self, pose_stack: PoseStack):
        pbt = pose_stack.packed_block_types
        bt = pose_stack.block_type_ind64
        is_metal = (pbt.metal_coordination_conn_metal[bt.clamp_min(0)] >= 0).any(-1)
        return not bool((is_metal & (bt >= 0)).any())

    def get_pose_score_term_function(self):
        from tmol.score.metal.potentials._compiled import (
            _metal_coordination_pose_scores,
        )

        return _metal_coordination_pose_scores

    def get_rotamer_score_term_function(self):
        from tmol.score.metal.potentials._compiled import (
            _metal_coordination_rotamer_scores,
        )

        return _metal_coordination_rotamer_scores

    def get_score_term_attributes(self, pose_stack: PoseStack):
        pbt = pose_stack.packed_block_types
        self.check_donors(pose_stack)
        return [
            pose_stack.inter_residue_connections,
            pbt.conn_atom,
            pbt.metal_coordination_conn_metal,
            pbt.metal_coordination_conn_virt,
            pbt.metal_coordination_conn_key,
            pbt.metal_coordination_site_params,
            pbt.metal_coordination_fan_atoms,
            pbt.metal_coordination_fan_params,
            pbt.metal_coordination_bridge_internal_metal,
            pbt.metal_coordination_bridge_internal_d0,
            pbt.metal_coordination_bridge_params,
        ]

    def check_donors(self, pose_stack: PoseStack):
        """Every filled site's donor atom must have a metal-donor distance."""
        host = pose_stack.packed_block_types.metal_coordination_host
        validate_donors(
            pose_stack.block_type_ind64.detach().cpu().numpy(),
            pose_stack.inter_residue_connections64.detach().cpu().numpy(),
            host["metal_coordination_conn_virt"],
            host["metal_coordination_conn_key"],
        )
