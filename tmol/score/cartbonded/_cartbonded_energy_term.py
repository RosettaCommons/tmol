import torch
import numpy
import attrs


from tmol.score._atom_type_dependent_term import AtomTypeDependentTerm
from tmol.score._annotation_cache import (
    AnnotationKey,
    _cached_annotation,
    _store_annotation,
)

from tmol.database import ParameterDatabase

from tmol.chemical import RefinedResidueType
from tmol.pose._packed_block_types import PackedBlockTypes
from tmol.pose._pose_stack import PoseStack
from tmol_core.scoring.cartbonded import (  # noqa: F401
    CartBondedPreparation,
    CartBondedBlockAnnotations,  # noqa: F401
    CROSS_RES_PREFIX,  # noqa: F401
)


@attrs.define(auto_attribs=True, slots=True, frozen=True)
class CartBondedPackedBlockTypesAnnotations:
    cartbonded_subgraphs: torch.Tensor
    cartbonded_subgraph_offsets: torch.Tensor
    cartbonded_subgraph_type_counts: torch.Tensor
    cartbonded_subgraph_type_offsets: torch.Tensor
    cartbonded_subgraph_param_indices: torch.Tensor
    cartbonded_max_subgraphs_per_block: int
    cartbonded_atom_unique_id_index: dict
    atom_unique_ids: torch.Tensor
    atom_wildcard_ids: torch.Tensor
    atom_cross_ids: torch.Tensor
    cartbonded_params_hash_keys: torch.Tensor
    cartbonded_params_hash_values: torch.Tensor
    connection_hash_keys: torch.Tensor
    connection_spans: torch.Tensor
    connection_paths: torch.Tensor
    atom_is_rosetta: torch.Tensor
    rosetta_typed: frozenset


class CartBondedEnergyTerm(AtomTypeDependentTerm, CartBondedPreparation):
    device: torch.device  # = attr.ib()
    improper_roots: set()

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        super(CartBondedEnergyTerm, self).__init__(param_db=param_db, device=device)

        CartBondedPreparation.__init__(
            self, param_db.scoring.cartbonded, param_db.scoring.genbonded.rosetta_typed
        )
        self._block_annotation_key = AnnotationKey.from_sources(self.cart_database)
        self._packed_annotation_key = AnnotationKey.from_sources(
            param_db.chemical,
            self.cart_database,
            param_db.scoring.genbonded,
            settings=(self.device,),
        )
        self._fragment_annotation_key = AnnotationKey.from_sources(
            settings=(self.device,)
        )
        self._ownership_annotation_key = AnnotationKey.from_sources(
            param_db.scoring.genbonded, settings=(self.device,)
        )

    @classmethod
    def class_name(cls):
        return "CartBonded"

    @classmethod
    def score_types(cls):
        import tmol.score.terms._cartbonded_creator

        return tmol.score.terms._cartbonded_creator.CartBondedTermCreator.score_types()

    def n_bodies(self):
        return 2

    def setup_block_type(self, block_type: RefinedResidueType):
        super(CartBondedEnergyTerm, self).setup_block_type(block_type)
        cached = _cached_annotation(
            block_type, "_cartbonded_annotation", self._block_annotation_key
        )
        if cached is not None:
            return cached

        cb_block_ann = self.prepare_block(block_type)
        return _store_annotation(
            block_type,
            "_cartbonded_annotation",
            self._block_annotation_key,
            cb_block_ann,
        )

    def setup_packed_block_types(
        self, packed_block_types: PackedBlockTypes
    ):  # noqa: C901
        super(CartBondedEnergyTerm, self).setup_packed_block_types(packed_block_types)

        fragment = _cached_annotation(
            packed_block_types,
            "_cartbonded_fragment_annotation",
            self._fragment_annotation_key,
        )
        if fragment is None:
            fragment = torch.as_tensor(
                numpy.asarray(
                    [
                        block_type.is_ligand_fragment
                        for block_type in packed_block_types.active_block_types
                    ],
                    dtype=numpy.int32,
                ),
                device=self.device,
            )
            packed_block_types.cartbonded_is_fragment = fragment
            _store_annotation(
                packed_block_types,
                "_cartbonded_fragment_annotation",
                self._fragment_annotation_key,
                fragment,
                fields=("cartbonded_is_fragment",),
            )
        previous = _cached_annotation(
            packed_block_types,
            "_cartbonded_annotation",
            self._packed_annotation_key,
        )
        if previous is not None:
            return previous
        # Capture returned block annotations; cache eviction or another setup
        # must not switch the parameters while this packed set is assembled.
        block_annotations = [
            self.setup_block_type(bt) for bt in packed_block_types.active_block_types
        ]

        host = self.prepare_packed(
            packed_block_types.active_block_types,
            block_annotations,
            atom_ids=packed_block_types.atom_unique_id_data,
        )

        def on_device(name, value):
            if name == "atom_is_rosetta":
                return self._ownership_mask(packed_block_types, value)
            if isinstance(value, numpy.ndarray):
                if name in ("atom_unique_ids", "atom_wildcard_ids", "atom_cross_ids"):
                    if value is getattr(packed_block_types.atom_unique_id_data, name):
                        return getattr(packed_block_types, name)
                return torch.from_numpy(value).to(self.device)
            return value

        cb_pbt_ann = CartBondedPackedBlockTypesAnnotations(
            **{
                field.name: on_device(field.name, getattr(host, field.name))
                for field in attrs.fields(CartBondedPackedBlockTypesAnnotations)
            }
        )
        packed_block_types.cartbonded_atom_is_rosetta = cb_pbt_ann.atom_is_rosetta
        return _store_annotation(
            packed_block_types,
            "_cartbonded_annotation",
            self._packed_annotation_key,
            cb_pbt_ann,
            fields=("cartbonded_atom_is_rosetta",),
        )

    def _ownership_mask(self, packed_block_types, prepared=None):
        cached = _cached_annotation(
            packed_block_types,
            "_cartbonded_ownership_annotation",
            self._ownership_annotation_key,
        )
        if cached is not None:
            return cached
        mask = (
            self.ownership_mask(packed_block_types.active_block_types)
            if prepared is None
            else prepared
        )
        value = torch.tensor(mask, dtype=torch.int32, device=self.device)
        return _store_annotation(
            packed_block_types,
            "_cartbonded_ownership_annotation",
            self._ownership_annotation_key,
            value,
        )

    def setup_poses(self, poses: PoseStack):
        super(CartBondedEnergyTerm, self).setup_poses(poses)

    def get_pose_score_term_function(self):
        from tmol.score.cartbonded.potentials._compiled import _cartbonded_pose_scores

        return _cartbonded_pose_scores

    def get_packing_rotamer_score_term_function(self):
        from tmol.score.cartbonded.potentials._compiled import (
            _iter_packing_rotamer_scores,
        )

        return _iter_packing_rotamer_scores

    def get_rotamer_score_term_function(self):
        from tmol.score.cartbonded.potentials._compiled import (
            _cartbonded_rotamer_scores,
        )

        return _cartbonded_rotamer_scores

    def get_score_term_attributes(self, pose_stack):
        pbt = pose_stack.packed_block_types

        pbt_cb_ann = self.setup_packed_block_types(pbt)
        return [
            pose_stack.inter_residue_connections,
            pbt.atom_paths_from_conn,
            pbt_cb_ann.atom_unique_ids,
            pbt_cb_ann.atom_wildcard_ids,
            pbt_cb_ann.atom_is_rosetta,
            pbt.cartbonded_is_fragment,
            pbt_cb_ann.atom_cross_ids,
            pbt_cb_ann.connection_hash_keys,
            pbt_cb_ann.connection_spans,
            pbt_cb_ann.connection_paths,
            pbt_cb_ann.cartbonded_params_hash_keys,
            pbt_cb_ann.cartbonded_params_hash_values,
            pbt_cb_ann.cartbonded_subgraphs,
            pbt_cb_ann.cartbonded_subgraph_offsets,
            pbt_cb_ann.cartbonded_subgraph_type_counts,
            pbt_cb_ann.cartbonded_subgraph_type_offsets,
            pbt_cb_ann.cartbonded_subgraph_param_indices,
        ]
