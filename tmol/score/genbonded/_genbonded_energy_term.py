"""GenBonded energy term: torsional sub-term of cart-bonded using generic
atom-type parameters rather than per-residue atom-name parameters.

Key differences from CartBondedEnergyTerm:
  - Torsional interactions (proper torsions and improper torsions).
  - Parameter lookup is by atom chemical type (e.g. CS, CD, C*, X) rather
    than by (residue_name, atom_name).  The database carries a hierarchy that
    maps each concrete type to a sequence of fall-back types.
  - Intra-block parameters (proper and improper torsions) are resolved at
    setup_block_type time and stored as a single dense array, tagged by type
    (tag=0 proper, tag=1 improper).  Shape per entry: Vec<Int,5> for the
    4-atom subgraph + type tag, Vec<Real,5> for parameters.
  - Inter-block torsion parameters are stored in a hash table keyed by
    (type1, type2, type3, type4, bond_bin), with the same parameter
    priority and reversed matches as the CPU database lookup.
  - Bond type of the central bond is tracked through the pipeline and used
    for both intra (Python-time lookup) and inter (GPU-time hash lookup).
"""

import torch


from tmol.score._atom_type_dependent_term import AtomTypeDependentTerm
from tmol.score._annotation_cache import (
    AnnotationKey,
    _cached_annotation,
    _store_annotation,
)

from tmol.database import ParameterDatabase

from tmol.score.genbonded.potentials._compiled import (
    genbonded_pose_scores,
    genbonded_rotamer_scores,
)

from tmol.chemical import RefinedResidueType
from tmol.pose._packed_block_types import PackedBlockTypes
from tmol.pose._pose_stack import PoseStack
from tmol.utility.weak_identity_cache import WeakIdentityLRU

from tmol_core.scoring.genbonded import (  # noqa: F401
    GenBondedPreparation,
    MAX_HIER_DEPTH,  # noqa: F401
    _linear_centers,  # noqa: F401
)

_INTER_TABLE_CACHE = WeakIdentityLRU()


_PACKED_FIELDS = (
    "genbonded_intra_subgraphs",
    "genbonded_intra_subgraph_offsets",
    "genbonded_intra_params",
    "genbonded_atom_type_hierarchy",
    "genbonded_atom_is_rosetta",
    "genbonded_connection_bond_bins",
    "genbonded_conn_scored_elsewhere",
    "genbonded_source_atom_index",
    "genbonded_source_block_type_index",
    "genbonded_inter_torsion_hash_keys",
    "genbonded_inter_torsion_hash_values",
    "genbonded_inter_improper_hash_keys",
    "genbonded_inter_improper_hash_values",
)


class GenBondedEnergyTerm(AtomTypeDependentTerm, GenBondedPreparation):
    device: torch.device

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        super(GenBondedEnergyTerm, self).__init__(param_db=param_db, device=device)
        GenBondedPreparation.__init__(
            self, param_db.chemical, param_db.scoring.genbonded
        )
        self._block_annotation_key = AnnotationKey.from_sources(
            param_db.chemical, self.gen_database
        )
        self._packed_annotation_key = AnnotationKey.from_sources(
            param_db.chemical, self.gen_database, settings=(self.device,)
        )

    @classmethod
    def class_name(cls):
        return "GenBonded"

    @classmethod
    def score_types(cls):
        import tmol.score.terms._genbonded_creator

        return tmol.score.terms._genbonded_creator.GenBondedTermCreator.score_types()

    def n_bodies(self):
        return 2

    # ------------------------------------------------------------------
    # Block-type setup
    # ------------------------------------------------------------------

    def setup_block_type(self, block_type: RefinedResidueType):
        super(GenBondedEnergyTerm, self).setup_block_type(block_type)
        previous = _cached_annotation(
            block_type, "_genbonded_annotation", self._block_annotation_key
        )
        if previous is not None:
            return previous

        annotation = self.prepare_block(block_type)
        for name, value in zip(
            (
                "genbonded_intra_subgraphs",
                "genbonded_intra_params",
                "genbonded_atom_type_hierarchy",
            ),
            annotation,
        ):
            setattr(block_type, name, value)
        return _store_annotation(
            block_type,
            "_genbonded_annotation",
            self._block_annotation_key,
            annotation,
            fields=(
                "genbonded_intra_subgraphs",
                "genbonded_intra_params",
                "genbonded_atom_type_hierarchy",
            ),
        )

    # ------------------------------------------------------------------
    # Packed-block-types setup
    # ------------------------------------------------------------------

    def setup_packed_block_types(  # noqa: C901
        self, packed_block_types: PackedBlockTypes
    ):
        super(GenBondedEnergyTerm, self).setup_packed_block_types(packed_block_types)
        previous = _cached_annotation(
            packed_block_types,
            "_genbonded_annotation",
            self._packed_annotation_key,
        )
        if previous is not None:
            return previous

        block_types = packed_block_types.active_block_types
        # Capture each returned annotation, so another setup cannot switch the
        # parameters while this packed set is being assembled.
        block_values = [self.setup_block_type(bt) for bt in block_types]
        inter_tables = _INTER_TABLE_CACHE.get_or_create(
            self.gen_database, self.device, self._build_inter_tables
        )
        values = (
            tuple(
                torch.from_numpy(arr).to(device=self.device)
                for arr in self.prepare_packed(block_types, block_values)
            )
            + inter_tables
        )
        for name, value in zip(_PACKED_FIELDS, values):
            setattr(packed_block_types, name, value)
        return _store_annotation(
            packed_block_types,
            "_genbonded_annotation",
            self._packed_annotation_key,
            values,
            fields=_PACKED_FIELDS,
            bindings=tuple(
                (block_type, field)
                for block_type in packed_block_types.active_block_types
                for field in (
                    "genbonded_intra_subgraphs",
                    "genbonded_intra_params",
                    "genbonded_atom_type_hierarchy",
                )
            ),
        )

    def _build_inter_tables(self):
        return tuple(
            torch.from_numpy(arr).to(device=self.device) for arr in self.inter_tables()
        )

    # ------------------------------------------------------------------
    # Pose setup
    # ------------------------------------------------------------------

    def setup_poses(self, poses: PoseStack):
        super(GenBondedEnergyTerm, self).setup_poses(poses)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    score_only_in_no_grad = True

    def get_pose_score_term_function(self):
        return genbonded_pose_scores

    def get_packing_rotamer_score_term_function(self):
        from tmol.score.genbonded.potentials._compiled import (
            _iter_packing_rotamer_scores,
        )

        return _iter_packing_rotamer_scores

    def get_rotamer_score_term_function(self):
        return genbonded_rotamer_scores

    def get_score_term_attributes(self, pose_stack: PoseStack):
        pbt = pose_stack.packed_block_types
        parameters = self.setup_packed_block_types(pbt)
        return [
            pose_stack.inter_residue_connections,
            pbt.atom_paths_from_conn,
            *parameters,
        ]
