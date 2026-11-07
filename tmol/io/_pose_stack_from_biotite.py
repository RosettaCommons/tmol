import copy
from collections import defaultdict  # noqa: F401
from typing import TYPE_CHECKING

import attr
import torch
import numpy
import toolz
import biotite
import biotite.structure
import logging

from tmol_core.io.atom_metadata import (  # noqa: F401
    populate_optional_atom_metadata as _populate_optional_atom_metadata,
)
from tmol_core.io.atom_array import (  # noqa: F401
    METAL_ORIGIN,
    template_array as _template_array,
    is_metal as _is_metal,
    metal_coordination_bond_mask as _metal_coordination_bond_mask,
    with_metal_coordination_typed as _with_metal_coordination_typed,
    metal_coordination_from_biotite as _metal_coordination_from_biotite,
    covalent_bonds_from_biotite as _covalent_bonds_from_biotite,
)

from tmol.types._functional import _validate_args
from tmol.chemical import ResidueTypeSet
from tmol.database import ParameterDatabase
from tmol.database.chemical import _metal_table, _site_connections  # noqa: F401
from tmol.io._atomworks_reader import (  # noqa: F401
    INFERRED_POLYMER_BOND,
    _renumbered_decreasing_chains,
)
from tmol.io._input_geometry import _rebuild_coincident_hydrogens
from tmol.io.details._select_from_canonical import (  # noqa: F401
    _validate_ligand_hydrogens,
)  # noqa: F401
from tmol.io._input_termini import (  # noqa: F401
    EXPLICIT_TERMINI,
    _validate_connection_hydrogens,
    _with_stated_termini,
)
from tmol.io._canonical_ordering import _only_coordinates_a_metal, CanonicalOrdering
from tmol.io._protonation import (  # noqa: F401
    PROTONATION_VARIANT,
    _database_forms,
    _residues_lacking_hydrogens,
    _with_atomworks_hydrogens,
)
from tmol.io._canonical_form import CanonicalForm
from tmol.io._pose_stack_deconstruction import (  # noqa: F401
    _canonical_form_from_pose_stack,
)
from tmol.io._build_context import PoseBuildContext
from tmol.pose._packed_block_types import PackedBlockTypes
from tmol.pose._pose_stack import PoseStack
from tmol.utility import resolve_device

from tmol_core.io import structure_context as _structure_context
from tmol_core.io.atom37 import (
    Atom37MappingError,
    atom37_mapping as _atom37_mapping,
    required_mainchain_entries as _required_mainchain_entries,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from tmol.ligand._fragmentation import FragmentedLigandPoseMapping

_MAX_PREPARED_BATCH_SIZES = 4


def _clone_pose_topology(pose_stack: PoseStack) -> PoseStack:
    """Clone caller-mutable pose data while retaining shared chemical types."""
    result = pose_stack.clone()
    result.pdb_info = copy.deepcopy(pose_stack.pdb_info)
    result.split_block_mapping = copy.deepcopy(pose_stack.split_block_mapping)
    return result


@attr.s(auto_attribs=True, frozen=True, slots=True)
class _PreparedAtom37PoseTopology:
    """Fixed pose layout plus the masks needed to rebuild missing leaf atoms."""

    pose_stack: PoseStack
    canonical_atom_mapping: torch.Tensor
    pose_atom_mapping: torch.Tensor
    block_leaf_atom_is_missing: torch.Tensor
    pose_atom_is_missing: torch.Tensor
    block_has_missing_atoms: torch.Tensor
    real_atoms: torch.Tensor

    @classmethod
    def from_pose(
        cls,
        pose_stack: PoseStack,
        canonical_coords: torch.Tensor,
        canonical_atom_mapping: torch.Tensor,
        pose_atom_mapping: torch.Tensor,
        block_has_missing_atoms: torch.Tensor,
    ) -> "_PreparedAtom37PoseTopology":
        canonical_atom_mapping = canonical_atom_mapping.to(torch.int64)
        pose_atom_mapping = pose_atom_mapping.to(torch.int64)
        block_leaf_atom_is_missing = torch.zeros(
            (
                pose_stack.n_poses,
                pose_stack.max_n_blocks,
                pose_stack.max_n_block_atoms,
            ),
            dtype=torch.bool,
            device=pose_stack.device,
        )
        pose_atom_is_missing = torch.zeros(
            pose_stack.coords.shape[:2], dtype=torch.bool, device=pose_stack.device
        )
        pose_template = _clone_pose_topology(pose_stack)
        return cls(
            pose_stack=pose_template,
            canonical_atom_mapping=canonical_atom_mapping,
            pose_atom_mapping=pose_atom_mapping,
            block_leaf_atom_is_missing=block_leaf_atom_is_missing,
            pose_atom_is_missing=pose_atom_is_missing,
            block_has_missing_atoms=block_has_missing_atoms,
            real_atoms=pose_stack.real_atoms,
        )

    def pose_from_canonical(self, canonical_coords: torch.Tensor) -> PoseStack:
        """Rebind coordinates and rebuild leaf atoms in the prepared pose layout."""
        mapping = self.canonical_atom_mapping
        pose_mapping = self.pose_atom_mapping
        source_coords = canonical_coords[mapping[:, 0], mapping[:, 1], mapping[:, 2]]
        finite = torch.isfinite(source_coords).all(dim=-1)
        pose_ind = pose_mapping[:, 0]
        pose_atom = pose_mapping[:, 1]
        block_ind = mapping[:, 1]
        block_atom = (
            pose_atom - self.pose_stack.block_coord_offset64[pose_ind, block_ind]
        )
        block_leaf_atom_is_missing = self.block_leaf_atom_is_missing.clone()
        block_leaf_atom_is_missing[
            pose_ind[~finite], block_ind[~finite], block_atom[~finite]
        ] = True
        pose_atom_is_missing = self.pose_atom_is_missing.clone()
        pose_atom_is_missing[pose_ind[~finite], pose_atom[~finite]] = True
        coords = torch.zeros_like(self.pose_stack.coords)
        coords[pose_ind[finite], pose_atom[finite]] = source_coords[finite]

        pbt = self.pose_stack.packed_block_types
        from tmol.io.details._build_missing_leaf_atoms import (
            _apply_h_geometric_completion,
        )
        from tmol.io.details.compiled._compiled import _gen_pose_leaf_atoms

        coords = _gen_pose_leaf_atoms(
            coords,
            block_leaf_atom_is_missing,
            pose_atom_is_missing,
            self.pose_stack.block_coord_offset,
            self.pose_stack.block_type_ind,
            self.pose_stack.inter_residue_connections,
            pbt.n_atoms,
            pbt.atom_downstream_of_conn,
            pbt.build_missing_leaf_atom_icoor_ann.anc_uaids,
            pbt.build_missing_leaf_atom_icoor_ann.geom,
            pbt.build_missing_leaf_atom_icoor_ann.anc_uaids_backup,
            pbt.build_missing_leaf_atom_icoor_ann.geom_backup,
        )
        coords = _apply_h_geometric_completion(
            pbt,
            coords,
            block_leaf_atom_is_missing,
            self.pose_stack.block_coord_offset,
            self.pose_stack.block_type_ind,
            self.pose_stack.inter_residue_connections,
        )
        pose_stack = _clone_pose_topology(self.pose_stack)
        pose_stack.coords = coords
        return pose_stack

    def has_missing_nonleaf(self, canonical_coords: torch.Tensor) -> bool:
        """Return whether replay would require sidechain packing."""
        mapping = self.canonical_atom_mapping
        pose_mapping = self.pose_atom_mapping
        source_coords = canonical_coords[mapping[:, 0], mapping[:, 1], mapping[:, 2]]
        missing = ~torch.isfinite(source_coords).all(dim=-1)
        if not bool(torch.any(missing)):
            return False

        pose_ind = pose_mapping[:, 0]
        pose_atom = pose_mapping[:, 1]
        block_ind = mapping[:, 1]
        block_atom = (
            pose_atom - self.pose_stack.block_coord_offset64[pose_ind, block_ind]
        )
        block_type = self.pose_stack.block_type_ind64[pose_ind, block_ind]
        is_leaf = self.pose_stack.packed_block_types.is_leaf_atom[
            block_type, block_atom
        ]
        return bool(torch.any(missing & ~is_leaf))


@attr.s(auto_attribs=True, frozen=True, slots=True)
class PreparedAtom37PoseBuilder:
    """Bind immutable Biotite topology for repeated Atom37 pose construction.

    Calls accept float32 coordinates shaped ``[n_poses, n_tokens, 37, 3]`` on
    the context's device. The first call for a batch size prepares its fixed
    pose topology; up to four recently used batch sizes are cached. Inputs with
    coordinate-dependent atom presence, metal geometry, or ambiguous histidine
    hydrogens use the normal uncached construction path.

    The builder owns a mutable topology cache and is not safe for concurrent
    calls. Use one builder per calling thread when pose construction overlaps.
    """

    context: PoseBuildContext
    canonical_template: CanonicalForm
    mapped_token_id: torch.Tensor
    mapped_slot: torch.Tensor
    mapped_residue: torch.Tensor
    mapped_atom: torch.Tensor
    max_token_id: int
    required_mainchain_entries: tuple[tuple[int, int, str], ...]
    fragment_mapping: "FragmentedLigandPoseMapping | None" = None
    _topology_cache_safe: bool = True
    _pose_topologies: dict[int, _PreparedAtom37PoseTopology] = attr.ib(
        factory=dict, eq=False, repr=False
    )

    def __call__(
        self,
        atom37_coords: torch.Tensor,
        *,
        opt_h: bool = False,
    ) -> PoseStack:
        """Build a differentiable pose batch.

        Args:
            atom37_coords: Float32 coordinates shaped
                ``[n_poses, n_tokens, 37, 3]`` on the context's device.
            opt_h: Optimize hydrogen positions after construction. Disabled by
                default so finite input coordinates are preserved.

        Returns:
            A pose whose coordinates remain differentiable with respect to
            ``atom37_coords``.
        """
        canonical_coords = self._canonical_coords(atom37_coords)
        if not self._topology_cache_safe:
            return _pose_stack_from_canonical_form_and_context(
                self._canonical_form(canonical_coords),
                self.context,
                no_optH=not opt_h,
                atom37_coords=atom37_coords,
                fragment_mapping=self.fragment_mapping,
            )
        n_poses = atom37_coords.shape[0]
        topology = self._pose_topologies.pop(n_poses, None)
        topology_was_cached = topology is not None
        if topology is None:
            cf = self._canonical_form(canonical_coords)
            pose_stack, details = _pose_stack_from_canonical_form_and_context(
                cf,
                self.context,
                no_optH=True,
                atom37_coords=atom37_coords,
                fragment_mapping=self.fragment_mapping,
                return_atom_mapping=True,
            )
            block_has_missing_atoms = details["block_has_missing_atoms"]
            if bool(torch.any(block_has_missing_atoms)):
                if not opt_h:
                    return pose_stack
                return _pose_stack_from_canonical_form_and_context(
                    cf,
                    self.context,
                    no_optH=False,
                    atom37_coords=atom37_coords,
                    fragment_mapping=self.fragment_mapping,
                )
            topology = _PreparedAtom37PoseTopology.from_pose(
                pose_stack,
                canonical_coords,
                details["can_atom_mapping"],
                details["ps_atom_mapping"],
                block_has_missing_atoms,
            )
        else:
            if topology.has_missing_nonleaf(canonical_coords):
                self._pose_topologies[n_poses] = topology
                return _pose_stack_from_canonical_form_and_context(
                    self._canonical_form(canonical_coords),
                    self.context,
                    no_optH=not opt_h,
                    atom37_coords=atom37_coords,
                    fragment_mapping=self.fragment_mapping,
                )
            pose_stack = topology.pose_from_canonical(canonical_coords)
        if len(self._pose_topologies) >= _MAX_PREPARED_BATCH_SIZES:
            self._pose_topologies.pop(next(iter(self._pose_topologies)))
        self._pose_topologies[n_poses] = topology

        if opt_h:
            from tmol.pack._build_missing_sidechains import _build_missing_sidechains

            pose_stack = _build_missing_sidechains(
                pose_stack,
                self.context._opth_score_function,
                self.context._dunbrack_sampler,
                topology.block_has_missing_atoms,
                no_optH=False,
                has_missing_atoms=False,
            )
            pose_stack = _restore_canonical_input_coords(
                pose_stack,
                canonical_coords,
                topology.canonical_atom_mapping,
                topology.pose_atom_mapping,
            )
        # Pose construction validates the initial topology. Check the initial
        # packed result too, but do not force a CUDA-to-host synchronization on
        # every replay of an already validated fixed topology.
        if opt_h and not topology_was_cached:
            _assert_no_nan_coords(pose_stack, topology.real_atoms)
        return pose_stack

    def _canonical_coords(self, atom37_coords: torch.Tensor) -> torch.Tensor:
        """Overlay one coordinate batch without expanding static topology data."""
        device = self.context.packed_block_types.device
        _validate_atom37_coords(atom37_coords, device)
        if self.max_token_id >= atom37_coords.shape[1]:
            raise Atom37MappingError(
                f"token_id {self.max_token_id} exceeds atom37_coords token count "
                f"{atom37_coords.shape[1]}"
            )

        n_poses = atom37_coords.shape[0]
        template = self.canonical_template
        template_n_poses = template.coords.shape[0]
        if template_n_poses not in (1, n_poses):
            raise ValueError(
                f"Biotite structure has {template_n_poses} poses but "
                f"atom37_coords has {n_poses}"
            )

        coords = template.coords
        coords = (
            coords.clone()
            if template_n_poses == n_poses
            else coords.expand(n_poses, *coords.shape[1:]).clone()
        )
        source_coords = atom37_coords[:, self.mapped_token_id, self.mapped_slot]
        _validate_mapped_atom37_triplets(
            source_coords, self.mapped_token_id, self.mapped_slot
        )
        coords[:, self.mapped_residue, self.mapped_atom] = source_coords
        _validate_effective_mainchain_coords(coords, self.required_mainchain_entries)
        return coords

    def _canonical_form(self, coords: torch.Tensor) -> CanonicalForm:
        """Expand the static canonical topology for an uncached batch size."""
        template = self.canonical_template
        n_poses = coords.shape[0]

        def tensor_for_poses(value):
            if value is None or value.shape[0] == n_poses:
                return value
            return value.expand(n_poses, *value.shape[1:]).clone()

        def array_for_poses(value):
            if value is None or value.shape[0] == n_poses:
                return value
            return numpy.repeat(value, n_poses, axis=0)

        def bonds_for_poses(value):
            if value is None or template.coords.shape[0] == n_poses:
                return value
            bonds = value.repeat(n_poses, 1)
            bonds[:, 0] = torch.arange(n_poses, device=coords.device).repeat_interleave(
                value.shape[0]
            )
            return bonds

        return CanonicalForm(
            chain_id=tensor_for_poses(template.chain_id),
            res_types=tensor_for_poses(template.res_types),
            coords=coords,
            res_labels=array_for_poses(template.res_labels),
            residue_insertion_codes=array_for_poses(template.residue_insertion_codes),
            chain_labels=array_for_poses(template.chain_labels),
            atom_occupancy=array_for_poses(template.atom_occupancy),
            atom_b_factor=array_for_poses(template.atom_b_factor),
            disulfides=bonds_for_poses(template.disulfides),
            res_not_connected=tensor_for_poses(template.res_not_connected),
            cyclic_bonds=bonds_for_poses(template.cyclic_bonds),
            covalent_bonds=bonds_for_poses(template.covalent_bonds),
            metal_coordination=bonds_for_poses(template.metal_coordination),
            residue_annotations=array_for_poses(template.residue_annotations),
            metal_origins=array_for_poses(template.metal_origins),
            protonation_variants=tensor_for_poses(template.protonation_variants),
        )


@_validate_args
def prepare_atom37_pose_builder(
    biotite_structure: biotite.structure.AtomArray | biotite.structure.AtomArrayStack,
    context: PoseBuildContext,
) -> PreparedAtom37PoseBuilder:
    """Prepare a callable for repeatedly binding Atom37 coordinates to topology.

    This is the campaign-oriented counterpart to
    :func:`pose_stack_from_atom37_and_topology`: immutable residue identity,
    connectivity, fragmentation, and Atom37 routing are resolved once. Calling
    the returned builder with a coordinate tensor constructs a differentiable
    pose while retaining TMol's usual missing-atom behavior. The returned
    builder preserves finite input hydrogens by default; pass ``opt_h=True`` to
    optimize them after construction.
    """
    device = context.packed_block_types.device
    fragment_mapping = None
    if context.fragment_definitions:
        from tmol.ligand._fragmentation import _expand_fragmented_ligands

        biotite_structure, fragment_mapping = _expand_fragmented_ligands(
            biotite_structure, context.fragment_definitions
        )

    from tmol_core.io.canonical_atom_array import prepare_canonical_atom_array

    plan = prepare_canonical_atom_array(
        biotite_structure,
        context.canonical_ordering,
        missing_density_distance_threshold=0.0,
        filter_missing_mainchain=False,
        chemical_database=context.parameter_database.chemical,
        cut_partners=context.cut_covalent_partners,
    )
    filtered = plan.structure
    canonical_template = _canonical_form_from_host_plan(plan, device)
    token_id, slot, mapped_residue, mapped_atom = _atom37_mapping(
        filtered, plan.valid_atom_mask, plan.valid_res_inds, plan.valid_atom_inds
    )
    mapped_token_id = torch.as_tensor(token_id, device=device)
    mapped_slot = torch.as_tensor(slot, device=device)
    mapped_residue = torch.as_tensor(mapped_residue, device=device)
    mapped_atom = torch.as_tensor(mapped_atom, device=device)
    his_inds = context.canonical_ordering.his_inds
    ambiguous_his = False
    if his_inds.his_co_aa_inds:
        from tmol.io.details._his_taut_resolution import _his_atom_inds_tensor

        indices = _his_atom_inds_tensor(his_inds, device)[:, 4:7].long()
        types = canonical_template.res_types
        rows = (
            torch.zeros_like(types)
            if len(indices) == 1
            else types.clamp(0, len(indices) - 1)
        )
        ambiguous_atom_inds = indices[rows].clamp_min(0)
        is_his = torch.isin(
            canonical_template.res_types,
            torch.tensor(his_inds.his_co_aa_inds, device=device),
        )
        ambiguous_his = bool(
            torch.any(
                is_his.unsqueeze(-1)
                & torch.isfinite(
                    torch.gather(
                        canonical_template.coords,
                        2,
                        ambiguous_atom_inds.unsqueeze(-1).expand(-1, -1, -1, 3),
                    )
                ).all(dim=-1)
            )
        )
    return PreparedAtom37PoseBuilder(
        context=context,
        canonical_template=canonical_template,
        mapped_token_id=mapped_token_id,
        mapped_slot=mapped_slot,
        mapped_residue=mapped_residue,
        mapped_atom=mapped_atom,
        max_token_id=int(token_id.max()),
        required_mainchain_entries=_required_mainchain_entries(
            filtered, context.canonical_ordering
        ),
        fragment_mapping=fragment_mapping,
        # Metal virtuals require fitted site geometry, not leaf-atom icoors.
        # Donor selection can also change when the input coordinates change.
        topology_cache_safe=not ambiguous_his and not bool(_is_metal(filtered).any()),
    )


@_validate_args
def build_context_from_atom_array(
    biotite_structure: biotite.structure.AtomArray | biotite.structure.AtomArrayStack,
    torch_device: torch.device,
    param_db: ParameterDatabase | None = None,
    prepare_ligands: bool = False,
    ligand_ph: float = 7.4,
    strict_atom_types: bool = False,
    strict_ligands: bool = True,
    ligand_params_files: list[str] | None = None,
    chem_comp_types: dict | None = None,
    ligand_seed: int | None = None,
    ligand_hydrogens: str = "names",
) -> PoseBuildContext:
    """Build the structure-independent construction context.

    The returned context holds only database/ligand-derived pieces (canonical
    ordering, residue-type set, packed block types, parameter database); it does
    not depend on the input structure's coordinates and can be reused across
    structures sharing the same ligand(s). ``biotite_structure`` is used only to
    detect and prepare ligands (when ``prepare_ligands=True``).

    Args:
        biotite_structure: Input AtomArray or AtomArrayStack. Used only for
            ligand detection/preparation when ``prepare_ligands=True``.
        torch_device: Target torch device.
        param_db: Optional parameter database. When provided, canonical ordering,
            residue types, and packed block types are built from this database.
            If prepare_ligands=True, it is extended with ligand data. If None,
            defaults are used.
        prepare_ligands: If True, detect and prepare non-standard residues
            (via ``tmol.ligand``, which uses RDKit for atom typing and
            residue-type construction).
        ligand_ph: pH AtomWorks protonates residues lacking hydrogens at before
            ligands are prepared (default 7.4, only used when
            prepare_ligands=True).
        strict_atom_types: If True, unknown ligand atom types raise errors
            instead of using a fallback element heuristic.
        strict_ligands: If True (default), raise when a detected ligand cannot
            be prepared and registered (instead of silently dropping it during
            pose construction). Pass False to fall back to warn-and-skip. Only
            used when prepare_ligands=True.
        ligand_params_files: Optional list of tmol YAML params file paths.
            Residues defined in these files skip the RDKit/OB pipeline.
        chem_comp_types: ``{comp_id: type}`` from the input file's
            ``_chem_comp`` table (see
            ``tmol.ligand._detect._chem_comp_types_from_cif``), which says whether a
            residue belongs to a polymer where the file does not number it
            along a sequence. Only used when prepare_ligands=True.
        ligand_seed: Fixed RNG seed for the conformer each prepared residue
            is built from, making preparation reproducible. Only used when
            prepare_ligands=True.
        ligand_hydrogens: "rebuild" names prepared ligands' hydrogens by parent;
            "input" and "names" take the input's names where its hydrogens pair
            up parent by parent. Only used when prepare_ligands=True.

    Returns:
        PoseBuildContext containing canonical ordering, packed block
        types, parameter database, and residue type set.
    """
    torch_device = resolve_device(torch_device)

    def _prepare_ligands(*args, **kwargs):
        from tmol.ligand import prepare_ligands as prepare

        return prepare(*args, **kwargs)

    prepared = _structure_context.prepare_structure_context(
        biotite_structure,
        parameter_database=param_db,
        default_database=ParameterDatabase.get_default,
        prepare_ligands=prepare_ligands,
        ligand_ph=ligand_ph,
        strict_atom_types=strict_atom_types,
        strict_ligands=strict_ligands,
        ligand_params_files=ligand_params_files,
        chem_comp_types=chem_comp_types,
        ligand_seed=ligand_seed,
        ligand_hydrogens=ligand_hydrogens,
        input_hydrogens=_with_input_hydrogens,
        ligand_preparer=_prepare_ligands,
    )
    if prepared.use_default_context:
        return _default_pose_build_context_for(prepared.structure, torch_device)
    db = prepared.parameter_database
    if prepare_ligands:
        co = prepared.canonical_ordering
        rts = _restype_set_sharing_default(db.chemical)
        pbt = PackedBlockTypes.from_restype_list(
            rts.chem_db, rts, rts.residue_types, torch_device
        )
    else:
        co, rts, pbt = _derived_types_for_param_db(db, torch_device)
    return PoseBuildContext(
        canonical_ordering=co,
        packed_block_types=pbt,
        parameter_database=db,
        restype_set=rts,
        fragment_definitions=prepared.fragment_definitions,
        cut_covalent_partners=prepared.cut_covalent_partners,
    )


@_validate_args
def pose_stack_from_atom_array(  # noqa: C901
    biotite_structure: biotite.structure.AtomArray | biotite.structure.AtomArrayStack,
    torch_device: torch.device,
    param_db: ParameterDatabase | None = None,
    missing_density_distance_threshold: float = 2.4,
    no_optH: bool = False,
    prepare_ligands: bool = False,
    ligand_ph: float = 7.4,
    strict_atom_types: bool = False,
    strict_ligands: bool = True,
    ligand_params_files: list[str] | None = None,
    chem_comp_types: dict | None = None,
    ligand_seed: int | None = None,
    ligand_hydrogens: str = "names",
    packer_seed: int | None = None,
    return_context: bool = False,
    context: PoseBuildContext | None = None,
    atom37_coords: torch.Tensor | None = None,
    **kwargs: object,
) -> PoseStack | tuple[PoseStack, dict] | tuple[PoseStack, PoseBuildContext]:
    """Build a PoseStack from an AtomArray or AtomArrayStack.

    To score many structures that share the same ligand(s) efficiently, build
    the (expensive, structure-independent) context once and reuse it::

        context = build_context_from_atom_array(struct0, dev, prepare_ligands=True)
        for struct in structures:
            pose_stack = pose_stack_from_atom_array(struct, dev, context=context)

    Reusing a context skips rebuilding the parameter database, canonical
    ordering, residue-type set, and packed block types; only the per-structure
    canonical form is recomputed (see the ``context`` arg).

    Missing non-polymer atoms use prepared conformer geometry and resolved
    coordinate anchors. If only an attachment's endpoints are resolved, its
    first declared torsion sample and the next resolved partner atom can orient
    the missing component. These are starting conformers for scoring/packing,
    not recovered experimental coordinates. Supplied heavy-atom coordinates
    stay unchanged; insufficient or degenerate references still raise.

    Args:
        biotite_structure: A Biotite AtomArray or AtomArrayStack.
        torch_device: Target PyTorch device.
        param_db: Optional ParameterDatabase. When provided, conversion and pose
            construction use this database. If prepare_ligands=True, it is
            extended with ligand data. Mutually exclusive with ``context``.
        missing_density_distance_threshold: Distance threshold in Angstroms.
            Adjacent polymer residues whose connection atoms exceed this distance
            are disconnected only where topology was inferred. A supplied bond
            across a gap raises; set to 0 to retain it. Default is 2.4.
        no_optH: Residues the input gives no hydrogens take AtomWorks'
            protonation state: database residues then get hydrogens built by
            tmol, other residues those AtomWorks places. When True,
            preserve finite input hydrogen coordinates, except coincident
            hydrogen-parent pairs, which are rebuilt with a warning,
            and build only missing hydrogens and heavy-atom sidechains. When
            False (default), residues with complete heavy atoms are packed with
            OptHSampler to optimize hydrogen positions and NHQ flips, while
            residues with missing heavy atoms are rebuilt with
            DunbrackChiSampler. Generated ligand hydrogens follow
            ``ligand_hydrogens`` instead.
        prepare_ligands: If True, detect and prepare non-standard residues
            (see ``build_context_from_atom_array`` for details).
        ligand_ph: pH AtomWorks protonates residues lacking hydrogens at
            (default 7.4).
        strict_atom_types: If True, unknown ligand atom types raise errors
            instead of using a fallback element heuristic.
        strict_ligands: If True (default), raise when a detected ligand cannot
            be prepared and registered, instead of silently dropping it. Pass
            False to warn-and-skip. Only used when prepare_ligands=True.
        ligand_params_files: Optional list of tmol YAML params file paths.
        chem_comp_types: ``{comp_id: type}`` from the input file's
            ``_chem_comp`` table, which says whether a residue belongs to a
            polymer where the file does not number it along a sequence. Only
            used when prepare_ligands=True.
        ligand_seed: Fixed RNG seed for the conformer each prepared residue
            is built from, making preparation reproducible. Only used when
            prepare_ligands=True.
        ligand_hydrogens: Hydrogens of generated ligand types. "input" takes the
            input's names where its hydrogens pair up with the prepared ones
            parent by parent, and keeps each input hydrogen at bond length from
            the parent its name gives it. "names" (default) takes the names but
            rebuilds every coordinate. "rebuild" names them by parent and
            rebuilds every coordinate.
        packer_seed: Seed of the packer that places hydrogens and builds missing
            side chains; unseeded, it continues torch's global random state.
        return_context: If True, return ``(pose_stack, PoseBuildContext)``.
        context: Reusable context from ``build_context_from_atom_array``. It must
            be on ``torch_device`` and is mutually exclusive with ``param_db``
            and ``prepare_ligands=True``.
        atom37_coords: Optional coordinates shaped ``[pose, token, 37, xyz]``.
            When supplied, mapped coordinates are read from this tensor using
            the input structure's integer ``token_id`` and ``atom37_slot``
            annotations. Wholly finite mapped triplets are authoritative;
            all-NaN mapped triplets are missing and do not fall back to the
            Biotite coordinates. Unmapped finite Biotite atoms remain context,
            and absent leaf atoms are completed normally. Partial-NaN or
            infinite mapped triplets raise :class:`Atom37MappingError`. The
            resulting pose coordinates remain connected to this tensor for
            autograd. Geometry-based missing-density and
            additional-disulfide detection are disabled so topology is fixed.
        **kwargs: Additional arguments passed to pose_stack_from_canonical_form.

    Returns:
        PoseStack when no optional values requested and return_context is False.
        ``(PoseStack, PoseBuildContext)`` when return_context is True.
        ``(PoseStack, dict)`` when optional return values were requested via kwargs.
        Fragmented poses expose their block mapping as
        ``pose_stack.split_block_mapping``.
    """
    torch_device = resolve_device(torch_device)

    if context is not None:
        if param_db is not None or ligand_params_files:
            raise ValueError(
                "Pass either context= or param_db=/ligand_params_files, not both; the context "
                "already carries its parameter database."
            )
        if prepare_ligands:
            raise ValueError(
                "context= already contains prepared ligands; do not also pass "
                "prepare_ligands=True."
            )
        context_device = context.packed_block_types.device
        if context_device.type != torch_device.type or (
            context_device.type == "cuda" and context_device.index != torch_device.index
        ):
            raise ValueError(
                "context was built for device "
                f"'{context.packed_block_types.device}' but torch_device is "
                f"'{torch_device}'; they must match."
            )
    else:
        if prepare_ligands and atom37_coords is None:
            biotite_structure = _with_input_hydrogens(
                biotite_structure,
                ligand_ph,
                None,
                (param_db or _paramdb_for_biotite()).chemical,
                kwargs.get("find_additional_metal_coordination", True),
            )
        context = build_context_from_atom_array(
            biotite_structure,
            torch_device,
            param_db=param_db,
            prepare_ligands=prepare_ligands,
            ligand_ph=ligand_ph,
            strict_atom_types=strict_atom_types,
            strict_ligands=strict_ligands,
            ligand_params_files=ligand_params_files,
            chem_comp_types=chem_comp_types,
            ligand_seed=ligand_seed,
            ligand_hydrogens=ligand_hydrogens,
        )
    if atom37_coords is None:
        biotite_structure = _with_input_hydrogens(
            biotite_structure,
            ligand_ph,
            context.canonical_ordering,
            context.parameter_database.chemical,
            kwargs.get("find_additional_metal_coordination", True),
        )

    fragment_mapping = None
    if context.fragment_definitions:
        from tmol.ligand._fragmentation import _expand_fragmented_ligands

        biotite_structure, fragment_mapping = _expand_fragmented_ligands(
            biotite_structure, context.fragment_definitions
        )

    # The canonical form is per-structure, so it is always computed here for the
    # given structure (never carried in the reusable context).
    cf = canonical_form_from_biotite(
        biotite_structure,
        torch_device,
        co=context.canonical_ordering,
        missing_density_distance_threshold=missing_density_distance_threshold,
        atom37_coords=atom37_coords,
        _cut_covalent_partners=context.cut_covalent_partners,
        _chemical_database=(
            context.parameter_database.chemical if atom37_coords is not None else None
        ),
    )
    disconnected = kwargs.pop("res_not_connected", None)
    if disconnected is not None:
        if (
            disconnected.shape != (*cf.chain_id.shape, 2)
            or disconnected.dtype != torch.bool
            or disconnected.device != torch_device
        ):
            raise ValueError(
                "res_not_connected must be a boolean [pose, residue, 2] tensor on the pose device"
            )
        bonds = cf.covalent_bonds
        if bonds is not None:
            ports = context.canonical_ordering.polymer_conn_inds
            remove = torch.zeros(len(bonds), dtype=torch.bool, device=torch_device)
            for direction, names in enumerate(
                (ports.down_atom_for_co_restype, ports.up_atom_for_co_restype)
            ):
                lookup = torch.tensor(names, device=torch_device)
                for column in (1, 3):
                    pose, residue = bonds[:, 0], bonds[:, column]
                    remove |= disconnected[pose, residue, direction] & (
                        bonds[:, column + 1] == lookup[cf.res_types[pose, residue]]
                    )
            bonds = bonds[~remove]
        cf = attr.evolve(cf, res_not_connected=disconnected, covalent_bonds=bonds)

    return _pose_stack_from_canonical_form_and_context(
        cf,
        context,
        no_optH=no_optH,
        atom37_coords=atom37_coords,
        fragment_mapping=fragment_mapping,
        return_context=return_context,
        packer_seed=packer_seed,
        ligand_hydrogens=ligand_hydrogens,
        **kwargs,
    )


def _pose_stack_from_canonical_form_and_context(
    cf: CanonicalForm,
    context: PoseBuildContext,
    *,
    no_optH: bool,
    atom37_coords: torch.Tensor | None,
    fragment_mapping=None,
    return_context: bool = False,
    packer_seed: int | None = None,
    **kwargs: object,
) -> PoseStack | tuple[PoseStack, dict] | tuple[PoseStack, PoseBuildContext]:
    """Build a pose from a canonical form and a reusable build context.

    This is the single construction path. Every entry point -- canonical
    amino-acid Atom37 tensors, an Atom37 batch bound to a fixed topology, a
    parsed CIF/PDB structure -- reduces its input to a
    :py:class:`~tmol.io.CanonicalForm` plus a
    :py:class:`~tmol.io.PoseBuildContext` and finishes here. The entry points
    differ only in how they *derive* those two objects, not in how a pose is
    built from them.

    The context carries the canonical ordering and packed block types, so
    noncanonical residues, ligands, and covalent links are handled by the same
    code as standard amino acids: nothing here inspects an ``AtomArray``.
    Callers that already hold a canonical form and a context -- notably
    repeated guidance or search steps over one fixed topology -- should call
    this directly rather than re-deriving topology per batch.

    Args:
      cf: Canonical-form tensors describing residue identity and coordinates.
      context: Structure-independent chemistry resolved once.
      no_optH: Preserve finite input hydrogens instead of optimizing them.
      atom37_coords: When supplied, the autograd-tracked source of ``cf``'s
        coordinates; retained so gradients survive hydrogen rebuilding.
      fragment_mapping: Mapping produced when fragmented ligands were expanded.
      return_context: Also return the context used.
      packer_seed: Seed of the packer, as for ``pose_stack_from_atom_array``.

    For ordinary structure inputs, coincident bonded heavy atoms raise before
    packing; hydrogens coincident with their parent are rebuilt without changing
    protonation. Differentiable ``atom37_coords`` retain their supplied geometry.

    Returns:
      The constructed pose, optionally with atom mappings or the context.
    """
    from tmol.io._pose_stack_construction import _pose_stack_from_canonical_form
    from tmol.pack._build_missing_sidechains import _build_missing_sidechains
    from tmol.pack import FixedAAChiSampler
    from tmol.pack.rotamer._single_residue_kinforest import (
        _construct_single_residue_kinforest,
    )

    # amino-acid side chains a rotamer sampler covers are left for the packer
    samplers = (context._dunbrack_sampler, FixedAAChiSampler())

    def packer_atoms(rt):
        from tmol_core.pack.input_completion import packer_atom_mask

        return packer_atom_mask(rt, samplers, _construct_single_residue_kinforest)

    if atom37_coords is not None:
        # Both searches are geometric, so leaving either on makes the chemistry
        # depend on how close the coordinates happen to be -- and, because the
        # topology is cached, on which frame of a trajectory arrived first.
        kwargs.setdefault("find_additional_disulfides", False)
        kwargs.setdefault("find_additional_cyclic_closures", False)

    caller_requested_atom_mapping = bool(kwargs.get("return_atom_mapping", False))
    if atom37_coords is not None:
        # Coordinate rebuilding / hydrogen optimization may return a detached
        # coordinate tensor. Keep the canonical-to-pose mapping so the finite
        # input coordinates can be restored afterward without name matching.
        kwargs["return_atom_mapping"] = True

    return_block_has_missing_atoms = bool(
        kwargs.pop("return_block_has_missing_atoms", False)
    )
    result = _pose_stack_from_canonical_form(
        context.canonical_ordering,
        context.packed_block_types,
        *cf,
        return_block_has_missing_atoms=True,
        packer_atoms=packer_atoms,
        **kwargs,
    )

    pose_stack, opt_return_vals = result
    if atom37_coords is None:
        pose_stack = _rebuild_coincident_hydrogens(pose_stack)
    if fragment_mapping is not None:
        from tmol.ligand._fragmentation import _apply_fragment_connections

        pose_stack = _apply_fragment_connections(pose_stack, fragment_mapping)
        fragment_mapping = pose_stack.split_block_mapping
    block_has_missing_atoms = opt_return_vals["block_has_missing_atoms"]

    has_missing_atoms = block_has_missing_atoms is not None and bool(
        torch.any(block_has_missing_atoms)
    )
    if has_missing_atoms:
        _assert_no_ligand_with_missing_atoms(pose_stack, block_has_missing_atoms)

    needs_packing = block_has_missing_atoms is not None and (
        has_missing_atoms or not no_optH
    )
    if needs_packing:
        sfxn = (
            context._packing_score_function
            if has_missing_atoms
            else context._opth_score_function
        )
        dunbrack_sampler = context._dunbrack_sampler
        na_sampler = context._na_sampler if has_missing_atoms else None

        if has_missing_atoms:
            logger.info(
                "%i blocks with missing heavy atoms",
                torch.count_nonzero(block_has_missing_atoms),
            )
        pose_stack = _build_missing_sidechains(
            pose_stack,
            sfxn,
            dunbrack_sampler,
            block_has_missing_atoms,
            no_optH=no_optH,
            na_sampler=na_sampler,
            has_missing_atoms=has_missing_atoms,
            seed=packer_seed,
        )

    # Canonical construction may rebuild noncanonical hydrogens even without
    # packing. Supplied Atom37 coordinates remain authoritative on the initial
    # call, just as they are when replaying a prepared topology.
    if atom37_coords is not None:
        pose_stack = _restore_canonical_input_coords(
            pose_stack,
            cf.coords,
            opt_return_vals["can_atom_mapping"],
            opt_return_vals["ps_atom_mapping"],
        )
    if atom37_coords is not None and not caller_requested_atom_mapping:
        del opt_return_vals["can_atom_mapping"]
        del opt_return_vals["ps_atom_mapping"]

    if fragment_mapping is not None:
        pose_stack.split_block_mapping = fragment_mapping
    _assert_no_nan_coords(pose_stack)

    # This code tries to faithfully return what the caller expects based on the optional
    # return values that they requested. Since we override the return_block_has_missing_atoms
    # bool to True, we cannot just count on the existence or absence of optional returned vals
    if return_context:
        return pose_stack, context
    if len(opt_return_vals) > (0 if return_block_has_missing_atoms else 1):
        return pose_stack, opt_return_vals
    return pose_stack


def _restore_canonical_input_coords(
    pose_stack: PoseStack,
    canonical_coords: torch.Tensor,
    canonical_atom_mapping: torch.Tensor,
    pose_atom_mapping: torch.Tensor,
) -> PoseStack:
    """Restore finite canonical inputs after coordinate rebuilding or packing.

    TMol's packing pipeline deliberately treats coordinates as values rather
    than as an autograd graph. Atom37 callers need the reverse behavior: keep
    rebuilt/optimized coordinates for missing atoms and hydrogens, but route
    every finite input atom back to the differentiable canonical tensor. The
    atom mapping returned by pose construction makes this a pair of indexed
    tensor operations and avoids matching residue or atom names.
    """
    canonical_atom_mapping = canonical_atom_mapping.to(torch.int64)
    pose_atom_mapping = pose_atom_mapping.to(torch.int64)
    source_coords = canonical_coords[
        canonical_atom_mapping[:, 0],
        canonical_atom_mapping[:, 1],
        canonical_atom_mapping[:, 2],
    ]
    coords = pose_stack.coords.clone()
    finite = torch.isfinite(source_coords).all(dim=-1)
    coords[
        pose_atom_mapping[finite, 0],
        pose_atom_mapping[finite, 1],
    ] = source_coords[finite]
    result = copy.copy(pose_stack)
    result.coords = coords
    return result


def _assert_no_ligand_with_missing_atoms(
    pose_stack: PoseStack, block_has_missing_atoms: "torch.Tensor"
) -> None:
    """Reject ligand gaps left unresolved by the coordinate builder.

    Available construction frames have already been used. Polymer rotamer
    sampling cannot place an unanchored ligand or resolve the remaining gaps.
    """
    pbt = pose_stack.packed_block_types
    block_type_ind = pose_stack.block_type_ind
    block_coord_offset = pose_stack.block_coord_offset
    coords = pose_stack.coords
    pdb_info = getattr(pose_stack, "pdb_info", None)

    flagged = torch.nonzero(block_has_missing_atoms, as_tuple=False).cpu().tolist()
    bad: list[str] = []
    for pi, bi in flagged:
        bt_ind = int(block_type_ind[pi, bi].item())
        if bt_ind < 0:
            continue
        bt = pbt.active_block_types[bt_ind]
        if bt.properties.polymer.is_polymer:
            continue  # protein/nucleic — handled by sidechain rebuild

        n_ats = len(bt.atoms)
        atom_start = int(block_coord_offset[pi, bi].item())
        block_coords = coords[pi, atom_start : atom_start + n_ats]
        missing_mask = torch.isnan(block_coords).any(dim=-1)
        missing_names = [
            bt.atoms[ai].name
            for ai in torch.nonzero(missing_mask, as_tuple=False).flatten().tolist()
        ]

        label = ""
        if pdb_info is not None and pdb_info.residue_labels is not None:
            chain = pdb_info.chain_labels[pi, bi]
            resid = pdb_info.residue_labels[pi, bi]
            label = f" chain={chain} resid={resid}"
        bad.append(
            f"pose={pi} block={bi} bt={bt.name}{label} "
            f"missing_atoms={missing_names}"
        )

    if bad:
        raise RuntimeError(
            "Ligand (non-polymer) block(s) have missing heavy atoms; "
            "the available construction frames cannot place them. "
            "Provide enough resolved anchors or a complete ligand structure "
            "before calling pose_stack_from_atom_array:\n  " + "\n  ".join(bad)
        )


def _assert_no_nan_coords(
    pose_stack: PoseStack, real_atoms: torch.Tensor | None = None
) -> None:
    """Raise a descriptive error if any real atom in the PoseStack has NaN coords.

    Reports the offending pose, residue label/chain, block-type name, and atom
    name so failures in the auto-parsing pipeline (ligand prep, leaf-atom
    rebuild, sidechain build) can be traced to a specific residue.
    """
    coords = pose_stack.coords
    real = pose_stack.real_atoms if real_atoms is None else real_atoms
    nan_atom_mask = torch.isnan(coords).any(dim=-1) & real
    if not torch.any(nan_atom_mask):
        return

    pbt = pose_stack.packed_block_types
    block_coord_offset = pose_stack.block_coord_offset
    block_type_ind = pose_stack.block_type_ind
    pdb_info = getattr(pose_stack, "pdb_info", None)

    bad: list[str] = []
    nan_idxs = torch.nonzero(nan_atom_mask, as_tuple=False).cpu().tolist()
    for pi, at_idx in nan_idxs:
        valid_block_mask = block_type_ind[pi] >= 0
        valid_block_inds = torch.nonzero(valid_block_mask, as_tuple=False).flatten()
        offsets = block_coord_offset[pi, valid_block_inds]
        sel = torch.nonzero(offsets <= at_idx, as_tuple=False).flatten()
        if sel.numel() == 0:
            continue
        bi = int(valid_block_inds[sel[-1]].item())
        offset_in_block = at_idx - int(block_coord_offset[pi, bi].item())
        bt = pbt.active_block_types[int(block_type_ind[pi, bi].item())]
        atom_name = (
            bt.atoms[offset_in_block].name
            if 0 <= offset_in_block < len(bt.atoms)
            else f"#{offset_in_block}"
        )
        label = ""
        if pdb_info is not None and pdb_info.residue_labels is not None:
            chain = pdb_info.chain_labels[pi, bi]
            resid = pdb_info.residue_labels[pi, bi]
            label = f" chain={chain} resid={resid}"
        bad.append(
            f"pose={pi} block={bi} bt={bt.name}{label} atom={atom_name} "
            f"(global_atom_idx={at_idx})"
        )

    head = bad[:20]
    tail = f"\n  ... and {len(bad) - 20} more" if len(bad) > 20 else ""
    raise RuntimeError(
        "NaN coordinates produced by pose_stack_from_atom_array:\n  "
        + "\n  ".join(head)
        + tail
    )


@_validate_args
def atom_array_from_pose_stack(
    pose_stack: PoseStack,
    co: CanonicalOrdering | None = None,
    merge_fragments: bool = True,
    include_virtual_atoms: bool = False,
) -> biotite.structure.AtomArray | biotite.structure.AtomArrayStack:
    """Convert a PoseStack to an AtomArray or AtomArrayStack.

    Args:
        pose_stack: Pose stack to convert.
        co: Canonical ordering used for conversion. Provide the ordering that
            was used when ligands or custom residue types are present. By
            default it is built from the pose's own chemical database when
            the pose holds types outside the standard ordering.
        merge_fragments: Restore fragmented ligands to their original residue
            identity. Set to False to keep fragment residues separate.
        include_virtual_atoms: Also write virtual atoms, such as a metal's
            site virtuals. A metal split out of a component is then left as
            its own residue, which its virtuals belong to; otherwise it is put
            back in the component it came from.

    Returns:
        Biotite AtomArray for single-pose or AtomArrayStack for multi-pose,
        with every bond the poses' residue types and connections declare:
        bond orders within residues, and chemical bonds between them, metal
        coordination typed COORDINATION.
    """
    from types import SimpleNamespace
    from tmol.io._chain_deduction import _chain_inds_for_pose_stack
    from tmol_core.io.structure_export import atom_array_from_host_pose

    if co is None:
        co = _canonical_ordering_for_pose_stack(pose_stack)
    host = SimpleNamespace(
        coords=pose_stack.coords.detach().cpu().numpy(),
        block_type_ind64=pose_stack.block_type_ind64.cpu().numpy(),
        block_coord_offset=pose_stack.block_coord_offset.cpu().numpy(),
        inter_residue_connections64=pose_stack.inter_residue_connections64.cpu().numpy(),
        packed_block_types=pose_stack.packed_block_types,
        pdb_info=pose_stack.pdb_info,
        split_block_mapping=pose_stack.split_block_mapping,
    )
    return atom_array_from_host_pose(
        host,
        co,
        _chain_inds_for_pose_stack(pose_stack),
        merge_fragments=merge_fragments,
        include_virtual_atoms=include_virtual_atoms,
    )


def _canonical_ordering_for_pose_stack(pose_stack: PoseStack) -> CanonicalOrdering:
    """Default Biotite ordering, or one built from the pose's own chemical database.

    Generated ligand and custom residue types are absent from the default
    ordering, but the pose's packed block types carry the database they came from.
    """
    co = _canonical_ordering_for_biotite()
    known = set(co.restype_io_equiv_classes)
    block_types = pose_stack.packed_block_types.active_block_types
    if all(bt.io_equiv_class in known for bt in block_types):
        return co
    return CanonicalOrdering.from_chemdb(pose_stack.packed_block_types.chem_db)


def _with_metals_rejoined(structure, block_for_atom, origins):
    from tmol_core.io.structure_export import _with_metals_rejoined

    return _with_metals_rejoined(structure, block_for_atom, origins)


def _bonds_from_pose_stack(pose_stack, structure, block_for_atom):
    from types import SimpleNamespace
    from tmol_core.io.structure_export import _bonds_from_pose_stack

    pose = SimpleNamespace(
        block_type_ind64=pose_stack.block_type_ind64.cpu().numpy(),
        inter_residue_connections64=pose_stack.inter_residue_connections64.cpu().numpy(),
        packed_block_types=pose_stack.packed_block_types,
    )
    return _bonds_from_pose_stack(pose, structure, block_for_atom)


from tmol_core.io.canonical_atom_mapping import (  # noqa: E402,F401
    map_atoms_to_canonical as _map_atoms_to_canonical,
    map_neutral_terminal_hydrogens,
)


def _renumbered_for_cif(structure):
    from tmol_core.io.structure_export import _renumbered_for_cif

    return _renumbered_for_cif(structure)


_without_metal_coordination_bonds = _structure_context._without_metal_coordination_bonds


_metal_atom_names = _structure_context._metal_atom_names


_chelated_metals = _structure_context._chelated_metals


# per atom: "res_id<TAB>ins_code<TAB>res_name<TAB>atom_name" of the component
#    a split-out metal came from; empty elsewhere

# input annotations a pose keeps per residue and writes back on export
RESIDUE_ANNOTATIONS = ("hetero", "is_polymer", "chain_type")


_residue_annotations = _structure_context._residue_annotations


_metal_origins = _structure_context._metal_origins


_with_peptide_tautomers = _structure_context._with_peptide_tautomers


_with_input_chemistry_normalized = _structure_context._with_input_chemistry_normalized


_with_chelated_metals_split = _structure_context._with_chelated_metals_split


_ion_elements = _structure_context._ion_elements


_without_chelated_metals = _structure_context._without_chelated_metals


def _bonds_for_poses(bonds, n_poses, torch_device):
    """Prefix each bond row with its pose index."""
    from tmol_core.io.canonical_atom_array import bonds_for_poses

    values = bonds_for_poses(bonds, n_poses)
    return None if values is None else torch.as_tensor(values, device=torch_device)


_res_names_for_structure = _structure_context._res_names_for_structure


_validate_filtered_covalent_partners = (
    _structure_context._validate_filtered_covalent_partners
)


_filter_supported_atoms_and_connectivity = (
    _structure_context._filter_supported_atoms_and_connectivity
)


def _break_polymer_gaps(
    not_connected, bonds, inferred, coords, restypes, chain_id, co, threshold
):
    from tmol_core.io.polymer_gaps import break_polymer_gaps

    return break_polymer_gaps(
        not_connected,
        bonds,
        inferred,
        coords.cpu().numpy(),
        restypes,
        chain_id,
        co,
        threshold,
    )


from tmol_core.io.polymer_gaps import (  # noqa: E402,F401
    orient_polymer_gap_flags as _orient_polymer_gap_flags,
)

_extract_residue_metadata = _structure_context._extract_residue_metadata


def _populate_canonical_coords(
    biotite_structure: biotite.structure.AtomArray | biotite.structure.AtomArrayStack,
    torch_device: torch.device,
    co: CanonicalOrdering,
    biotite_residues,
    valid_atom_mask,
    valid_res_inds,
    valid_atom_inds,
):
    n_poses = 1
    if isinstance(biotite_structure, biotite.structure.AtomArrayStack):
        n_poses = biotite_structure.coord.shape[0]

    tmol_coords = torch.full(
        (n_poses, len(biotite_residues), co.max_n_canonical_atoms, 3),
        numpy.nan,
        dtype=torch.float32,
        device=torch_device,
    )
    biotite_coords = torch.as_tensor(biotite_structure.coord, device=torch_device)
    if biotite_coords.ndim == 2:
        biotite_coords = biotite_coords.unsqueeze(0)
    tmol_coords[:, valid_res_inds, valid_atom_inds] = biotite_coords[:, valid_atom_mask]
    return tmol_coords, n_poses


def _validate_atom37_coords(
    atom37_coords: torch.Tensor, torch_device: torch.device
) -> None:
    """Validate the tensor contract shared by direct and prepared adapters."""
    if atom37_coords.ndim != 4 or atom37_coords.shape[-2:] != (37, 3):
        raise ValueError(
            "atom37_coords must have shape [n_poses, n_tokens, 37, 3]; "
            f"got {tuple(atom37_coords.shape)}"
        )
    if atom37_coords.dtype != torch.float32:
        raise TypeError(
            "atom37_coords must have dtype torch.float32; " f"got {atom37_coords.dtype}"
        )
    if atom37_coords.device != torch_device:
        raise ValueError(
            f"atom37_coords is on '{atom37_coords.device}' but torch_device is "
            f"'{torch_device}'; they must match"
        )


def _validate_mapped_atom37_triplets(
    source_coords: torch.Tensor,
    mapped_token_id: torch.Tensor,
    mapped_slot: torch.Tensor,
) -> None:
    """Reject mapped triplets that are neither wholly finite nor all-NaN."""
    has_infinity = torch.isinf(source_coords).any(dim=-1)
    nan_count = torch.isnan(source_coords).sum(dim=-1)
    partial_nan = (nan_count > 0) & (nan_count < 3)
    malformed = has_infinity | partial_nan
    if not bool(torch.any(malformed)):
        return

    details = []
    for pose, mapped_atom in (
        torch.nonzero(malformed, as_tuple=False).cpu().tolist()[:20]
    ):
        reason = (
            "contains infinity" if has_infinity[pose, mapped_atom] else "partial NaN"
        )
        details.append(
            f"pose={pose} token_id={int(mapped_token_id[mapped_atom])} "
            f"atom37_slot={int(mapped_slot[mapped_atom])} ({reason})"
        )
    count = int(torch.count_nonzero(malformed))
    tail = f"; ... and {count - 20} more" if count > 20 else ""
    raise Atom37MappingError(
        "Malformed mapped Atom37 coordinate triplet(s): "
        + "; ".join(details)
        + tail
        + ". Each mapped triplet must be wholly finite or exactly "
        "[NaN, NaN, NaN] for a missing atom."
    )


def _validate_effective_mainchain_coords(
    canonical_coords: torch.Tensor,
    required_mainchain_entries: tuple[tuple[int, int, str], ...],
) -> None:
    """Require every pose's effective mainchain coordinates to be finite."""
    if not required_mainchain_entries:
        return

    # Gather every (pose, entry) at once. Testing one entry at a time costs a
    # device synchronization each, which otherwise dominates repeated replay.
    device = canonical_coords.device
    residues = torch.tensor(
        [entry[0] for entry in required_mainchain_entries],
        dtype=torch.int64,
        device=device,
    )
    atoms = torch.tensor(
        [entry[1] for entry in required_mainchain_entries],
        dtype=torch.int64,
        device=device,
    )
    absent = ~torch.isfinite(canonical_coords[:, residues, atoms]).all(dim=-1)
    if not bool(absent.any()):
        return

    # Entry-major, pose-ascending: the order the per-entry loop reported.
    missing = [
        f"pose={pose} residue={required_mainchain_entries[entry][2]}"
        for entry, pose in torch.nonzero(absent.t(), as_tuple=False).cpu().tolist()
    ]

    head = missing[:20]
    tail = f"; ... and {len(missing) - 20} more" if len(missing) > 20 else ""
    raise Atom37MappingError(
        "Required mainchain coordinates are missing from both usable "
        "Atom37/Biotite coordinate sources: "
        + "; ".join(head)
        + tail
        + ". Supply a wholly finite mapped Atom37 triplet or, for an unmapped "
        "atom, a finite Biotite reference coordinate. Mapped all-NaN triplets "
        "are missing and never fall back to the reference."
    )


def _populate_canonical_coords_from_atom37(
    atom37_coords: torch.Tensor,
    biotite_structure: biotite.structure.AtomArray | biotite.structure.AtomArrayStack,
    torch_device: torch.device,
    co: CanonicalOrdering,
    biotite_residues,
    valid_atom_mask: numpy.ndarray,
    valid_res_inds: numpy.ndarray,
    valid_atom_inds: numpy.ndarray,
) -> tuple[torch.Tensor, int]:
    """Overlay mapped atom37 coordinates on the Biotite canonical coordinates.

    Mapped tensor values replace their matching Biotite atoms through one
    differentiable indexed assignment. Wholly finite triplets are authoritative
    and all-NaN triplets are missing; only unmapped atoms retain reference
    coordinates. Partial-NaN and infinite mapped triplets are rejected.

    Returns:
        The canonical coordinate tensor and its pose count.
    """
    _validate_atom37_coords(atom37_coords, torch_device)
    token_id, slot, mapped_res_inds, mapped_atom_inds = _atom37_mapping(
        biotite_structure,
        valid_atom_mask,
        valid_res_inds,
        valid_atom_inds,
        atom37_coords.shape[1],
    )

    reference_coords, reference_n_poses = _populate_canonical_coords(
        biotite_structure,
        torch_device,
        co,
        biotite_residues,
        valid_atom_mask,
        valid_res_inds,
        valid_atom_inds,
    )
    n_poses = atom37_coords.shape[0]
    if reference_n_poses not in (1, n_poses):
        raise ValueError(
            f"Biotite structure has {reference_n_poses} poses but atom37_coords "
            f"has {n_poses}"
        )
    if reference_n_poses == 1 and n_poses != 1:
        reference_coords = reference_coords.expand(n_poses, -1, -1, -1).clone()

    mapped_token_id = torch.as_tensor(token_id, device=torch_device)
    mapped_slot = torch.as_tensor(slot, device=torch_device)
    source_coords = atom37_coords[:, mapped_token_id, mapped_slot]
    _validate_mapped_atom37_triplets(source_coords, mapped_token_id, mapped_slot)
    mapped_res_inds = torch.as_tensor(mapped_res_inds, device=torch_device)
    mapped_atom_inds = torch.as_tensor(mapped_atom_inds, device=torch_device)
    reference_coords[:, mapped_res_inds, mapped_atom_inds] = source_coords
    return reference_coords, n_poses


_normalize_input_identifiers = _structure_context._normalize_input_identifiers


def _with_input_hydrogens(biotite_structure, ph, co, chemdb, find_metal_coordination):
    return _structure_context._with_input_hydrogens(
        biotite_structure,
        ph,
        co,
        chemdb,
        find_metal_coordination,
        detect_metal_bonds=_detected_metal_bonds,
        default_canonical_ordering=_canonical_ordering_for_biotite,
    )


def _detected_metal_bonds(biotite_structure, co, chemdb):
    """Canonical storage conversion around common metal-input chemistry."""
    from types import SimpleNamespace

    template = _template_array(biotite_structure)
    known = numpy.isin(
        template.res_name, [*co.restype_io_equiv_classes, *co.name3_aliases]
    )
    cf = canonical_form_from_biotite(
        biotite_structure if known.all() else template[known],
        torch.device("cpu"),
        co=co,
    )
    fields = (
        "res_types",
        "coords",
        "metal_sites",
        "metal_coordination",
        "disulfides",
        "chain_labels",
        "res_labels",
        "residue_insertion_codes",
    )
    values = {name: getattr(cf, name) for name in fields}
    host = SimpleNamespace(
        **{
            name: (
                value.detach().cpu().numpy()
                if isinstance(value, torch.Tensor)
                else value
            )
            for name, value in values.items()
        }
    )
    return _structure_context.detected_metal_bonds_for_canonical(
        biotite_structure, co, chemdb, host
    )


def canonical_form_from_biotite(
    biotite_structure: biotite.structure.AtomArray | biotite.structure.AtomArrayStack,
    torch_device: torch.device,
    co: CanonicalOrdering | None = None,
    missing_density_distance_threshold: float = 2.4,
    atom37_coords: torch.Tensor | None = None,
    _filter_missing_mainchain: bool = True,
    _cut_covalent_partners: frozenset[str] = frozenset(),
    _chemical_database=None,
) -> CanonicalForm:
    """Convert a Biotite AtomArray or AtomArrayStack to a CanonicalForm.

    This function bridges between Biotite's data structures and tmol's internal
    representation by converting atom and residue information from string-based
    identifiers to tmol's canonical integer-based indexing system.

    Args:
        biotite_structure: A Biotite AtomArray (single structure) or
            AtomArrayStack (multiple structures) containing the molecular data.
            Must contain atom coordinates, residue names, atom names, chain IDs,
            and optionally B-factors and occupancy values.
        torch_device: PyTorch device (e.g., torch.device('cuda') or torch.device('cpu'))
            where the resulting tensors should be allocated.
        co: A CanonicalForm in case you want to use a non-default database (and thus may need
            a different mapping)
        missing_density_distance_threshold: Maximum distance in angstroms for
            treating a polymer gap as missing density rather than a chain break.
        atom37_coords: Optional autograd-tracked coordinate tensor
            of shape [n_poses, n_tokens, 37, 3]. When provided, coordinates are
            sourced from this tensor (routed by the ``token_id`` and ``atom37_slot``
            annotations on ``biotite_structure``); all-NaN mapped triplets are
            treated as missing while unmapped finite reference atoms remain
            context. The geometry-based missing-density check is skipped so the
            topology stays fixed and gradients flow. See
            :func:`~tmol.io.pose_stack_from_atom37_and_topology`.

    Returns:
        CanonicalForm: A data structure containing:
            - chain_id: Tensor mapping residues to chain indices
            - res_types: Tensor mapping residues to tmol residue type indices
            - coords: 4D tensor of atomic coordinates (poses x residues x atoms x 3)
            - res_labels: Original residue sequence numbers from the structure
            - residue_insertion_codes: PDB insertion codes for residues
            - chain_labels: Original chain identifiers from the structure
            - atom_occupancy: Optional tensor of atom occupancy values
            - atom_b_factor: Optional tensor of atom B-factor values
            - disulfides: Explicit cysteine sulfur bonds, including unresolved SG
            - res_not_connected: Tensor describing whether two consecutive residues
              should be treated as chemically bonded.

    """
    torch_device = resolve_device(torch_device)
    if co is None:
        co = _canonical_ordering_for_biotite()

    if atom37_coords is not None:
        _validate_atom37_coords(atom37_coords, torch_device)

    from tmol_core.io.canonical_atom_array import prepare_canonical_atom_array

    plan = prepare_canonical_atom_array(
        biotite_structure,
        co,
        missing_density_distance_threshold=missing_density_distance_threshold,
        filter_missing_mainchain=_filter_missing_mainchain,
        cut_partners=_cut_covalent_partners,
        atom37_n_poses=None if atom37_coords is None else atom37_coords.shape[0],
        chemical_database=_chemical_database,
    )
    coordinates = None
    if atom37_coords is not None:
        coordinates, _ = _populate_canonical_coords_from_atom37(
            atom37_coords,
            plan.structure,
            torch_device,
            co,
            plan.residue_names,
            plan.valid_atom_mask,
            plan.valid_res_inds,
            plan.valid_atom_inds,
        )
        _validate_effective_mainchain_coords(
            coordinates, _required_mainchain_entries(plan.structure, co)
        )
    return _canonical_form_from_host_plan(plan, torch_device, coordinates)


def _canonical_form_from_host_plan(plan, torch_device, coordinates=None):
    """Place one common canonical plan, retaining an optional differentiable overlay."""
    fields = plan.fields.copy()
    fields["coords"] = (
        torch.as_tensor(fields["coords"], device=torch_device)
        if coordinates is None
        else coordinates
    )
    for name in (
        "chain_id",
        "res_types",
        "res_not_connected",
        "disulfides",
        "covalent_bonds",
        "metal_coordination",
        "protonation_variants",
    ):
        if fields[name] is not None:
            fields[name] = torch.as_tensor(fields[name], device=torch_device)
    return CanonicalForm(**fields)


@toolz.functoolz.memoize
def _paramdb_for_biotite() -> ParameterDatabase:
    """For Biotite, let's just get the default param DB.
    We shouldn't need a subset since we're mapping from strings(?)"""

    return ParameterDatabase.get_default()


@toolz.functoolz.memoize
def _restype_set_for_biotite() -> ResidueTypeSet:
    paramdb = _paramdb_for_biotite()
    return ResidueTypeSet.from_database(paramdb.chemical)


@_validate_args
@toolz.functoolz.memoize
def _canonical_ordering_for_biotite() -> CanonicalOrdering:
    """Construct the CanonicalOrdering object to use for Biotite.
    This wont be used as a typical CanonicalOrdering object, since
    we aren't mapping from int-to-int, and instead are going from
    string-to-int.
    """

    paramdb = _paramdb_for_biotite()
    return CanonicalOrdering.from_chemdb(paramdb.chemical)


@_validate_args
@toolz.functoolz.memoize
def _packed_block_types_for_biotite(device: torch.device) -> PackedBlockTypes:
    """Construct the PackedBlockTypes (PBT) object that will used for Biotite.
    We'll use the defaults since anything might show up in a Biotite AtomArray.
    Some things may show up in the AtomArrays that are not handled by this
    PBT, but that is work for the future.
    """

    restype_set = _restype_set_for_biotite()

    # metal-only types widen every pose packed with them; see default_packed_block_types
    active = [
        rt for rt in restype_set.residue_types if not _only_coordinates_a_metal(rt)
    ]

    return PackedBlockTypes.from_restype_list(
        restype_set.chem_db, restype_set, active, device
    )


@_validate_args
@toolz.functoolz.memoize
def _packed_block_types_for_biotite_with_metals(
    device: torch.device,
) -> PackedBlockTypes:
    """The Biotite packed block types, plus the ions and clusters a structure's
    metals need whether or not anything coordinates them."""
    restype_set = _restype_set_for_biotite()

    return PackedBlockTypes.from_restype_list(
        restype_set.chem_db, restype_set, restype_set.residue_types, device
    )


def _default_pose_build_context_for(
    structure, device: torch.device
) -> PoseBuildContext:
    """The process-wide context, packed wide enough for what this structure holds."""
    return _default_pose_build_context(device, bool(_is_metal(structure).any()))


def _restype_set_sharing_default(chemical_db) -> ResidueTypeSet:
    """chemical_db's residue types, reusing the default set's objects (and their
    cached annotations) where chemical_db extends the default database."""
    try:
        return _restype_set_for_biotite().extended(chemical_db)
    except ValueError:
        return ResidueTypeSet.from_database(chemical_db)


@_validate_args
@toolz.functoolz.memoize
def _default_pose_build_context(
    device: torch.device, with_metals: bool
) -> PoseBuildContext:
    """Return the process-wide construction context for the default database."""
    packed = (
        _packed_block_types_for_biotite_with_metals
        if with_metals
        else _packed_block_types_for_biotite
    )
    return PoseBuildContext(
        canonical_ordering=_canonical_ordering_for_biotite(),
        packed_block_types=packed(device),
        parameter_database=_paramdb_for_biotite(),
        restype_set=_restype_set_for_biotite(),
    )


def _derived_types_for_param_db(
    param_db: ParameterDatabase, device: torch.device
) -> tuple[CanonicalOrdering, ResidueTypeSet, PackedBlockTypes]:
    """Build canonical ordering and packed block types from a DB."""
    co = CanonicalOrdering.from_chemdb(param_db.chemical)
    rts = _restype_set_sharing_default(param_db.chemical)
    pbt = PackedBlockTypes.from_restype_list(
        rts.chem_db, rts, rts.residue_types, device
    )
    return co, rts, pbt


@_validate_args
def _biotite_from_canonical_form(
    cf: CanonicalForm,
    co: CanonicalOrdering | None = None,
    include_virtual_atoms: bool = False,
) -> biotite.structure.AtomArray | biotite.structure.AtomArrayStack:
    """Export coordinates and chemical/author labels using a shared atom layout.

    Multi-model arrays require identical residue and atom annotations. Their atom
    layout is the union of resolved atoms; absent coordinates remain NaN. Missing
    author labels default to internal chain IDs and one-based residue positions.
    Virtual atoms are left out unless ``include_virtual_atoms``. This host-array
    export detaches coordinates from autograd. A canonical form carries no
    residue types, so the result has no bonds; atom_array_from_pose_stack adds them.
    """
    return _biotite_from_canonical_form_impl(cf, co, include_virtual_atoms)[0]


def _biotite_from_canonical_form_impl(cf, co, include_virtual_atoms):
    from tmol_core.io.structure_export import _biotite_from_canonical_form_impl

    return _biotite_from_canonical_form_impl(
        _host_export_canonical_form(cf),
        co if co is not None else _canonical_ordering_for_biotite(),
        include_virtual_atoms,
    )


def _poses_have_identical_metadata(cf):
    from tmol_core.io.structure_export import _poses_have_identical_metadata

    return _poses_have_identical_metadata(_host_export_canonical_form(cf))


def _host_export_canonical_form(cf):
    """Detach only the arrays needed by the common host structure exporter."""
    from types import SimpleNamespace

    names = (
        "coords",
        "res_types",
        "chain_id",
        "res_labels",
        "chain_labels",
        "residue_insertion_codes",
        "atom_occupancy",
        "atom_b_factor",
        "metal_origins",
        "residue_annotations",
    )
    return SimpleNamespace(
        **{
            name: (
                value.detach().cpu().numpy()
                if isinstance(value, torch.Tensor)
                else value
            )
            for name in names
            for value in (getattr(cf, name),)
        }
    )
