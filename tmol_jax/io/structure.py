"""JAX structure input over the common chemistry preparation policy."""

from dataclasses import dataclass
from functools import cached_property
from types import SimpleNamespace

import jax
import numpy as np

from tmol_core.io import structure_context as common
from tmol_core.weak_identity_cache import WeakIdentityLRU
from tmol_jax.database import ParameterDatabase
from tmol_jax.io.canonical_ordering import CanonicalOrdering, default_canonical_ordering
from tmol_jax.io.residue_blocks import PoseBuildContext
from tmol_jax.pose.packed_block_types import PackedBlockTypes


@dataclass(frozen=True)
class AtomArrayBuildContext(PoseBuildContext):
    canonical_ordering: object
    fragment_definitions: tuple = ()
    cut_covalent_partners: frozenset = frozenset()

    @cached_property
    def _packing_score_function(self):
        from tmol_jax.score import beta2016_score_function

        return beta2016_score_function(self.parameter_database)

    @cached_property
    def _opth_score_function(self):
        from tmol_jax.score import beta2016_score_function, ScoreType

        score = beta2016_score_function(self.parameter_database)
        from tmol_core.pack.input_completion import OPTH_INVARIANT_TERMS

        for name in OPTH_INVARIANT_TERMS:
            score.set_weight(ScoreType[name], 0)
        return score

    @cached_property
    def _dunbrack_sampler(self):
        from tmol_jax.pack import DunbrackChiSampler

        return DunbrackChiSampler.from_database(self.parameter_database)

    @cached_property
    def _na_sampler(self):
        from tmol_jax.pack import NaChiRotamerSampler

        return NaChiRotamerSampler.from_database(self.parameter_database)


_CONTEXTS = WeakIdentityLRU()


def _device(device):
    if device is None:
        return jax.numpy.asarray(0).device
    if isinstance(device, str):
        platform, _, index = device.partition(":")
        return jax.devices(platform)[int(index or 0)]
    return device


def _detected_metal_bonds(structure, ordering, chemistry):
    from .pose_stack_from_biotite import canonical_form_from_biotite

    template = common._template_array(structure)
    known = np.isin(
        template.res_name, [*ordering.restype_io_equiv_classes, *ordering.name3_aliases]
    )
    cf = canonical_form_from_biotite(
        structure if known.all() else template[known],
        canonical_ordering=ordering,
        prepare_input=True,
    )
    host = SimpleNamespace(
        res_types=cf["res_types"],
        coords=cf["coords"],
        metal_sites=cf.get("metal_sites"),
        metal_coordination=cf.get("metal_coordination"),
        disulfides=cf.get("disulfides"),
        chain_labels=cf["chain_labels"],
        res_labels=cf["res_num"],
        residue_insertion_codes=cf["res_ins_codes"],
    )
    return common.detected_metal_bonds_for_canonical(
        structure, ordering, chemistry, host
    )


def _input_hydrogens(structure, ph, ordering, chemistry, find_metal_coordination):
    return common._with_input_hydrogens(
        structure,
        ph,
        ordering,
        chemistry,
        find_metal_coordination,
        detect_metal_bonds=_detected_metal_bonds,
        default_canonical_ordering=default_canonical_ordering,
    )


def build_context_from_atom_array(
    biotite_structure,
    device=None,
    param_db=None,
    prepare_ligands=False,
    ligand_ph=7.4,
    strict_atom_types=False,
    strict_ligands=True,
    ligand_params_files=None,
    chem_comp_types=None,
    ligand_seed=None,
    ligand_hydrogens="names",
):
    """Prepare reusable chemistry and parameter context for molecular AtomArrays."""
    from tmol_jax.ligand import prepare_ligands as ligand_preparer

    device = _device(device)
    with jax.default_device(device):
        prepared = common.prepare_structure_context(
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
            input_hydrogens=_input_hydrogens,
            ligand_preparer=ligand_preparer,
        )
        database = prepared.parameter_database

        def create():
            ordering = prepared.canonical_ordering or CanonicalOrdering.from_chemdb(
                database.chemical
            )
            return AtomArrayBuildContext(
                database,
                PackedBlockTypes.default(database.chemical),
                ordering,
                prepared.fragment_definitions,
                prepared.cut_covalent_partners,
            )

        if prepared.use_default_context:
            return _CONTEXTS.get_or_create(database, (device,), create)
        return create()


def pose_stack_from_atom_array(
    biotite_structure,
    device=None,
    *,
    param_db=None,
    context=None,
    prepare_ligands=False,
    ligand_ph=7.4,
    strict_atom_types=False,
    strict_ligands=True,
    ligand_params_files=None,
    chem_comp_types=None,
    ligand_seed=None,
    ligand_hydrogens="names",
    no_optH=False,
    packer_seed=None,
    return_context=False,
    missing_density_distance_threshold=2.4,
    find_additional_metal_coordination=True,
    **kwargs,
):
    """Build a molecular pose using shared normalization, chemistry and input H.

    Shared sampler policy selects missing-sidechain rebuilding and OptH packing.
    Fragment cut bonds and original identities use the common fragment plan.
    Pass the returned context's parameter database to scoring functions. Input
    preparation is eager.
    """
    from .pose_stack_from_biotite import canonical_form_from_biotite
    from .pose_stack_construction import pose_stack_from_canonical_form

    device = _device(device)
    with jax.default_device(device):
        if context is not None:
            if param_db is not None or ligand_params_files or prepare_ligands:
                raise ValueError(
                    "context already carries prepared chemistry; do not also pass a database or preparation options"
                )
            if context.packed_block_types.n_atoms.device != device:
                raise ValueError("context and requested device must match")
        else:
            if prepare_ligands:
                biotite_structure = _input_hydrogens(
                    biotite_structure,
                    ligand_ph,
                    None,
                    (param_db or ParameterDatabase.get_default()).chemical,
                    find_additional_metal_coordination,
                )
            context = build_context_from_atom_array(
                biotite_structure,
                device,
                param_db,
                prepare_ligands,
                ligand_ph,
                strict_atom_types,
                strict_ligands,
                ligand_params_files,
                chem_comp_types,
                ligand_seed,
                ligand_hydrogens,
            )
        structure = _input_hydrogens(
            biotite_structure,
            ligand_ph,
            context.canonical_ordering,
            context.parameter_database.chemical,
            find_additional_metal_coordination,
        )
        fragment_mapping = None
        if context.fragment_definitions:
            from tmol_core.ligand._fragmentation import _expand_fragmented_ligands

            structure, fragment_mapping = _expand_fragmented_ligands(
                structure, context.fragment_definitions
            )
        cf = canonical_form_from_biotite(
            structure,
            canonical_ordering=context.canonical_ordering,
            missing_density_distance_threshold=missing_density_distance_threshold,
            prepare_input=True,
            cut_partners=context.cut_covalent_partners,
        )
        cf["canonical_ordering"] = context.canonical_ordering
        cf["find_additional_metal_coordination"] = find_additional_metal_coordination
        # These are shared discrete chemistry inputs, not constructor options.
        for name in (
            "find_additional_disulfides",
            "find_additional_cyclic_closures",
            "protonation_variants",
            "metal_sites",
            "require_all_metal_donors",
        ):
            if name in kwargs:
                cf[name] = kwargs.pop(name)
        from .input_completion import input_packer_atoms

        pose, extra = pose_stack_from_canonical_form(
            cf,
            context.packed_block_types,
            return_block_has_missing_atoms=True,
            packer_atoms=lambda rt: input_packer_atoms(context)(rt),
            compact_padding=True,
            build_missing_context=True,
            ligand_hydrogens=ligand_hydrogens,
            **kwargs,
        )
        if fragment_mapping is not None:
            from tmol_jax.ligand.fragment_imposition import apply_fragment_connections

            pose = apply_fragment_connections(pose, fragment_mapping)
        from .input_completion import complete_input

        pose = complete_input(
            pose,
            extra["block_has_missing_atoms"],
            context,
            no_optH=no_optH,
            seed=packer_seed,
        )
        return (pose, context) if return_context else pose


def pose_stack_from_file(
    source,
    device=None,
    format=None,
    *,
    model=1,
    assembly_id=None,
    residue_start=None,
    residue_end=None,
    **kwargs,
):
    """Read molecular files with the common reader and build a JAX pose.

    Remaining options are passed to :func:`pose_stack_from_atom_array`.
    Residue bounds select a half-open range using the file's physical order.
    """
    import biotite.structure as struc
    from tmol_core.io import atom_array_from_file

    array = atom_array_from_file(source, format, model=model, assembly_id=assembly_id)
    if residue_start is not None or residue_end is not None:
        boundaries = struc.get_residue_starts(array, add_exclusive_stop=True)
        start, stop, _ = slice(residue_start, residue_end).indices(len(boundaries) - 1)
        array = array[boundaries[start] : boundaries[stop]]
    return pose_stack_from_atom_array(array, device, **kwargs)
