"""JAX scoring frontend for TMol; no PyTorch runtime dependency."""

from importlib import import_module as _import_module
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tmol-jax")
except PackageNotFoundError:
    __version__ = "0+source"


def has_cuda():
    """Whether the native adapter includes CUDA, independently of JAX devices."""
    from .score._native import libraries

    return "cuda" in libraries()


_LAZY_ATTRS = {}
for _module, _names in (
    (
        "tmol_jax.io",
        (
            "ResidueBlock",
            "pose_stack_from_pdb",
            "pose_stack_from_residue_blocks",
            "extended_pose_stack_from_sequences",
            "atom_array_from_pose_stack",
            "build_context_from_atom_array",
            "pose_stack_from_atom_array",
            "pose_stack_from_file",
            "atom_array_from_file",
            "file_from_atom_array",
            "file_from_pose_stack",
            "PreparedAtom37PoseBuilder",
            "atom37_slot_map_for_ordering",
            "pose_stack_from_atom37",
            "pose_stack_from_atom37_and_topology",
            "prepare_atom37_pose_builder",
            "prepare_atom37_tensor_builder",
            "atomworks_from_pose_stack",
            "pose_stack_from_canonical_aa_atom37",
            "prepare_canonical_aa_atom37_builder",
        ),
    ),
    (
        "tmol_jax.optimization",
        (
            "run_cart_min",
            "run_cart_relax",
            "run_kin_min",
        ),
    ),
    (
        "tmol_jax.kinematics",
        (
            "CartesianMoveMap",
            "EdgeType",
            "FoldForest",
            "MoveMap",
            "PoseStackKinematics",
            "set_named_torsions",
        ),
    ),
    (
        "tmol_jax.database",
        ("ParameterDatabase",),
    ),
    (
        "tmol_jax.chemical.patched_chemdb",
        ("PatchedChemicalDatabase",),
    ),
    (
        "tmol_jax.io.canonical_ordering",
        ("CanonicalOrdering",),
    ),
    (
        "tmol_jax.pack",
        (
            "PackerPalette",
            "PackerTask",
            "pack_rotamers",
            "mutate_residue",
            "mutate_to_sequence",
        ),
    ),
    (
        "tmol_jax.pose",
        (
            "ConstraintSet",
            "PoseStack",
            "PoseStackBuilder",
            "PackedBlockTypes",
            "PDBInfo",
            "get_named_torsions",
            "get_torsion_names",
            "get_sequence",
            "get_ligand_mask",
            "get_polymer_type_mask",
            "metal_blocks",
            "residue_label",
            "atom_label",
            "block_for_residue_label",
            "blocks_for_residue_labels",
            "atom_for_residue_label",
            "constraint_atoms_for_labels",
        ),
    ),
    (
        "tmol_jax.score",
        (
            "ScoreFunction",
            "ScoreType",
            "beta2016_score_function",
        ),
    ),
    (
        "tmol_jax.relax",
        ("fast_relax",),
    ),
    (
        "tmol_jax.score.fragment_interactions",
        (
            "calculate_fragment_interactions",
            "render_fragment_interactions",
        ),
    ),
    (
        "tmol_jax.io.visualize",
        (
            "overlay_view",
            "selection_gallery",
            "switchable_view",
            "view",
        ),
    ),
    (
        "tmol_core.ligand._fragmentation",
        ("LigandFragmentDefinition",),
    ),
    (
        "tmol_jax.ligand.fragment_imposition",
        ("recombine_fragmented_ligands",),
    ),
):
    _LAZY_ATTRS.update(dict.fromkeys(_names, _module))
del _module, _names

__all__ = ["has_cuda", *_LAZY_ATTRS]
_SUBMODULES = {
    "chemical",
    "database",
    "io",
    "kinematics",
    "optimization",
    "pack",
    "pose",
    "relax",
    "score",
}


def __getattr__(name):
    module_name = _LAZY_ATTRS.get(name)
    if module_name is not None:
        value = getattr(_import_module(module_name), name)
    elif name in _SUBMODULES:
        value = _import_module(f"{__name__}.{name}")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__) | _SUBMODULES)
