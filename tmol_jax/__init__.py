"""JAX scoring frontend for TMol; no PyTorch runtime dependency."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("tmol-jax")
except PackageNotFoundError:
    __version__ = "0+source"

from .io import ResidueBlock, pose_stack_from_pdb, pose_stack_from_residue_blocks
from .optimization import run_cart_min, run_cart_relax, run_kin_min
from .kinematics import (
    CartesianMoveMap,
    EdgeType,
    FoldForest,
    MoveMap,
    PoseStackKinematics,
)
from .database import ParameterDatabase
from .chemical.patched_chemdb import PatchedChemicalDatabase
from .io.canonical_ordering import CanonicalOrdering
from .pack import PackerPalette, PackerTask, pack_rotamers
from .pose import ConstraintSet, PoseStack, PoseStackBuilder, PackedBlockTypes, PDBInfo
from .score import ScoreFunction, ScoreType, beta2016_score_function
from .pose import get_named_torsions, get_torsion_names
from .relax import fast_relax

from .score.fragment_interactions import (
    calculate_fragment_interactions,
    render_fragment_interactions,
)

__all__ = [
    "calculate_fragment_interactions",
    "render_fragment_interactions",
    "fast_relax",
    "get_named_torsions",
    "get_torsion_names",
    "pose_stack_from_pdb",
    "pose_stack_from_residue_blocks",
    "ResidueBlock",
    "ConstraintSet",
    "PackerPalette",
    "PackerTask",
    "pack_rotamers",
    "ParameterDatabase",
    "PatchedChemicalDatabase",
    "CanonicalOrdering",
    "PoseStack",
    "PoseStackBuilder",
    "PackedBlockTypes",
    "PDBInfo",
    "FoldForest",
    "EdgeType",
    "PoseStackKinematics",
    "MoveMap",
    "CartesianMoveMap",
    "run_cart_min",
    "run_kin_min",
    "run_cart_relax",
    "ScoreFunction",
    "ScoreType",
    "beta2016_score_function",
]


_STRUCTURE_INPUT = {
    "atom_array_from_pose_stack",
    "build_context_from_atom_array",
    "pose_stack_from_atom_array",
    "pose_stack_from_file",
}
__all__ += sorted(_STRUCTURE_INPUT)

_STRUCTURE_FILES = {
    "atom_array_from_file",
    "file_from_atom_array",
    "file_from_pose_stack",
}
__all__ += sorted(_STRUCTURE_FILES)

_FRAGMENT_API = {
    "LigandFragmentDefinition": "tmol_core.ligand._fragmentation",
    "recombine_fragmented_ligands": "tmol_jax.ligand.fragment_imposition",
}
__all__ += sorted(_FRAGMENT_API)


_ATOM37_INPUT = {
    "atom37_slot_map_for_ordering",
    "prepare_atom37_pose_builder",
    "prepare_atom37_tensor_builder",
    "pose_stack_from_atom37",
    "pose_stack_from_atom37_and_topology",
    "PreparedAtom37PoseBuilder",
}
__all__ += sorted(_ATOM37_INPUT)


_ATOMWORKS_INPUT = {
    "prepare_canonical_aa_atom37_builder",
    "pose_stack_from_canonical_aa_atom37",
    "atomworks_from_pose_stack",
}
__all__ += sorted(_ATOMWORKS_INPUT)


def __getattr__(name):
    if name in _FRAGMENT_API:
        from importlib import import_module

        value = getattr(import_module(_FRAGMENT_API[name]), name)
        globals()[name] = value
        return value
    if name in _ATOMWORKS_INPUT | _ATOM37_INPUT | _STRUCTURE_INPUT | _STRUCTURE_FILES:
        from importlib import import_module

        value = getattr(import_module(".io", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(name)
