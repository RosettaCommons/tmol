"""JAX scoring frontend for TMol; no PyTorch runtime dependency."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("tmol-jax")
except PackageNotFoundError:
    __version__ = "0+source"

from .io import ResidueBlock, pose_stack_from_pdb, pose_stack_from_residue_blocks
from .optimization import run_cart_min, run_cart_relax, run_kin_min
from .kinematics import FoldForest, MoveMap, PoseStackKinematics
from .database import ParameterDatabase
from .chemical.patched_chemdb import PatchedChemicalDatabase
from .io.canonical_ordering import CanonicalOrdering
from .pack import PackerPalette, PackerTask, pack_rotamers
from .pose import ConstraintSet, PoseStack, PoseStackBuilder, PackedBlockTypes, PDBInfo
from .score import ScoreFunction, ScoreType, beta2016_score_function
from .pose import get_named_torsions, get_torsion_names
from .relax import fast_relax

__all__ = [
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
    "PoseStackKinematics",
    "MoveMap",
    "run_cart_min",
    "run_kin_min",
    "run_cart_relax",
    "ScoreFunction",
    "ScoreType",
    "beta2016_score_function",
]


_STRUCTURE_INPUT = {
    "build_context_from_atom_array",
    "pose_stack_from_atom_array",
    "pose_stack_from_file",
}
__all__ += sorted(_STRUCTURE_INPUT)


def __getattr__(name):
    if name in _STRUCTURE_INPUT:
        from importlib import import_module

        value = getattr(import_module(".io.structure", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(name)
