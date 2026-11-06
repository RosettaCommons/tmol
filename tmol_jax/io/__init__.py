"""Protein structure loading and writing."""

from .pdb_parsing import parse_pdb, pose_stack_from_pdb
from .canonical_ordering import (
    CanonicalOrdering,
    default_canonical_ordering,
    default_packed_block_types,
    canonical_form_from_pdb,
)
from .pose_stack_construction import pose_stack_from_canonical_form
from .write_pose_stack_pdb import atom_records_from_pose_stack, write_pose_stack_pdb
from .pose_stack_from_biotite import pose_stack_from_biotite
from .residue_blocks import (
    PoseBuildContext,
    ResidueBlock,
    pose_stack_from_residue_blocks,
)

__all__ = [
    "parse_pdb",
    "pose_stack_from_pdb",
    "pose_stack_from_biotite",
    "pose_stack_from_residue_blocks",
    "PoseBuildContext",
    "ResidueBlock",
    "CanonicalOrdering",
    "default_canonical_ordering",
    "default_packed_block_types",
    "canonical_form_from_pdb",
    "pose_stack_from_canonical_form",
    "atom_records_from_pose_stack",
    "write_pose_stack_pdb",
]


_STRUCTURE_INPUT = {
    "atom_array_from_pose_stack",
    "build_context_from_atom_array",
    "pose_stack_from_atom_array",
    "pose_stack_from_file",
}
__all__ += sorted(_STRUCTURE_INPUT)


_ATOM37_INPUT = {
    "atom37_slot_map_for_ordering",
    "prepare_atom37_pose_builder",
    "PreparedAtom37PoseBuilder",
}
__all__ += sorted(_ATOM37_INPUT)


def __getattr__(name):
    if name in _ATOM37_INPUT:
        from importlib import import_module

        value = getattr(import_module(".atom37", __name__), name)
        globals()[name] = value
        return value
    if name in _STRUCTURE_INPUT:
        from importlib import import_module

        module = (
            ".structure_export"
            if name == "atom_array_from_pose_stack"
            else ".structure"
        )
        value = getattr(import_module(module, __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(name)
