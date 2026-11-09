"""``tmol_jax.pose`` -- PoseStack, PackedBlockTypes, and builders (JAX port)."""

from __future__ import annotations

import importlib

_EXPORTS = {
    "select_blocks": "tmol_jax.pose.block_selection",
    "repeat_pose_stack": "tmol_jax.pose.block_selection",
    "get_named_torsions": "tmol_jax.pose.query",
    "get_torsion_names": "tmol_jax.pose.query",
    "ConstraintSet": "tmol_jax.pose.constraint_set",
    "PoseStack": "tmol_jax.pose.pose_stack",
    "PackedBlockTypes": "tmol_jax.pose.packed_block_types",
    "PoseStackBuilder": "tmol_jax.pose.pose_stack_builder",
    "PDBInfo": "tmol_jax.pose.pdb_info",
}

_EXPORTS.update(
    {
        name: "tmol_jax.pose.query"
        for name in (
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
        )
    }
)


def __getattr__(name):
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module 'tmol_jax.pose' has no attribute {name!r}")
    return getattr(importlib.import_module(module_path), name)


__all__ = list(_EXPORTS)
