"""``tmol_jax.pose`` -- PoseStack, PackedBlockTypes, and builders (JAX port)."""

from __future__ import annotations

import importlib

_EXPORTS = {
    "get_named_torsions": "tmol_jax.pose.query",
    "get_torsion_names": "tmol_jax.pose.query",
    "ConstraintSet": "tmol_jax.pose.constraint_set",
    "PoseStack": "tmol_jax.pose.pose_stack",
    "PackedBlockTypes": "tmol_jax.pose.packed_block_types",
    "PoseStackBuilder": "tmol_jax.pose.pose_stack_builder",
    "PDBInfo": "tmol_jax.pose.pdb_info",
}


def __getattr__(name):
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module 'tmol_jax.pose' has no attribute {name!r}")
    return getattr(importlib.import_module(module_path), name)


__all__ = list(_EXPORTS)
