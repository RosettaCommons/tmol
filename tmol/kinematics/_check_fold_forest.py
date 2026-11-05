"""Compatibility exports for shared compiled fold-forest validation."""

from tmol_core.kinematics.check_fold_forest import (
    _mark_polymeric_bonds_in_foldforest_edges as _mark_polymeric_bonds_in_foldforest_edges,
    _bfs_proper_forest as _bfs_proper_forest,
    _ensure_jumps_numbered_and_distinct as _ensure_jumps_numbered_and_distinct,
    _validate_fold_forest_jit as _validate_fold_forest_jit,
    _append_bad_edge_errors as _append_bad_edge_errors,
    _append_cycle_errors as _append_cycle_errors,
    _append_bad_jumps_errors as _append_bad_jumps_errors,
    _append_unrooted_start_errors as _append_unrooted_start_errors,
    validate_fold_forest as _validate_fold_forest,
)

__all__ = [
    "_mark_polymeric_bonds_in_foldforest_edges",
    "_bfs_proper_forest",
    "_ensure_jumps_numbered_and_distinct",
    "_validate_fold_forest_jit",
    "_append_bad_edge_errors",
    "_append_cycle_errors",
    "_append_bad_jumps_errors",
    "_append_unrooted_start_errors",
    "_validate_fold_forest",
]
