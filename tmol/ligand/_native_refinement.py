"""Explicit CPU refinement executor using the installed TMol native library."""

from functools import lru_cache

from tmol_core.ligand._native_refinement import NativeRefinementBackend
from tmol_core.ligand.native_refinement import NativeConformerExecution


@lru_cache(maxsize=1)
def native_conformer_execution():
    from tmol import _cpp_lib

    _cpp_lib._ensure_loaded()
    return NativeConformerExecution(NativeRefinementBackend(_cpp_lib._native.__file__))
