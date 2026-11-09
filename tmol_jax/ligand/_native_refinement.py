"""CPU library selection for the common host ligand refinement executor."""

from functools import lru_cache


@lru_cache(maxsize=1)
def native_conformer_execution():
    """Return the explicit native CPU candidate without importing Torch.

    Pass this executor through ``conformer_execution`` during preparation.
    Parameter preparation is eager host chemistry; JAX scoring of the prepared
    pose retains its independent CPU/CUDA execution. Default adoption requires
    complete molecular and workflow validation, not pointwise kernel agreement.
    """
    from tmol_core.ligand._native_refinement import NativeRefinementBackend
    from tmol_core.ligand.native_refinement import NativeConformerExecution
    from tmol_jax_native import library_path

    path = library_path("cpu")
    if path is None:
        raise RuntimeError("native ligand refinement requires the CPU native library")
    return NativeConformerExecution(NativeRefinementBackend(path))
