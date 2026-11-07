"""One native host executor retains the complete shared ligand workflow contract."""

from importlib import import_module

import pytest

from .test_ligand_preparation import (
    test_public_ligand_preparation_scoring_and_minimization as _check_preparation,
)


@pytest.mark.parametrize("source", ("l_alanine", "d_alanine", "aspirin", "carboxylate"))
def test_native_preparation_custom_database_and_molecular_science(backend, source):
    frontend = "tmol_jax" if backend.framework == "jax" else "tmol"
    execution = import_module(frontend + ".ligand._native_refinement")
    _check_preparation(backend, source, execution.native_conformer_execution())
