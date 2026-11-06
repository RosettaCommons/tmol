"""Host structure chemistry must remain usable without either frontend."""

import os
from pathlib import Path
import subprocess
import sys


def test_protonation_and_terminal_policy_import_without_frameworks():
    root = Path(__file__).resolve().parents[2]
    script = """
import importlib.abc
import sys
class BlockFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'jax', 'tmol'}:
            raise ImportError('framework import: ' + fullname)
sys.meta_path.insert(0, BlockFrameworks())
from tmol_core.chemical.default import default_patched_chemical_database
from tmol_core.io.protonation_input import _database_forms
from tmol_core.io.input_termini import _connection_hydrogen_capacities
database = default_patched_chemical_database()
assert _database_forms(database)
assert _connection_hydrogen_capacities(database)
assert not {'torch', 'jax', 'tmol'}.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=dict(os.environ, PYTHONPATH=str(root)),
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
