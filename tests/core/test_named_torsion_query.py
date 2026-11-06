"""The common query policy remains independent of coordinate frameworks."""

import os
from pathlib import Path
import subprocess
import sys


def test_query_selection_and_group_planning_import_without_frameworks():
    code = """
import importlib.abc, sys
class RejectFramework(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('tmol', 'torch', 'tmol_jax', 'jax'):
            raise AssertionError(fullname)
sys.meta_path.insert(0, RejectFramework())
import numpy as np
from types import SimpleNamespace as NS
from tmol_core.pose.query import torsion_requests, format_named_torsions
from tmol_core.pack.conjugated import group_task_rows, group_sample_layout
kind = NS(name='custom', torsions=[NS(name='linkage'), NS(name='chi1')])
indices = np.array([[0,-1],[0,0]])
requests = torsion_requests((kind,), indices, [0,1], [0,1], ['chi1','linkage'])
assert requests == [(0,0,1,'chi1'),(1,1,0,'linkage')]
values = format_named_torsions(requests,np.array([2.,3.]),indices.shape,[0,1],[0,1],None)
assert values == [[{'chi1':2.},{}],[{}, {'linkage':3.}]]
assert len(torsion_requests((kind,),indices)) == 6
assert not any(name in sys.modules for name in ('torch','jax','tmol','tmol_jax'))
"""
    source = str(Path(__file__).parents[2])
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env={**os.environ, "PYTHONPATH": source},
    )
