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
from tmol_core.pack.conjugated import anchor_library_rows
kind = NS(name='custom', torsions=[NS(name='linkage'), NS(name='chi1')])
indices = np.array([[0,-1],[0,0]])
requests = torsion_requests((kind,), indices, [0,1], [0,1], ['chi1','linkage'])
assert requests == [(0,0,1,'chi1'),(1,1,0,'linkage')]
values = format_named_torsions(requests,np.array([2.,3.]),indices.shape,[0,1],[0,1],None)
assert values == [[{'chi1':2.},{}],[{}, {'linkage':3.}]]
assert len(torsion_requests((kind,),indices)) == 6
# Original anchors retain all repeated library samples; design types do not leak
# into the source rows, including the same block index in another pose.
anchors = [NS(pose=0,anchor=1), NS(pose=1,anchor=1), NS(pose=0,anchor=0)]
rows = anchor_library_rows(anchors, np.array([[0,2],[0,3]]),
    np.array([0,0,1,1]), np.ones(4,int), np.array([2,9,3,8]),
    np.array([3,1,0,2,0,2]))
assert set(rows) == {(0,1),(1,1)}
np.testing.assert_array_equal(rows[0,1], [2,4])
np.testing.assert_array_equal(rows[1,1], [3,5])
assert not any(name in sys.modules for name in ('torch','jax','tmol','tmol_jax'))
"""
    source = str(Path(__file__).parents[2])
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        cwd=source,
        env={**os.environ, "PYTHONPATH": source},
    )
