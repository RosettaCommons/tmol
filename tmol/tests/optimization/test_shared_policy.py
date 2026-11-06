"""Independent convergence invariants with both framework imports prohibited."""

import subprocess
import sys
from pathlib import Path


def test_common_convergence_preserves_pending_resets_without_frameworks():
    root = Path(__file__).resolve().parents[3]
    code = r"""
import sys
class BlockFrameworks:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'jax', 'tmol'):
            raise ImportError('framework import prohibited: ' + fullname)
sys.meta_path.insert(0, BlockFrameworks())
import numpy as np
from types import SimpleNamespace
from tmol_core.optimization.segmented import SegmentedLBFGSPolicy
from tmol_core.optimization.armijo import armijo_linesearch_segmented
from tmol_core.optimization.lbfgs import compact_lbfgs_direction
class Execution(SegmentedLBFGSPolicy):
    verbose = False
    _n_segments = 3
    _zeros_like = staticmethod(np.zeros_like)
    _where = staticmethod(np.where)
    _abs = staticmethod(np.abs)
    _copy = staticmethod(lambda old, new: new)
    _count_true = staticmethod(np.count_nonzero)
    _seg_amax = staticmethod(lambda grad: grad.max(axis=1))
ctx = SimpleNamespace(
    fixed_iterations=False, converged=np.zeros(3, bool), stalled=np.zeros(3, bool),
    needs_reset=np.array([False, True, False]), small_steps=np.array([1, 1, 0]),
    flat_grad=np.array([[2., 1.], [2., 1.], [1e-8, 1e-9]]),
    gradtol=1e-6, prev_loss_vec=np.array([2., 5., 7.]),
    loss_vec=np.array([2., 5., 3.]), atol=1e-4, rtol=None, patience=2,
)
p = Execution()
assert not p._check_segment_convergence(ctx, 1)
assert ctx.converged.tolist() == [True, False, True]
assert ctx.small_steps.tolist() == [2, 1, 0]  # Rejected step must not count.
ctx.needs_reset[:] = False
assert p._check_segment_convergence(ctx, 2)
ctx.fixed_iterations = True
assert not p._check_segment_convergence(ctx, 3)
assert not any(m.split('.')[0] in ('torch', 'jax', 'tmol') for m in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=root, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
