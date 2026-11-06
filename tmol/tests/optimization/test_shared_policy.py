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


def test_common_setup_preserves_segment_identity_and_checkpoint_history():
    root = Path(__file__).resolve().parents[3]
    code = r"""
import sys
class BlockFrameworks:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'jax', 'tmol'):
            raise ImportError('framework import prohibited: ' + fullname)
sys.meta_path.insert(0, BlockFrameworks())
import numpy as np
from tmol_core.optimization.segmented import SegmentedLBFGSPolicy
from tmol_core.optimization.setup import LBFGS_DEFAULTS, migrate_settings, validate_settings
class Execution(SegmentedLBFGSPolicy):
    verbose = False
    _segment_vector = staticmethod(lambda x: np.asarray(x, dtype=np.int64).reshape(-1))
    _parameter_size = staticmethod(np.size)
    _all_true = staticmethod(np.all)
    _max_index = staticmethod(lambda x: int(np.max(x)))
    _segment_counts = staticmethod(lambda x, n: np.bincount(x, minlength=n))
    _dense_segment_order = staticmethod(lambda x, n, s: np.array_equal(x, np.repeat(np.arange(n), s)))
    _segment_ranks = staticmethod(lambda ids, counts: np.array([np.count_nonzero(ids[:i] == v) for i, v in enumerate(ids)]))
    _empty_like = staticmethod(np.empty_like)
    _clone = staticmethod(np.copy)
    _allocate = staticmethod(lambda shape, ref, kind, zero: (np.zeros if zero else np.empty)(shape, dtype={'real': ref.dtype, 'bool': bool, 'index': np.int64}[kind]))
    _state_cast = staticmethod(lambda x, kind: x.astype(bool if kind == 'bool' else np.int64, copy=False))
    def _setup_parameter(self): return self.x
    def _setup_group(self): return self.group
    def _setup_state(self, param): return self.state
    def _setup_vectors(self, param): return param, np.ones_like(param)
    def _dtype_tolerance(self): return np.finfo(self.x.dtype).eps ** 0.5
p = Execution()
p.x = np.arange(5, dtype=float)
p.group = dict(LBFGS_DEFAULTS, max_iter=3)
p.state = {}
p._last_loss_vec = np.array([1., 2.])
p._init_segments([1, 0, 1, 0, 1])
assert (p._n_segments, p._segment_size, p._segments_are_dense) == (2, 3, False)
assert p._pad_index.tolist() == [3, 0, 4, 1, 5]
ctx = p._step_setup(lambda: 3.)
assert ctx.history_size == 2 and ctx.old_dirs_mat.shape == (2, 2, 3)
assert p.state['grad_pad'].tolist() == [[0., 0., 0.], [0., 0., 0.]]
p.state['history_count'] = 1
p.state['old_dirs_mat'][0] = 7.
p.state['converged'] = np.array([0., 1.])
d = p.state['d']
again = p._step_setup(lambda: 3.)
assert again.d is d and again.old_dirs_mat is ctx.old_dirs_mat
assert again.history_count == 1 and np.all(again.old_dirs_mat[0] == 7.)
assert again.converged.dtype == bool and again.converged.tolist() == [False, True]
assert again.state['func_evals'] == 2
legacy = dict(rtol=None, atol=0., gradtol=None)
migrate_settings(legacy, 0.125)
assert legacy == dict(rtol=0.125, atol=0., gradtol=None, patience=1)
validate_settings(legacy, False)
assert not any(m.split('.')[0] in ('torch', 'jax', 'tmol') for m in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=root, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
