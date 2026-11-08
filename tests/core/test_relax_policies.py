"""Protocol invariants independent of either tensor framework."""

from types import SimpleNamespace

import numpy as np
import pytest

from tmol_core.optimization.segmented import SegmentedLBFGSPolicy
from tmol_core.optimization.setup import LBFGS_DEFAULTS
from tmol_core.pack.annealing import annealer_trajectory_counts
from tmol_core.relax import (
    DEFAULT_RELAX_SCHEDULE,
    KINEMATIC_RELAX_LBFGS_OPTIONS,
    _normalize_schedule,
    relax_pack_min_step,
)


@pytest.mark.parametrize(
    "high,cpu,gpu",
    [
        (4, (10, 2), (10, 2)),
        (5, (12, 3), (10, 2)),
        (127, (317, 79), (310, 77)),
        (128, (320, 80), (320, 80)),
    ],
)
def test_trajectory_counts_preserve_backend_rounding(high, cpu, gpu):
    assert annealer_trajectory_counts(high, cpu=True) == (high, *cpu)
    assert annealer_trajectory_counts(high) == (high, *gpu)


@pytest.mark.parametrize("count", [0, 3, -1, 2**31])
def test_invalid_trajectory_counts_fail_before_allocation(count):
    with pytest.raises(ValueError):
        annealer_trajectory_counts(count)


class ConvergenceExecutor(SegmentedLBFGSPolicy):
    """Only reductions/storage needed to exercise the real shared policy."""

    _zeros_like = staticmethod(np.zeros_like)
    _abs = staticmethod(np.abs)
    _seg_amax = staticmethod(lambda x: np.max(x, axis=1))
    _where = staticmethod(np.where)
    _count_true = staticmethod(np.count_nonzero)
    _copy = staticmethod(lambda old, new: new.copy())
    verbose = False
    _n_segments = 4


def test_gradient_gate_requires_energy_patience_except_failed_search():
    # Four simultaneous segments: high gradient, insufficient patience,
    # satisfied patience, and a failed line search with small gradient.
    c = SimpleNamespace(
        fixed_iterations=False,
        gradtol=1.0,
        atol=0.01,
        rtol=None,
        patience=3,
        flat_grad=np.array([[2.0], [0.5], [0.5], [0.5]]),
        loss_vec=np.ones(4),
        prev_loss_vec=np.ones(4),
        small_steps=np.array([2, 0, 2, 0]),
        needs_reset=np.array([False, False, False, True]),
        converged=np.zeros(4, bool),
        stalled=np.zeros(4, bool),
    )
    runner = ConvergenceExecutor()
    assert not runner._check_segment_convergence(c, 4)
    np.testing.assert_array_equal(c.converged, [False, False, True, True])
    c.gradtol = None
    runner._check_segment_convergence(c, 5)
    assert c.converged[0]  # With no gradient gate, energy patience suffices.
    c.fixed_iterations = True
    before = c.small_steps.copy()
    assert runner._check_segment_convergence(c, 6) is False
    np.testing.assert_array_equal(c.small_steps, before)
    assert LBFGS_DEFAULTS["history_size"] == 200


def test_stage_tolerance_and_lifetime_contract():
    assert [step[3] for step in _normalize_schedule(DEFAULT_RELAX_SCHEDULE)] == [
        0.01,
        0.01,
        0.01,
        5e-4,
    ]
    assert _normalize_schedule([0.25])[0][3] is None
    assert KINEMATIC_RELAX_LBFGS_OPTIONS["gradtol"] == 180 / np.pi
    events = []
    execution = SimpleNamespace(
        task=lambda pose, palette: object(),
        pack=lambda pose, score, task, verbose: pose + 1,
        score_type=SimpleNamespace(fa_ljrep="rep", constraint="cst"),
    )
    score = SimpleNamespace(set_weight=lambda term, value: events.append((term, value)))

    class Minimize:
        def __call__(self, pose, score, *, verbose, **kwargs):
            events.append((pose, kwargs))
            return pose + 1

        def release_retained_state(self):
            events.append("released")

    for tol, expected in [(None, {}), (5e-4, {"rtol": 5e-4})]:
        events.clear()
        result = relax_pack_min_step(
            0,
            score,
            None,
            0.04,
            0.051,
            1.0,
            [],
            Minimize(),
            False,
            min_rtol=tol,
            execution=execution,
        )
        assert result == 2
        assert events == [
            ("rep", 0.04),
            ("cst", 1.0),
            ("rep", 0.051),
            (1, expected),
            "released",
        ]
