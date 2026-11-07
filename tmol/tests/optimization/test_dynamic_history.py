"""A dynamic circular offset must not change the dense BFGS invariant."""

import numpy as np
import pytest
import torch

from tmol_core.optimization.lbfgs import compact_lbfgs_direction
from tmol.optimization._lbfgs_armijo import _TorchCompactLBFGS


class _DynamicOffset(int):
    def __bool__(self):
        raise AssertionError("dynamic history offset was converted to a host bool")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("layout", ["single", "batched", "einsum"])
def test_dynamic_history_matches_dense_bfgs(torch_device, dtype, layout):
    rng = np.random.default_rng(74)
    poses = 1 if layout == "single" else 2
    steps = rng.normal(size=(5, poses, 6))
    differences = steps * np.arange(1, 7)[None, None, :]
    steps[1, 0] = differences[1, 0] = 0  # An absent slot in only one segment.
    gradient = rng.normal(size=(poses, 6))
    g, y, s = [
        torch.tensor(v, device=torch_device, dtype=dtype)
        for v in (gradient, differences, steps)
    ]
    # The oracle uses the exact stored input values at either coordinate dtype.
    gradient, differences, steps = [v.cpu().double().numpy() for v in (g, y, s)]
    for start in range(5):
        expected = []
        for pose in range(poses):
            inverse = np.eye(6)
            for index in list(range(start, 5)) + list(range(start)):
                sv, yv = steps[index, pose], differences[index, pose]
                if not np.any(sv):
                    continue
                rho = 1 / np.dot(yv, sv)
                left = np.eye(6) - rho * np.outer(sv, yv)
                inverse = left @ inverse @ left.T + rho * np.outer(sv, sv)
            expected.append(-inverse @ gradient[pose])
        kwargs = dict(
            execution=_TorchCompactLBFGS,
            single_segment=layout == "single",
            use_bmm=layout == "batched",
        )
        actual = compact_lbfgs_direction(
            g, y, s, _DynamicOffset(start), rotate_history=True, **kwargs
        )
        baseline = compact_lbfgs_direction(g, y, s, start, **kwargs)
        assert actual.dtype == dtype and actual.device == g.device
        torch.testing.assert_close(actual, baseline, atol=0, rtol=0)
        np.testing.assert_allclose(
            actual.cpu().numpy(),
            expected,
            atol=2e-6 if dtype == torch.float32 else 2e-12,
            rtol=2e-6 if dtype == torch.float32 else 2e-12,
        )
