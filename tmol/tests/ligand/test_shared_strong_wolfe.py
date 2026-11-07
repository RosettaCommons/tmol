"""Match the mature optimizer's evaluated trajectory, not a second oracle loop."""

import pytest
import torch

from tmol.ligand._conformer_generation import _LigandLBFGS, _TorchLBFGSExecution
from tmol_core.optimization.strong_wolfe import _strong_wolfe


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("problem", ["quadratic", "rosenbrock", "stationary"])
def test_ligand_optimizer_matches_mature_trajectory(dtype, problem):
    def objective(x):
        if problem == "rosenbrock":
            return (
                (1 - x[:-1]).square() + 100 * (x[1:] - x[:-1].square()).square()
            ).sum()
        if problem == "stationary":
            return (x - initial).square().sum()
        return ((x - 1).square() * x.new_tensor([1, 2, 4, 8])).sum()

    initial = torch.tensor([-1.2, 1.0, -0.7, 0.5], dtype=dtype)
    runs = []
    for cls in (torch.optim.LBFGS, _LigandLBFGS):
        x = initial.clone().requires_grad_(True)
        optimizer = cls([x], max_iter=20, history_size=2, line_search_fn="strong_wolfe")
        trace = []

        def closure():
            x.grad = None
            loss = objective(x)
            loss.backward()
            trace.append((x.detach().clone(), loss.detach().clone(), x.grad.clone()))
            return loss

        result = optimizer.step(closure)
        runs.append((x.detach().clone(), trace, result, optimizer))
    reference, candidate = runs
    assert len(reference[1]) == len(candidate[1]) == candidate[2]["n_eval"]
    assert (
        candidate[2]["n_iter"] == reference[3].state[reference[3]._params[0]]["n_iter"]
    )
    for expected, actual in zip(reference[1], candidate[1]):
        for e, a in zip(expected, actual):
            torch.testing.assert_close(a, e, rtol=0, atol=0)
    torch.testing.assert_close(candidate[0], reference[0], rtol=0, atol=0)
    assert candidate[2]["fun"] <= float(objective(initial))


def test_strong_wolfe_satisfies_descent_and_curvature():
    x = torch.tensor([2.0, -3.0], dtype=torch.float64)
    target = x.new_tensor([0.5, 1.0])
    gradient = 2 * (x - target)
    direction = -gradient
    initial = float((x - target).square().sum())
    gtd = gradient.dot(direction)

    def evaluate(base, step, direction):
        trial = torch.add(base, direction, alpha=step)
        return float((trial - target).square().sum()), 2 * (trial - target)

    value, grad, step, evaluations = _strong_wolfe(
        _TorchLBFGSExecution, evaluate, x, 1.0, direction, initial, gradient, gtd
    )
    assert value <= initial + 1e-4 * step * gtd
    assert abs(grad.dot(direction)) <= -0.9 * gtd
    assert evaluations > 0


def test_optional_stopping_rules_and_line_search_budget():
    from tmol_core.optimization.strong_wolfe import lbfgs_minimize

    x = torch.tensor([-1.2, 1.0, -0.7, 0.5], dtype=torch.float64)
    calls, observations = [], []

    def evaluate(x):
        value = 1e4 + ((x - 1).pow(4)).sum()
        calls.append(x.clone())
        return value, 4 * (x - 1).pow(3)

    def observe(iteration, coordinates, value, gradient):
        observations.append((iteration, len(calls), value, coordinates.clone()))

    result = lbfgs_minimize(
        evaluate,
        x,
        _TorchLBFGSExecution,
        max_iter=10,
        max_eval=float("inf"),
        tolerance_grad=0,
        tolerance_change=None,
        relative_loss_tolerance=1e-2,
        c1=1e-3,
        c2=0.8,
        line_search_max_iter=1,
        accepted_step=observe,
    )
    assert result["reason"] == "relative_loss_tolerance"
    assert result["n_iter"] == 1
    assert result["n_eval"] == len(calls) == 2
    assert len(observations) == 1
    assert observations[0][0] == 1
    torch.testing.assert_close(observations[0][3], result["x"], rtol=0, atol=0)
    assert result["fun"] < float(evaluate(x)[0])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_two_loop_matches_scaled_dense_inverse_hessian(dtype):
    import numpy as np
    from tmol_core.optimization.lbfgs import two_loop_direction
    from tmol_core.optimization.strong_wolfe import _HostTwoLoop

    rng = np.random.default_rng(74)
    steps = torch.tensor(rng.normal(size=(4, 6)), dtype=dtype)
    differences = steps * torch.arange(1, 7, dtype=dtype)
    gradient = torch.tensor(rng.normal(size=6), dtype=dtype)
    rho = 1 / (steps * differences).sum(dim=1)
    scale = 0.375
    for count in (0, 2, 4):
        # Dense BFGS updates are an independent matrix identity, including an
        # empty history and a partially filled fixed-capacity buffer.
        inverse = np.eye(6) * scale
        for index in range(count):
            s = steps[index].double().numpy()
            y = differences[index].double().numpy()
            r = float(rho[index])
            left = np.eye(6) - r * np.outer(s, y)
            inverse = left @ inverse @ left.T + r * np.outer(s, s)
        expected = -inverse @ gradient.double().numpy()
        result = two_loop_direction(
            gradient,
            steps,
            differences,
            rho,
            scale,
            count,
            _HostTwoLoop(_TorchLBFGSExecution),
        )
        np.testing.assert_allclose(
            result.numpy(),
            expected,
            rtol=2e-6 if dtype == torch.float32 else 2e-12,
            atol=2e-6 if dtype == torch.float32 else 2e-12,
        )
        assert result.dtype == dtype


def test_compiled_direction_hook_retains_optimizer_history_policy():
    from tmol_core.optimization.lbfgs import two_loop_direction
    from tmol_core.optimization.strong_wolfe import _HostTwoLoop, lbfgs_minimize

    counts = []

    class Hook(_TorchLBFGSExecution):
        @staticmethod
        def two_loop(gradient, steps, differences, rho, scale):
            counts.append(len(steps))
            return two_loop_direction(
                gradient,
                torch.stack(steps),
                torch.stack(differences),
                torch.stack(rho),
                scale,
                len(steps),
                _HostTwoLoop(_TorchLBFGSExecution),
            )

    def evaluate(x):
        diagonal = x.new_tensor([1, 2, 4, 8])
        return ((x - 1).square() * diagonal).sum(), 2 * (x - 1) * diagonal

    x = torch.tensor([-1.2, 1, -0.7, 0.5], dtype=torch.float64)
    plain = lbfgs_minimize(
        evaluate, x, _TorchLBFGSExecution, max_iter=30, history_size=2
    )
    hooked = lbfgs_minimize(evaluate, x, Hook, max_iter=30, history_size=2)
    torch.testing.assert_close(hooked["x"], plain["x"], rtol=0, atol=0)
    assert hooked["n_eval"] == plain["n_eval"]
    assert max(counts) == 2
    assert 1 in counts
    assert hooked["fun"] < 1e-8
