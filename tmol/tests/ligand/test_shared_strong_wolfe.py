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
