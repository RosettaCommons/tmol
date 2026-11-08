"""Independent optimizer oracles, with conditional native transport checks."""

import numpy as np
import pytest

from .test_armijo_execution import _services


def _wolfe_minimize(backend, objective, start, *, value_and_grad=None, **settings):
    if backend.framework == "jax":
        from tmol_jax.optimization import lbfgs_minimize

        return lbfgs_minimize(
            objective, start, value_and_grad=value_and_grad, **settings
        )
    from tmol.ligand._conformer_generation import _TorchLBFGSExecution
    from tmol_core.optimization.strong_wolfe import lbfgs_minimize

    mask = settings.pop("grad_mask", None)
    gtol = settings.pop("gtol", 1e-5)

    def evaluate(x):
        if value_and_grad is None:
            x = x.detach().requires_grad_()
            value = objective(x)
            (gradient,) = backend.module.autograd.grad(value, x)
            value, gradient = value.detach(), gradient.detach()
        else:
            value, gradient = value_and_grad(x)
        return value, gradient if mask is None else gradient * mask

    result = lbfgs_minimize(
        evaluate,
        start,
        _TorchLBFGSExecution,
        max_iter=settings.pop("max_iter", 100),
        history_size=settings.pop("history", 5),
        tolerance_grad=gtol,
        tolerance_change=None,
        relative_loss_tolerance=settings.pop("ftol", 1e-6),
        max_eval=float("inf"),
        **settings,
    )
    result["converged"] = (
        float(result["grad"].abs().max()) <= gtol
        or result["reason"] == "relative_loss_tolerance"
    )
    return result


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_fixed_history_matches_dense_inverse_hessian_after_wrap(backend, dtype):
    if backend.framework != "jax":
        pytest.skip("JAX fixed-capacity compiled history transport")
    import jax
    import jax.numpy as jnp
    from tmol_jax.optimization._strong_wolfe_execution import JaxLBFGSExecution

    # Independently form inverse BFGS matrices, rather than another two-loop
    # implementation. Exercise empty, partial, full, wrapped and reused histories.
    with jax.enable_x64(True), jax.default_device(backend.device):
        rng = np.random.default_rng(731)
        size, capacity = 7, 3
        gradient = jnp.asarray(rng.normal(size=size), dtype=dtype)
        execution = JaxLBFGSExecution(gradient, capacity)
        steps, differences, rho = [], [], []
        for iteration in range(8):
            if iteration:
                step = jnp.asarray(rng.normal(size=size), dtype=dtype)
                difference = step * jnp.asarray(np.arange(1, size + 1), dtype=dtype)
                steps = (steps + [step])[-capacity:]
                differences = (differences + [difference])[-capacity:]
                rho = (rho + [1 / jnp.dot(step, difference)])[-capacity:]
            scale = 0.3
            inverse = np.eye(size) * scale
            identity = np.eye(size)
            for step, difference in zip(steps, differences):
                s, y = np.asarray(step, dtype=np.float64), np.asarray(
                    difference, dtype=np.float64
                )
                curvature = 1 / np.dot(s, y)
                transform = identity - curvature * np.outer(s, y)
                inverse = transform @ inverse @ transform.T + curvature * np.outer(s, s)
            expected = -inverse @ np.asarray(gradient)
            # Reusing history must preserve the molecular device even when a
            # different default device is active for host scalar construction.
            for default_device in (gradient.device, jax.devices("cpu")[0]):
                with jax.default_device(default_device):
                    actual = execution.two_loop(
                        gradient, steps, differences, rho, scale
                    )
                assert (
                    actual.device == gradient.device and actual.dtype == gradient.dtype
                )
                np.testing.assert_allclose(
                    actual,
                    expected,
                    rtol=3e-6,
                    atol=3e-7 if dtype == np.float32 else 2e-14,
                )
            assert execution.steps.shape == (capacity, size)
            assert execution.count == min(iteration, capacity)


def test_invalid_trial_does_not_report_convergence(backend):
    _, array, _, _, _ = _services(backend, np.float32)
    xp = backend.module
    if backend.framework == "jax":
        import jax.numpy as xp

    # A finite initial point with no finite improving trial must remain in place,
    # and exhaustion of the line search must not be reported as convergence.
    def objective(x):
        return xp.where(x[0] == 0, array(1.0), array(np.inf)), xp.ones_like(x)

    result = _wolfe_minimize(
        backend,
        None,
        array([0.0]),
        max_iter=3,
        line_search_max_iter=2,
        value_and_grad=objective,
    )
    np.testing.assert_array_equal(backend.numpy(result["x"]), [0])
    assert np.isfinite(result["fun"])
    assert not result["converged"]

    if backend.framework == "jax":
        import jax.numpy as jnp
        from tmol_jax.optimization import lbfgs_minimize

        with pytest.raises(FloatingPointError, match="finite"):
            lbfgs_minimize(lambda x: jnp.sum(x), jnp.asarray([jnp.nan]))
        with pytest.raises(ValueError, match="same shape"):
            lbfgs_minimize(
                lambda x: jnp.sum(x * x),
                jnp.ones(2),
                grad_mask=jnp.ones(3),
            )


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("operation", ("abs_sum", "dot"))
def test_step_reductions_preserve_dtype_device_and_array_transformations(
    backend, dtype, operation
):
    context, array, _, _, _ = _services(backend, dtype)
    if backend.framework == "jax":
        from tmol_jax.optimization._strong_wolfe_execution import (
            JaxStrongWolfeExecution as Execution,
        )
    else:
        from tmol.ligand._conformer_generation import _TorchLBFGSExecution as Execution
    with context:
        # These integer reductions are exactly representable in both dtypes.
        values = array([-4, 2, -1, 8])
        if operation == "abs_sum":
            norm = Execution.abs_sum
            expected, derivative = 15, [-1, 1, -1, 1]
        else:
            weights = array([1, 2, 3, 4])

            def norm(x):
                return Execution.dot(x, weights)

            expected, derivative = 29, weights
        result = norm(values)
        assert result.dtype == values.dtype and result.device == values.device
        np.testing.assert_array_equal(backend.numpy(result), expected)
        if backend.framework == "jax":
            import jax
            import jax.numpy as jnp

            result = jax.jit(norm)(values)
            assert result.dtype == values.dtype and result.device == values.device
            np.testing.assert_array_equal(result, expected)
            np.testing.assert_array_equal(
                jax.jit(jax.vmap(norm))(jnp.stack((values, values * 2))),
                [expected, expected * 2],
            )
            gradient = jax.grad(norm)(values)
        else:
            values.requires_grad_()
            (gradient,) = backend.module.autograd.grad(norm(values), values)
        np.testing.assert_array_equal(
            backend.numpy(gradient),
            backend.numpy(derivative) if operation == "dot" else derivative,
        )


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_host_control_quadratic_retains_dtype_and_frozen_coordinate(backend, dtype):
    context, array, _, _, _ = _services(backend, dtype)
    with context:
        # Coupled positive-definite quadratic: freezing y=5 shifts the minimizing
        # x to 2 - (5 + 3)/2 = -2. This checks the constrained physical optimum.
        start = array([0, 5, 9])

        def objective(x):
            a, b, c = x - array([2, -3, 4])
            return a * a + a * b + b * b + 3 * c * c

        result = _wolfe_minimize(
            backend,
            objective,
            start,
            grad_mask=array([1, 0, 1]),
            max_iter=30,
            gtol=1e-6 if dtype == np.float32 else 1e-10,
            ftol=0,
        )
        assert result["x"].dtype == start.dtype
        assert result["x"].device == start.device
        assert result["x"][1] == start[1]
        np.testing.assert_allclose(
            backend.numpy(result["x"]),
            [-2, 5, 4],
            rtol=0,
            atol=3e-6 if dtype == np.float32 else 2e-10,
        )
        np.testing.assert_allclose(result["fun"], 48, rtol=0, atol=2e-6)

    if dtype == np.float32:
        for history in (1, 5, 50):
            target = array([2.0, -3.0, 4.0])

            def objective(x):
                return ((x - target) ** 2).sum()

            result = _wolfe_minimize(
                backend,
                objective,
                array([0.0, 5.0, 9.0]),
                grad_mask=array([1.0, 0.0, 1.0]),
                history=history,
            )
            assert result["converged"]
            result_x = backend.numpy(result["x"])
            target_x = backend.numpy(target)
            np.testing.assert_allclose(result_x[[0, 2]], target_x[[0, 2]], atol=1e-5)
            assert float(result["x"][1]) == 5.0
            assert float(result["fun"]) <= 64.0 + 1e-5


def test_cubic_line_search_quadratic(backend):
    _, array, _, _, _ = _services(backend, np.float32)
    if backend.framework == "jax":
        from tmol_jax.optimization.line_search import (
            _cubic_interp,
            strong_wolfe_line_search,
        )
    else:
        from tmol.ligand._conformer_generation import _TorchLBFGSExecution
        from tmol_core.optimization.strong_wolfe import _cubic_interpolate, strong_wolfe

        def _cubic_interp(*args):
            return _cubic_interpolate(_TorchLBFGSExecution, *(array(x) for x in args))

        def strong_wolfe_line_search(fn, x, direction, f0, g0, *, alpha0, c2):
            def evaluate(base, step, vector):
                return fn(base + step * vector)

            value, gradient, alpha, _ = strong_wolfe(
                _TorchLBFGSExecution,
                evaluate,
                x,
                alpha0,
                direction,
                float(f0),
                g0,
                g0.dot(direction),
                c2=c2,
            )
            return alpha, value, gradient

    assert _cubic_interp(0.0, 4.0, -4.0, 3.0, 1.0, 2.0) == pytest.approx(2.0)
    assert _cubic_interp(3.0, 1.0, 2.0, 0.0, 4.0, -4.0) == pytest.approx(2.0)
    calls = []

    def value_and_grad(x):
        calls.append(float(x[0]))
        return ((x - 2) ** 2).sum(), 2 * (x - 2)

    x = array([0.0])
    alpha, value, gradient = strong_wolfe_line_search(
        value_and_grad,
        x,
        array([1.0]),
        array(4.0),
        array([-4.0]),
        alpha0=5.0,
        c2=0.1,
    )
    assert alpha == pytest.approx(2.0, abs=1e-6)
    assert len(calls) == 2
    assert float(value) < 1e-12
    np.testing.assert_allclose(backend.numpy(gradient), 0, atol=1e-6)


def test_compact_execution_uses_fresh_coordinate_masks_and_parameter_snapshots(backend):
    if backend.framework != "jax":
        pytest.skip("JAX prepared compact objective cache contract")
    import jax
    import jax.numpy as jnp
    from tmol_core.scoring.elec import partial_charge_table, partial_charges_for_block
    from .molecules import DOMAIN_CASES

    b = backend.molecule(DOMAIN_CASES["mixed_batch"], terms=("fa_elec",))
    score, pose = b.score_function, b.pose
    types = pose.packed_block_types.active_block_types
    previous = [getattr(bt, "elec_partial_charges", None) for bt in types]
    table = partial_charge_table(b.parameter_database.scoring.elec)
    with jax.default_device(b.device), jax.enable_x64():
        x = pose.coords.astype(jnp.float64)
        active = jax.device_put(np.array([0], np.int32), b.device)
        original = score.render_whole_pose_scoring_module(pose)
        selected = jax.device_put(
            np.arange(0, x.shape[1] * 3, 3, dtype=np.int32), b.device
        )
        value, derivative = original.compact_objective(x, selected, active)
        expected = derivative(x.reshape(-1)[selected])
        assert np.linalg.norm(expected[0]) > 1e-3 and np.linalg.norm(expected[1]) > 1e-3
        try:
            for bt, override in zip(types, previous):
                charges = partial_charges_for_block(bt, table, overrides=override)
                bt.elec_partial_charges = {
                    atom.name: float(2 * q) for atom, q in zip(bt.atoms, charges)
                }
            changed = score.render_whole_pose_scoring_module(pose)
            _, updated = changed.compact_objective(x, selected, active)
            for actual, before in zip(updated(x.reshape(-1)[selected]), expected):
                np.testing.assert_allclose(actual, 4 * before, rtol=2e-10, atol=2e-10)
            for actual, before in zip(derivative(x.reshape(-1)[selected]), expected):
                np.testing.assert_allclose(actual, before, rtol=2e-10, atol=2e-10)
            # Same cardinality, different components and a fresh non-rigid reference.
            selected = selected + 1
            reference = x * 1.01
            function, gradient = changed.compact_objective(reference, selected, active)
            compact = reference.reshape(-1)[selected]
            direction = jax.device_put(
                np.random.default_rng(127).normal(size=compact.shape), b.device
            )
            h = 1e-5
            numeric = (
                function(compact + h * direction).sum()
                - function(compact - h * direction).sum()
            ) / (2 * h)
            analytic = (gradient(compact)[1] * direction).sum()
            np.testing.assert_allclose(numeric, analytic, rtol=2e-6, atol=2e-6)
        finally:
            for bt, override in zip(types, previous):
                if override is None:
                    del bt.elec_partial_charges
                else:
                    bt.elec_partial_charges = override


def analytic_cartesian_fixture(backend, monkeypatch, *, repulsive=False):
    """One exact quadratic fixture for Cartesian masks and relax configuration."""
    from dataclasses import dataclass, replace
    from types import SimpleNamespace
    import jax.numpy as jnp
    from tmol_jax.score.energy_term import EnergyTerm
    from tmol_jax.score import score_function as scoring
    from tmol_jax.score import ScoreFunction, ScoreType

    channel = ScoreType.fa_ljrep if repulsive else ScoreType.fa_ljatr

    @dataclass(frozen=True)
    class Pose:
        coords: object
        packed_block_types: object = None

        @property
        def atom_is_real(self):
            return jnp.asarray([[True, True, False]])

        def replace(self, **updates):
            return replace(self, **updates)

    class QuadraticTerm(EnergyTerm):
        def __init__(self, database):
            pass

        def score_types_provided(self):
            return (channel,)

        def setup_for_pose_stack(self, pose, block_types):
            return None

        def __call__(self, pose, state, coords_override=None):
            return {channel: (coords_override**2).sum(axis=(1, 2))}

    original = scoring.importlib.import_module
    monkeypatch.setattr(
        scoring.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(QuadraticTerm=QuadraticTerm)
            if name == "tmol_jax.score.fake.fake_energy_term"
            else original(name)
        ),
    )
    monkeypatch.setitem(scoring._TERM_FOR_TYPE, channel, ("fake", "QuadraticTerm"))
    score = ScoreFunction(None)
    score.set_weight(channel, 0.55 if repulsive else 1.0)
    pose = Pose(
        backend.module.device_put(np.ones((1, 3, 3), np.float32), backend.device)
    )
    return pose, score
