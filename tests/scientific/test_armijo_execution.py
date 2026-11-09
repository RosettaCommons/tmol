"""Independent objective and inverse-Hessian contracts for shared optimization."""

from contextlib import nullcontext

import numpy as np
import pytest


def _services(backend, dtype):
    if backend.framework == "jax":
        from tmol_jax.optimization._armijo_native import ArmijoExecution, finalize
        from tmol_jax.optimization._compact_lbfgs import compact_direction

        context = backend.module.enable_x64(True)

        def array(value, kind=dtype):
            return backend.module.device_put(np.asarray(value, kind), backend.device)

        return context, array, ArmijoExecution, finalize, compact_direction
    from tmol.optimization._lbfgs_armijo import (
        _TorchArmijoExecution,
        _lbfgs_two_loop_impl,
    )
    from tmol.optimization._armijo_compiled import _armijo_finalize

    def array(value, kind=dtype):
        return backend.module.as_tensor(np.asarray(value, kind), device=backend.device)

    return (
        nullcontext(),
        array,
        _TorchArmijoExecution,
        _armijo_finalize,
        _lbfgs_two_loop_impl,
    )


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_segmented_armijo_acceptance_inactive_and_failed_objectives(backend, dtype):
    from tmol_core.optimization.armijo import armijo_linesearch_segmented

    context, array, execution, finalize, _ = _services(backend, dtype)
    with context:
        initial = array([1, 0.1, 4, 7, 4e-8])
        searching = array([True, True, True, False, True], bool)
        derivative, phi0 = array([-4] * 5), array([2] * 5)

        def objective(alpha):
            value = 0.5 * (2 - 2 * alpha) ** 2
            if backend.framework == "jax":
                return value.at[4].set(np.nan)
            value[4] = np.nan
            return value

        accepted, values, evaluations, status, _ = armijo_linesearch_segmented(
            objective,
            derivative,
            phi0,
            initial,
            searching,
            minstep=1e-6,
            execution=execution,
            combine_checks=True,
        )
        step, failed = finalize(status, derivative, accepted, initial, searching, 1e-6)
        atol = 1e-6 if dtype == np.float32 else 1e-12
        np.testing.assert_allclose(
            backend.numpy(accepted), [1, 0.2, 1, 0, 0], atol=atol, rtol=0
        )
        np.testing.assert_allclose(
            backend.numpy(values), [0, 1.28, 0, 2, 2], atol=atol, rtol=0
        )
        np.testing.assert_array_equal(backend.numpy(status), [0, 0, 0, 0, 3])
        np.testing.assert_allclose(
            backend.numpy(step), [1, 0.2, 1, 7, 0.5], atol=atol, rtol=0
        )
        np.testing.assert_array_equal(backend.numpy(failed), [False] * 4 + [True])
        assert evaluations == 2

        if backend.framework == "jax":
            import jax
            import jax.numpy as jnp
            from tmol_jax.optimization._armijo_native import classify, start

            with jax.default_device(backend.device):
                # Check a single native transition independently of policy iteration.
                for integer in (np.int32, np.int64):
                    alpha = jnp.asarray([[1, 0.1, 4], [0.5, 0.25, 2]], dtype=dtype)
                    searching = jnp.ones(3, bool)

                    def run(row):
                        selected = start(searching, row)
                        return classify(
                            searching,
                            selected,
                            0.5 * (2 - 2 * selected) ** 2,
                            jnp.full(3, 2, dtype=dtype),
                            jnp.full(3, -4, dtype=dtype),
                            0.8,
                            0.1,
                            status_dtype=integer,
                        )

                    accepted, values, status, active = jax.jit(jax.vmap(run))(alpha)
                    assert accepted.dtype == values.dtype == dtype
                    assert status.dtype == integer
                    np.testing.assert_allclose(
                        accepted, [[1, 0.1, 0], [0.5, 0.25, 0]], rtol=0, atol=1e-7
                    )
                    np.testing.assert_array_equal(status, [[0, 1, 2], [0, 1, 2]])
                    np.testing.assert_array_equal(active, [[False, True, True]] * 2)
                    np.testing.assert_allclose(
                        values, [[0, 1.62, 2], [0.5, 1.125, 2]], rtol=0, atol=2e-7
                    )

                if dtype == np.float32:
                    with jax.enable_x64(False):
                        alpha = start(
                            jnp.asarray([True, False]),
                            jnp.asarray([1.0, 7.0], dtype=jnp.float32),
                        )
                        np.testing.assert_array_equal(alpha, [1, 0])
                        assert not jax.config.x64_enabled
                    with pytest.raises(RuntimeError, match="shape|extent|Armijo"):
                        start(
                            jnp.asarray([True]), jnp.ones(2, jnp.float32)
                        ).block_until_ready()


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_compact_direction_matches_dense_bfgs_with_wrapped_partial_history(
    backend, dtype
):
    context, array, _, _, direction = _services(backend, dtype)
    random = np.random.default_rng(74)
    steps = random.normal(size=(5, 2, 6)).astype(dtype)
    differences = steps * np.arange(1, 7, dtype=dtype)[None, None, :]
    steps[1, 0], differences[1, 0] = 0, 0
    gradient = random.normal(size=(2, 6)).astype(dtype)
    expected = []
    for pose in range(2):
        inverse = np.eye(6)
        for index in [3, 4, 0, 1, 2]:
            s, y = steps[index, pose].astype(float), differences[index, pose].astype(
                float
            )
            if not np.any(s):
                continue
            rho = 1 / np.dot(y, s)
            left = np.eye(6) - rho * np.outer(s, y)
            inverse = left @ inverse @ left.T + rho * np.outer(s, s)
        expected.append(-inverse @ gradient[pose])
    with context:
        g = array(gradient)
        actual = direction(g, array(differences), array(steps), history_start=3)
    assert actual.dtype == g.dtype
    np.testing.assert_allclose(
        backend.numpy(actual),
        expected,
        atol=2e-6 if dtype == np.float32 else 2e-12,
        rtol=2e-6 if dtype == np.float32 else 2e-12,
    )


def minimize(backend, objective, initial, segment_ids, mask=None, **settings):
    """Bind the same objective to each frontend's native optimizer execution."""
    _, array, _, _, _ = _services(backend, np.asarray(initial).dtype)
    x = array(initial)
    ids = array(segment_ids, np.int32)
    if backend.framework == "jax":
        from tmol_jax.optimization import armijo_minimize

        return armijo_minimize(
            objective, x, segment_ids=ids, grad_mask=mask, **settings
        )
    from tmol.optimization._lbfgs_armijo import LBFGS_Armijo

    x = x.clone().requires_grad_()
    optimizer = LBFGS_Armijo([x], segment_ids=ids, **settings)

    def closure():
        optimizer.zero_grad()
        values = objective(x)
        values.sum().backward()
        if mask is not None:
            x.grad *= array(mask, np.bool_)
        return values

    initial_value = optimizer.step(closure, lambda: objective(x))
    state = optimizer.state[x]
    return dict(
        x=x.detach(),
        initial_fun=initial_value.detach(),
        fun=objective(x).detach().sum(),
        grad=x.grad.detach(),
        converged=state["converged"],
        stalled=state["stalled"],
        state=state,
    )


@pytest.mark.parametrize("dense", (True, False))
@pytest.mark.parametrize("value_only", (True, False))
def test_full_segmented_optimization_matches_independent_objectives(
    backend, dense, value_only
):
    context, array, _, _, _ = _services(backend, np.float64)
    with context:
        ids = np.repeat(np.arange(3), 3) if dense else np.array([0, 1, 0, 2, 1, 1, 2])
        initial = np.linspace(-2, 3, len(ids))
        target = np.linspace(1, 2, len(ids))
        initial[ids == 0] = target[ids == 0]
        stiffness = np.linspace(0.5, 4, len(ids))
        if backend.framework == "jax":
            import jax.numpy as xp
        else:
            xp = backend.module

        def objective(values, labels=ids, center=target, spring=stiffness):
            d = values - array(center)
            terms = 0.5 * array(spring) * d**2 + 0.05 * d**4
            return xp.stack(
                [
                    terms[array(np.flatnonzero(labels == i), np.int32)].sum()
                    for i in range(int(labels.max()) + 1)
                ]
            )

        settings = dict(
            max_iter=60,
            history_size=3,
            rtol=None,
            atol=None,
            gradtol=1e-6,
            value_only_line_search=value_only,
        )
        result = minimize(backend, objective, initial, ids, **settings)
        np.testing.assert_allclose(
            backend.numpy(result["x"]), target, rtol=0, atol=2e-6
        )
        np.testing.assert_array_equal(backend.numpy(result["converged"]), True)
        np.testing.assert_array_equal(backend.numpy(result["stalled"]), False)
        for segment in range(3):
            pick = ids == segment
            solo = minimize(
                backend,
                lambda value: objective(
                    value, np.zeros(pick.sum(), int), target[pick], stiffness[pick]
                ),
                initial[pick],
                np.zeros(pick.sum(), int),
                **settings,
            )
            np.testing.assert_allclose(
                backend.numpy(result["x"])[pick],
                backend.numpy(solo["x"]),
                rtol=0,
                atol=1e-10,
            )
        assert result["state"]["history_count"] == 3
        assert result["state"]["history_start"] != 0 or result["state"]["n_iter"] > 4

    if backend.framework == "jax" and not dense and value_only:
        import jax
        from tmol_jax.optimization import armijo_minimize

        device = backend.device
        with jax.default_device(device):
            x = jax.device_put(
                np.array([[2.0, -1.0, 3.0], [-3.0, 1.0, 2.0]], np.float32), device
            )
            ids = jax.device_put(np.array([[0] * 3, [1] * 3], np.int32), device)
            observed = []

            def objective(value):
                return ((value - 0.2) ** 2).sum(1)

            compiled = jax.jit(objective)

            def value_only(value):
                observed.append(value.shape)
                return compiled(value)

            result = armijo_minimize(
                objective,
                x,
                segment_ids=ids,
                value_only=value_only,
                value_only_line_search=True,
                max_iter=3,
                history_size=2,
                fixed_iterations=True,
            )
            assert observed and all(shape == x.shape for shape in observed)
            assert np.isfinite(result["x"]).all()
            assert objective(result["x"]).sum() < objective(x).sum()


@pytest.mark.parametrize(
    "name",
    (
        "ligand",
        "connected_noncanonical",
        "d_peptide",
        "modified_dna",
        "metal_mixed",
        "mixed_batch",
    ),
)
def test_mature_armijo_full_molecular_objective_and_fixed_atoms(backend, name):
    from .molecules import DOMAIN_CASES

    b = backend.molecule(DOMAIN_CASES[name], terms=())
    b.use_beta2016()
    initial = b.coords().copy()
    mask = np.broadcast_to(
        b.numpy(b.pose.real_atoms if b.framework == "torch" else b.pose.atom_is_real)[
            ..., None
        ],
        initial.shape,
    ).copy()
    mask[:, 0] = False
    ids = np.broadcast_to(
        np.arange(initial.shape[0])[:, None, None], initial.shape
    ).copy()
    scorer = (
        b.scorer
        if b.framework == "torch"
        else b.score_function.render_whole_pose_scoring_module(b.pose)
    )
    result = minimize(
        b,
        scorer,
        initial,
        ids,
        mask,
        max_iter=6,
        history_size=3,
        fixed_iterations=True,
        value_only_line_search=True,
    )
    final = b.numpy(result["x"])
    np.testing.assert_array_equal(final[~mask], initial[~mask])
    np.testing.assert_array_equal(b.coords(), initial)
    assert np.isfinite(final).all()
    assert b.numpy(result["fun"]) < b.numpy(result["initial_fun"]) - 1e-3
    assert np.linalg.norm(final - initial) > 1e-4
    channels, gradients = b.channel_values_and_gradients(final)
    assert np.isfinite(channels).all() and np.isfinite(gradients).all()


@pytest.mark.parametrize("kind", ("cartesian", "kinematic"))
@pytest.mark.parametrize("name", ("ligand", "d_peptide", "mixed_batch"))
def test_public_molecular_armijo_minimizers(backend, kind, name):
    from .molecules import DOMAIN_CASES

    b = backend.molecule(DOMAIN_CASES[name], terms=())
    b.use_beta2016()
    if b.framework == "jax":
        import tmol_jax as frontend
    else:
        import tmol as frontend
    settings = dict(
        max_iter=6, history_size=3, fixed_iterations=True, value_only_line_search=True
    )
    options = dict(optimizer_kwargs=settings)
    before = b.coords().copy()
    if kind == "cartesian":
        result = frontend.run_cart_min(b.pose, b.score_function, **options)
    else:
        movement = frontend.MoveMap.from_pose_stack(b.pose)
        movement.move_all_named_torsions = True
        if b.framework == "jax":
            result = frontend.run_kin_min(b.pose, b.score_function, movement, **options)
        else:
            forest = frontend.FoldForest.reasonable_fold_forest(b.pose)
            result = frontend.run_kin_min(
                b.pose, b.score_function, forest, movement, **options
            )
    np.testing.assert_array_equal(b.coords(), before)
    after = b.numpy(result.coords)
    assert np.isfinite(after).all()
    assert np.linalg.norm(after - before) > 1e-4
    assert b.scores(after).sum() < b.scores(before).sum() - 1e-3

    if b.framework == "jax" and name == "mixed_batch" and kind == "kinematic":
        import jax
        import jax.numpy as jnp
        from tmol_jax import MoveMap, PoseStackKinematics

        with jax.default_device(b.device), jax.enable_x64():
            pose = b.pose.replace(coords=b.pose.coords.astype(jnp.float64))
            kinematics = PoseStackKinematics(pose)
            movement = MoveMap.from_pose_stack(pose)
            movement.move_all_named_torsions = True
            selected = jax.device_put(
                np.flatnonzero(
                    np.asarray(kinematics.resolve_move_map(movement))
                ).astype(np.int32),
                b.device,
            )
            active = jax.device_put(np.arange(pose.n_poses, dtype=np.int32), b.device)
            scorer = b.score_function.render_whole_pose_scoring_module(pose)
            original = kinematics.bind(pose.coords)
            _, before = scorer.compact_objective(
                original.initial_dofs, selected, active, original
            )
            saved = before(original.initial_dofs.reshape(-1)[selected])
            changed = kinematics.bind(pose.coords * 1.01)
            value, derivative = scorer.compact_objective(
                changed.initial_dofs, selected, active, changed
            )
            compact = changed.initial_dofs.reshape(-1)[selected]
            actual = derivative(compact)
            expected = scorer(changed(changed.initial_dofs))
            np.testing.assert_allclose(actual[0], expected, rtol=2e-10, atol=2e-10)
            assert np.linalg.norm(actual[1]) > 1e-3
            direction = jax.device_put(
                np.random.default_rng(362).normal(size=compact.shape), b.device
            )
            h = 1e-5
            numeric = value(compact + h * direction) - value(compact - h * direction)
            numeric = numeric.sum() / (2 * h)
            analytic = (actual[1] * direction).sum()
            np.testing.assert_allclose(numeric, analytic, rtol=2e-5, atol=2e-5)
            for current, expected in zip(
                before(original.initial_dofs.reshape(-1)[selected]), saved
            ):
                np.testing.assert_allclose(current, expected, rtol=2e-10, atol=2e-10)
