"""Public pair selection conserves molecular energy and physical derivatives."""

from contextlib import nullcontext

import numpy as np
import pytest

from .test_block_pair_scoring import CASES


def _array(backend, value):
    if backend.framework == "jax":
        return backend.module.device_put(value, backend.device)
    return backend.module.as_tensor(value, device=backend.device)


@pytest.mark.parametrize("name", ("ligand", "connected_noncanonical", "mixed_batch"))
def test_selected_pair_partition_and_physical_gradient(backend, name):
    b = backend.molecule(CASES[name], terms=())
    _, weights = b.use_beta2016()
    with b.module.enable_x64() if b.framework == "jax" else nullcontext():
        pair = b.score_function.render_block_pair_scoring_module(b.pose)
        x = _array(b, b.coords().astype(np.float64))
        count = b.pose.max_n_blocks
        all_pairs = np.array(
            [(i, j) for i in range(count) for j in range(count)], np.int32
        )
        left, right = (_array(b, part) for part in (all_pairs[::2], all_pairs[1::2]))
        # Disjoint selections reconstruct independent whole-pose energies,
        # including zero-weight channels and asymmetric native pair ownership.
        whole = b.scores(b.numpy(x), sum_terms=False, apply_weights=False)
        raw = pair.unweighted_scores(x)
        np.testing.assert_allclose(
            b.numpy(raw).sum((-2, -1)), whole, rtol=2e-8, atol=2e-8
        )
        for weighted in (False, True):
            selected = pair.score_interactions(
                x, left, sum_terms=False, apply_weights=weighted
            ) + pair.score_interactions(
                x, right, sum_terms=False, apply_weights=weighted
            )
            expected = whole * weights[:, None] if weighted else whole
            np.testing.assert_allclose(
                b.numpy(selected), expected, rtol=2e-8, atol=2e-8
            )
        assert np.linalg.norm(whole) > 1e-3
        empty = _array(b, np.empty((0, 2), np.int32))
        np.testing.assert_array_equal(b.numpy(pair.score_interactions(x, empty)), 0)
        doubled = _array(b, np.concatenate((all_pairs[::2], all_pairs[::2])))
        np.testing.assert_allclose(
            b.numpy(pair.score_interactions(x, doubled)),
            2 * b.numpy(pair.score_interactions(x, left)),
            rtol=2e-12,
            atol=2e-12,
        )

        def objective(coords):
            return pair.score_interactions(coords, left).sum()

        if b.framework == "jax":
            gradient = b.module.jit(b.module.grad(objective))(x)
        else:
            x = x.detach().requires_grad_(True)
            gradient = b.module.autograd.grad(objective(x), x)[0]
        assert np.linalg.norm(b.numpy(gradient)) > 1e-3
        direction = np.random.default_rng(241).normal(size=x.shape)
        direction = _array(b, direction / np.linalg.norm(direction))
        finite = (
            objective(x + 1e-5 * direction) - objective(x - 1e-5 * direction)
        ) / 2e-5
        np.testing.assert_allclose(
            b.numpy((gradient * direction).sum()), b.numpy(finite), rtol=2e-5, atol=2e-5
        )

        if b.framework == "jax":
            _check_prepared_selections(b, pair, x)
    if b.framework == "jax" and name == "mixed_batch":
        _check_pair_parameter_snapshot(b)


def test_selected_pair_rejects_malformed_indices(backend):
    b = backend.molecule(CASES["mixed_batch"], terms=("fa_ljatr",))
    pair = b.score_function.render_block_pair_scoring_module(b.pose)
    x = b.pose.coords
    with pytest.raises(ValueError, match="shape"):
        pair.score_interactions(x, _array(b, np.array([0, 1], np.int32)))
    with pytest.raises(TypeError, match="integer"):
        pair.score_interactions(x, _array(b, np.array([[0, 1]], np.float32)))

    if b.framework == "jax":
        _check_dynamic_selections(b, pair, x)


def _check_dynamic_selections(b, pair, x):
    import jax
    import jax.numpy as jnp

    pairs = jax.device_put(np.array([[0, 0], [-1, -1]], np.int32), b.device)
    execute = jax.jit(jax.vmap(pair.score_interactions, in_axes=(0, None)))
    result = execute(jnp.stack((x, x + 0.5)), pairs)
    assert result.device == b.device
    np.testing.assert_allclose(result[0], result[1], rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(result[0], pair.score_interactions(x, pairs))
    invalid = pairs.at[0, 0].set(b.pose.max_n_blocks)
    with pytest.raises(IndexError, match="out of range"):
        pair.score_interactions(x, invalid)
    assert np.isnan(np.asarray(execute(jnp.stack((x, x)), invalid))).all()


def _check_prepared_selections(b, pair, x):
    import jax
    import jax.numpy as jnp

    last = b.pose.max_n_blocks - 1
    selections = [
        np.array([[0, 0], [0, last]], np.int32),
        np.array([[last, last], [0, last]], np.int32),
    ]
    observed = []
    for summed, weighted in ((True, True), (False, False)):
        for selection in selections:
            indices = jax.device_put(selection, b.device)
            prepared = pair.prepare_interactions(
                indices, sum_terms=summed, apply_weights=weighted
            )

            def reference(x):
                return pair.score_interactions(
                    x, indices, sum_terms=summed, apply_weights=weighted
                )

            expected = reference(x)
            expected_gradient = jax.grad(lambda x: reference(x).sum())(x)
            actual, gradient = prepared.value_and_grad()(x)
            assert actual.device == gradient.device == b.device
            np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-10)
            np.testing.assert_allclose(
                gradient, expected_gradient, rtol=2e-10, atol=2e-10
            )
            np.testing.assert_allclose(
                prepared.jitted()(x), expected, rtol=2e-10, atol=2e-10
            )
            batched = jax.jit(jax.vmap(prepared))(jnp.stack((x, x + 0.25)))
            np.testing.assert_allclose(
                batched, jnp.stack((expected, expected)), rtol=2e-10, atol=2e-10
            )
            observed.append(np.asarray(actual))
    if last:
        assert np.max(np.abs(observed[0] - observed[1])) > 1e-3


def _check_pair_parameter_snapshot(b):
    import attrs
    import jax
    import jax.numpy as jnp

    b.molecule(CASES["mixed_batch"], terms=("fa_ljatr",))
    with jax.enable_x64():
        x = b.pose.coords.astype(jnp.float64)
        indices = jax.device_put(np.array([[0, 0], [0, 1]], np.int32), b.device)
        original = b.score_function.render_block_pair_scoring_module(
            b.pose
        ).prepare_interactions(indices)
        value, gradient = original.value_and_grad()(x)
        assert np.linalg.norm(gradient) > 1e-3
        db = b.parameter_database
        # Quadrupling well depth doubles its stored square-root parameters
        # exactly, giving an independent fourfold energy/force invariant.
        ljlk = attrs.evolve(
            db.scoring.ljlk,
            atom_type_parameters=tuple(
                attrs.evolve(p, lj_wdepth=4 * p.lj_wdepth)
                for p in db.scoring.ljlk.atom_type_parameters
            ),
        )
        changed = attrs.evolve(db, scoring=attrs.evolve(db.scoring, ljlk=ljlk))
        b.use_parameter_database(changed)
        fresh = b.score_function.render_block_pair_scoring_module(
            b.pose
        ).prepare_interactions(indices)
        actual, actual_gradient = fresh.value_and_grad()(x)
        np.testing.assert_allclose(actual, 4 * value, rtol=2e-12, atol=2e-11)
        np.testing.assert_allclose(
            actual_gradient, 4 * gradient, rtol=2e-12, atol=2e-11
        )
        old_value, old_gradient = original.value_and_grad()(x)
        np.testing.assert_allclose(old_value, value, rtol=2e-12, atol=2e-11)
        np.testing.assert_allclose(old_gradient, gradient, rtol=2e-12, atol=2e-11)
