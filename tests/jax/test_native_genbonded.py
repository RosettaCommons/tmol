"""JAX transformations around real shared bonded native molecular execution."""

import attrs
import os
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.scientific.execution import Execution
from tests.scientific.molecules import CASES, BACKBONE_CASES
from tmol_jax import ScoreFunction, ScoreType


@pytest.fixture(
    params=[
        name
        for name in os.environ.get("TMOL_TEST_BACKENDS", "jax-cpu").split(",")
        if name.startswith("jax-")
    ]
)
def backend_name(request):
    return request.param


@pytest.mark.parametrize(
    "terms",
    [
        ("gen_torsions",),
        ("rama", "omega"),
        (
            "cart_lengths",
            "cart_angles",
            "cart_torsions",
            "cart_impropers",
            "cart_hxltorsions",
        ),
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_native_coordinate_transforms_and_pose_independence(dtype, backend_name, terms):
    with jax.enable_x64(dtype == np.float64):
        case = BACKBONE_CASES["mixed"] if "rama" in terms else CASES["mixed_batch"]
        backend = Execution(backend_name).molecule(case, terms=terms)
        pose = backend.pose
        score = backend.score_function.render_whole_pose_scoring_module(pose)
        x = pose.coords.astype(dtype)
        direction = jnp.asarray(
            np.random.default_rng(183).normal(size=x.shape), dtype=dtype
        )
        direction /= jnp.linalg.norm(direction)
        gradient = jax.jit(jax.grad(lambda x: score(x).sum()))(x)
        value, tangent = jax.jit(lambda x, d: jax.jvp(score, (x,), (d,)))(x, direction)
        expected = (gradient * direction).sum(axis=(1, 2))
        assert np.linalg.norm(tangent) > 1e-3
        np.testing.assert_allclose(tangent, expected, rtol=2e-6, atol=2e-6)
        jacobian = jax.jit(jax.jacrev(score))(x)
        np.testing.assert_array_equal(jacobian[0, 1], 0)
        np.testing.assert_array_equal(jacobian[1, 0], 0)
        np.testing.assert_allclose(jacobian[0, 0], gradient[0], rtol=2e-6, atol=2e-6)
        samples = jnp.stack([x, x + direction * 0.01])
        values = jax.jit(jax.vmap(score))(samples)
        gradients = jax.jit(jax.vmap(jax.grad(lambda x: score(x).sum())))(samples)
        np.testing.assert_allclose(values[0], value, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(gradients[0], gradient, rtol=2e-6, atol=2e-6)
        assert values.dtype == gradient.dtype == dtype
        # Summed float32 torsions lose precision for sub-milliangstrom changes;
        # 5e-3 along this normalized direction balances cancellation/truncation.
        step = 5e-3 if dtype == np.float32 else 1e-5
        numeric = (score(x + step * direction) - score(x - step * direction)) / (
            2 * step
        )
        np.testing.assert_allclose(
            tangent,
            numeric,
            rtol=3e-3 if dtype == np.float32 else 2e-6,
            atol=3e-3 if dtype == np.float32 else 2e-6,
        )


def test_native_custom_parameters_change_values_and_gradients_without_stale_cache(
    backend_name,
):
    backend = Execution(backend_name).molecule(CASES["ligand"])
    pose = backend.pose
    old = backend.score_function.render_whole_pose_scoring_module(pose)
    db = backend.parameter_database
    gen = db.scoring.genbonded
    changed = attrs.evolve(
        gen,
        torsions=tuple(
            attrs.evolve(
                p,
                **{
                    name: 2 * getattr(p, name)
                    for name in ("k1", "k2", "k3", "k4", "offset")
                },
            )
            for p in gen.torsions
        ),
        impropers=tuple(attrs.evolve(p, k=2 * p.k) for p in gen.impropers),
    )
    new_db = attrs.evolve(db, scoring=attrs.evolve(db.scoring, genbonded=changed))
    sf = ScoreFunction(new_db)
    sf.set_weight(ScoreType.gen_torsions, 1)
    new = sf.render_whole_pose_scoring_module(pose)
    value, gradient = jax.jit(jax.value_and_grad(lambda x: old(x).sum()))(pose.coords)
    new_value, new_gradient = jax.jit(jax.value_and_grad(lambda x: new(x).sum()))(
        pose.coords
    )
    assert value > 0 and np.linalg.norm(gradient) > 0
    np.testing.assert_allclose(new_value, 2 * value, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(new_gradient, 2 * gradient, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(old(pose.coords).sum(), value)


def test_native_mixed_molecule_minimization_preserves_frozen_atoms(backend_name):
    from tmol_jax import run_cart_min

    backend = Execution(backend_name).molecule(CASES["mixed_batch"])
    pose = backend.pose
    before = np.asarray(pose.coords).copy()
    mask = np.asarray(pose.atom_is_real).copy()
    mask[:, :3] = False
    with jax.default_device(backend.device):
        minimized, result = run_cart_min(
            pose,
            backend.score_function,
            move_mask=mask,
            max_iter=15,
            return_result=True,
        )
    after = np.asarray(minimized.coords)
    np.testing.assert_array_equal(after[~mask], before[~mask])
    np.testing.assert_array_equal(pose.coords, before)
    assert np.isfinite(after).all()
    assert float(result["fun"]) < result["initial_fun"] - 1e-3
    assert np.linalg.norm(after[mask] - before[mask]) > 1e-3


@pytest.mark.parametrize("term", ["gen_torsions", "cart_lengths", "rama"])
def test_empty_native_pose_batch_preserves_value_and_gradient_shapes(
    backend_name, term
):
    backend = Execution(backend_name).molecule(BACKBONE_CASES["mixed"], terms=(term,))
    pose = backend.pose[:0]
    score = backend.score_function.render_whole_pose_scoring_module(pose)
    values = jax.jit(score)(pose.coords)
    gradient = jax.jit(jax.grad(lambda x: score(x).sum()))(pose.coords)
    assert values.shape == (0,) and gradient.shape == pose.coords.shape
