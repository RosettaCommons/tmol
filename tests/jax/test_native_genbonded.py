"""JAX transformations around real shared bonded native molecular execution."""

import attrs
import os
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.scientific.execution import Execution
from tests.scientific.molecules import CASES, BACKBONE_CASES, MolecularCase
from tests.scientific.metal_molecules import METAL_CASES
from tests.scientific.na_molecules import NA_CASES
from tests.scientific.dunbrack_molecules import DUNBRACK_CASES
from tests.scientific.hbond_molecules import HBOND_CASES
from tests.scientific.disulfide_molecules import DISULFIDE_CASES

from tmol_jax import ScoreFunction, ScoreType

_NATIVE_CASES = {
    "hbond": HBOND_CASES["mixed_batch"],
    "disulfide": DISULFIDE_CASES["mixed_batch"],
    "rama": BACKBONE_CASES["mixed"],
    "dunbrack_rot": DUNBRACK_CASES["mixed_batch"],
    "fa_ljatr": CASES["mixed_batch"],
    "fa_elec": CASES["mixed_batch"],
    "metal_coordination": METAL_CASES["mixed_batch"],
    "na_torsion": NA_CASES["mixed_batch"],
}


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
        ("dunbrack_rot", "dunbrack_rotdev", "dunbrack_semirot"),
        ("fa_ljatr", "fa_ljrep", "fa_lk", "fa_elec"),
        ("fa_ljatr", "fa_ljrep", "fa_lk"),
        ("fa_elec",),
        ("hbond",),
        ("disulfide",),
        ("lk_ball_iso", "lk_ball", "lk_bridge", "lk_bridge_uncpl"),
        ("rama", "omega"),
        ("metal_coordination",),
        ("na_torsion", "na_torsion_well"),
        (
            "cart_lengths",
            "cart_angles",
            "cart_torsions",
            "cart_impropers",
            "cart_hxltorsions",
        ),
    ],
    ids=lambda terms: "+".join(terms),
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_native_coordinate_transforms_and_pose_independence(dtype, backend_name, terms):
    with jax.enable_x64(dtype == np.float64):
        case = _NATIVE_CASES.get(terms[0], CASES["mixed_batch"])
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
        assert value.device == gradient.device == backend.device
        # Summed float32 torsions lose precision for sub-milliangstrom changes;
        # 5e-3 along this normalized direction balances cancellation/truncation.
        step = 5e-3 if dtype == np.float32 else 1e-5
        if terms == ("hbond",) and dtype == np.float64:
            # Degree-ten native HB polynomials cancel at 1e-5; the 1e-2..1e-4
            # convergence window supports 1e-3 without weakening tolerances.
            step = 1e-3
        numeric = (score(x + step * direction) - score(x - step * direction)) / (
            2 * step
        )
        np.testing.assert_allclose(
            tangent,
            numeric,
            rtol=3e-3 if dtype == np.float32 else 2e-6,
            atol=3e-3 if dtype == np.float32 else 2e-6,
        )


def test_native_nonbonded_bounds_follow_changed_coordinates(backend_name):
    """One rendered scorer must find interactions that begin outside its cutoff."""
    terms = ("fa_ljatr", "fa_ljrep", "fa_lk", "fa_elec")
    case = MolecularCase(CASES["ligand"].parameter_files, (("LG1", "LG1"),), ((1, 1),))
    backend = Execution(backend_name).molecule(case, terms=terms)
    near = backend.pose.coords
    offset = int(backend.pose.block_coord_offset[0, 1])
    far = near.at[:, offset:, 0].add(100)
    pose = backend.pose.replace(coords=far)
    weights = (0.7, -0.2, 1.3, 0.5)

    def prepare(names):
        score_function = ScoreFunction(backend.parameter_database)
        for name, weight in zip(terms, weights):
            if name in names:
                score_function.set_weight(ScoreType[name], weight)
        return score_function.render_whole_pose_scoring_module(pose)

    fused = prepare(terms)
    ljlk, elec = prepare(terms[:3]), prepare(terms[3:])
    fused_vg = jax.jit(jax.value_and_grad(lambda x: fused(x).sum()))
    split_vg = jax.jit(jax.value_and_grad(lambda x: (ljlk(x) + elec(x)).sum()))
    results = []
    for coords in (far, near, far):
        value, gradient = fused_vg(coords)
        split_value, split_gradient = split_vg(coords)
        np.testing.assert_allclose(value, split_value, atol=1e-4, rtol=2e-5)
        np.testing.assert_allclose(gradient, split_gradient, atol=1e-4, rtol=2e-5)
        results.append((value, gradient))
    assert abs(float(results[1][0] - results[0][0])) > 1e-3
    np.testing.assert_array_equal(results[0][0], results[2][0])
    np.testing.assert_allclose(results[0][1], results[2][1], atol=1e-5, rtol=1e-6)


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


@pytest.mark.parametrize(
    "term",
    [
        "gen_torsions",
        "cart_lengths",
        "fa_ljatr",
        "fa_elec",
        "rama",
        "metal_coordination",
        "na_torsion",
        "hbond",
        "lk_ball",
        "disulfide",
    ],
)
def test_empty_native_pose_batch_preserves_value_and_gradient_shapes(
    backend_name, term
):
    case = _NATIVE_CASES.get(term, BACKBONE_CASES["mixed"])
    backend = Execution(backend_name).molecule(case, terms=(term,))
    pose = backend.pose[:0]
    score = backend.score_function.render_whole_pose_scoring_module(pose)
    values = jax.jit(score)(pose.coords)
    gradient = jax.jit(jax.grad(lambda x: score(x).sum()))(pose.coords)
    assert values.shape == (0,) and gradient.shape == pose.coords.shape
