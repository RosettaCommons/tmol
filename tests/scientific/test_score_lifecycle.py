"""Retained score channels and rendered snapshots obey the mature public API."""

from importlib import import_module

import numpy as np
import pytest

from .molecules import BACKBONE_CASES, CASES


def evaluate(b, scorer, *, summed=False):
    if b.framework == "jax":
        import jax

        if summed:
            value, grad = jax.jit(jax.value_and_grad(lambda x: scorer(x).sum()))(
                b.pose.coords
            )
        else:
            value = scorer(b.pose.coords)
            grad = jax.grad(lambda x: scorer(x).sum())(b.pose.coords)
    else:
        x = b.pose.coords.detach().clone().requires_grad_(True)
        value = scorer(x)
        if summed:
            value = value.sum()
        grad = b.module.autograd.grad(value.sum(), x)[0]
    return b.numpy(value), b.numpy(grad)


@pytest.mark.parametrize(
    "weighted,channels",
    (
        ("fa_ljatr", ("fa_ljatr", "fa_ljrep", "fa_lk")),
        ("omega", ("rama", "omega")),
        ("gen_torsions", ("gen_torsions",)),
    ),
)
def test_retained_zero_channels_removal_and_old_snapshot_are_consistent(
    backend, weighted, channels
):
    if weighted == "gen_torsions":
        _check_generic_database_snapshot(backend)
        return
    b = backend.molecule(BACKBONE_CASES["mixed"], terms=(weighted,))
    sf = b.score_function
    st = import_module("tmol_jax" if b.framework == "jax" else "tmol").ScoreType
    channel = st[weighted]
    assert tuple(x.name for x in sf.all_score_types()) == channels
    assert len(sf.two_body_terms()) == 1
    assert not sf.one_body_terms() and not sf.multi_body_terms()
    assert sf.pre_work_initialization(b.pose) is None
    raw, gradients = b.channel_values_and_gradients(b.coords())
    assert raw.shape == (len(channels), b.pose.n_poses)
    assert np.linalg.norm(raw[0]) > 1e-4
    weights = np.array([name == weighted for name in channels], np.float32)
    np.testing.assert_array_equal(b.numpy(sf.weights_tensor()), weights)
    old = sf.render_whole_pose_scoring_module(b.pose)
    values, gradient = evaluate(b, old)
    index = channels.index(weighted)
    np.testing.assert_allclose(values, raw[index], rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(gradient, gradients[index], rtol=3e-5, atol=3e-5)
    sf.remove_term_for_score_type(channel)
    assert not sf.all_terms() and not sf.all_score_types()
    assert b.numpy(sf.weights_tensor()).size == 0
    assert not sf.score_type_covered_by_contained_term(channel)
    sf.set_weight(channel, -2)
    assert sf.pre_work_initialization(b.pose) is None
    fresh_values, fresh_gradient = evaluate(
        b, sf.render_whole_pose_scoring_module(b.pose)
    )
    np.testing.assert_allclose(fresh_values, -2 * values, rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(fresh_gradient, -2 * gradient, rtol=3e-5, atol=3e-5)
    old_values, old_gradient = evaluate(b, old)
    for repeated, reference in ((old_values, values), (old_gradient, gradient)):
        # Independent unchanged-scorer CUDA controls show float32 atomic-sum
        # variation before mutation, with unchanged float64-oracle error after
        # mutation. Bound only this fixed-fixture repeat check at two epsilons
        # of its value/force scale; keep CPU snapshots bit-exact.
        if b.requested_device == "cuda":
            atol = 2 * np.finfo(reference.dtype).eps * np.max(np.abs(reference))
            np.testing.assert_allclose(repeated, reference, rtol=0, atol=atol)
        else:
            np.testing.assert_array_equal(repeated, reference)
    sf.set_weight(channel, 0)
    assert not sf.all_terms() and not sf.all_score_types()


def test_public_term_order_retains_native_channel_groups(backend):
    b = backend.molecule(
        BACKBONE_CASES["mixed"], terms=("gen_torsions", "omega", "fa_ljatr")
    )
    sf = b.score_function
    terms, channels = sf.get_sorted_terms(reversed(sf.all_terms()))
    assert terms == sf.all_terms()
    assert tuple(st.name for st in channels) == (
        "fa_ljatr",
        "fa_ljrep",
        "fa_lk",
        "rama",
        "omega",
        "gen_torsions",
    )


def _check_generic_database_snapshot(b):
    import attrs

    b.molecule(CASES["ligand"])
    old = b.score_function.render_whole_pose_scoring_module(b.pose)
    db = b.parameter_database
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
    value, gradient = evaluate(b, old, summed=True)
    value = value.sum()
    assert value > 0 and np.linalg.norm(gradient) > 0
    b.use_parameter_database(
        attrs.evolve(db, scoring=attrs.evolve(db.scoring, genbonded=changed))
    )
    new = b.score_function.render_whole_pose_scoring_module(b.pose)
    new_value, new_gradient = evaluate(b, new, summed=True)
    np.testing.assert_allclose(new_value.sum(), 2 * value, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(new_gradient, 2 * gradient, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(b.numpy(old(b.pose.coords)).sum(), value)
