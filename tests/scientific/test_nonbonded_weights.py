"""Weighted native totals preserve the public decomposed scoring contract."""

import numpy as np
import pytest

from .test_score_contract import check_coordinate_transforms

from .molecules import CASES, MolecularCase
from .test_lk_ball import TERMS as LK_BALL_TERMS
from .test_molecular_nonbonded import TERMS as NONBONDED_TERMS


@pytest.mark.parametrize(
    "group,weights",
    [
        ("nonbonded", (0.7, 0.44, 0.0, -0.3)),
        ("neighbor_bounds", (0.7, -0.2, 1.3, 0.5)),
        ("lk_ball", (0.2, -0.7, 0, 0.5)),
        ("lk_ball", (0, 0, 0, -1.3)),
    ],
)
def test_nonbonded_custom_weights_and_gradients(backend, group, weights):
    if group == "neighbor_bounds":
        _check_neighbor_bounds(backend, weights)
        return
    retained_group = group == "lk_ball"
    terms = LK_BALL_TERMS if retained_group else NONBONDED_TERMS
    name = "connected_noncanonical" if retained_group else "mixed_batch"
    backend.molecule(CASES[name], terms=terms)
    x = backend.coords()
    values, gradients = backend.channel_values_and_gradients(x)
    weights = np.asarray(weights, dtype=None if retained_group else x.dtype)
    if retained_group and backend.framework == "jax":
        channels = backend.score_function.render_whole_pose_scoring_module(
            backend.pose, sum_terms=False, apply_weights=False
        )
    for score_type, weight in zip(
        (
            backend.score_function.active_score_types()
            if backend.framework == "jax"
            else backend.score_function.all_score_types()
        ),
        weights,
    ):
        backend.score_function.set_weight(score_type, float(weight))
    if backend.framework == "torch":
        backend._render_torch()
    if not retained_group:
        total, gradient = backend.value_and_grad(x)
        np.testing.assert_allclose(total, weights @ values, rtol=3e-6, atol=3e-5)
        np.testing.assert_allclose(
            gradient, np.einsum("s,spac->pac", weights, gradients), rtol=5e-5, atol=5e-5
        )
        contributions = backend.scores(x, sum_terms=False)
        np.testing.assert_allclose(
            contributions.sum(axis=0), total, rtol=3e-6, atol=3e-5
        )
        return
    assert backend.channel_names() == terms
    for apply_weights, coefficients in ((True, weights), (False, np.ones(len(terms)))):
        if backend.framework == "jax":
            jax, jnp = backend.module, backend.module.numpy
            function = backend.score_function.render_whole_pose_scoring_module(
                backend.pose, apply_weights=apply_weights
            )

            def reference(x):
                return (jnp.asarray(coefficients, x.dtype)[:, None] * channels(x)).sum()

            actual = jax.jit(jax.value_and_grad(lambda x: function(x).sum()))(
                backend.pose.coords
            )
            expected = jax.jit(jax.value_and_grad(reference))(backend.pose.coords)
        else:
            torch = backend.module
            source = torch.tensor(x, device=backend.device, requires_grad=True)
            value = backend.scorer(source, apply_weights=apply_weights).sum()
            actual = (
                backend.numpy(value),
                backend.numpy(torch.autograd.grad(value, source)[0]),
            )
            expected = (
                (coefficients @ values).sum(),
                np.einsum("s,spac->pac", coefficients, gradients),
            )
        for actual_part, expected_part in zip(actual, expected):
            np.testing.assert_allclose(actual_part, expected_part, rtol=3e-6, atol=3e-6)


def test_independent_nonbonded_terms_do_not_require_other_parameters(backend):
    import attrs

    case = CASES["mixed_batch"]
    terms = ("fa_ljatr", "fa_ljrep", "fa_lk", "fa_elec")
    backend.molecule(case, terms=terms)
    x = backend.coords()
    expected, expected_gradients = backend.channel_values_and_gradients(x)
    database = backend.parameter_database
    for keep_electrostatics in (False, True):
        absent = "ljlk" if keep_electrostatics else "elec"
        field = (
            "atom_type_parameters" if keep_electrostatics else "atom_charge_parameters"
        )
        missing = attrs.evolve(getattr(database.scoring, absent), **{field: ()})
        db = attrs.evolve(
            database, scoring=attrs.evolve(database.scoring, **{absent: missing})
        )
        selected = terms[-1:] if keep_electrostatics else terms[:3]
        rows = slice(3, 4) if keep_electrostatics else slice(0, 3)
        backend.molecule(case, terms=selected, parameter_database=db)
        np.testing.assert_array_equal(backend.coords(), x)
        value, gradient = backend.channel_values_and_gradients(x)
        np.testing.assert_allclose(value, expected[rows], rtol=3e-6, atol=3e-5)
        np.testing.assert_allclose(
            gradient, expected_gradients[rows], rtol=5e-5, atol=5e-5
        )

    check_coordinate_transforms(backend, case, ("fa_elec",), empty_term="fa_elec")


def _check_neighbor_bounds(b, weights):
    from importlib import import_module
    import attrs

    front = import_module("tmol_jax" if b.framework == "jax" else "tmol")
    case = MolecularCase(CASES["ligand"].parameter_files, (("LG1", "LG1"),), ((1, 1),))
    b.molecule(case, terms=NONBONDED_TERMS)
    near = b.pose.coords
    offset = int(b.pose.block_coord_offset[0, 1])
    far = near.at[:, offset:, 0].add(100) if b.framework == "jax" else near.clone()
    if b.framework == "torch":
        far[:, offset:, 0] += 100
    pose = (
        b.pose.replace(coords=far)
        if b.framework == "jax"
        else attrs.evolve(b.pose, coords=far)
    )

    def prepare(names):
        sf = front.ScoreFunction(
            b.parameter_database, *((b.device,) if b.framework == "torch" else ())
        )
        for name, weight in zip(NONBONDED_TERMS, weights):
            if name in names:
                sf.set_weight(front.ScoreType[name], weight)
        if b.framework == "torch":
            b.pose, b.score_function = pose, sf
            b._render_torch()
            return b.scorer
        return sf.render_whole_pose_scoring_module(pose)

    fused = prepare(NONBONDED_TERMS)
    ljlk, elec = prepare(NONBONDED_TERMS[:3]), prepare(NONBONDED_TERMS[3:])

    def prepare_value_and_grad(score):
        if b.framework == "jax":
            return b.module.jit(b.module.value_and_grad(lambda x: score(x).sum()))

        def evaluate(x):
            x = x.detach().clone().requires_grad_()
            value = score(x).sum()
            return value, b.module.autograd.grad(value, x)[0]

        return evaluate

    fused_vg = prepare_value_and_grad(fused)
    split_vg = prepare_value_and_grad(lambda x: ljlk(x) + elec(x))
    results = []
    for coords in (far, near, far):
        actual = tuple(map(b.numpy, fused_vg(coords)))
        expected = tuple(map(b.numpy, split_vg(coords)))
        for value, reference in zip(actual, expected):
            np.testing.assert_allclose(value, reference, atol=1e-4, rtol=2e-5)
        results.append(actual)
    assert abs(float(results[1][0] - results[0][0])) > 1e-3
    np.testing.assert_array_equal(results[0][0], results[2][0])
    np.testing.assert_allclose(results[0][1], results[2][1], atol=1e-5, rtol=1e-6)
