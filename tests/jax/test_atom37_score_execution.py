"""Prepared source-coordinate scoring retains fresh chemistry and derivatives."""

import jax
import attrs
import numpy as np
import pytest

from tests.scientific.test_atom37_input import atom37_fixture, bind_atom37
from tests.jax.test_atom37_transforms import execution


@pytest.mark.parametrize("name", ("ligand", "hydroxyproline", "zinc"))
def test_prepared_source_scores_match_coordinate_ad_and_fresh_bindings(name):
    b, builder, source, _, context = atom37_fixture(execution(), name, count=1)
    x, bind = bind_atom37(b, builder, source)
    b.pose, b.parameter_database = bind(x), context.parameter_database
    b.use_beta2016()
    scorer = b.score_function.render_whole_pose_scoring_module(b.pose)
    mapped = scorer.value_and_grad(coordinate_map=bind)
    reference = jax.jit(
        jax.value_and_grad(lambda y: scorer(bind(y).coords.astype(np.float64)).sum())
    )
    # The score precision follows the stored pose, as in ordinary coordinate AD.
    direct = jax.jit(jax.value_and_grad(lambda y: scorer(bind(y).coords).sum()))
    energy, gradient = mapped(x)
    expected, expected_gradient = direct(x)
    np.testing.assert_allclose(energy.sum(), expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=2e-5, atol=2e-5)
    assert np.isfinite(gradient).all() and np.linalg.norm(gradient) > 1e-3
    precise_energy, precise_gradient = reference(x)
    precise = scorer.value_and_grad(coordinate_map=bind, coordinate_dtype=np.float64)(x)
    assert precise[0].dtype == np.float64 and precise[1].dtype == x.dtype
    np.testing.assert_allclose(precise[0].sum(), precise_energy, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(precise[1], precise_gradient, rtol=2e-5, atol=2e-5)
    with jax.enable_x64(False), pytest.raises(ValueError, match="x64 enabled"):
        scorer.value_and_grad(coordinate_map=bind, coordinate_dtype=np.float64)(x)

    direction = np.random.default_rng(183).normal(size=source.shape).astype(np.float32)
    direction = jax.device_put(np.where(np.isfinite(source), direction, 0), b.device)
    shifted = x + 0.03 * direction
    # Check two step sizes at a displaced geometry: the unmodified zinc fixture
    # crosses a score discontinuity inside the original +/-0.001 interval.
    # Its original values/gradients are still compared with direct AD above.
    shifted_gradient = mapped(shifted)[1]
    for step in (1e-3, 2e-4):
        numerical = (
            mapped(shifted + step * direction)[0].sum()
            - mapped(shifted - step * direction)[0].sum()
        ) / (2 * step)
        np.testing.assert_allclose(
            numerical, (shifted_gradient * direction).sum(), rtol=4e-3, atol=4e-3
        )

    fresh_bind = builder.prepare(shifted)
    fresh_pose = fresh_bind(shifted)
    # A changed stored weight must not reuse an earlier scorer's parameter data.
    from tmol_jax import ScoreType

    b.score_function.set_weight(ScoreType.cart_lengths, 0.7)
    fresh_scorer = b.score_function.render_whole_pose_scoring_module(fresh_pose)
    fresh = fresh_scorer.value_and_grad(coordinate_map=fresh_bind)
    values, gradients = jax.jit(jax.vmap(fresh))(jax.numpy.stack([x, shifted]))
    for row, coords in enumerate((x, shifted)):
        expected, expected_gradient = jax.value_and_grad(
            lambda y: fresh_scorer(fresh_bind(y).coords).sum()
        )(coords)
        np.testing.assert_allclose(values[row].sum(), expected, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(
            gradients[row], expected_gradient, rtol=2e-5, atol=2e-5
        )
    # Reusing the original bound function still returns its original snapshot.
    restored = mapped(x)
    np.testing.assert_allclose(restored[0], energy, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(restored[1], gradient, rtol=2e-5, atol=2e-5)


def test_prepared_source_execution_observes_custom_charge_parameters():
    from tmol_jax import ScoreFunction, ScoreType

    b, builder, source, _, context = atom37_fixture(execution(), "ligand", count=1)
    x, bind = bind_atom37(b, builder, source)
    pose = bind(x)
    database = context.parameter_database
    executions = []
    for factor in (1, 2):
        changed = attrs.evolve(
            database,
            scoring=attrs.evolve(
                database.scoring,
                elec=attrs.evolve(
                    database.scoring.elec,
                    atom_charge_parameters=tuple(
                        attrs.evolve(record, charge=record.charge * factor)
                        for record in database.scoring.elec.atom_charge_parameters
                    ),
                ),
            ),
        )
        score = ScoreFunction(changed)
        score.set_weight(ScoreType.fa_elec, 1.0)
        executions.append(
            score.render_whole_pose_scoring_module(pose).value_and_grad(
                coordinate_map=bind, coordinate_dtype=np.float64
            )
        )
    original, doubled = [execute(x) for execute in executions]
    assert np.linalg.norm(original[0]) > 1e-3 and np.linalg.norm(original[1]) > 1e-3
    # Coulomb energies and forces are bilinear in the two atom charges.
    np.testing.assert_allclose(doubled[0], original[0] * 4, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(doubled[1], original[1] * 4, rtol=2e-5, atol=2e-5)
    restored = executions[0](x)
    np.testing.assert_allclose(restored[0], original[0], rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(restored[1], original[1], rtol=2e-5, atol=2e-5)
