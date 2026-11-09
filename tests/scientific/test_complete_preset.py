"""Complete molecular presets must retain every channel and its derivative."""

import numpy as np
import pytest

from .disulfide_molecules import DISULFIDE_CASES
from .hbond_molecules import HBOND_CASES
from .molecules import DOMAIN_CASES

CASES = {
    **DOMAIN_CASES,
    "hydrogen_bonded_mixed": HBOND_CASES["mixed_batch"],
    "disulfide_mixed": DISULFIDE_CASES["mixed_batch"],
}
CHANNELS = {
    "fa_ljatr",
    "fa_ljrep",
    "fa_lk",
    "fa_elec",
    "hbond",
    "lk_ball_iso",
    "lk_ball",
    "lk_bridge",
    "lk_bridge_uncpl",
    "omega",
    "rama",
    "disulfide",
    "metal_coordination",
    "cart_lengths",
    "cart_angles",
    "cart_torsions",
    "cart_impropers",
    "cart_hxltorsions",
    "dunbrack_rot",
    "dunbrack_rotdev",
    "dunbrack_semirot",
    "ref",
    "gen_torsions",
    "na_torsion",
    "na_torsion_well",
}


@pytest.mark.parametrize("name", CASES)
def test_complete_preset_molecular_derivatives(backend, name):
    backend.molecule(CASES[name], terms=())
    channels, weights = backend.use_beta2016()
    assert len(channels) == 25 and set(channels) == CHANNELS
    coords = backend.coords()
    values, gradients = backend.channel_values_and_gradients(coords)
    assert values.shape == (25, len(coords))
    assert np.isfinite(values).all() and np.isfinite(gradients).all()
    total, gradient = backend.value_and_grad(coords)
    np.testing.assert_allclose(
        total, (weights[:, None] * values).sum(axis=0), rtol=3e-6, atol=3e-5
    )
    np.testing.assert_allclose(
        gradient,
        np.einsum("c,cpai->pai", weights, gradients),
        rtol=3e-5,
        atol=3e-5,
    )
    direction = np.random.default_rng(910).normal(size=coords.shape)
    direction /= np.linalg.norm(direction)
    # The same coordinates in float64 avoid cancellation from large repulsive
    # energies. This checks every channel, including channels absent in a case.
    finite_difference = backend.float64_directional_derivative(
        coords, direction, step=1e-4
    )
    np.testing.assert_allclose(
        np.einsum("cpai,pai->cp", gradients, direction),
        finite_difference,
        rtol=2e-3,
        atol=2e-3,
    )
    if name in ("ligand", "connected_noncanonical", "mixed_batch"):
        assert np.max(np.abs(values[channels.index("gen_torsions")])) > 1e-3
    if name == "hydrogen_bonded_mixed":
        assert np.all(np.abs(values[channels.index("hbond")]) > 1)
    if name == "disulfide_mixed":
        assert np.all(np.abs(values[channels.index("disulfide")]) > 1e-3)

    if backend.framework == "jax" and name == "ligand":
        _check_declared_weights()
    if backend.framework == "jax" and name == "connected_noncanonical":
        _check_ineligible_channels(backend)


def _check_declared_weights():
    import yaml
    from tmol_jax import ScoreType, beta2016_score_function
    from tmol_jax.database import default_directory

    expected = yaml.safe_load(
        (default_directory() / "score_functions/beta2016.sfxn").read_text()
    )["weights"]
    score = beta2016_score_function()
    # The mature default stores float32 weights even for float64 coordinates.
    assert score.weight_dtype == np.dtype(np.float32)
    assert {st.name: weight for st, weight in score.weights.items()} == {
        name: float(np.float32(weight)) for name, weight in expected.items()
    }
    assert ScoreType.na_torsion in score.active_score_types()
    assert ScoreType.metal_coordination in score.active_score_types()


def _check_ineligible_channels(backend):
    import jax
    from tmol_jax import ScoreFunction, ScoreType

    sf = ScoreFunction(backend.parameter_database)
    for st in (
        ScoreType.na_torsion,
        ScoreType.na_torsion_well,
        ScoreType.metal_coordination,
    ):
        sf.set_weight(st, 1)
    score = sf.render_whole_pose_scoring_module(backend.pose, sum_terms=False)
    assert score(backend.pose.coords).shape == (3, 1)
    np.testing.assert_array_equal(jax.jit(score)(backend.pose.coords), 0)
    np.testing.assert_array_equal(
        jax.jit(jax.grad(lambda x: score(x).sum()))(backend.pose.coords), 0
    )
