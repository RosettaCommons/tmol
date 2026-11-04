"""Water overlap saturates at one and absent waters have finite zero gradients."""

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.score.lk_ball.potentials import LK_BALL_DEFAULTS, lk_bridge_fraction


def test_water_bridge_saturation_and_empty_water_mask():
    first = jnp.zeros(3)
    second = jnp.array([np.sqrt(8 / 3) * LK_BALL_DEFAULTS.lkb_water_dist, 0, 0])
    waters = jnp.zeros((2, 3))

    def score(w, mask):
        return lk_bridge_fraction(first, second, w, waters, waters_i_mask=mask)

    full = jnp.ones(2, dtype=bool)
    empty = jnp.zeros(2, dtype=bool)
    np.testing.assert_allclose(score(waters, full), 1.0, atol=1e-6)
    np.testing.assert_array_equal(score(waters, empty), 0.0)
    np.testing.assert_array_equal(jax.grad(score)(waters, empty), 0.0)
    assert np.isfinite(jax.grad(score)(waters, full)).all()


def test_lkball_sigma_mixed_hydroxyl_donor_acceptor_precedence():
    from tmol_jax.score.lk_ball.potentials import lj_sigma

    # OH can donate and accept; the other atom can donate but not accept.
    for first, second in (([1, 1, 1], [1, 0, 0]), ([1, 0, 0], [1, 1, 1])):
        donor_i, hydroxyl_i, acceptor_i = first
        donor_j, hydroxyl_j, acceptor_j = second
        sigma = lj_sigma(
            1.55,
            1.75,
            float(donor_i),
            float(donor_j),
            float(hydroxyl_i),
            float(hydroxyl_j),
            0.0,
            0.0,
            float(acceptor_i),
            float(acceptor_j),
            lj_hbond_OH_donor_dis=2.6,
        )
        np.testing.assert_allclose(sigma, 2.6)
