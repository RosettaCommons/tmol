"""Near-planar torsions retain accurate values and finite derivatives."""

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.score.constraint.functions import torsion


def test_native_planar_torsion_gradient_and_undefined_geometry():
    angles = np.array([0, 1e-5, np.pi - 1e-5, np.pi], dtype=np.float32)
    coords = np.tile(
        np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]]), (4, 1, 1)
    ).astype(np.float32)
    coords[:, 3, 1] = np.cos(angles)
    coords[:, 3, 2] = np.sin(angles)

    def energy(x):
        phi = torsion(x)
        return jnp.cos(3 * phi - 0.4).sum()

    gradient = jax.jit(jax.grad(energy))
    expected = -3 * np.sin(3 * angles.astype(np.float64) - 0.4) * np.cos(angles)
    np.testing.assert_allclose(
        gradient(jnp.asarray(coords))[:, 3, 2], expected, atol=3e-6
    )
    # Coincident atoms do not define a torsion. Preserve the canonical Torch
    # native derivative contract rather than the discarded prototype mask.
    assert np.isnan(gradient(jnp.zeros_like(coords))).all()
