"""Block attribution conserves independent whole-pose physics and derivatives."""

from contextlib import nullcontext
import numpy as np
import pytest

from .test_complete_preset import CASES as PRESET_CASES
from .metal_molecules import METAL_CASES
from .dunbrack_molecules import DUNBRACK_CASES

CASES = {
    **PRESET_CASES,
    "metal_bridge": METAL_CASES["bridging_donor"],
    "semirotameric": DUNBRACK_CASES["l_peptide"],
}


@pytest.mark.parametrize(
    "name",
    (
        "ligand",
        "connected_noncanonical",
        "d_peptide",
        "modified_dna",
        "metal_mixed",
        "mixed_batch",
        "hydrogen_bonded_mixed",
        "disulfide_mixed",
        "metal_bridge",
        "semirotameric",
    ),
)
def test_block_pair_conserves_all_channel_values_and_gradients(backend, name):
    b = backend.molecule(CASES[name], terms=())
    b.use_beta2016()
    with b.module.enable_x64() if b.framework == "jax" else nullcontext():
        pair = b.score_function.render_block_pair_scoring_module(b.pose)
        coords = b.coords().astype(np.float64)
        expected_e, expected_g = b.channel_values_and_gradients(coords)
        if b.framework == "jax":
            x = b.module.device_put(coords, b.device)
            matrix = b.module.jit(
                lambda x: pair(x, sum_terms=False, apply_weights=False)
            )(x)
            gradients = b.module.jit(
                b.module.jacrev(
                    lambda x: pair(x, sum_terms=False, apply_weights=False).sum(
                        (1, 2, 3)
                    )
                )
            )(x)
        else:
            x = b.module.tensor(coords, device=b.device, requires_grad=True)
            matrix = pair(x, sum_terms=False, apply_weights=False)
            gradients = b.module.stack(
                [
                    b.module.autograd.grad(e.sum(), x, retain_graph=True)[0]
                    for e in matrix
                ]
            )
        matrix, gradients = b.numpy(matrix), b.numpy(gradients)
        ref = b.channel_names().index("ref")
        physical = [i for i in range(len(expected_e)) if i != ref]
        np.testing.assert_allclose(
            matrix[physical].sum((-2, -1)), expected_e[physical], rtol=2e-8, atol=2e-8
        )
        # Mature reference tables and whole-pose reference accumulation are
        # float32 even when coordinate-dependent physics executes in float64.
        diagonal = np.diagonal(matrix[ref], axis1=-2, axis2=-1)
        # Different reduction trees retain the standard float32 summation
        # bound; reference terms must still be exactly diagonal and force-free.
        off_diagonal = matrix[ref].copy()
        ids = np.arange(off_diagonal.shape[-1])
        off_diagonal[:, ids, ids] = 0
        np.testing.assert_array_equal(off_diagonal, 0)
        np.testing.assert_array_equal(gradients[ref], 0)
        n = max(1, diagonal.shape[-1] - 1)
        epsilon = np.finfo(np.float32).eps
        bound = n * epsilon / (1 - n * epsilon) * np.abs(diagonal).sum(-1)
        assert np.all(np.abs(diagonal.sum(-1) - expected_e[ref]) <= bound)
        np.testing.assert_allclose(gradients, expected_g, rtol=2e-8, atol=2e-8)
        assert np.linalg.norm(expected_g) > 1e-3

        if b.framework == "jax" and name == "mixed_batch":
            _check_pair_transforms(b, pair, x)


def _check_pair_transforms(b, pair, x):
    import jax
    import jax.numpy as jnp

    moved = x.at[0, int(b.pose.block_coord_offset[0, 2]) :].add(
        jnp.array([20.0, 0.0, 0.0])
    )
    samples = jnp.stack((x, moved))
    actual = jax.jit(jax.vmap(pair))(samples)
    whole = b.score_function.render_whole_pose_scoring_module(b.pose)
    expected = jax.jit(jax.vmap(whole))(samples)
    assert actual.device == b.device
    np.testing.assert_allclose(actual.sum((-2, -1)), expected, atol=2e-8, rtol=2e-8)
    assert np.max(np.abs(np.asarray(actual[1] - actual[0]))) > 1e-3
    weights = jnp.asarray(np.random.default_rng(718).normal(size=actual.shape[1:]))
    objective = jax.jit(lambda x: jnp.sum(pair(x) * weights))
    gradient = jax.jit(jax.grad(objective))(x)
    direction = jnp.asarray(np.random.default_rng(44).normal(size=x.shape))
    direction /= jnp.linalg.norm(direction)
    finite = (objective(x + 1e-5 * direction) - objective(x - 1e-5 * direction)) / 2e-5
    np.testing.assert_allclose(
        jnp.sum(gradient * direction), finite, atol=2e-6, rtol=2e-6
    )
