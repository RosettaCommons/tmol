"""One molecular input and invariant set for every supported execution lane."""

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def peptide():
    return Path(__file__).parent / "data/peptide.pdb"


def test_weighted_channels_and_coordinate_derivative(backend, peptide):
    backend.load(peptide)
    coords = backend.coords()
    atoms = [
        line for line in peptide.read_text().splitlines() if line.startswith("ATOM")
    ]
    expected_ids = [(int(line[22:26]) - 1, line[12:16].strip()) for line in atoms]
    actual_ids = [
        (block, atom.name)
        for block in range(backend.pose.max_n_blocks)
        for atom in backend.pose.block_type(0, block).atoms
    ]
    assert actual_ids == expected_ids
    expected_coords = np.array(
        [[float(line[start : start + 8]) for start in (30, 38, 46)] for line in atoms],
        dtype=np.float32,
    )
    assert coords.shape == (1, len(expected_ids), 3)
    # Loaders may idealize hydrogens; evaluate identical coordinates only after
    # checking the complete residue/atom order against the fixture.
    coords = expected_coords[None]
    coords = coords - coords.mean(axis=1, keepdims=True)
    total, gradient = backend.value_and_grad(coords)
    weighted = backend.scores(coords, sum_terms=False)
    assert np.isfinite(total).all() and np.isfinite(gradient).all()
    assert gradient.shape == coords.shape and np.linalg.norm(gradient) > 0
    np.testing.assert_allclose(weighted.sum(axis=0), total, rtol=2e-5, atol=2e-5)

    # A normalized deterministic direction checks the same full coordinate
    # derivative in every lane. Centering avoids rounding sub-millangstrom
    # perturbations against large absolute coordinates. Differences at 2e-3 A balance
    # cancellation against truncation; use mixed absolute/relative error.
    direction = (
        np.random.default_rng(141).normal(size=coords.shape).astype(coords.dtype)
    )
    direction /= np.linalg.norm(direction)
    step = np.float32(2e-3)
    finite_difference = (
        backend.scores(coords + step * direction)
        - backend.scores(coords - step * direction)
    ) / (2 * step)
    analytic = (gradient * direction).sum(axis=(1, 2))
    np.testing.assert_allclose(analytic, finite_difference, rtol=3e-3, atol=3e-3)


def check_coordinate_transforms(
    b, case, terms, *, empty_term=None, empty_case=None, prepared_check=None
):
    # Portable term checks already run above; retain JAX transform/FFI capabilities.
    if b.framework != "jax":
        return
    for dtype in (np.float32, np.float64):
        with b.module.enable_x64(dtype == np.float64):
            b.molecule(case, terms=terms)
            x = b.pose.coords.astype(dtype)
            score = b.score_function.render_whole_pose_scoring_module(b.pose)
            if prepared_check is not None:
                prepared_check(b, x)
            direction = b.module.device_put(
                np.random.default_rng(183).normal(size=x.shape).astype(dtype), b.device
            )
            direction /= b.module.numpy.linalg.norm(direction)
            gradient = b.module.jit(b.module.grad(lambda x: score(x).sum()))(x)
            value, tangent = b.module.jit(lambda x, d: b.module.jvp(score, (x,), (d,)))(
                x, direction
            )
            jacobian = b.module.jit(b.module.jacrev(score))(x)
            samples = b.module.numpy.stack((x, x + direction * 0.01))
            values = b.module.jit(b.module.vmap(score))(samples)
            gradients = b.module.jit(
                b.module.vmap(b.module.grad(lambda x: score(x).sum()))
            )(samples)
            np.testing.assert_allclose(
                b.numpy(values[0]), b.numpy(value), rtol=2e-06, atol=2e-06
            )
            np.testing.assert_allclose(
                b.numpy(gradients[0]), b.numpy(gradient), rtol=2e-06, atol=2e-06
            )
            assert values.dtype == gradient.dtype == dtype
            expected = b.numpy((gradient * direction).sum(axis=(1, 2)))
            assert np.linalg.norm(b.numpy(tangent)) > 0.001
            np.testing.assert_allclose(
                b.numpy(tangent), expected, rtol=2e-06, atol=2e-06
            )
            np.testing.assert_array_equal(b.numpy(jacobian[0, 1]), 0)
            np.testing.assert_array_equal(b.numpy(jacobian[1, 0]), 0)
            np.testing.assert_allclose(
                b.numpy(jacobian[0, 0]), b.numpy(gradient[0]), rtol=2e-06, atol=2e-06
            )
            assert value.device == gradient.device == b.device
            step = (
                0.005
                if dtype == np.float32
                else 0.001 if terms == ("hbond",) else 1e-05
            )
            numeric = (score(x + step * direction) - score(x - step * direction)) / (
                2 * step
            )
            tolerance = 0.003 if dtype == np.float32 else 2e-06
            np.testing.assert_allclose(
                b.numpy(tangent), b.numpy(numeric), rtol=tolerance, atol=tolerance
            )
    if empty_term is not None:
        b.molecule(case if empty_case is None else empty_case, terms=(empty_term,))
        pose = b.pose[:0]
        score = b.score_function.render_whole_pose_scoring_module(pose)
        values = b.module.jit(score)(pose.coords)
        gradient = b.module.jit(b.module.grad(lambda x: score(x).sum()))(pose.coords)
        assert values.shape == (0,) and gradient.shape == pose.coords.shape
