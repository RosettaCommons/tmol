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
