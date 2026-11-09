"""Shared nonzero chiral disulfide and parameter-context invariants."""

from dataclasses import replace

import attrs
import numpy as np
import pytest

from .test_score_contract import check_coordinate_transforms

from .disulfide_molecules import DISULFIDE_CASES
from .execution import Execution


@pytest.mark.parametrize("name", DISULFIDE_CASES)
def test_disulfide_coordinate_derivatives_and_translation(backend, name):
    backend.molecule(DISULFIDE_CASES[name], terms=("disulfide",))
    x = backend.coords()
    value, gradient = backend.value_and_grad(x)
    assert np.isfinite(value).all() and np.isfinite(gradient).all()
    assert np.all(np.abs(value) > 1e-3) and np.linalg.norm(gradient) > 1e-3
    direction = np.random.default_rng(421).normal(size=x.shape).astype(x.dtype)
    direction /= np.linalg.norm(direction)
    step = 5e-3
    numeric = (
        backend.scores(x + step * direction) - backend.scores(x - step * direction)
    ) / (2 * step)
    np.testing.assert_allclose(
        (gradient * direction).sum(axis=(1, 2)), numeric, rtol=3e-3, atol=3e-3
    )
    np.testing.assert_allclose(gradient.sum(axis=1), 0, atol=3e-5)
    if name == "ld":
        # Published-equation golden for this committed 6DMZ derivative;
        # independently established in Torch's test_chirality.py.
        np.testing.assert_allclose(value, -0.048889146701669, atol=3e-5, rtol=3e-5)

    if name == "mixed_batch":
        check_coordinate_transforms(
            backend, DISULFIDE_CASES[name], ("disulfide",), empty_term="disulfide"
        )

        if backend.framework == "jax":
            with backend.module.default_device(backend.device):
                _check_protein_disulfide_reference()


@pytest.mark.parametrize("pair", ["ll", "ld"])
def test_disulfide_mirror_and_permutation_invariants(backend, pair):
    case = DISULFIDE_CASES[pair]
    backend.molecule(case, terms=("disulfide",))
    x = backend.coords()
    value, gradient = backend.value_and_grad(x)
    reflected = replace(
        case,
        sequences=(
            tuple("CYD" if name == "DCYD" else "DCYD" for name in case.sequences[0]),
        ),
    )
    mirror = Execution(backend.name).molecule(reflected, terms=("disulfide",))
    mirror_value, mirror_gradient = mirror.value_and_grad(-x)
    np.testing.assert_allclose(mirror_value, value, atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(mirror_gradient, -gradient, atol=3e-5, rtol=3e-5)
    swapped = replace(case, sequences=(case.sequences[0][::-1],))
    reverse = Execution(backend.name).molecule(swapped, terms=("disulfide",))
    assert x.shape[1] % 2 == 0
    width = x.shape[1] // 2
    permutation = np.r_[np.arange(width, 2 * width), np.arange(width)]
    reverse_value, reverse_gradient = reverse.value_and_grad(x[:, permutation])
    np.testing.assert_allclose(reverse_value, value, atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(
        reverse_gradient[:, permutation], gradient, atol=3e-5, rtol=3e-5
    )


def scaled_disulfide_database(db):
    params = db.scoring.disulfide.global_parameters
    scaled = attrs.evolve(
        params,
        **{
            name: 2 * getattr(params, name)
            for name in ("wt_dih_ss", "wt_dih_cs", "wt_ang", "wt_len", "shift")
        },
    )
    changed = attrs.evolve(
        db,
        scoring=attrs.evolve(
            db.scoring,
            disulfide=attrs.evolve(db.scoring.disulfide, global_parameters=scaled),
        ),
    )
    return changed


def test_disulfide_custom_weights_scale_energy_and_gradients(backend):
    case = DISULFIDE_CASES["ld"]
    backend.molecule(case, terms=("disulfide",))
    value, gradient = backend.value_and_grad(backend.coords())
    changed = scaled_disulfide_database(backend.parameter_database)
    other = Execution(backend.name).molecule(
        case, terms=("disulfide",), parameter_database=changed
    )
    new_value, new_gradient = other.value_and_grad(other.coords())
    np.testing.assert_allclose(new_value, 2 * value, atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(new_gradient, 2 * gradient, atol=3e-5, rtol=3e-5)
    np.testing.assert_array_equal(backend.scores(backend.coords()), value)


def _check_protein_disulfide_reference():
    from pathlib import Path
    from tmol_jax import pose_stack_from_pdb, beta2016_score_function

    """Unweighted disulfide energy of 3PLC's four bonds, from TMol's Torch scorer."""
    from tmol_jax.score import ScoreType

    pose = pose_stack_from_pdb(
        Path(__file__).parents[2] / "tmol/tests/data/pdb/3plc.pdb"
    )
    score_function = beta2016_score_function()
    score = score_function.render_whole_pose_scoring_module(
        pose, sum_terms=False, apply_weights=False
    )
    channel = score_function.active_score_types().index(ScoreType.disulfide)
    energy = np.asarray(score(pose.coords))[channel, 0]
    np.testing.assert_allclose(energy, -3.25716, atol=2e-3)
