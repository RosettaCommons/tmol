"""Shared nonzero chiral disulfide and parameter-context invariants."""

from dataclasses import replace

import attrs
import numpy as np
import pytest

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
