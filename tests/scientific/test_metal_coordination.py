"""Nonzero sites, virtual fans and bridges share one scientific contract."""

import attrs
import numpy as np
import pytest

from .execution import Execution
from .metal_molecules import METAL_CASES, MetalCase


@pytest.mark.parametrize("name", METAL_CASES)
def test_metal_coordination_values_and_coordinate_derivatives(backend, name):
    backend.molecule(METAL_CASES[name], terms=("metal_coordination",))
    x = backend.coords()
    values, gradient = backend.value_and_grad(x)
    assert np.isfinite(values).all() and np.isfinite(gradient).all()
    assert np.all(np.abs(values) > 1e-3) and np.linalg.norm(gradient) > 1e-3
    direction = np.random.default_rng(514).normal(size=x.shape).astype(x.dtype)
    direction /= np.linalg.norm(direction)
    step = 5e-3
    numeric = (
        backend.scores(x + step * direction) - backend.scores(x - step * direction)
    ) / (2 * step)
    np.testing.assert_allclose(
        (gradient * direction).sum(axis=(1, 2)), numeric, rtol=3e-3, atol=3e-3
    )
    np.testing.assert_allclose(gradient.sum(axis=1), 0, atol=2e-3)


def test_metal_custom_widths_scale_energy_and_coordinate_gradient(backend):
    case = METAL_CASES["bridging_donor"]
    backend.molecule(case, terms=("metal_coordination",))
    values, gradient = backend.value_and_grad(backend.coords())
    db = backend.parameter_database
    params = db.scoring.metal_coordination
    widths = params.global_parameters
    scaled = attrs.evolve(
        params,
        global_parameters=attrs.evolve(
            widths,
            **{
                name: 2 * getattr(widths, name)
                for name in ("radial_sd", "lateral_sd", "fan_sd", "bridge_width")
            },
        ),
        ions=tuple(
            attrs.evolve(
                row,
                **{
                    name: None if getattr(row, name) is None else 2 * getattr(row, name)
                    for name in ("radial_sd", "lateral_sd")
                },
            )
            for row in params.ions
        ),
        well_depths=tuple(
            attrs.evolve(row, depth=row.depth / 4) for row in params.well_depths
        ),
    )
    changed = attrs.evolve(
        db, scoring=attrs.evolve(db.scoring, metal_coordination=scaled)
    )
    other = Execution(backend.name).molecule(
        case, terms=("metal_coordination",), parameter_database=changed
    )
    np.testing.assert_array_equal(backend.coords(), other.coords())
    new_value, new_gradient = other.value_and_grad(other.coords())
    np.testing.assert_allclose(new_value, values / 4, rtol=3e-6, atol=3e-5)
    np.testing.assert_allclose(new_gradient, gradient / 4, rtol=3e-6, atol=3e-5)
    np.testing.assert_array_equal(backend.scores(backend.coords()), values)


def test_metal_connection_requires_parameterized_donor(backend):
    case = MetalCase(
        (),
        (("ZN_tetrahedral", "ASP:metal_CB"),),
        ((1, 1),),
        (("ASP", "CB", 1),),
        (((0, "site1", 1, "metal_CB"),),),
    )
    with pytest.raises(ValueError, match="not a donor"):
        backend.molecule(case, terms=("metal_coordination",))
        backend.scores(backend.coords())
