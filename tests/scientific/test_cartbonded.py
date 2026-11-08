"""Shared Cartesian-bonded physics on real molecular topologies."""

import numpy as np
import pytest

from .test_score_contract import check_coordinate_transforms

from .molecules import BACKBONE_CASES, DOMAIN_CASES, MolecularCase

CARTESIAN_CASES = {
    **DOMAIN_CASES,
    "hydroxyls": MolecularCase((), (("SER", "THR", "TYR"),), ((3,),)),
}

TERMS = (
    "cart_lengths",
    "cart_angles",
    "cart_torsions",
    "cart_impropers",
    "cart_hxltorsions",
)


@pytest.mark.parametrize("name", CARTESIAN_CASES)
def test_cartesian_bonded_channels_and_coordinate_derivatives(backend, name):
    backend.molecule(CARTESIAN_CASES[name], terms=TERMS)
    x = backend.coords()
    values, gradients = backend.channel_values_and_gradients(x)
    assert np.isfinite(values).all() and np.isfinite(gradients).all()
    assert np.abs(values).sum() > 1e-3
    assert np.linalg.norm(gradients) > 1e-3
    # All five channels are checked, including genuinely absent interactions.
    direction = np.random.default_rng(195).normal(size=x.shape).astype(x.dtype)
    direction /= np.linalg.norm(direction)
    step = 5e-3
    numeric = (
        backend.scores(x + step * direction, sum_terms=False, apply_weights=False)
        - backend.scores(x - step * direction, sum_terms=False, apply_weights=False)
    ) / (2 * step)
    analytic = (gradients * direction).sum(axis=(2, 3))
    np.testing.assert_allclose(analytic, numeric, rtol=3e-3, atol=3e-3)
    np.testing.assert_allclose(gradients.sum(axis=2), 0, atol=1e-3)

    if name == "mixed_batch":
        check_coordinate_transforms(
            backend,
            CARTESIAN_CASES[name],
            TERMS,
            empty_term="cart_lengths",
            empty_case=BACKBONE_CASES["mixed"],
        )


@pytest.mark.parametrize("name", ("ligand", "connected_noncanonical"))
def test_cartesian_spring_context_reuse_preserves_values_and_gradients(backend, name):
    import attrs

    backend.molecule(DOMAIN_CASES[name], terms=("cart_lengths",))
    pose = backend.pose
    palette = pose.packed_block_types
    coords = backend.coords()
    expected = backend.value_and_grad(coords)
    assert np.max(np.abs(expected[0])) > 1e-3
    assert np.linalg.norm(expected[1]) > 1e-3
    database = backend.parameter_database
    cart = database.scoring.cartbonded

    def springs(record):
        return attrs.evolve(
            record,
            length_parameters=tuple(
                attrs.evolve(group, K=3.25 * group.K)
                for group in record.length_parameters
            ),
        )

    changed = type(cart).from_cartres_dict(
        {name: springs(record) for name, record in cart.residue_params.items()},
        connection_params=tuple(map(springs, cart.connection_params)),
    )
    scaled = attrs.evolve(
        database, scoring=attrs.evolve(database.scoring, cartbonded=changed)
    )
    for context, factor in (
        (database, 1),
        (scaled, 3.25),
        (database, 1),
        (scaled, 3.25),
    ):
        backend.use_parameter_database(context)
        value, gradient = backend.value_and_grad(coords)
        for actual, reference in zip((value, gradient), expected):
            np.testing.assert_allclose(actual, factor * reference, rtol=3e-6, atol=3e-5)
        assert backend.pose is pose and backend.pose.packed_block_types is palette
        np.testing.assert_array_equal(backend.coords(), coords)
