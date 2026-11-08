"""Connected backbone values and derivatives across shared molecular cases."""

import numpy as np
import pytest

from .test_score_contract import check_coordinate_transforms

from .molecules import BACKBONE_CASES, DOMAIN_CASES


@pytest.mark.parametrize("name", BACKBONE_CASES)
def test_connected_backbone_channels_and_coordinate_derivatives(backend, name):
    backend.molecule(BACKBONE_CASES[name], terms=("rama", "omega"))
    x = backend.coords()
    values, gradients = backend.channel_values_and_gradients(x)
    assert np.isfinite(values).all() and np.isfinite(gradients).all()
    assert np.all(np.abs(values).sum(axis=1) > 1e-3)
    assert np.all(np.linalg.norm(gradients.reshape(2, -1), axis=1) > 1e-3)
    direction = np.random.default_rng(827).normal(size=x.shape).astype(x.dtype)
    direction /= np.linalg.norm(direction)
    step = 5e-3
    numeric = (
        backend.scores(x + step * direction, sum_terms=False, apply_weights=False)
        - backend.scores(x - step * direction, sum_terms=False, apply_weights=False)
    ) / (2 * step)
    np.testing.assert_allclose(
        (gradients * direction).sum(axis=(2, 3)), numeric, rtol=3e-3, atol=3e-3
    )
    np.testing.assert_allclose(gradients.sum(axis=2), 0, atol=2e-4)

    if name == "mixed":
        check_coordinate_transforms(
            backend, BACKBONE_CASES[name], ("rama", "omega"), empty_term="rama"
        )

        if backend.framework == "jax":
            _check_variant_tables()


def test_absent_backbone_reference_does_not_invent_peptide_energy(backend):
    backend.molecule(DOMAIN_CASES["connected_noncanonical"], terms=("rama", "omega"))
    values, gradients = backend.channel_values_and_gradients(backend.coords())
    # B3Q/B3D have real bonded parameters and peptide connections, but their
    # committed records assign no backbone statistical reference.
    np.testing.assert_array_equal(values, 0)
    np.testing.assert_array_equal(gradients, 0)


def _check_variant_tables():
    from tmol_jax.database import ParameterDatabase
    from tmol_jax.io import default_packed_block_types

    database = ParameterDatabase.get_default()
    block_types = default_packed_block_types().active_block_types
    by_name = {block_type.name: block_type for block_type in block_types}
    from tmol_core.scoring.backbone_torsion import BackboneTorsionPreparation

    rama, omega = database.scoring.rama, database.scoring.omega_bbdep
    prep = BackboneTorsionPreparation(
        rama.rama_lookup,
        omega.bbdep_omega_lookup,
        [row.table_id for row in rama.rama_tables],
        [row.table_id for row in omega.bbdep_omega_tables],
    )
    for variant, reference in (
        ("HIS_D", "HIS"),
        ("HIS_POS", "HIS"),
        ("CYD", "CYS"),
        ("ALA:nterm", "ALA"),
        ("ALA:cterm", "ALA"),
        ("PRO:nterm", "PRO"),
    ):
        actual, expected = (
            prep.prepare_block(by_name[name]) for name in (variant, reference)
        )
        np.testing.assert_array_equal(actual.rama_table_inds, expected.rama_table_inds)
        np.testing.assert_array_equal(
            actual.omega_table_inds, expected.omega_table_inds
        )
        np.testing.assert_array_equal(actual.is_pro, expected.is_pro)
