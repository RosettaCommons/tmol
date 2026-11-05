import numpy as np

from tmol_jax.database import ParameterDatabase
from tmol_jax.io import default_packed_block_types


def test_rama_reference_selects_variant_tables_and_proline_identity():
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
