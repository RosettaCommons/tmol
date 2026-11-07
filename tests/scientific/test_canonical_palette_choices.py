"""Canonical input must retain supplied tautomer and unused design choices."""

import numpy as np
import pytest

from .molecules import DOMAIN_CASES, MolecularCase
from .test_canonical_molecular_input import canonical_input, rebuild_canonical
from .test_opth_sampling import opth_task


@pytest.mark.parametrize("identity", ("HIS", "DHIS"))
def test_canonical_input_preserves_tautomers_and_custom_design_palette(
    backend, identity
):
    parameter_files = DOMAIN_CASES["ligand"].parameter_files
    names = (identity, identity + "_D", "GLY", "LG1")
    b = backend.molecule(
        MolecularCase(parameter_files, (names,), ((1, 1, 1, 1),)), terms=()
    )
    original_types = {rt.name for rt in b.pose.packed_block_types.active_block_types}
    database = b.parameter_database
    target = MolecularCase(parameter_files, ((identity,),), ((1,),))
    co, canonical = canonical_input(target, database)
    rebuild_canonical(b, co, canonical)
    assert original_types <= {
        rt.name for rt in b.pose.packed_block_types.active_block_types
    }
    assert b.parameter_database is database
    np.testing.assert_array_equal(b.coords(), target.blocks(database)[0][0][2][None])
    task, sampler, concrete = opth_task(b)
    fixed = concrete(task)
    counts, owners, payload = sampler.create_samples_for_poses(b.pose, fixed)
    # Two protonation tautomers, each with the original and flipped ring.
    assert int(b.numpy(counts).sum()) == 4
    selected = b.numpy(fixed.cons_bt_block_type)[b.numpy(owners)]
    active = b.pose.packed_block_types.active_block_types
    assert {active[int(i)].name for i in selected} == {identity, identity + "_D"}
    b.use_beta2016()
    values, gradients = b.channel_values_and_gradients(b.coords())
    assert values.shape[0] == 25
    assert np.isfinite(values).all() and np.isfinite(gradients).all()
    assert np.linalg.norm(gradients) > 1
    # GLY was absent from the input but remains a legal requested design choice.
    front = __import__("tmol_jax" if b.framework == "jax" else "tmol")
    design = front.PackerTask(b.pose, front.PackerPalette())
    design.restrict_absent_name3s({"GLY"})
    choices = b.numpy(design.per_block_considered_block_types)
    allowed = b.numpy(design.per_block_is_block_type_allowed)
    assert {active[int(i)].name for i in choices[allowed]} == {"GLY"}
