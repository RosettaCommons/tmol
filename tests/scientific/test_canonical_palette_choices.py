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


@pytest.mark.parametrize("identity", ("LEU", "DLEU"))
def test_zero_dunbrack_input_retains_nonzero_design_scores_and_forces(
    backend, identity
):
    from .test_dunbrack import TERMS
    from .test_molecular_sampling import sampling_task
    from .rotamer_execution import RotamerExecution

    files = DOMAIN_CASES["ligand"].parameter_files
    palette = MolecularCase(
        files, (("ALA", "DALA", "LEU", "DLEU", "LG1"),), ((1,) * 5,)
    )
    b = backend.molecule(palette, terms=TERMS)
    database = b.parameter_database
    original = "DALA" if identity == "DLEU" else "ALA"
    target = MolecularCase(files, ((original,) * 3,), ((3,),))
    co, canonical = canonical_input(target, database)
    rebuild_canonical(b, co, canonical)
    values, gradients = b.channel_values_and_gradients(b.coords())
    np.testing.assert_array_equal(values, 0)
    np.testing.assert_array_equal(gradients, 0)
    original_palette = b.pose.packed_block_types
    original_names = tuple(rt.name for rt in original_palette.active_block_types)

    # An inactive input term must still score newly sampled design identities.
    front = __import__("tmol_jax" if b.framework == "jax" else "tmol")
    _, sampler, _ = sampling_task(b)
    task = front.PackerTask(b.pose, front.PackerPalette(noncanonical_design=True))
    name3 = next(
        rt.name3 for rt in original_palette.active_block_types if rt.name == identity
    )
    task.restrict_absent_name3s({name3})
    task.add_conformer_sampler(sampler)
    execution = RotamerExecution(b, family="dunbrack", task=task)
    rotamers = execution.rotamers
    counts = b.numpy(rotamers.n_rots_for_block)[0]
    assert np.all(counts[:3] > 1)
    selected = b.numpy(rotamers.rot_offset_for_block)[0, :3].astype(int)
    types = b.numpy(rotamers.block_type_ind_for_rot)
    assert all(
        original_palette.active_block_types[types[i]].name == identity for i in selected
    )
    coords = b.numpy(rotamers.coords).astype(np.float64)
    coords += np.random.default_rng(1802).normal(0, 0.01, coords.shape)
    weights = np.zeros((3, len(types)))
    weights[:, selected] = 1
    values, gradient, _ = execution.value_and_grad(coords, weights)

    reference = type(b)(b.name).molecule(
        MolecularCase(files, ((identity,) * 3,), ((3,),)),
        terms=TERMS,
        parameter_database=database,
    )
    physical = reference.coords().astype(np.float64)
    offsets = b.numpy(rotamers.coord_offset_for_rot)
    expected_gradient = np.zeros_like(coords)
    atom_count = reference.pose.block_type(0, 0).n_atoms
    reference_offsets = reference.numpy(reference.pose.block_coord_offset)[0]
    for block, rotamer in enumerate(selected):
        start, out = int(offsets[rotamer]), int(reference_offsets[block])
        physical[0, out : out + atom_count] = coords[start : start + atom_count]
    expected_values, forces = reference.channel_values_and_gradients(physical)
    assert np.linalg.norm(expected_values) > 1
    assert np.linalg.norm(forces) > 1
    for block, rotamer in enumerate(selected):
        start, out = int(offsets[rotamer]), int(reference_offsets[block])
        expected_gradient[start : start + atom_count] = forces[
            :, 0, out : out + atom_count
        ].sum(0)
    np.testing.assert_allclose(
        (values * weights).sum(1), expected_values[:, 0], rtol=3e-6, atol=3e-5
    )
    np.testing.assert_allclose(gradient, expected_gradient, rtol=3e-5, atol=3e-5)
    assert b.pose.packed_block_types is original_palette
    assert (
        tuple(rt.name for rt in original_palette.active_block_types) == original_names
    )
    assert b.parameter_database is database
