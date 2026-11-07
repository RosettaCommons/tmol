"""Current conformers preserve molecular coordinates and identity exactly."""

import numpy as np
import pytest

from .molecules import DOMAIN_CASES
from .test_current_conformers import current_samplers
from .test_molecular_sampling import block_mask


def coordinate_dihedral(xyz):
    """Independent plane-normal geometry, retaining the coordinates' precision."""
    a, b, c, d = np.asarray(xyz, dtype=np.float64)
    axis = c - b
    n1, n2 = np.cross(b - a, axis), np.cross(axis, d - c)
    return np.arctan2(
        np.dot(np.cross(n1, n2), axis / np.linalg.norm(axis)), np.dot(n1, n2)
    )


@pytest.mark.parametrize(
    "case_name",
    (
        "ligand",
        "connected_noncanonical",
        "d_peptide",
        "modified_dna",
        "metal_mixed",
        "mixed_batch",
    ),
)
def test_current_conformer_assembly_preserves_molecular_atoms(backend, case_name):
    backend.molecule(DOMAIN_CASES[case_name], terms=("ref",))
    if backend.framework == "jax":
        from tmol_jax.pack import PackerTask, SetPackerTask, PackerPalette
        from tmol_jax.pack.rotamers import build_rotamers
    else:
        from tmol import PackerTask, PackerPalette
        from tmol.pack import SetPackerTask, build_rotamers
    task = PackerTask(backend.pose, PackerPalette())
    task.restrict_to_repacking()
    current, fallback = current_samplers(backend)
    task.add_conformer_sampler(current)
    if not any(s.sampler_name() == "FallbackSampler" for s in task.conformer_samplers):
        task.add_conformer_sampler(fallback)
    fixed = np.zeros(backend.numpy(task.is_real_block).shape, bool)
    fixed[0, 0] = True
    task.disable_packing_by_block_mask(block_mask(backend, fixed))
    original = backend.coords().copy()
    if backend.framework == "torch":
        backend.pose.coords = backend.pose.coords.detach().requires_grad_()
    concrete = SetPackerTask.from_packer_task(task)
    poses, result = build_rotamers(
        backend.pose,
        concrete,
        backend.parameter_database.chemical,
    )
    assert poses is backend.pose
    assert result.coords.device == backend.pose.coords.device
    np.testing.assert_array_equal(backend.coords(), original)
    counts = backend.numpy(result.n_rots_for_block)
    np.testing.assert_array_equal(
        counts, backend.numpy(task.is_real_block).astype(np.int32)
    )
    assert result.n_rotamers_total == counts.sum()
    block_types = backend.numpy(backend.pose.block_type_ind)
    offsets = backend.numpy(backend.pose.block_coord_offset)
    pbt = backend.pose.packed_block_types
    for i, (pose, block, block_type, offset) in enumerate(
        zip(
            *(
                backend.numpy(getattr(result, name))
                for name in (
                    "pose_for_rot",
                    "block_ind_for_rot",
                    "block_type_ind_for_rot",
                    "coord_offset_for_rot",
                )
            )
        )
    ):
        assert block_type == block_types[pose, block]
        n_atoms = pbt.active_block_types[block_type].n_atoms
        np.testing.assert_array_equal(
            backend.numpy(result.coords)[offset : offset + n_atoms],
            original[pose, offsets[pose, block] : offsets[pose, block] + n_atoms],
        )
        np.testing.assert_array_equal(
            backend.numpy(result.pose_ind_for_atom)[offset : offset + n_atoms], pose
        )
    # Each fixed residue is represented by the input, despite no allowed type.
    assert counts[0, 0] == 1
    # Every real atom occurs once, including fixed residues. Padding must have
    # zero derivative; the known quadratic has curvature two on real atoms.
    expected = np.zeros_like(original)
    curvature = np.zeros_like(original)
    for pose, block in np.argwhere(block_types >= 0):
        offset = offsets[pose, block]
        size = pbt.active_block_types[block_types[pose, block]].n_atoms
        expected[pose, offset : offset + size] = (
            2 * original[pose, offset : offset + size]
        )
        curvature[pose, offset : offset + size] = 0.5
    if backend.framework == "jax":
        jax = backend.module
        # Cardinality preparation is eager; differentiate its fixed execution
        # plan, the same graph consumed by subsequent build_rotamers calls.
        execute = concrete._assembly_execution[2]
        payloads = tuple(() for _ in concrete.conformer_samplers)
        grad = jax.grad(lambda x: (execute(x, payloads) ** 2).sum())
        gradient, hvp = jax.jit(
            lambda x: jax.jvp(grad, (x,), (jax.numpy.full_like(x, 0.25),))
        )(backend.pose.coords)
    else:
        torch = backend.module
        gradient = torch.autograd.grad(
            result.coords.square().sum(), backend.pose.coords, create_graph=True
        )[0]
        hvp = torch.autograd.grad((gradient * 0.25).sum(), backend.pose.coords)[0]
    np.testing.assert_array_equal(backend.numpy(gradient), expected)
    np.testing.assert_array_equal(backend.numpy(hvp), curvature)


@pytest.mark.parametrize(
    "case_name", ("l_peptide", "d_peptide", "variants", "mixed_batch")
)
def test_sampled_conformer_coordinates_match_requested_dihedrals(backend, case_name):
    from .dunbrack_molecules import DUNBRACK_CASES
    from .test_molecular_sampling import sampling_task

    backend.molecule(DUNBRACK_CASES[case_name], terms=("ref",))
    task, sampler, concrete = sampling_task(backend)
    _, fallback = current_samplers(backend)
    if not any(s.sampler_name() == "FallbackSampler" for s in task.conformer_samplers):
        task.add_conformer_sampler(fallback)
    fixed = concrete(task)
    counts, choices, defining, angles = map(
        backend.numpy, sampler.sample_chi_for_poses(backend.pose, fixed)
    )
    if backend.framework == "jax":
        from tmol_jax.pack.rotamers import build_rotamers
    else:
        from tmol.pack import build_rotamers
    original = backend.coords().copy()
    _, result = build_rotamers(backend.pose, fixed, backend.parameter_database.chemical)
    checked = check_sampled_dihedrals(
        backend, fixed, result, counts, choices, defining, angles, original
    )
    assert checked > 10
    exported = result.to_atom_array(backend.pose.packed_block_types)
    np.testing.assert_array_equal(exported.coord, backend.numpy(result.coords))
    np.testing.assert_array_equal(
        np.unique(exported.model), np.arange(result.n_rotamers_total)
    )
    assert set(exported.chain_id) == {"A"}
    elements = {
        atom_type.name: atom_type.element
        for atom_type in backend.parameter_database.chemical.atom_types
    }
    for model, (type_index, block) in enumerate(
        zip(
            backend.numpy(result.block_type_ind_for_rot),
            backend.numpy(result.block_ind_for_rot),
        )
    ):
        residue = backend.pose.packed_block_types.active_block_types[type_index]
        atoms = exported[exported.model == model]
        assert set(atoms.res_id) == {block + 1}
        assert set(atoms.res_name) == {residue.name3}
        np.testing.assert_array_equal(atoms.atom_name, [a.name for a in residue.atoms])
        np.testing.assert_array_equal(
            atoms.element, [elements[a.atom_type] for a in residue.atoms]
        )


def check_sampled_dihedrals(
    backend, fixed, result, counts, choices, defining, angles, original
):
    """Measure public chemical torsions and retained backbone independently."""
    from tmol_core.pack.chi import chi_moves_fourth_atom

    coordinates = backend.numpy(result.coords)
    assert np.isfinite(coordinates).all()
    np.testing.assert_array_equal(backend.coords(), original)
    pbt = backend.pose.packed_block_types
    result_types = backend.numpy(result.block_type_ind_for_rot)
    result_poses = backend.numpy(result.pose_for_rot)
    result_blocks = backend.numpy(result.block_ind_for_rot)
    offsets = backend.numpy(result.coord_offset_for_rot)
    checked = 0
    for choice in np.flatnonzero(counts):
        pose = backend.numpy(fixed.cons_bt_pose)[choice]
        block = backend.numpy(fixed.cons_bt_block)[choice]
        type_index = backend.numpy(fixed.cons_bt_block_type)[choice]
        residue = pbt.active_block_types[type_index]
        selected = np.flatnonzero(
            (result_poses == pose)
            & (result_blocks == block)
            & (result_types == type_index)
        )
        assert len(selected) == counts[choice]
        for rotamer, row in zip(
            selected, np.flatnonzero(choices == choice), strict=True
        ):
            xyz = coordinates[offsets[rotamer] : offsets[rotamer] + residue.n_atoms]
            # Atom identities come from the public chemical torsion definition,
            # independent of the native correction's per-defining-atom table.
            for column, atom in enumerate(defining[row]):
                if atom < 0:
                    continue
                torsion = residue.torsion_to_uaids.get(f"chi{column + 1}")
                if torsion is None:
                    continue
                atoms = np.asarray(torsion)[:, 0]
                if np.any(atoms < 0):
                    continue
                # A ring-closing chi controls an alternate branch; its fixed
                # ring geometry is not an independently adjustable dihedral.
                if not chi_moves_fourth_atom(residue, int(atom), int(atoms[3])):
                    continue
                measured = coordinate_dihedral(xyz[atoms])
                error = np.arctan2(
                    np.sin(measured - angles[row, column]),
                    np.cos(measured - angles[row, column]),
                )
                assert abs(error) < 2e-5, (residue.name, column, error)
                checked += 1
            # Repacking preserves the input backbone in its original frame.
            names = residue.properties.polymer.mainchain_atoms or ()
            retained = [residue.atom_to_idx[name] for name in names]
            source_offset = backend.numpy(backend.pose.block_coord_offset)[pose, block]
            np.testing.assert_allclose(
                xyz[retained],
                original[pose, source_offset + np.asarray(retained, dtype=np.int32)],
                atol=2e-5,
                rtol=0,
            )
    return checked
