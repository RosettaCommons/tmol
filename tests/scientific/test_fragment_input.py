"""Fragment input retains physical molecules, atoms and full molecular physics."""

import numpy as np
import pytest

from .fragment_molecules import fragment_fixture


def _frontend(backend):
    if backend.framework == "jax":
        import tmol_jax as frontend
    else:
        import tmol as frontend
    return frontend


def _build(backend, atoms, parameters):
    return _frontend(backend).pose_stack_from_atom_array(
        atoms,
        backend.device,
        prepare_ligands=True,
        ligand_params_files=[str(parameters)],
        no_optH=True,
        return_context=True,
    )


@pytest.mark.parametrize("repeated,mixed", ((False, True), (True, False)))
def test_fragment_input_preserves_each_physical_ligand(backend, repeated, mixed):
    whole, annotated, parameters = fragment_fixture(repeated=repeated, mixed=mixed)
    pose, context = _build(backend, annotated, parameters)
    mapping = pose.split_block_mapping
    assert mapping is not None
    groups = {(e.pose_ind, e.group_ind) for e in mapping.entries}
    assert len(groups) == (2 if repeated else 1)
    by_identity = {}
    for entry in mapping.entries:
        group = by_identity.setdefault((entry.pose_ind, entry.group_ind), [])
        group.append(entry)
    offsets = backend.numpy(pose.block_coord_offset)
    for entries in by_identity.values():
        first = entries[0]
        source = whole[
            (whole.chain_id == first.orig_chain_label)
            & (whole.res_id == first.orig_residue_label)
            & (whole.res_name == "LG1")
        ]
        assert len(source) > 0
        expected = {name: xyz for name, xyz in zip(source.atom_name, source.coord)}
        seen = set()
        for entry in entries:
            block = pose.block_type(entry.pose_ind, entry.block_ind)
            offset = offsets[entry.pose_ind, entry.block_ind]
            for atom_index, atom in enumerate(block.atoms):
                if atom.name in expected:
                    np.testing.assert_allclose(
                        backend.numpy(pose.coords)[entry.pose_ind, offset + atom_index],
                        expected[atom.name],
                        atol=2e-5,
                        rtol=2e-5,
                    )
                    seen.add(atom.name)
        assert seen == set(expected)
    backend.pose, backend.parameter_database = pose, context.parameter_database
    channels, _ = backend.use_beta2016()
    values, gradients = backend.channel_values_and_gradients(backend.coords())
    assert (
        len(channels) == 25
        and np.isfinite(values).all()
        and np.isfinite(gradients).all()
    )
    gen = channels.index("gen_torsions")
    assert np.any(np.abs(values[gen]) > 1e-3) and np.linalg.norm(gradients[gen]) > 1e-3


def _physical_atoms(backend, pose):
    entries = {
        (entry.pose_ind, entry.block_ind): entry
        for entry in (
            pose.split_block_mapping.entries if pose.split_block_mapping else ()
        )
    }
    offsets = backend.numpy(pose.block_coord_offset)
    types = backend.numpy(pose.block_type_ind)
    result = {}
    for p, b in zip(*np.nonzero(types >= 0)):
        entry = entries.get((p, b))
        identity = (
            (entry.orig_chain_label, entry.orig_residue_label, entry.orig_ins_code)
            if entry is not None
            else (
                pose.pdb_info.chain_labels[p][b],
                pose.pdb_info.residue_labels[p, b],
                pose.pdb_info.residue_insertion_codes[p][b],
            )
        )
        for i, atom in enumerate(pose.block_type(p, b).atoms):
            key = (p, *identity, atom.name)
            assert key not in result
            result[key] = (p, int(offsets[p, b]) + i)
    return result


@pytest.mark.parametrize("repeated,mixed", ((False, True), (True, False)))
def test_fragment_cut_preserves_all_term_values_and_physical_gradients(
    backend, repeated, mixed
):
    from contextlib import nullcontext

    whole, annotated, parameters = fragment_fixture(repeated=repeated, mixed=mixed)
    output = []
    context = (
        backend.module.enable_x64() if backend.framework == "jax" else nullcontext()
    )
    with context:
        for atoms in (whole, annotated):
            pose, chemistry = _build(backend, atoms, parameters)
            backend.pose, backend.parameter_database = (
                pose,
                chemistry.parameter_database,
            )
            backend.use_beta2016()
            coords = backend.coords().astype(np.float64)
            values, gradient = backend.channel_values_and_gradients(coords)
            identity = _physical_atoms(backend, pose)
            output.append((identity, coords, values, gradient))
    left, right = output
    assert left[0].keys() == right[0].keys()
    np.testing.assert_allclose(right[2], left[2], rtol=2e-6, atol=2e-6)
    for atom in left[0]:
        p, a = left[0][atom]
        q, b = right[0][atom]
        np.testing.assert_array_equal(left[1][p, a], right[1][q, b])
        np.testing.assert_allclose(
            left[3][:, p, a],
            right[3][:, q, b],
            rtol=2e-6,
            atol=2e-6,
            err_msg=str(atom),
        )


def _unsplit(backend, pose):
    if backend.framework == "jax":
        from tmol_jax.ligand.fragment_imposition import unsplit_pose_stack
    else:
        from tmol.ligand._fragmentation import _unsplit_pose_stack as unsplit_pose_stack
    return unsplit_pose_stack(pose)


@pytest.mark.parametrize("repeated,mixed", ((False, True), (True, False)))
def test_fragment_reconstruction_retains_atoms_metadata_and_derivatives(
    backend, repeated, mixed
):
    import attrs

    _, annotated, parameters = fragment_fixture(repeated=repeated, mixed=mixed)
    pose, chemistry = _build(backend, annotated, parameters)
    identity = _physical_atoms(backend, pose)
    count = pose.coords.shape[1]
    pose.pdb_info.atom_occupancy[:] = np.linspace(0.2, 0.9, count)
    pose.pdb_info.atom_b_factor[:] = np.arange(count) / 3 + 2
    recovered = _unsplit(backend, pose)
    actual = _physical_atoms(backend, recovered)
    assert identity.keys() == actual.keys()
    for atom, (p, a) in identity.items():
        q, b = actual[atom]
        np.testing.assert_array_equal(
            backend.numpy(pose.coords)[p, a], backend.numpy(recovered.coords)[q, b]
        )
        for field in ("atom_occupancy", "atom_b_factor"):
            assert (
                getattr(pose.pdb_info, field)[p, a]
                == getattr(recovered.pdb_info, field)[q, b]
            )
    # A native differentiable gather must preserve coordinate precision and the
    # derivative of a nonlinear loss over every original atom.
    if backend.framework == "torch":
        coords = (
            pose.coords.detach().clone().to(backend.module.float64).requires_grad_()
        )
        result = _unsplit(backend, attrs.evolve(pose, coords=coords))
        (gradient,) = backend.module.autograd.grad((result.coords**2).sum(), coords)
        assert result.coords.dtype == coords.dtype
    else:
        import jax.numpy as jnp

        with backend.module.enable_x64():
            coords = pose.coords.astype(jnp.float64)
            gradient = backend.module.grad(
                lambda x: jnp.sum(_unsplit(backend, pose.replace(coords=x)).coords ** 2)
            )(coords)
            assert (
                _unsplit(backend, pose.replace(coords=coords)).coords.dtype
                == coords.dtype
            )
    np.testing.assert_allclose(
        backend.numpy(gradient), 2 * backend.numpy(coords), atol=0, rtol=0
    )
    backend.pose, backend.parameter_database = recovered, chemistry.parameter_database
    backend.use_beta2016()
    values, gradient = backend.channel_values_and_gradients(backend.coords())
    assert np.isfinite(values).all() and np.isfinite(gradient).all()


def test_fragment_reconstruction_preserves_constraint_atoms_and_parameters(backend):
    import attrs

    if backend.framework == "jax":
        from tmol_jax.pose.constraint_set import ConstraintSet
    else:
        from tmol.pose._constraint_set import ConstraintSet
    _, annotated, parameters = fragment_fixture(repeated=True, mixed=False)
    pose, _ = _build(backend, annotated, parameters)
    entries = pose.split_block_mapping.entries
    atoms = np.array([[[0, entry.block_ind, 0] for entry in entries[:3]]], np.int32)

    def function(x, parameters):
        return ((x[:, 0] - x[:, 1]) ** 2).sum(axis=-1) * parameters[:, 0]

    params = np.array([[2.75]], np.float32)
    if backend.framework == "torch":
        params = backend.module.as_tensor(params, device=backend.device)
        atoms = backend.module.as_tensor(atoms, device=backend.device)
    else:
        params = backend.module.device_put(params, backend.device)
    constraints = ConstraintSet.create_empty(backend.device, len(pose)).add_constraints(
        function, atoms, params
    )
    pose = attrs.evolve(pose, constraint_set=constraints)
    recovered = _unsplit(backend, pose)
    actual = recovered.constraint_set
    np.testing.assert_array_equal(backend.numpy(actual.constraint_params), [[2.75]])
    assert actual.constraint_functions == constraints.constraint_functions
    before = backend.numpy(constraints.constraint_atoms)
    after = backend.numpy(actual.constraint_atoms)
    assert np.array_equal(before[..., 0] >= 0, after[..., 0] >= 0)
    for source, target in zip(before[before[..., 0] >= 0], after[after[..., 0] >= 0]):
        p, b, a = source
        q, c, d = target
        left = backend.numpy(pose.coords)[
            p, backend.numpy(pose.block_coord_offset)[p, b] + a
        ]
        right = backend.numpy(recovered.coords)[
            q, backend.numpy(recovered.block_coord_offset)[q, c] + d
        ]
        np.testing.assert_array_equal(left, right)
    # Three constrained fragments belong to exactly two original molecules.
    np.testing.assert_array_equal(
        backend.numpy(actual.constraint_num_unique_blocks), [2]
    )


def test_fragment_default_input_packing_and_minimization(backend):
    _, annotated, parameters = fragment_fixture(mixed=True)
    pose, context = _frontend(backend).pose_stack_from_atom_array(
        annotated,
        backend.device,
        prepare_ligands=True,
        ligand_params_files=[str(parameters)],
        no_optH=False,
        packer_seed=73,
        return_context=True,
    )
    backend.pose, backend.parameter_database = pose, context.parameter_database
    backend.use_beta2016()
    original = backend.coords().copy()
    initial = backend.scores(original)
    movable = np.zeros(original.shape[:2], bool)
    for entry in pose.split_block_mapping.entries:
        offset = int(
            backend.numpy(pose.block_coord_offset)[entry.pose_ind, entry.block_ind]
        )
        count = len(pose.block_type(entry.pose_ind, entry.block_ind).atoms)
        movable[entry.pose_ind, offset : offset + count] = True
    result = backend.minimize(movable, iterations=3)
    final = backend.scores(result)
    assert np.isfinite(final).all() and np.all(final < initial - 1e-4)
    np.testing.assert_array_equal(result[~movable], original[~movable])
    np.testing.assert_array_equal(backend.coords(), original)
