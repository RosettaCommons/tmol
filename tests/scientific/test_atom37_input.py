"""Prepared Atom37 inputs retain molecular identity and source derivatives."""

from contextlib import nullcontext
from importlib import import_module
import numpy as np
import pytest

from .molecules import DOMAIN_CASES, BACKBONE_CASES
from .metal_molecules import METAL_CASES
from .test_atom_array_input import bonded_array, CASES as CONNECTED_CASES

CASES = {name: DOMAIN_CASES[name] for name in ("ligand", "d_peptide", "dna", "rna")}
CASES.update(
    {name: CONNECTED_CASES[name] for name in ("cyclic", "heterochiral_disulfide")}
)
CASES.update(
    hydroxyproline=BACKBONE_CASES["hydroxyproline"], zinc=METAL_CASES["zinc_site"]
)


def atom37_fixture(backend, name, count=2):
    case = CASES.get(name, BACKBONE_CASES.get(name))
    b = backend.molecule(case, terms=())
    frontend = import_module("tmol_jax" if b.framework == "jax" else "tmol")
    atoms = bonded_array(b, case)
    ordinary, context = frontend.pose_stack_from_atom_array(
        atoms,
        b.device,
        param_db=b.parameter_database,
        no_optH=True,
        ligand_hydrogens="input",
        missing_density_distance_threshold=0.0,
        return_context=True,
    )
    atoms = frontend.atom_array_from_pose_stack(ordinary)
    atoms.set_annotation("token_id", np.arange(len(atoms), dtype=np.int64))
    atoms.set_annotation("atom37_slot", np.ones(len(atoms), np.int64))
    coordinates = np.full((count, len(atoms), 37, 3), np.nan, np.float32)
    for p in range(count):
        coordinates[p, :, 1] = atoms.coord + p * np.array([1, 2, 3], np.float32)
    io = import_module(frontend.__name__ + ".io")
    builder = io.prepare_atom37_pose_builder(atoms, context)
    return b, builder, coordinates, ordinary, context


def bind_atom37(backend, builder, coordinates):
    if backend.framework == "jax":
        x = backend.module.device_put(coordinates, backend.device)
        bind = builder.prepare(x)
    else:
        x = backend.module.tensor(
            coordinates, device=backend.device, requires_grad=True
        )
        bind = builder
    return x, bind


def coordinate_masks(b, pose, context):
    """Physical atoms exclude fitted metal virtuals; heavy input must be retained."""
    types, offsets = map(b.numpy, (pose.block_type_ind, pose.block_coord_offset))
    heavy = np.zeros(pose.coords.shape[:2], bool)
    physical = np.zeros_like(heavy)
    pbt = pose.packed_block_types
    hydrogens = b.numpy(pbt.atom_is_hydrogen)
    for p, row in enumerate(types):
        for r, t in enumerate(row):
            if t < 0:
                continue
            bt = pbt.active_block_types[t]
            virtual = context.canonical_ordering.restypes_virtual_atoms.get(
                bt.io_equiv_class, ()
            )
            for a, atom in enumerate(bt.atoms):
                real = atom.name not in virtual
                physical[p, offsets[p, r] + a] = real
                heavy[p, offsets[p, r] + a] = real and not hydrogens[t, a]
    return heavy, physical


@pytest.mark.parametrize("name", CASES)
def test_atom37_molecular_identity_batch_and_source_gradient(backend, name):
    b, builder, coordinates, ordinary, context = atom37_fixture(backend, name)
    x, bind = bind_atom37(b, builder, coordinates)
    pose = bind(x)
    expected = b.numpy(ordinary.coords)
    heavy, physical = coordinate_masks(b, pose, context)
    np.testing.assert_allclose(
        b.numpy(pose.coords)[0], expected[0], rtol=2e-5, atol=2e-5
    )
    # First construction and replay must preserve the same finite source atoms;
    # fitting a new metal frame must also leave every replay coordinate finite.
    replayed = bind(x)
    assert np.isfinite(b.numpy(replayed.coords)).all()
    np.testing.assert_allclose(
        b.numpy(replayed.coords), b.numpy(pose.coords), rtol=2e-5, atol=2e-5
    )
    difference = b.numpy(pose.coords)[1] - b.numpy(pose.coords)[0]
    np.testing.assert_allclose(
        difference[physical[1]],
        np.broadcast_to([1, 2, 3], difference[physical[1]].shape),
        rtol=2e-5,
        atol=2e-5,
    )
    types = b.numpy(pose.block_type_ind)
    names = [
        [pose.packed_block_types.active_block_types[t].name for t in row if t >= 0]
        for row in types
    ]
    assert names[0] == names[1]
    b.pose, b.parameter_database = pose, context.parameter_database
    b.use_beta2016()
    precision = b.module.enable_x64() if b.framework == "jax" else nullcontext()
    with precision:
        scorer = b.score_function.render_whole_pose_scoring_module(pose)
        if b.framework == "jax":

            def objective(source):
                return scorer(bind(source).coords.astype(np.float64)).sum()

            energy, gradient = b.module.jit(b.module.value_and_grad(objective))(x)
            translated_gradient = b.module.jit(
                b.module.grad(
                    lambda shift: (
                        bind(x + shift).coords
                        * b.module.numpy.asarray(physical)[..., None]
                    ).sum()
                )
            )(b.module.numpy.zeros(3, np.float32))
        else:
            energy = scorer(pose.coords.double()).sum()
            gradient = b.module.autograd.grad(energy, x, retain_graph=True)[0]
            shift = b.module.zeros(
                3, dtype=b.module.float32, device=b.device, requires_grad=True
            )
            translated_gradient = b.module.autograd.grad(
                (
                    bind(x + shift).coords
                    * b.module.as_tensor(physical, device=b.device)[..., None]
                ).sum(),
                shift,
            )[0]
        assert (
            np.isfinite(b.numpy(energy)).all() and np.isfinite(b.numpy(gradient)).all()
        )
        assert np.linalg.norm(b.numpy(gradient)[:, :, 1]) > 1e-3
        np.testing.assert_array_equal(b.numpy(gradient)[:, :, 0], 0)
        np.testing.assert_array_equal(b.numpy(gradient)[:, :, 2:], 0)
        np.testing.assert_allclose(
            b.numpy(translated_gradient), physical.sum(), rtol=2e-5, atol=2e-5
        )
        terms, _ = b.channel_values_and_gradients(
            b.numpy(pose.coords).astype(np.float64)
        )
        assert terms.shape == (25, 2) and np.isfinite(terms).all()
        if name == "ligand":
            assert abs(terms[b.channel_names().index("gen_torsions")]).max() > 1e-3

    if b.framework != "jax" or name not in (
        "ligand",
        "hydroxyproline",
        "zinc",
        "d_peptide",
    ):
        return
    jax, jnp = b.module, b.module.numpy
    # Keep the original single-pose transform/prepared-score cases distinct
    # from the two-pose physical-atom contract above, sharing their setup.
    b, builder, source, ordinary, context = atom37_fixture(backend, name, count=1)
    x, bind = bind_atom37(b, builder, source)
    if name in ("ligand", "hydroxyproline", "zinc"):
        # This derivative includes fitted metal virtuals; Torch intentionally
        # checks physical atoms above because its inferred site fit detaches.
        ordinary = bind(x)
        shifts = jax.device_put(np.array([[0, 0, 0], [1, 2, 3]], np.float32), b.device)
        coords = jax.jit(jax.vmap(lambda shift: bind(x + shift).coords))(shifts)
        real = np.asarray(ordinary.atom_is_real)
        np.testing.assert_allclose(
            np.asarray(coords[1] - coords[0])[real],
            np.broadcast_to([1, 2, 3], (real.sum(), 3)),
            rtol=2e-5,
            atol=2e-5,
        )
        assert coords.device == b.device
        derivative = jax.jit(jax.grad(lambda shift: bind(x + shift).coords.sum()))(
            shifts[0]
        )
        np.testing.assert_allclose(derivative, real.sum(), rtol=2e-5, atol=2e-5)
    else:
        from tmol_core.io.atom37 import Atom37MappingError

        with pytest.raises(Atom37MappingError, match="wholly finite"):
            builder.prepare(x.at[0, 0, 1, 0].set(jnp.nan))
        with pytest.raises(Atom37MappingError, match="wholly finite"):
            builder.prepare(x.at[0, 0, 1, 0].set(jnp.inf))
        bind = builder.prepare(x)
        with pytest.raises(ValueError, match="batch size"):
            bind(jnp.concatenate((x, x)))
        # Unrouted extra tokens do not change the molecule or acquire derivatives.
        padded = jnp.pad(x, ((0, 0), (0, 2), (0, 0), (0, 0)), constant_values=jnp.nan)
        np.testing.assert_array_equal(bind(padded).coords, bind(x).coords)
        padded_gradient = jax.grad(lambda y: jnp.sin(bind(y).coords).sum())(padded)
        np.testing.assert_array_equal(padded_gradient[:, -2:], 0)
        bad = jax.jit(lambda y: bind(y).coords)(x.at[0, 0, 1, 0].set(jnp.nan))
        assert np.isnan(np.asarray(bad)).all()
        from tmol_jax import atom_array_from_pose_stack

        atoms = atom_array_from_pose_stack(ordinary)
        hydrogen = int(np.flatnonzero(atoms.element == "H")[-1])
        missing = x.at[0, hydrogen, 1].set(jnp.nan)
        rebuilt = jax.jit(lambda y: bind(y).coords)(missing)
        assert np.isfinite(np.asarray(rebuilt)).all()
        gradient = jax.jit(jax.grad(lambda y: jnp.sin(bind(y).coords).sum()))(missing)
        assert np.isfinite(np.asarray(gradient)).all()
        np.testing.assert_array_equal(gradient[0, hydrogen], 0)
        assert np.linalg.norm(np.asarray(gradient)) > 1
        # Removing a nonleaf mainchain atom changes the prepared chemistry contract.
        carbon = int(np.flatnonzero(atoms.atom_name == "CA")[0])
        with pytest.raises(Atom37MappingError, match="mainchain"):
            builder.prepare(x.at[0, carbon, 1].set(jnp.nan))
        invalid = jax.jit(lambda y: bind(y).coords)(x.at[0, carbon, 1].set(jnp.nan))
        assert np.isnan(np.asarray(invalid)).all()
    b.pose, b.parameter_database = bind(x), context.parameter_database
    b.use_beta2016()
    with jax.enable_x64():
        if name in ("ligand", "d_peptide", "hydroxyproline"):
            scorer = b.score_function.render_whole_pose_scoring_module(b.pose)
            objective = jax.jit(
                lambda y: scorer(bind(y).coords.astype(jnp.float64)).sum()
            )
            gradient = jax.jit(jax.grad(objective))(x)
            direction = np.where(
                np.isfinite(source),
                np.random.default_rng(413).normal(size=source.shape),
                0,
            ).astype(np.float32)
            direction /= np.linalg.norm(direction)
            direction = jax.device_put(direction, b.device)
            h = 3e-3
            numerical = (
                objective(x + h * direction) - objective(x - h * direction)
            ) / (2 * h)
            analytic = jnp.sum(gradient * direction)
            np.testing.assert_allclose(numerical, analytic, rtol=2e-3, atol=2e-3)
        if name in ("ligand", "hydroxyproline", "zinc"):
            scorer = b.score_function.render_whole_pose_scoring_module(b.pose)
            mapped = scorer.value_and_grad(coordinate_map=bind)
            reference = jax.jit(
                jax.value_and_grad(
                    lambda y: scorer(bind(y).coords.astype(np.float64)).sum()
                )
            )
            # The score precision follows the stored pose, as in ordinary coordinate AD.
            direct = jax.jit(jax.value_and_grad(lambda y: scorer(bind(y).coords).sum()))
            energy, gradient = mapped(x)
            expected, expected_gradient = direct(x)
            np.testing.assert_allclose(energy.sum(), expected, rtol=2e-5, atol=2e-5)
            np.testing.assert_allclose(
                gradient, expected_gradient, rtol=2e-5, atol=2e-5
            )
            assert np.isfinite(gradient).all() and np.linalg.norm(gradient) > 1e-3
            precise_energy, precise_gradient = reference(x)
            precise = scorer.value_and_grad(
                coordinate_map=bind, coordinate_dtype=np.float64
            )(x)
            assert precise[0].dtype == np.float64 and precise[1].dtype == x.dtype
            np.testing.assert_allclose(
                precise[0].sum(), precise_energy, rtol=2e-10, atol=2e-10
            )
            np.testing.assert_allclose(
                precise[1], precise_gradient, rtol=2e-5, atol=2e-5
            )
            with (
                jax.enable_x64(False),
                pytest.raises(ValueError, match="x64 enabled"),
            ):
                scorer.value_and_grad(coordinate_map=bind, coordinate_dtype=np.float64)(
                    x
                )
            direction = (
                np.random.default_rng(183).normal(size=source.shape).astype(np.float32)
            )
            direction = jax.device_put(
                np.where(np.isfinite(source), direction, 0), b.device
            )
            shifted = x + 0.03 * direction
            # Check two step sizes at a displaced geometry: the unmodified zinc fixture
            # crosses a score discontinuity inside the original +/-0.001 interval.
            # Its original values/gradients are still compared with direct AD above.
            shifted_gradient = mapped(shifted)[1]
            for step in (1e-3, 2e-4):
                numerical = (
                    mapped(shifted + step * direction)[0].sum()
                    - mapped(shifted - step * direction)[0].sum()
                ) / (2 * step)
                np.testing.assert_allclose(
                    numerical,
                    (shifted_gradient * direction).sum(),
                    rtol=4e-3,
                    atol=4e-3,
                )
            fresh_bind = builder.prepare(shifted)
            fresh_pose = fresh_bind(shifted)
            # A changed stored weight must not reuse an earlier scorer's parameter data.
            from tmol_jax import ScoreType

            b.score_function.set_weight(ScoreType.cart_lengths, 0.7)
            fresh_scorer = b.score_function.render_whole_pose_scoring_module(fresh_pose)
            fresh = fresh_scorer.value_and_grad(coordinate_map=fresh_bind)
            values, gradients = jax.jit(jax.vmap(fresh))(jax.numpy.stack([x, shifted]))
            for row, coords in enumerate((x, shifted)):
                expected, expected_gradient = jax.value_and_grad(
                    lambda y: fresh_scorer(fresh_bind(y).coords).sum()
                )(coords)
                np.testing.assert_allclose(
                    values[row].sum(), expected, rtol=2e-5, atol=2e-5
                )
                np.testing.assert_allclose(
                    gradients[row], expected_gradient, rtol=2e-5, atol=2e-5
                )
            # Reusing the original bound function still returns its original snapshot.
            restored = mapped(x)
            np.testing.assert_allclose(restored[0], energy, rtol=2e-5, atol=2e-5)
            np.testing.assert_allclose(restored[1], gradient, rtol=2e-5, atol=2e-5)
        if name == "ligand":
            import attrs
            from tmol_jax import ScoreFunction, ScoreType

            pose = bind(x)
            database = context.parameter_database
            executions = []
            for factor in (1, 2):
                changed = attrs.evolve(
                    database,
                    scoring=attrs.evolve(
                        database.scoring,
                        elec=attrs.evolve(
                            database.scoring.elec,
                            atom_charge_parameters=tuple(
                                attrs.evolve(record, charge=record.charge * factor)
                                for record in database.scoring.elec.atom_charge_parameters
                            ),
                        ),
                    ),
                )
                score = ScoreFunction(changed)
                score.set_weight(ScoreType.fa_elec, 1.0)
                executions.append(
                    score.render_whole_pose_scoring_module(pose).value_and_grad(
                        coordinate_map=bind, coordinate_dtype=np.float64
                    )
                )
            original, doubled = [execute(x) for execute in executions]
            assert (
                np.linalg.norm(original[0]) > 1e-3
                and np.linalg.norm(original[1]) > 1e-3
            )
            # Coulomb energies and forces are bilinear in the two atom charges.
            np.testing.assert_allclose(
                doubled[0], original[0] * 4, rtol=2e-10, atol=2e-10
            )
            np.testing.assert_allclose(
                doubled[1], original[1] * 4, rtol=2e-5, atol=2e-5
            )
            restored = executions[0](x)
            np.testing.assert_allclose(restored[0], original[0], rtol=2e-10, atol=2e-10)
            np.testing.assert_allclose(restored[1], original[1], rtol=2e-5, atol=2e-5)


def atom37_opth_fixture(backend, start, stop):
    from pathlib import Path
    import biotite.structure as struc
    import biotite.structure.io as structure_io

    b = backend
    frontend = import_module("tmol_jax" if b.framework == "jax" else "tmol")
    atoms = structure_io.load_structure(
        Path(__file__).parents[2] / "tmol/tests/data/pdb/1ubq.pdb"
    )
    starts = struc.get_residue_starts(atoms, add_exclusive_stop=True)
    atoms = atoms[starts[start] : starts[stop]].copy()
    atoms = atoms[atoms.element != "H"]
    atoms.set_annotation("token_id", np.arange(len(atoms), dtype=np.int64))
    atoms.set_annotation("atom37_slot", np.ones(len(atoms), np.int64))
    source = np.full((1, len(atoms), 37, 3), np.nan, np.float32)
    source[0, :, 1] = atoms.coord
    io = import_module(frontend.__name__ + ".io")
    context = io.build_context_from_atom_array(atoms, b.device)
    builder = io.prepare_atom37_pose_builder(atoms, context)
    x, plain = bind_atom37(b, builder, source)
    if b.framework == "jax":
        bound = builder.prepare(x, opt_h=True, seed=0)
    else:
        b.module.manual_seed(0)

        def bound(y):
            return builder(y, opt_h=True)

    return b, atoms, source, context, x, plain, bound


@pytest.mark.parametrize("start,stop", ((0, 2), (66, 69)))
def test_atom37_opth_preserves_observed_atoms_and_frozen_leaf_derivatives(
    backend, start, stop
):
    b, atoms, source, context, x, plain, bound = atom37_opth_fixture(
        backend, start, stop
    )
    pose, reference = bound(x), plain(x)
    types, offsets = map(b.numpy, (pose.block_type_ind, pose.block_coord_offset))
    mask = np.zeros(pose.coords.shape[:2], bool)
    for block, t in enumerate(types[0]):
        if t < 0:
            continue
        bt = pose.packed_block_types.active_block_types[t]
        label = pose.pdb_info.residue_labels[0, block]
        for a, atom in enumerate(bt.atoms):
            source_atom = np.flatnonzero(
                (atoms.res_id == label) & (atoms.atom_name == atom.name)
            )
            if len(source_atom):
                target = offsets[0, block] + a
                mask[0, target] = True
                np.testing.assert_array_equal(
                    b.numpy(pose.coords)[0, target], atoms.coord[source_atom[0]]
                )
    frozen = pose.block_type(0, 0)
    hydrogens = b.numpy(pose.packed_block_types.atom_is_hydrogen)[
        types[0, 0], : len(frozen.atoms)
    ].astype(bool)
    indices = np.flatnonzero(hydrogens)
    np.testing.assert_array_equal(
        b.numpy(pose.coords)[0, indices], b.numpy(reference.coords)[0, indices]
    )
    if b.framework == "jax":
        observed = b.module.jit(
            b.module.grad(
                lambda y: (
                    bound(y).coords * b.module.numpy.asarray(mask)[..., None]
                ).sum()
            )
        )(x)
        derivative = b.module.jit(
            b.module.grad(
                lambda y: b.module.numpy.square(bound(y).coords[0, indices]).sum()
            )
        )(x)
        expected = b.module.jit(
            b.module.grad(
                lambda y: b.module.numpy.square(plain(y).coords[0, indices]).sum()
            )
        )(x)
    else:
        observed = b.module.autograd.grad(
            pose.coords[b.module.as_tensor(mask, device=b.device)].sum(),
            x,
            retain_graph=True,
        )[0]
        derivative = b.module.autograd.grad(
            pose.coords[0, indices].square().sum(), x, retain_graph=True
        )[0]
        expected = b.module.autograd.grad(
            reference.coords[0, indices].square().sum(), x, retain_graph=True
        )[0]
    desired = np.zeros(source.shape, np.float32)
    desired[:, :, 1] = 1
    np.testing.assert_array_equal(b.numpy(observed), desired)
    np.testing.assert_array_equal(b.numpy(derivative), b.numpy(expected))
    assert np.linalg.norm(b.numpy(expected)) > 0
    b.pose, b.parameter_database = pose, context.parameter_database
    b.use_beta2016()
    values, gradients = b.channel_values_and_gradients(b.numpy(pose.coords))
    assert (
        values.shape[0] == 25
        and np.isfinite(values).all()
        and np.isfinite(gradients).all()
    )

    if b.framework != "jax" or (start, stop) != (0, 2):
        return
    jax = b.module
    with jax.enable_x64():
        score = b.score_function.render_whole_pose_scoring_module(b.pose)
        mapped = score.value_and_grad(coordinate_map=bound, coordinate_dtype=np.float64)
        objective = jax.jit(lambda y: score(bound(y).coords.astype(np.float64)).sum())
        expected, gradient = jax.value_and_grad(objective)(x)
        actual = mapped(x)
        np.testing.assert_allclose(actual[0].sum(), expected, rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(actual[1], gradient, rtol=2e-5, atol=2e-5)
        assert np.isfinite(actual[1]).all() and np.linalg.norm(actual[1]) > 1e-3
        direction = np.where(
            np.isfinite(source),
            np.random.default_rng(619).normal(size=source.shape),
            0,
        ).astype(np.float32)
        direction /= np.linalg.norm(direction)
        direction = jax.device_put(direction, b.device)
        for step in (1e-3, 3e-3):
            numerical = (
                mapped(x + step * direction)[0].sum()
                - mapped(x - step * direction)[0].sum()
            ) / (2 * step)
            np.testing.assert_allclose(
                numerical, (actual[1] * direction).sum(), rtol=4e-3, atol=4e-3
            )


@pytest.mark.parametrize("name", ("ligand", "hydroxyproline", "d_peptide", "zinc"))
def test_atom37_array_override_uses_source_coordinates_and_parameter_context(
    backend, name
):
    b, builder, source, ordinary, context = atom37_fixture(backend, name)
    frontend = import_module("tmol_jax" if b.framework == "jax" else "tmol")
    io = import_module(frontend.__name__ + ".io")
    atoms = frontend.atom_array_from_pose_stack(ordinary)
    atoms.set_annotation("token_id", np.arange(len(atoms), dtype=np.int64))
    atoms.set_annotation("atom37_slot", np.ones(len(atoms), np.int64))
    atoms.coord += np.array([40, 10, -20], np.float32)
    x, bind = bind_atom37(b, builder, source)
    expected = bind(x)
    actual, returned = frontend.pose_stack_from_atom_array(
        atoms,
        b.device,
        context=context,
        atom37_coords=x,
        no_optH=True,
        return_context=True,
    )
    direct = io.pose_stack_from_atom37_and_topology(x, atoms, context, no_optH=True)
    assert returned is context
    for pose in (actual, direct):
        np.testing.assert_array_equal(
            b.numpy(pose.block_type_ind), b.numpy(expected.block_type_ind)
        )
        np.testing.assert_array_equal(
            b.numpy(pose.inter_residue_connections),
            b.numpy(expected.inter_residue_connections),
        )
        np.testing.assert_allclose(
            b.numpy(pose.coords), b.numpy(expected.coords), rtol=2e-5, atol=2e-5
        )
        b.pose, b.parameter_database = pose, returned.parameter_database
        b.use_beta2016()
        values, gradient = b.channel_values_and_gradients(b.numpy(pose.coords))
        assert (
            values.shape == (25, 2)
            and np.isfinite(values).all()
            and np.isfinite(gradient).all()
        )


@pytest.mark.parametrize("name", ("hydroxyproline", "d_peptide"))
def test_atom37_missing_noncanonical_sidechain_uses_shared_native_packing(
    backend, name
):
    b, builder, source, ordinary, context = atom37_fixture(backend, name, count=1)
    frontend = import_module("tmol_jax" if b.framework == "jax" else "tmol")
    atoms = frontend.atom_array_from_pose_stack(ordinary)
    middle = np.unique(atoms.res_id)[1]
    missing = (atoms.res_id == middle) & ~np.isin(
        atoms.atom_name, ("N", "CA", "C", "O", "OXT")
    )
    assert np.count_nonzero(missing & (atoms.element != "H")) >= 2
    source[:, missing, 1] = np.nan
    x, bind = bind_atom37(b, builder, source)
    pose = bind(x)
    assert np.isfinite(b.numpy(pose.coords)).all()
    exported = frontend.atom_array_from_pose_stack(pose)
    for atom in atoms[~missing]:
        found = np.flatnonzero(
            (exported.res_id == atom.res_id) & (exported.atom_name == atom.atom_name)
        )
        assert len(found) == 1
        np.testing.assert_array_equal(exported.coord[found[0]], atom.coord)
    b.pose, b.parameter_database = pose, context.parameter_database
    b.use_beta2016()
    if b.framework == "jax":
        scorer = b.score_function.render_whole_pose_scoring_module(pose)
        energy, gradient = b.module.jit(
            b.module.value_and_grad(lambda y: scorer(bind(y).coords).sum())
        )(x)
    else:
        energy = b.scorer(pose.coords).sum()
        gradient = b.module.autograd.grad(energy, x)[0]
    assert np.isfinite(b.numpy(energy)).all() and np.isfinite(b.numpy(gradient)).all()
    np.testing.assert_array_equal(b.numpy(gradient)[:, missing], 0)
    assert np.linalg.norm(b.numpy(gradient)) > 1e-3
