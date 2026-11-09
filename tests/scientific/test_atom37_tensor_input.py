"""Tensor-only molecular input uses common identity and slot topology."""

from contextlib import nullcontext
from functools import partial
from importlib import import_module

import numpy as np
import pytest

from .test_atom37_input import atom37_fixture


def tensor_fixture(backend, name, default_layout=False):
    b, builder, _, ordinary, context = atom37_fixture(backend, name, count=1)
    template = builder.template if b.framework == "jax" else builder.canonical_template
    return tensor_input(b, template, context, default_layout)


def tensor_input(b, template, context, default_layout=False):
    def host(key):
        value = (
            template.get(
                key, template.get("res_gap") if key == "res_not_connected" else None
            )
            if isinstance(template, dict)
            else getattr(template, key)
        )
        return None if value is None else b.numpy(value)

    co = context.canonical_ordering
    io = import_module("tmol_jax.io" if b.framework == "jax" else "tmol.io")
    if default_layout:
        from tmol_core.io.atomworks_encoding import ATOMWORKS_ATOM37_NAMES

        names = ATOMWORKS_ATOM37_NAMES
    else:
        names = co.restypes_ordered_atom_names
    layout = b.numpy(io.atom37_slot_map_for_ordering(co, names, b.device))
    types, chains, canonical = map(host, ("res_types", "chain_id", "coords"))
    source = np.full((*types.shape, layout.shape[1], 3), np.nan, np.float32)
    for p, r in zip(*np.nonzero(types >= 0)):
        selected = layout[types[p, r]] >= 0
        source[p, r, selected] = canonical[p, r, layout[types[p, r], selected]]
    topology = dict(
        res_types=types,
        chain_id=chains,
        slot_map=None if default_layout else layout,
        **{
            key: host(key)
            for key in (
                "disulfides",
                "cyclic_bonds",
                "covalent_bonds",
                "res_not_connected",
            )
        },
    )
    return b, io, source, topology, context


def bind_tensor(b, io, source, topology, context):
    if b.framework == "jax":
        x = b.module.device_put(source, b.device)
        builder = io.prepare_atom37_tensor_builder(
            context=context, n_slots=source.shape[-2], **topology
        )
        return x, builder.prepare(x)
    topology = {
        key: None if value is None else b.module.as_tensor(value, device=b.device)
        for key, value in topology.items()
    }
    x = b.module.tensor(source, device=b.device, requires_grad=True)
    return x, partial(io.pose_stack_from_atom37, context=context, **topology)


@pytest.mark.parametrize(
    "name",
    (
        "canonical",
        "ligand",
        "d_peptide",
        "hydroxyproline",
        "dna",
        "cyclic",
        "heterochiral_disulfide",
    ),
)
def test_tensor_atom37_retains_source_and_full_energy_gradient(backend, name):
    b, io, source, topology, context = tensor_fixture(
        backend, name, default_layout=name == "canonical"
    )
    x, bind = bind_tensor(b, io, source, topology, context)
    pose = bind(x)
    assert np.isfinite(b.numpy(pose.coords)).all()
    b.pose, b.parameter_database = pose, context.parameter_database
    b.use_beta2016()
    precision = b.module.enable_x64() if b.framework == "jax" else nullcontext()
    with precision:
        scorer = b.score_function.render_whole_pose_scoring_module(pose)
        if b.framework == "jax":

            def objective(y):
                return scorer(bind(y).coords.astype(np.float64)).sum()

            energy, gradient = b.module.jit(b.module.value_and_grad(objective))(x)
            translation = b.module.jit(
                b.module.grad(lambda shift: bind(x + shift).coords.sum())
            )(b.module.numpy.zeros(3, dtype=np.float32))
        else:

            def objective(y):
                return scorer(bind(y).coords.double()).sum()

            energy = objective(x)
            gradient = b.module.autograd.grad(energy, x)[0]
            shift = b.module.zeros(3, device=b.device, requires_grad=True)
            translation = b.module.autograd.grad(bind(x + shift).coords.sum(), shift)[0]
        assert np.isfinite(b.numpy(energy)).all()
        assert np.isfinite(b.numpy(gradient)).all()
        np.testing.assert_array_equal(b.numpy(gradient)[~np.isfinite(source)], 0)
        np.testing.assert_allclose(
            b.numpy(translation), pose.coords.shape[1], rtol=2e-5, atol=2e-5
        )
        direction = (
            np.random.default_rng(312).normal(size=source.shape).astype(np.float32)
        )
        direction[~np.isfinite(source)] = 0
        delta = (
            b.module.device_put(direction, b.device)
            if b.framework == "jax"
            else b.module.as_tensor(direction, device=b.device)
        )
        step = 0.003
        finite_difference = b.numpy(
            (objective(x + step * delta) - objective(x - step * delta)) / (2 * step)
        )
        analytic = np.sum(b.numpy(gradient) * direction)
        np.testing.assert_allclose(analytic, finite_difference, rtol=3e-3, atol=3e-3)

    if b.framework == "jax" and name in ("ligand", "hydroxyproline"):
        jax = b.module
        shifts = jax.device_put(np.array([[0, 0, 0], [2, 3, -1]], np.float32), b.device)
        coords = jax.jit(jax.vmap(lambda shift: bind(x + shift).coords))(shifts)
        np.testing.assert_allclose(
            coords[1] - coords[0],
            np.broadcast_to(np.asarray(shifts[1]), coords[0].shape),
            atol=2e-5,
            rtol=2e-5,
        )
        invalid = x.at[0, 0, 0, 0].set(np.inf)
        assert np.isnan(np.asarray(jax.jit(lambda y: bind(y).coords)(invalid))).all()
        with pytest.raises(ValueError, match="shape"):
            bind(x[:, :-1])
        # No routing for a named ligand is a hard error, not ideal-coordinate input.
        if name == "ligand":
            with pytest.raises(ValueError, match="routes no atom"):
                io.prepare_atom37_tensor_builder(
                    topology["res_types"], topology["chain_id"], context
                )


def test_tensor_atom37_gap_suppresses_bond_without_inventing_termini(backend):
    b, io, source, topology, context = tensor_fixture(backend, "canonical", True)
    # Omit explicit exported polymer bonds: those correctly override a gap flag.
    topology["covalent_bonds"] = None
    x, plain = bind_tensor(b, io, source, topology, context)
    expected = plain(x)
    gaps = np.zeros((*topology["res_types"].shape, 2), bool)
    gaps[0, 0, 1] = gaps[0, 1, 0] = True
    topology["res_not_connected"] = gaps
    x, bind = bind_tensor(b, io, source, topology, context)
    pose = bind(x)
    assert [pose.block_type(0, r).name for r in range(3)] == [
        expected.block_type(0, r).name for r in range(3)
    ]
    links = b.numpy(pose.inter_residue_connections)[0]
    assert not np.any(links[0, :, 0] == 1)
    assert np.any(links[1, :, 0] == 2)
    assert np.isfinite(b.numpy(pose.coords)).all()
    b.pose, b.parameter_database = pose, context.parameter_database
    b.use_beta2016()
    energy, gradient = b.channel_values_and_gradients(b.numpy(pose.coords))
    assert energy.shape == (25, 1)
    assert np.isfinite(energy).all() and np.isfinite(gradient).all()


def test_tensor_atom37_ligand_returns_custom_identity_metadata(backend):
    b, io, source, topology, context = tensor_fixture(backend, "ligand")
    if b.framework == "jax":
        x = b.module.device_put(source, b.device)
        options = topology
    else:
        x = b.module.tensor(source, device=b.device, requires_grad=True)
        options = {
            key: None if value is None else b.module.as_tensor(value, device=b.device)
            for key, value in topology.items()
        }
    pose, details = io.pose_stack_from_atom37(
        x,
        context=context,
        **options,
        return_atom_mapping=True,
        return_chain_ind=True,
        return_block_has_missing_atoms=True,
    )
    canonical, packed = map(
        b.numpy, (details["can_atom_mapping"], details["ps_atom_mapping"])
    )
    pbt = pose.packed_block_types
    types, offsets = map(b.numpy, (pose.block_type_ind, pose.block_coord_offset))
    assert pbt.chem_db is context.parameter_database.chemical
    assert not b.numpy(details["block_has_missing_atoms"]).any()
    assert len(packed) == sum(
        len(pbt.active_block_types[t].atoms) for t in types.flat if t >= 0
    )
    for (p, r, a), (_, flat) in zip(canonical, packed):
        residue = pbt.active_block_types[types[p, r]]
        name = residue.atoms[flat - offsets[p, r]].name
        assert (
            context.canonical_ordering.restypes_ordered_atom_names[
                residue.io_equiv_class
            ][a]
            == name
        )
        np.testing.assert_array_equal(b.numpy(pose.coords)[p, flat], source[p, r, a])
    again, returned = io.pose_stack_from_atom37(
        x, context=context, **options, return_context=True, return_atom_mapping=True
    )
    assert returned is context
    np.testing.assert_array_equal(b.numpy(again.coords), b.numpy(pose.coords))


def real_tensor_fixture(b, source=("pdb", "1ubq.pdb"), with_reference=False):
    import biotite.structure as struc
    import biotite.structure.io as structure_io
    from .molecules import DATA

    atoms = structure_io.load_structure(DATA.joinpath(*source), model=1)
    assert isinstance(atoms, struc.AtomArray)
    io = import_module("tmol_jax.io" if b.framework == "jax" else "tmol.io")
    context = io.build_context_from_atom_array(atoms, b.device)
    options = dict(missing_density_distance_threshold=0.0)
    reference = None
    if b.framework == "jax":
        from tmol_jax.io.pose_stack_from_biotite import canonical_form_from_biotite

        canonical = canonical_form_from_biotite(
            atoms,
            canonical_ordering=context.canonical_ordering,
            prepare_input=True,
            **options,
        )
        if with_reference:
            reference = io.pose_stack_from_atom_array(
                atoms, b.device, context=context, no_optH=True, **options
            )
    else:
        canonical = io.canonical_form_from_biotite(
            atoms, b.device, co=context.canonical_ordering, **options
        )
        if with_reference:
            from tmol.io._pose_stack_from_biotite import (
                _pose_stack_from_canonical_form_and_context,
            )

            reference = _pose_stack_from_canonical_form_and_context(
                canonical, context, no_optH=True, atom37_coords=None
            )
    return (*tensor_input(b, canonical, context), reference)


@pytest.mark.parametrize(
    "source",
    (
        pytest.param(("pdb", "1ubq.pdb"), id="canonical_protein"),
        pytest.param(
            ("covalent_fixtures", "lys_biotin_1bdo.cif"), id="conjugated_lysine"
        ),
        pytest.param(("covalent_fixtures", "nglycan_tree_1ax2.cif"), id="glycan_tree"),
    ),
)
def test_tensor_atom37_matches_real_atom_array(backend, source):
    b, io, source, topology, context, reference = real_tensor_fixture(
        backend, source, with_reference=True
    )
    x, bind = bind_tensor(b, io, source, topology, context)
    pose = bind(x)
    assert pose.n_poses == reference.n_poses
    for key in ("block_type_ind", "inter_residue_connections"):
        np.testing.assert_array_equal(
            b.numpy(getattr(pose, key)), b.numpy(getattr(reference, key))
        )
    assert [bt.name for bt in pose.packed_block_types.active_block_types] == [
        bt.name for bt in reference.packed_block_types.active_block_types
    ]
    real = b.numpy(pose.real_atoms)
    rebuilt = np.zeros_like(real)
    offsets = b.numpy(pose.block_coord_offset)
    for p, types in enumerate(b.numpy(pose.block_type_ind)):
        for r, t in enumerate(types):
            if t < 0:
                continue
            bt = pose.packed_block_types.active_block_types[t]
            excluded = context.canonical_ordering.termini_only_atoms_by_class.get(
                bt.io_equiv_class, ()
            )
            for a, atom in enumerate(bt.atoms):
                if atom.name in excluded:
                    rebuilt[p, offsets[p, r] + a] = True
    shared_rebuilt = rebuilt & b.numpy(reference.real_atoms)
    assert shared_rebuilt.sum() <= 8, "more atoms went unrouted than expected"
    # Only class-specific terminus additions may use the wider geometric bound.
    for mask, tolerance in ((real & ~rebuilt, 1e-3), (shared_rebuilt, 0.1)):
        if b.framework == "torch":
            index = b.module.as_tensor(mask, device=b.device)
            b.module.testing.assert_close(
                pose.coords[index], reference.coords[index], rtol=0, atol=tolerance
            )
        else:
            assert pose.coords.dtype == reference.coords.dtype
            assert pose.coords.device == reference.coords.device
            np.testing.assert_allclose(
                b.numpy(pose.coords)[mask],
                b.numpy(reference.coords)[mask],
                rtol=0,
                atol=tolerance,
            )
    if b.framework == "jax":
        gradient = b.module.grad(lambda y: bind(y).coords[real].sum())(x)
    else:
        pose.coords[b.module.as_tensor(real, device=b.device)].sum().backward()
        gradient = x.grad
    assert np.count_nonzero(b.numpy(gradient)) > 0


def test_tensor_atom37_leaves_terminus_atoms_unmapped(backend):
    b, io, source, topology, context, _ = real_tensor_fixture(backend)
    co, slot_map = context.canonical_ordering, topology["slot_map"]
    by_class = co.termini_only_atoms_by_class
    assert by_class, "expected the database to define terminus-added atoms"
    checked = 0
    for row, name in enumerate(co.restype_io_equiv_classes):
        # H1/H2/H3 can be ordinary atoms of other classes: do not pool names.
        for slot, atom in enumerate(co.restypes_ordered_atom_names[name]):
            if atom in by_class.get(name, ()):
                assert slot_map[row, slot] == -1
                checked += 1
    assert checked > 0, "expected at least one terminus atom in the layout"


@pytest.mark.parametrize("invalid", ("oversize", "unroutable"))
def test_tensor_atom37_rejects_invalid_routing(backend, invalid):
    b, io, source, topology, context, _ = real_tensor_fixture(backend)
    co = context.canonical_ordering
    if invalid == "oversize":
        source = np.zeros((*source.shape[:2], source.shape[-2] + 1, 3), np.float32)
        message = "slot map covers"
    else:
        row = co.restype_io_equiv_classes.index("ALA")
        topology["slot_map"] = topology["slot_map"].copy()
        topology["slot_map"][row] = -1
        source = np.zeros((1, 1, source.shape[-2], 3), np.float32)
        topology.update(
            res_types=np.full((1, 1), row, np.int32),
            chain_id=np.zeros((1, 1), np.int32),
        )
        message = "routes no atom of residue type"
    topology = {key: topology[key] for key in ("res_types", "chain_id", "slot_map")}
    with pytest.raises(ValueError, match=message):
        x, bind = bind_tensor(b, io, source, topology, context)
        bind(x)
    if invalid == "unroutable" and b.framework == "torch":
        from tmol.io._pose_stack_from_atom37 import _canonical_form_from_atom37

        # Retain the legacy private-wrapper rejection as well as the public API.
        args = {
            key: b.module.as_tensor(value, device=b.device)
            for key, value in topology.items()
        }
        with pytest.raises(ValueError, match=message):
            _canonical_form_from_atom37(
                b.module.as_tensor(source, device=b.device),
                args["res_types"],
                args["chain_id"],
                co,
                slot_map=args["slot_map"],
            )
