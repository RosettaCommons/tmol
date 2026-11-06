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

    def host(key):
        value = (
            template.get(key) if isinstance(template, dict) else getattr(template, key)
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
