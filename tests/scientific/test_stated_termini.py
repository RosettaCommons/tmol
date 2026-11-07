"""Supplied terminal chemistry preserves disconnections despite a short gap."""

from importlib import import_module
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("bondless_consecutive", [False, True])
def test_neutral_terminus_preserves_atoms_ports_and_physics(
    backend, bondless_consecutive
):
    front = import_module("tmol_jax" if backend.framework == "jax" else "tmol")
    path = Path(__file__).parents[2] / (
        "tmol/tests/data/sweep_regressions/stated_terminus_5lh4.pdb"
    )
    atoms = front.atom_array_from_file(path)
    terminal = 132
    if bondless_consecutive:
        atoms.bonds = None
        terminal = 131
        atoms.res_id[atoms.res_id == 132] = terminal
    original = atoms.copy()
    pose, context = front.pose_stack_from_atom_array(
        atoms, backend.device, no_optH=True, return_context=True
    )
    assert np.all(backend.numpy(pose.inter_residue_connections)[0, :2, :, 0] < 0)
    block = next(
        i
        for i, label in enumerate(pose.pdb_info.residue_labels[0])
        if int(label) == terminal
    )
    residue = pose.block_type(0, block)
    hydrogen = backend.numpy(pose.packed_block_types.atom_is_hydrogen)[
        int(backend.numpy(pose.block_type_ind)[0, block])
    ]
    bonds = np.asarray(residue.bond_indices).reshape(-1, 2)
    nitrogen = residue.atom_to_idx["N"]
    assert np.count_nonzero((bonds[:, 0] == nitrogen) & hydrogen[bonds[:, 1]]) == 2
    supplied = atoms[(atoms.res_id == terminal) & np.isfinite(atoms.coord).all(-1)]
    offset = int(backend.numpy(pose.block_coord_offset)[0, block])
    slots = [offset + residue.atom_to_idx[name] for name in supplied.atom_name]
    np.testing.assert_allclose(
        backend.numpy(pose.coords)[0, slots], supplied.coord, atol=1e-5
    )
    np.testing.assert_array_equal(atoms.coord, original.coord)
    if original.bonds is None:
        assert atoms.bonds is None
    else:
        np.testing.assert_array_equal(atoms.bonds.as_array(), original.bonds.as_array())
    backend.pose, backend.parameter_database = pose, context.parameter_database
    backend.use_beta2016()
    values, gradients = backend.channel_values_and_gradients(backend.coords())
    assert (
        len(values) == 25 and np.isfinite(values).all() and np.isfinite(gradients).all()
    )
    assert np.linalg.norm(gradients) > 0
    before = backend.scores(backend.coords())
    minimized = backend.minimize(np.ones(pose.coords.shape[:2], bool), iterations=3)
    assert np.all(
        backend.scores(minimized) <= before + 1e-4 * np.maximum(1, np.abs(before))
    )


@pytest.mark.parametrize("ccd_h_name", [False, True])
def test_neutral_terminus_atom37_preserves_hydrogen_source_gradient(backend, ccd_h_name):
    front = import_module("tmol_jax" if backend.framework == "jax" else "tmol")
    path = Path(__file__).parents[2] / (
        "tmol/tests/data/sweep_regressions/stated_terminus_5lh4.pdb"
    )
    atoms = front.atom_array_from_file(path)
    atoms = atoms[atoms.res_id == 132]
    hydrogen = int(np.flatnonzero(atoms.atom_name == "H1")[0])
    if ccd_h_name:
        atoms.atom_name[hydrogen] = "H"
    atoms.set_annotation("token_id", np.arange(len(atoms), dtype=np.int64))
    atoms.set_annotation("atom37_slot", np.ones(len(atoms), dtype=np.int64))
    _, context = front.pose_stack_from_atom_array(
        atoms, backend.device, no_optH=True, return_context=True
    )
    builder = import_module(front.__name__ + ".io").prepare_atom37_pose_builder(
        atoms, context
    )
    coordinates = np.full((1, len(atoms), 37, 3), np.nan, np.float32)
    coordinates[0, :, 1] = atoms.coord
    coordinates[0, hydrogen, 1] += [0.05, 0.02, 0.01]
    if backend.framework == "jax":
        x = backend.module.device_put(coordinates, backend.device)
        bind = builder.prepare(x)
    else:
        x = backend.module.tensor(
            coordinates, device=backend.device, requires_grad=True
        )
        bind = builder
    pose = bind(x)
    h1 = pose.block_type(0, 0).atom_to_idx["H1"]
    np.testing.assert_allclose(
        backend.numpy(pose.coords)[0, h1], coordinates[0, hydrogen, 1], atol=1e-6
    )
    if backend.framework == "jax":
        gradient = backend.module.jit(
            backend.module.grad(lambda source: bind(source).coords[0, h1].sum())
        )(x)
    else:
        gradient = backend.module.autograd.grad(pose.coords[0, h1].sum(), x)[0]
    expected = np.zeros_like(coordinates)
    expected[0, hydrogen, 1] = 1
    np.testing.assert_array_equal(backend.numpy(gradient), expected)
    backend.pose, backend.parameter_database = pose, context.parameter_database
    backend.use_beta2016()
    values, gradients = backend.channel_values_and_gradients(backend.coords())
    assert len(values) == 25 and np.isfinite(values).all()
    assert np.isfinite(gradients).all() and np.linalg.norm(gradients) > 0
