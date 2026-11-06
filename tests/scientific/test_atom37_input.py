"""Prepared Atom37 inputs retain molecular identity and source derivatives."""

from contextlib import nullcontext
from importlib import import_module
import numpy as np
import pytest

from .molecules import DOMAIN_CASES, BACKBONE_CASES
from .metal_molecules import METAL_CASES
from .test_atom_array_input import bonded_array

CASES = {name: DOMAIN_CASES[name] for name in ("ligand", "d_peptide", "dna", "rna")}
CASES.update(
    hydroxyproline=BACKBONE_CASES["hydroxyproline"], zinc=METAL_CASES["zinc_site"]
)


def atom37_fixture(backend, name, count=2):
    b = backend.molecule(CASES[name], terms=())
    frontend = import_module("tmol_jax" if b.framework == "jax" else "tmol")
    atoms = bonded_array(b, CASES[name])
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
        b.numpy(pose.coords)[0][heavy[0]], expected[0][heavy[0]], rtol=2e-5, atol=2e-5
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
