"""Standard encoded protein compatibility over the full molecular core."""

from importlib import import_module
import numpy as np
import pytest

from tmol_core.io.atomworks_encoding import ATOMWORKS_NAME3S
from .test_atom37_tensor_input import tensor_fixture


def test_atomworks_padded_input_export_preserves_identity_and_gradients(backend):
    b, _, source, topology, context = tensor_fixture(backend, "canonical", True)
    co = context.canonical_ordering
    tokens = np.array(
        [
            [
                ATOMWORKS_NAME3S.index(co.restype_io_equiv_classes[t])
                for t in topology["res_types"][0]
            ]
        ],
        dtype=np.int32,
    )
    source = np.repeat(source, 2, axis=0)
    source[1] += np.array([1, 2, 3], np.float32)
    tokens = np.repeat(tokens, 2, axis=0)
    tokens[1, -1] = 0
    source[1, -1] = np.nan
    chains = np.zeros(tokens.shape, np.int32)
    io = import_module("tmol_jax.io" if b.framework == "jax" else "tmol.io")
    if b.framework == "jax":
        x = b.module.device_put(source, b.device)
        builder = io.prepare_canonical_aa_atom37_builder(tokens, chains, b.device)
        bind = builder.prepare(x)
    else:
        tokens_device = b.module.as_tensor(tokens, device=b.device)
        chains_device = b.module.as_tensor(chains, device=b.device)
        x = b.module.tensor(source, device=b.device, requires_grad=True)

        def bind(y):
            return io.pose_stack_from_canonical_aa_atom37(
                y, tokens_device, chains_device
            )

    pose = bind(x)
    output, encoded, chain_ids = io.atomworks_from_pose_stack(pose)
    np.testing.assert_array_equal(b.numpy(encoded), tokens)
    # Padding has no chain; compare chain identities only on real residues.
    np.testing.assert_array_equal(b.numpy(chain_ids)[tokens > 0], chains[tokens > 0])
    observed = np.isfinite(source).all(-1)
    np.testing.assert_array_equal(b.numpy(output)[observed], source[observed])
    assert np.isfinite(b.numpy(output)).all()
    expected = np.broadcast_to(observed[..., None], source.shape).astype(np.float32)
    # Export includes a newly reconstructed terminal OXT. Mask that slot to
    # isolate exact preservation of the supplied input coordinates below.
    if b.framework == "jax":
        derivative = b.module.jit(
            b.module.grad(
                lambda y: (
                    io.atomworks_from_pose_stack(bind(y))[0]
                    * b.module.numpy.asarray(observed)[..., None]
                ).sum()
            )
        )(x)
    else:
        output = io.atomworks_from_pose_stack(bind(x))[0]
        derivative = b.module.autograd.grad(
            (output * b.module.as_tensor(observed, device=b.device)[..., None]).sum(), x
        )[0]
    np.testing.assert_array_equal(b.numpy(derivative), expected)
    b.pose = pose
    b.use_beta2016()
    energy, gradient = b.channel_values_and_gradients(b.numpy(pose.coords))
    assert (
        energy.shape == (25, 2)
        and np.isfinite(energy).all()
        and np.isfinite(gradient).all()
    )


@pytest.mark.parametrize("invalid", (-1, 21, 100))
def test_atomworks_rejects_nonprotein_tokens(backend, invalid):
    b = backend
    io = import_module("tmol_jax.io" if b.framework == "jax" else "tmol.io")
    if b.framework == "jax":
        with pytest.raises(ValueError, match="protein"):
            io.prepare_canonical_aa_atom37_builder(
                np.array([[invalid]]), np.zeros((1, 1), np.int32), b.device
            )
    else:
        with pytest.raises(ValueError, match="protein"):
            io.pose_stack_from_canonical_aa_atom37(
                b.module.zeros((1, 1, 37, 3), device=b.device),
                b.module.tensor([[invalid]], device=b.device),
                b.module.zeros((1, 1), device=b.device, dtype=b.module.int32),
            )


def ubiquitin_encoding():
    from pathlib import Path
    import biotite.structure as struc
    import biotite.structure.io as structure_io
    from tmol_core.io.atomworks_encoding import ATOMWORKS_ATOM37_NAMES

    atoms = structure_io.load_structure(
        Path(__file__).parents[2] / "tmol/tests/data/pdb/1ubq.pdb", model=1
    )
    atoms = atoms[struc.filter_amino_acids(atoms)]
    rows = struc.get_all_residue_positions(atoms)
    names = atoms.res_name[struc.get_residue_starts(atoms)]
    source = np.full((1, len(names), 37, 3), np.nan, np.float32)
    for atom, row in zip(atoms, rows):
        slots = ATOMWORKS_ATOM37_NAMES[atom.res_name]
        if atom.atom_name in slots:
            source[0, row, slots.index(atom.atom_name)] = atom.coord
    tokens = np.array([[ATOMWORKS_NAME3S.index(name) for name in names]], np.int32)
    return source, tokens, np.zeros(tokens.shape, np.int32)


def test_atomworks_ubiquitin_full_score_and_source_gradient(backend):
    from contextlib import nullcontext

    b = backend
    io = import_module("tmol_jax.io" if b.framework == "jax" else "tmol.io")
    source, tokens, chains = ubiquitin_encoding()
    if b.framework == "jax":
        x = b.module.device_put(source, b.device)
        bind = io.prepare_canonical_aa_atom37_builder(tokens, chains, b.device).prepare(
            x
        )
    else:
        x = b.module.tensor(source, device=b.device, requires_grad=True)
        tokens = b.module.as_tensor(tokens, device=b.device)
        chains = b.module.as_tensor(chains, device=b.device)

        def bind(y):
            return io.pose_stack_from_canonical_aa_atom37(y, tokens, chains)

    b.pose = bind(x)
    frontend = import_module("tmol_jax" if b.framework == "jax" else "tmol")
    b.parameter_database = frontend.ParameterDatabase.get_default()
    b.use_beta2016()
    precision = b.module.enable_x64() if b.framework == "jax" else nullcontext()
    with precision:
        if b.framework == "jax":
            score = b.score_function.render_whole_pose_scoring_module(
                b.pose, sum_terms=False
            )

            def objective(y):
                energies = score(bind(y).coords.astype(np.float64))
                return energies.sum(), energies

            (_, energy), gradient = b.module.jit(
                b.module.value_and_grad(objective, has_aux=True)
            )(x)
        else:
            score = b.score_function.render_whole_pose_scoring_module(b.pose)
            energy = score(b.pose.coords.double(), sum_terms=False)
            gradient = b.module.autograd.grad(energy.sum(), x)[0]
        assert energy.shape == (25, 1)
        assert (
            np.isfinite(b.numpy(energy)).all() and np.isfinite(b.numpy(gradient)).all()
        )
        assert np.linalg.norm(b.numpy(gradient)) > 1
