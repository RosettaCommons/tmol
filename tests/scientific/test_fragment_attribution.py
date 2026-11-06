"""Original ligand/partner attribution survives chemical fragmentation."""

from contextlib import nullcontext
import numpy as np
import pytest

from .fragment_molecules import fragment_fixture
from .test_fragment_input import _frontend, _build, _physical_atoms


@pytest.mark.parametrize("repeated", (False, True))
def test_fragment_partner_energy_and_gradient_equal_uncut_ligands(backend, repeated):
    whole, fragmented, parameters = fragment_fixture(repeated=repeated, mixed=True)
    results = []
    with backend.module.enable_x64() if backend.framework == "jax" else nullcontext():
        for atoms, cut in ((whole, False), (fragmented, True)):
            pose, context = _build(backend, atoms, parameters)
            backend.pose, backend.parameter_database = pose, context.parameter_database
            backend.use_beta2016()
            types = backend.numpy(pose.block_type_ind)
            ligand = np.zeros(types.shape, bool)
            if cut:
                for entry in pose.split_block_mapping.entries:
                    ligand[entry.pose_ind, entry.block_ind] = True
            else:
                for p, b in zip(*np.nonzero(types >= 0)):
                    ligand[p, b] = pose.block_type(p, b).io_equiv_class == "LG1"
            assert ligand.sum() == (2 if repeated else 1) * (2 if cut else 1)
            partner = (types >= 0) & ~ligand
            if backend.framework == "jax":
                module = backend.module
                mask = module.device_put(partner, backend.device)
                x = module.device_put(
                    backend.coords().astype(np.float64), backend.device
                )
                if cut:
                    from tmol_jax import render_fragment_interactions

                    score = render_fragment_interactions(
                        pose, mask, sfxn=backend.score_function
                    )
                    assert len(score.mapping) == ligand.sum()

                    def evaluate(x):
                        return score(x).sum(-1)

                else:
                    import jax.numpy as jnp

                    pair = backend.score_function.render_block_pair_scoring_module(pose)
                    mask_pairs = ligand[:, :, None] & partner[:, None, :]
                    selection = jnp.asarray(mask_pairs | mask_pairs.swapaxes(-1, -2))

                    def evaluate(x):
                        return (pair(x, sum_terms=False) * selection).sum((-2, -1))

                e = module.jit(evaluate)(x)
                g = module.jit(module.grad(lambda x: evaluate(x).sum()))(x)
            else:
                import attrs

                module = backend.module
                mask = module.as_tensor(partner, device=backend.device)
                x = module.tensor(
                    backend.coords().astype(np.float64),
                    device=backend.device,
                    requires_grad=True,
                )
                if cut:
                    result = _frontend(backend).calculate_fragment_interactions(
                        attrs.evolve(pose, coords=x), mask, sfxn=backend.score_function
                    )
                    assert len(result.mapping) == ligand.sum()
                    e = result.scores.sum(-1)
                else:
                    pair = backend.score_function.render_block_pair_scoring_module(pose)
                    mask_pairs = ligand[:, :, None] & partner[:, None, :]
                    selection = module.as_tensor(
                        mask_pairs | mask_pairs.swapaxes(-1, -2), device=backend.device
                    )
                    e = (pair(x, sum_terms=False) * selection).sum((-2, -1))
                g = module.autograd.grad(e.sum(), x)[0]
            physical = _physical_atoms(backend, pose)
            keys = sorted(physical)
            gradient = np.array([backend.numpy(g)[physical[key]] for key in keys])
            results.append((backend.numpy(e), gradient, keys, backend.channel_names()))
        a, b = results
        assert a[2:] == b[2:]
        np.testing.assert_allclose(a[0], b[0], atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(a[1], b[1], atol=2e-6, rtol=2e-6)
        assert np.linalg.norm(b[0]) > 1e-3 and np.linalg.norm(b[1]) > 1e-3


def test_fragment_public_contract_validates_partner_and_column_identity(backend):
    _, fragmented, parameters = fragment_fixture(mixed=False)
    pose, context = _build(backend, fragmented, parameters)
    backend.pose, backend.parameter_database = pose, context.parameter_database
    backend.use_beta2016()
    frontend = _frontend(backend)

    def array(value):
        if backend.framework == "jax":
            return backend.module.device_put(value, backend.device)
        return backend.module.as_tensor(value, device=backend.device)

    empty = array(np.zeros(pose.block_type_ind.shape, bool))
    result = frontend.calculate_fragment_interactions(
        pose, empty, sfxn=backend.score_function
    )
    np.testing.assert_array_equal(backend.numpy(result.scores), 0)
    assert len(result.mapping) == 2 and result.scores.shape == (25, 1, 2)
    for mask, error, message in (
        (array(np.zeros(pose.block_type_ind.shape, np.float32)), TypeError, "boolean"),
        (array(np.zeros((1, 1), bool)), ValueError, "shape"),
        (
            array(np.ones(pose.block_type_ind.shape, bool)),
            ValueError,
            "fragment blocks",
        ),
    ):
        with pytest.raises(error, match=message):
            frontend.calculate_fragment_interactions(
                pose, mask, sfxn=backend.score_function
            )
    from tmol_core.io.split_block_mapping import SplitBlockMapping

    entry = pose.split_block_mapping.entries[0]
    with pytest.raises(ValueError, match="duplicate columns"):
        frontend.calculate_fragment_interactions(
            pose,
            empty,
            sfxn=backend.score_function,
            mapping=SplitBlockMapping(entries=(entry, entry)),
        )
