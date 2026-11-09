"""Score files preserve version, options, channel identity and molecular physics."""

from pathlib import Path
from contextlib import nullcontext
import numpy as np
import pytest
from .molecules import DOMAIN_CASES
from .test_fragment_input import _frontend


def _load(backend, path):
    if backend.framework == "jax":
        from tmol_jax import ScoreFunction

        return ScoreFunction.from_sfxn_file(path, backend.parameter_database)
    from tmol import ScoreFunction

    return ScoreFunction.from_sfxn_file(
        path, backend.parameter_database, backend.device
    )


def test_score_file_version_and_custom_options(backend, tmp_path):
    backend.molecule(DOMAIN_CASES["d_peptide"], terms=())
    path = tmp_path / "custom.sfxn"
    for version, message in ((None, "no 'version'"), ("99.0", "incompatible")):
        path.write_text(
            ("" if version is None else f"version: '{version}'\n")
            + "weights: {ref: 1.0}\n"
        )
        with pytest.raises(ValueError, match=message):
            _load(backend, path)
    # Mature minor-version compatibility and explicit D reference overrides.
    path.write_text(
        "version: '1.7'\nweights: {ref: 1.0}\noptions:\n  soft_rep: true\n  ref_weights: {DALA: 2.0, DPRO: 3.0, DVAL: 4.0}\n"
    )
    score = _load(backend, path)
    rendered = score.render_whole_pose_scoring_module(backend.pose)
    x = backend.pose.coords
    if backend.framework == "jax":
        energy = backend.module.jit(rendered)(x)
        gradient = backend.module.jit(backend.module.grad(lambda x: rendered(x).sum()))(
            x
        )
    else:
        x = x.detach().requires_grad_()
        energy = rendered(x)
        # Reference energies are coordinate independent; Torch returns an
        # unused gradient while JAX materializes its mathematical zero.
        gradient = backend.module.autograd.grad(energy.sum(), x, allow_unused=True)[0]
        if gradient is None:
            gradient = backend.module.zeros_like(x)
    np.testing.assert_array_equal(backend.numpy(energy), [9.0])
    np.testing.assert_array_equal(backend.numpy(gradient), 0)


@pytest.mark.parametrize(
    "name", ("ligand", "connected_noncanonical", "d_peptide", "mixed_batch")
)
def test_complete_soft_preset_keeps_all_channels_and_finite_derivatives(backend, name):
    b = backend.molecule(DOMAIN_CASES[name], terms=())
    path = Path(__file__).parents[2] / "tmol/database/score_functions/beta_soft.sfxn"
    b.score_function = _load(b, path)
    if b.framework == "torch":
        b._render_torch()
    with b.module.enable_x64() if b.framework == "jax" else nullcontext():
        x = b.coords().astype(np.float64)
        e, g = b.channel_values_and_gradients(x)
        assert (
            len(b.channel_names()) == 25
            and np.isfinite(e).all()
            and np.isfinite(g).all()
        )
        score_type = _frontend(b).ScoreType
        weights = np.array(
            [
                float(b.score_function.get_weight(getattr(score_type, n)))
                for n in b.channel_names()
            ]
        )
        value, gradient = b.value_and_grad(x)
        np.testing.assert_allclose(
            value, (e * weights[:, None]).sum(0), rtol=2e-10, atol=2e-10
        )
        np.testing.assert_allclose(
            gradient, np.einsum("c,cpai->pai", weights, g), rtol=2e-10, atol=2e-10
        )
        assert abs(e[b.channel_names().index("fa_ljrep")]).max() > 1e-3
