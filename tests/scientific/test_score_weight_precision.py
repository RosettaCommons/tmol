"""Default weight precision matches mature TMol independently of coordinates."""

from contextlib import nullcontext
import numpy as np
import pytest
from .molecules import DOMAIN_CASES


def test_nonbinary_weights_keep_default_storage_precision(backend, monkeypatch):
    b = backend.molecule(DOMAIN_CASES["ligand"], terms=("fa_ljrep",))
    with b.module.enable_x64() if b.framework == "jax" else nullcontext():
        x = b.coords().astype(np.float64)
        unweighted, unweighted_g = b.value_and_grad(x)
        from importlib import import_module

        score_type = import_module(
            "tmol_jax" if b.framework == "jax" else "tmol"
        ).ScoreType
        for weight in (0.55, 0.92):
            b.score_function.set_weight(score_type.fa_ljrep, weight)
            if b.framework == "torch":
                b._render_torch()
            value, gradient = b.value_and_grad(x)
            stored = float(np.float32(weight))
            np.testing.assert_allclose(
                value, unweighted * stored, rtol=2e-10, atol=2e-10
            )
            np.testing.assert_allclose(
                gradient, unweighted_g * stored, rtol=2e-10, atol=2e-10
            )

    if b.framework == "jax":
        with b.module.default_device(b.device):
            _check_double_weight_clone(monkeypatch)


def _check_double_weight_clone(monkeypatch):
    import jax
    import jax.numpy as jnp
    from types import SimpleNamespace
    from tmol_jax import ScoreFunction, ScoreType
    from .test_score_lifecycle import _quadratic_scorer

    scorer = _quadratic_scorer(monkeypatch)
    score = ScoreFunction(None, weight_dtype=np.float64)
    score.set_weight(ScoreType.fa_ljatr, 0.55)
    cloned = score.clone()
    assert cloned.weight_dtype == np.dtype(np.float64)
    assert cloned.get_weight(ScoreType.fa_ljatr) == 0.55
    with jax.enable_x64():
        pose = SimpleNamespace(
            coords=jnp.ones((1, 2, 3), jnp.float64), packed_block_types=None
        )
        value, gradient = cloned.jitted_value_and_grad(pose)(pose.coords)
        np.testing.assert_allclose(value, [6 * 0.55], rtol=2e-15, atol=2e-15)
        np.testing.assert_allclose(gradient, 2 * 0.55, rtol=2e-15, atol=2e-15)
    with pytest.raises(TypeError, match="float32 or float64"):
        ScoreFunction(None, weight_dtype=np.float16)
    with pytest.raises(ValueError, match="finite"):
        scorer.set_weight(ScoreType.fa_ljatr, 1e100)
