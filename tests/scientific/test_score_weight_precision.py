"""Default weight precision matches mature TMol independently of coordinates."""

from contextlib import nullcontext
import numpy as np
from .molecules import DOMAIN_CASES


def test_nonbinary_weights_keep_default_storage_precision(backend):
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
