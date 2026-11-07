"""Complete sparse scores conserve whole-pose science for input conformers."""

from contextlib import nullcontext

import numpy as np
import pytest

from .test_complete_preset import CASES
from .rotamer_execution import RotamerExecution


def complete_rotamer_scorer(backend, *, sampled=False):
    execution = RotamerExecution(backend, family="genbonded", sampled=sampled)
    backend.use_beta2016()
    scorer = backend.score_function.render_rotamer_scoring_module(
        backend.pose, execution.rotamers
    )
    if backend.framework == "jax":
        return execution, scorer.prepare(execution.rotamers.coords)

    def evaluate(coords):
        result = scorer(coords).coalesce()
        return result.values(), result.indices()

    return execution, evaluate


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_complete_rotamer_score_conserves_whole_pose(backend, name, dtype):
    precision = (
        backend.module.enable_x64() if backend.framework == "jax" else nullcontext()
    )
    with precision:
        backend.molecule(CASES[name], terms=("ref",))
        e, score = complete_rotamer_scorer(backend)
        xyz = backend.numpy(e.rotamers.coords).astype(dtype)
        x = e.array(xyz)
        if backend.framework == "jax":
            values, indices = backend.module.jit(score)(x)
            dx = backend.module.jit(backend.module.grad(lambda x: score(x)[0].sum()))(x)
        else:
            x = x.detach().requires_grad_()
            values, indices = score(x)
            dx = backend.module.autograd.grad(values.sum(), x)[0]
        values, indices, dx = map(backend.numpy, (values, indices, dx))
        total = np.zeros(backend.pose.n_poses)
        np.add.at(total, indices[0], values)
        whole, whole_gradient = backend.value_and_grad(backend.coords().astype(dtype))
        np.testing.assert_allclose(total, whole, rtol=3e-6, atol=3e-5)
        assert np.isfinite(values).all() and np.any(np.abs(dx) > 1e-3)
        pbt = backend.pose.packed_block_types
        for r, (p, b, t, begin) in enumerate(
            zip(
                *[
                    backend.numpy(getattr(e.rotamers, field))
                    for field in (
                        "pose_for_rot",
                        "block_ind_for_rot",
                        "block_type_ind_for_rot",
                        "coord_offset_for_rot",
                    )
                ]
            )
        ):
            count = len(pbt.active_block_types[t].atoms)
            offset = int(backend.numpy(backend.pose.block_coord_offset)[p, b])
            np.testing.assert_allclose(
                dx[begin : begin + count],
                whole_gradient[p, offset : offset + count],
                rtol=3e-5,
                atol=3e-5,
            )
