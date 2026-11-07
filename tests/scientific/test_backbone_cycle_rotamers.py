"""Two-residue cycles score both directed peptide bonds exactly once."""

from contextlib import nullcontext

import numpy as np
import pytest

from .molecules import CyclicPeptideCase
from .rotamer_execution import RotamerExecution


@pytest.mark.parametrize("sequence", (("ALA", "PRO"), ("DALA", "DPRO")))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_two_residue_cycle_sparse_values_and_full_gradients(backend, sequence, dtype):
    with backend.module.enable_x64() if backend.framework == "jax" else nullcontext():
        backend.molecule(
            CyclicPeptideCase((), (sequence,), ((2,),)), terms=("rama", "omega")
        )
        execution = RotamerExecution(backend, family="backbone")
        xyz = backend.numpy(execution.rotamers.coords).astype(dtype)
        values, indices = map(backend.numpy, execution.scores(xyz))
        np.testing.assert_array_equal(indices, [[0], [0], [1]])
        assert np.all(np.abs(values[:, 0]) > 1e-3)
        weights = np.asarray([[0.7], [1.3]], dtype=dtype)
        scored, gradient, _ = execution.value_and_grad(xyz, weights)
        whole, derivatives = backend.channel_values_and_gradients(
            backend.coords().astype(dtype)
        )
        order = [backend.channel_names().index(name) for name in ("rama", "omega")]
        tolerance = 2e-4 if dtype == np.float32 else 1e-9
        np.testing.assert_allclose(scored, whole[order], atol=tolerance, rtol=tolerance)
        expected = np.sum(derivatives[order] * weights[:, :, None, None], axis=0)
        for rot in range(
            execution.rotamers.n_rotamers_total
            if backend.framework == "jax"
            else len(execution.rotamers.pose_for_rot)
        ):
            r = execution.rotamers
            p, b = int(r.pose_for_rot[rot]), int(r.block_ind_for_rot[rot])
            start = int(r.coord_offset_for_rot[rot])
            source = int(backend.pose.block_coord_offset[p, b])
            count = backend.pose.block_type(p, b).n_atoms
            np.testing.assert_allclose(
                gradient[start : start + count],
                expected[p, source : source + count],
                atol=tolerance,
                rtol=tolerance,
            )
