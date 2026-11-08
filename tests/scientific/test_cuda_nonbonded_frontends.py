"""Direct CUDA coherence when both independent frontends are installed."""

import numpy as np
import pytest

from .execution import Execution
from .molecules import DOMAIN_CASES, MolecularCase
from .test_canonical_molecular_input import canonical_input, rebuild_canonical


@pytest.mark.parametrize(
    "case",
    [
        MolecularCase((), (("ALA", "LEU", "ARG", "PHE"),), ((4,),)),
        DOMAIN_CASES["mixed_batch"],
    ],
    ids=("canonical", "mixed_noncanonical_ligand"),
)
def test_cuda_nonbonded_frontends_preserve_channel_gradients(case):
    # Individual frontend/device domains have their own shared tests. This
    # additional comparison requires both runtimes on an actual CUDA host.
    executions = [Execution(name) for name in ("torch-cuda", "jax-cuda")]
    results = []
    for backend in executions:
        backend.molecule(case, terms=("fa_ljatr", "fa_ljrep", "fa_lk", "fa_elec"))
        ordering, canonical = canonical_input(case, backend.parameter_database)
        rebuild_canonical(backend, ordering, canonical)
        coords = backend.coords()
        assert coords.dtype == np.float32
        values, gradients = backend.channel_values_and_gradients(coords)
        assert np.all(np.abs(values) > 1e-3)
        assert np.all(np.linalg.norm(gradients.reshape(4, -1), axis=1) > 1e-3)
        results.append((coords, values, gradients))
    np.testing.assert_array_equal(results[0][0], results[1][0])
    for first, second in zip(results[0][1:], results[1][1:]):
        np.testing.assert_allclose(first, second, rtol=3e-6, atol=3e-5)
