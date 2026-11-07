"""Framework execution retains one native RNG stream through real Relax stages."""

import numpy as np
import pytest

from tests.scientific.execution import Execution
from tests.scientific.test_fast_relax_execution import check_fast_relax_molecular_stages
from tests.scientific.test_packing_seed import (
    SEEDS,
    CUDA_SEEDS,
    observe_native_seed_stream,
)


@pytest.mark.parametrize("platform", ("cpu", "cuda"))
def test_relax_advances_search_stream_between_molecular_stages(platform, monkeypatch):
    backend = Execution("jax-" + platform)
    monkeypatch.setenv("TMOL_PACK_MAX_POSES_PER_CHUNK", "2")
    observed, _ = observe_native_seed_stream(backend, monkeypatch)
    check_fast_relax_molecular_stages(backend, "mixed_batch", seed=17)
    expected = (CUDA_SEEDS if platform == "cuda" else SEEDS)[17]
    np.testing.assert_array_equal(
        np.asarray(observed, dtype=np.uint64), np.asarray(expected, dtype=np.uint64)
    )
