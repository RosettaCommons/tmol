"""Public search seeds match mature draws without touching global RNG state."""

from importlib import import_module

import numpy as np
import pytest

from .test_packing_execution import check_complete_packing
from .test_fast_relax_execution import check_fast_relax_molecular_stages

# Independent full-int64 Torch generator draws, reinterpreted as uint64.
SEEDS = {
    0: (10123822223749065263, 13192915183495924928),
    17: (5435609932269944049, 9787598041570588038),
    (1 << 64) - 1: (1800993050274709794, 16830492821160766535),
}
SEEDS[-1] = SEEDS[(1 << 64) - 1]  # Independently observed Torch seed alias.


CUDA_SEEDS = {
    0: [
        [7361108121267848589, 9531048246547148458, 459884446619952985],
        [17934684622398030043, 14590808054892559853, 8652275798068798352],
    ],
    17: [
        [7541213820192324597, 10336368738079625943, 4322329562509144791],
        [11906501371032699704, 5718183226368221346, 15448450724482114173],
    ],
    18446744073709551615: [
        [8260858497399736121, 12052255946454675714, 12893778847829730467],
        [1873168765288964148, 2394938694379724090, 5422658774283383720],
    ],
}


def observe_native_seed_stream(backend, monkeypatch):
    observed = []
    cuda = backend.requested_device == "cuda"
    if backend.framework == "jax":
        execution = import_module("tmol_jax.pack.packing")
        operation = "anneal_cuda" if cuda else "anneal_cpu"
        anneal = getattr(execution, operation)

        def record(*args, **kwargs):
            observed.append(
                np.asarray(kwargs["phase_seeds"]).view(np.uint64).ravel().tolist()
                if cuda
                else kwargs["seed"]
            )
            return anneal(*args, **kwargs)

        monkeypatch.setattr(execution, operation, record)
        front = import_module("tmol_jax.pack")
    else:
        import torch

        execution = import_module("tmol.pack.compiled._compiled")
        anneal = execution._pack_anneal

        def record(*args, **kwargs):
            device = backend.device if cuda else torch.device("cpu")

            def state():
                return (
                    torch.cuda.get_rng_state(device) if cuda else torch.get_rng_state()
                )

            generator = torch.Generator(device=device).set_state(state())
            draw = torch.empty(
                (3,) if cuda else (1,), dtype=torch.int64, device=device
            ).random_(-(1 << 63), None, generator=generator)
            result = anneal(*args, **kwargs)
            # The real native search consumes exactly this draw from Torch.
            torch.testing.assert_close(state(), generator.get_state())
            words = [int(value) & ((1 << 64) - 1) for value in draw.cpu()]
            observed.append(words if cuda else words[0])
            return result

        monkeypatch.setattr(execution, "_pack_anneal", record)
        front = import_module("tmol")

    return observed, front


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("chunked", (False, True))
def test_search_seed_stream_preserves_molecular_packing(
    backend, monkeypatch, seed, chunked
):
    monkeypatch.setenv("TMOL_PACK_MAX_POSES_PER_CHUNK", "1" if chunked else "2")
    cuda = backend.requested_device == "cuda"
    reference_seed = (1 << 64) - 1 if seed == -1 else seed
    expected = (CUDA_SEEDS if cuda else SEEDS)[reference_seed][: 2 if chunked else 1]
    observed, front = observe_native_seed_stream(backend, monkeypatch)
    pack = front.pack_rotamers

    def preserving_globals(*args, **kwargs):
        # Exercise the mature public verbosity option without changing search.
        kwargs["verbose"] = seed < 0
        numpy_state = np.random.get_state()
        if backend.framework == "torch":
            torch = backend.module
            torch_state = torch.get_rng_state()
            cuda_state = torch.cuda.get_rng_state(backend.device) if cuda else None
        result = pack(*args, **kwargs)
        for before, after in zip(numpy_state, np.random.get_state()):
            np.testing.assert_array_equal(before, after)
        if backend.framework == "torch":
            torch.testing.assert_close(torch.get_rng_state(), torch_state)
            if cuda:
                torch.testing.assert_close(
                    torch.cuda.get_rng_state(backend.device), cuda_state
                )
        return result

    monkeypatch.setattr(front, "pack_rotamers", preserving_globals)
    # Includes noncanonical/D/ligand identities, complete score and gradient,
    # real sampled choices, input/topology preservation and independent descent.
    check_complete_packing(
        backend, "mixed_batch", seed=seed, anneal_device="cuda" if cuda else "cpu"
    )
    np.testing.assert_array_equal(
        np.asarray(observed, dtype=np.uint64), np.asarray(expected, dtype=np.uint64)
    )

    if not chunked and seed in (17, -1):
        # The two Relax stages must advance one stream, not reseed each pack.
        # Reuse these existing rows instead of expanding the seed/chunk grid.
        monkeypatch.setattr(front, "pack_rotamers", pack)
        observed.clear()
        check_fast_relax_molecular_stages(backend, "mixed_batch", seed=seed)
        expected = (CUDA_SEEDS if cuda else SEEDS)[reference_seed]
        np.testing.assert_array_equal(
            np.asarray(observed, dtype=np.uint64), np.asarray(expected, dtype=np.uint64)
        )
