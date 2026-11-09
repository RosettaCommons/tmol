"""Shared independent-graph minima, trajectory schedules and cache boundaries."""

from contextlib import contextmanager, nullcontext

import numpy as np
import pytest


def independent_minimum_tables(counts, chunk_size, baseline=0, native_empty=False):
    """Build one host graph; preserve both historical empty adjacency layouts."""
    counts = np.asarray(counts, np.int32)
    n_res, n_rots = len(counts), int(counts.sum())
    n_for_res = np.zeros((2, n_res), np.int32)
    n_for_res[1] = counts
    offsets = np.zeros_like(n_for_res)
    offsets[1, 1:] = counts.cumsum()[:-1]
    edges = np.full((2, n_res, n_res), -1, np.int64)
    chunks, energies = [], []
    for i in range(n_res - 1):
        for a, b in ((i, i + 1), (i + 1, i)):
            if native_empty and not n_rots:
                continue
            edges[1, a, b] = len(chunks)
            for first in range(0, counts[a], chunk_size):
                for second in range(0, counts[b], chunk_size):
                    chunks.append(len(energies))
                    rows = np.arange(first, min(first + chunk_size, counts[a]))
                    cols = np.arange(second, min(second + chunk_size, counts[b]))
                    energies.extend((rows[:, None] + cols[None, :]).ravel())
    return (
        np.array([0, n_res], np.int32),
        np.array([0, n_rots], np.int32),
        np.zeros(2, np.int32),
        n_for_res,
        offsets,
        np.repeat(np.arange(n_res, dtype=np.int32), counts),
        np.r_[np.int64(0), (edges != -1).sum(2).ravel().cumsum()],
        np.nonzero(edges != -1)[2].astype(np.int32),
        edges[edges != -1],
        np.asarray(chunks, np.int64),
        np.concatenate([np.arange(n, dtype=np.float32) for n in counts]) + baseline,
        np.asarray(energies, np.float32),
    )


def _arrays(b, host):
    if b.framework == "torch":
        return tuple(b.module.as_tensor(x, device=b.device) for x in host)
    return tuple(
        b.module.device_put(
            x.view(np.uint8).reshape(-1, 8) if i in (6, 8, 9) else x, b.device
        )
        for i, x in enumerate(host)
    )


def _execution(b, args, chunk, trajectories, seed):
    if b.framework == "torch":
        # Direct dispatch follows tensor storage. The Apple CPU-pose lane tests
        # its CPU component here; molecular offload tests cover Metal execution.
        from tmol.pack.compiled._compiled import _pack_anneal

        def run(onebody):
            b.module.manual_seed(seed)
            return _pack_anneal(
                len(args[5]),
                *args[:6],
                chunk,
                *args[6:-2],
                onebody,
                args[-1],
                trajectories,
            )

        return run
    from tmol_jax.pack.annealing import anneal_cpu, anneal_cuda, cuda_phase_seeds

    kwargs = dict(chunk_size=chunk, n_trajectories=trajectories)
    cuda = b.requested_device == "cuda"
    if cuda:
        phase = (
            np.array(seed, np.uint64).view(np.uint8).reshape(3, 8)
            if isinstance(seed, tuple)
            else cuda_phase_seeds(seed, 0)
        )
        kwargs.update(
            max_rotamers=len(args[5]), phase_seeds=b.module.device_put(phase, b.device)
        )
    else:
        kwargs["seed"] = seed
    operation = anneal_cuda if cuda else anneal_cpu
    return lambda onebody: operation(*args[:-2], onebody, args[-1], **kwargs)


def _rng_state(b):
    if b.framework == "torch":
        return b.numpy(
            b.module.cuda.get_rng_state(b.device)
            if b.requested_device == "cuda"
            else b.module.get_rng_state()
        ).copy()


@contextmanager
def _cpu_threads(b, threads, monkeypatch):
    if b.framework == "torch":
        previous = b.module.get_num_threads()
        try:
            if threads is not None:
                b.module.set_num_threads(threads)
            yield b.module.get_num_threads()
        finally:
            b.module.set_num_threads(previous)
    else:
        from tmol_jax.pack.annealing import _cpu_thread_budget

        with monkeypatch.context() as context:
            if threads is not None:
                context.setenv("OMP_NUM_THREADS", str(threads))
            yield _cpu_thread_budget()


@pytest.mark.parametrize("counts", ([0, 0], [2, 3]))
@pytest.mark.parametrize(
    "threads,trajectories", ((None, 5), (None, 128), (1, 5), (8, 5), (8, 128))
)
def test_annealer_empty_pose_has_no_energy(
    backend, monkeypatch, counts, threads, trajectories
):
    b = backend
    cuda = b.requested_device == "cuda"
    if cuda and threads is not None:
        pytest.skip("explicit CPU thread schedule")
    baseline = 5 if threads is None else 3
    host = independent_minimum_tables(counts, 2, baseline, native_empty=threads is None)
    if cuda and not any(counts):
        host[0][:] = 0  # Original CUDA empty case has no residues in either pose.
    args = _arrays(b, host)
    before = tuple(b.numpy(x).copy() for x in args)
    seeds = (0, 42, (1 << 64) - 1) if threads is None else (73,)
    if cuda and b.framework == "jax":
        seeds = ((0, 0, 0), (42, 1 << 63, (1 << 64) - 1))
    # CUDA has no CPU thread budget; the context only governs CPU execution.
    context = _cpu_threads(b, threads, monkeypatch) if not cuda else nullcontext(8)
    with context as budget:
        width = (2 if cuda else 3) if trajectories == 5 else 80
        if not cuda and budget < 8:
            width = 1
        expected = np.repeat(
            [[0], [baseline * sum(n > 0 for n in counts)]], width, axis=1
        )
        for seed in seeds:
            run = _execution(b, args, 2, trajectories, seed)
            scores, choices = (
                b.module.jit(run)(args[-2])
                if cuda and b.framework == "jax"
                else run(args[-2])
            )
            np.testing.assert_array_equal(b.numpy(scores), expected)
            assert choices.shape == (2, width, 2)
            np.testing.assert_array_equal(b.numpy(choices)[0], 0)
            np.testing.assert_array_equal(
                b.numpy(choices)[1], -1 if not cuda and not any(counts) else 0
            )
            assert (
                b.numpy(scores).dtype == np.float32
                and b.numpy(choices).dtype == np.int32
            )
            assert scores.device == choices.device == b.device
            state = _rng_state(b)
            for first, again in zip((scores, choices), run(args[-2])):
                np.testing.assert_array_equal(b.numpy(first), b.numpy(again))
            if state is not None:
                np.testing.assert_array_equal(_rng_state(b), state)
            if b.framework == "jax" and threads is None and (cuda or seed == 42):
                values, assignments = b.module.jit(b.module.vmap(run))(
                    b.module.numpy.stack((args[-2], args[-2] + 1))
                )
                np.testing.assert_array_equal(b.numpy(values)[0], expected)
                np.testing.assert_array_equal(
                    b.numpy(values)[1],
                    expected + np.array([[0], [sum(n > 0 for n in counts)]]),
                )
                np.testing.assert_array_equal(
                    b.numpy(assignments), np.stack((b.numpy(choices),) * 2)
                )
                with pytest.raises(ValueError, match="differentiated"):
                    b.module.grad(lambda x: run(x)[0].sum())(args[-2])
        if b.framework == "jax" and not cuda and threads is None:
            from tmol_jax.pack.annealing import anneal_cpu

            run = _execution(b, args, 2, trajectories, 42)
            with b.module.enable_x64(), pytest.raises(TypeError, match="float32"):
                run(args[-2].astype(np.float64))
            for options, message in (
                (dict(seed=1 << 64), "64-bit"),
                (dict(chunk_size=0), "chunk size"),
                (dict(n_trajectories=3), "trajector"),
                (dict(n_trajectories=2**31), "trajector"),
            ):
                with pytest.raises(ValueError, match=message):
                    anneal_cpu(*args, **(dict(chunk_size=2, seed=0) | options))
    for original, value in zip(before, args):
        np.testing.assert_array_equal(b.numpy(value), original)


@pytest.mark.parametrize(
    "chunk_size,n_res", ((7, 3), (16, 1), (16, 2), (16, 128), (16, 129), (32, 3))
)
def test_cuda_annealer_assignment_cache_boundaries(backend, chunk_size, n_res):
    b = backend
    if b.requested_device != "cuda":
        pytest.skip("CUDA assignment-cache boundary")
    counts = [1] * n_res if n_res <= 2 else [35, 36, 17] if n_res == 3 else [2] * n_res
    args = _arrays(b, independent_minimum_tables(counts, chunk_size))
    scores, assignments = _execution(b, args, chunk_size, 128, 20260911)(args[-2])
    assert scores.shape == (2, 80) and assignments.shape == (2, 80, n_res)
    np.testing.assert_array_equal(b.numpy(scores), 0)
    np.testing.assert_array_equal(b.numpy(assignments), 0)


@pytest.mark.parametrize("n_res", (128, 129))
def test_cuda_annealer_fixed_seed_is_repeatedly_exact(backend, n_res):
    b = backend
    if b.requested_device != "cuda":
        pytest.skip("CUDA assignment-cache boundary")
    args = _arrays(b, independent_minimum_tables([2] * n_res, 16))
    run = _execution(b, args, 16, 128, 20260915)
    expected = None
    for _ in range(3):
        scores, assignments = run(args[-2])
        actual = (b.numpy(scores).copy(), b.numpy(assignments).copy(), _rng_state(b))
        if expected is not None:
            for value, previous in zip(actual, expected):
                if previous is not None:
                    np.testing.assert_array_equal(value, previous)
        expected = actual
