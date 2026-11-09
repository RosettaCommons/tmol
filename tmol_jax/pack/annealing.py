"""Array execution of the common CPU and CUDA annealing schedules."""

from functools import lru_cache, partial
import ctypes
import os

import jax
import numpy as np

from tmol_core.pack.annealing import (
    DEFAULT_ANNEAL_TRAJECTORIES,
    MIN_CPU_FUNNEL_THREADS,
    annealer_trajectory_counts,
    validate_anneal_trajectories,
)


def cuda_phase_seeds(seed, call):
    """Use the shared Philox generator for mature CUDA phase-seed draws."""
    from tmol_jax._host import invoke

    (values,) = invoke(17, np.uint64(seed).view(np.int64), call)
    return values.view(np.uint8).reshape(3, 8)


@lru_cache(maxsize=1)
def _register_cpu_annealer():
    from tmol_jax.score._native import libraries

    library = libraries().get("cpu")
    if library is None:
        raise RuntimeError("native CPU annealing requires the CPU adapter")
    target = "tmol_anneal_cpu"
    try:
        symbol = getattr(library, target)
        query = library.tmol_cpu_annealer_workspace
        threads = library.tmol_cpu_thread_count
    except AttributeError as error:
        raise RuntimeError(
            "rebuild the matching native CPU annealing adapter"
        ) from error
    jax.ffi.register_ffi_target(target, jax.ffi.pycapsule(symbol), platform="cpu")
    jax.ffi.register_ffi_target(
        "tmol_cpu_thread_count", jax.ffi.pycapsule(threads), platform="cpu"
    )
    query.argtypes, query.restype = [ctypes.c_int64] * 3, ctypes.c_uint64
    return target, query


@lru_cache(maxsize=1)
def cpu_thread_count():
    """Read the actual XLA CPU pool; the OMP budget may further limit search."""
    _register_cpu_annealer()
    with jax.ensure_compile_time_eval(), jax.default_device(jax.devices("cpu")[0]):
        value = jax.ffi.ffi_call(
            "tmol_cpu_thread_count", jax.ShapeDtypeStruct((), np.int32)
        )()
        return int(np.asarray(value))


def _cpu_thread_budget():
    available = cpu_thread_count()
    requested = os.environ.get("OMP_NUM_THREADS")
    return available if requested is None else min(available, max(1, int(requested)))


@partial(jax.jit, static_argnames=("chunk_size", "seed", "n_trajectories", "threads"))
def _execute_cpu(arguments, *, chunk_size, seed, n_trajectories, threads):
    poses, residues = arguments[3].shape
    rotamers = arguments[5].size
    target, query = _register_cpu_annealer()
    funnel = threads >= MIN_CPU_FUNNEL_THREADS
    final = annealer_trajectory_counts(n_trajectories, cpu=True)[2] if funnel else 1
    capacity = query(poses, residues, n_trajectories) if funnel else 0
    if capacity == (1 << 64) - 1:
        raise ValueError("invalid native CPU annealing workspace dimensions")
    shape = jax.ShapeDtypeStruct
    return jax.ffi.ffi_call(
        target,
        (
            shape((poses, final), np.float32),
            shape((poses, final, residues), np.int32),
            shape((poses, 1, residues), np.int32),
            shape((rotamers,), np.int32),
            shape((capacity,), np.uint8),
        ),
        vmap_method="sequential",
    )(
        *arguments,
        chunk_size=np.int64(chunk_size),
        seed=np.uint64(seed).view(np.int64),
        n_trajectories=np.int64(n_trajectories),
        threads=np.int64(threads),
    )[
        :2
    ]


def anneal_cpu(
    *arguments, chunk_size, seed, n_trajectories=DEFAULT_ANNEAL_TRAJECTORIES
):
    """Run the mature CPU search with an explicit reproducible uint64 seed.

    This internal entry consumes prepared interaction-graph arrays: pose sizes,
    pose counts/offsets, residue counts/one-body offsets, residue per rotamer,
    CSR row offsets/neighbor IDs/edge chunk offsets/chunk offsets, and one-/two-
    body energies. CSR int64 arrays use contiguous uint8 rows of eight bytes,
    preserving their capacity independently of JAX's float64 setting. Energies
    are float32, matching the mature CPU algorithm; no precision conversion is
    performed. Outputs are scores and local residue assignments.

    Inputs are immutable; XLA owns temporary and output arrays. Native execution
    supports jit/vmap on CPU, but discrete annealing has no derivative rule.
    A budget of at least eight native CPU threads enables the shared funnel;
    smaller budgets retain single-trajectory search. This is not a complete
    packing frontend.
    """
    if len(arguments) != 12:
        raise ValueError("CPU annealing requires twelve prepared graph arrays")
    if not isinstance(seed, (int, np.integer)) or not 0 <= seed < 1 << 64:
        raise ValueError("annealing seed must be an unsigned 64-bit integer")
    if not isinstance(chunk_size, (int, np.integer)) or not 0 < chunk_size < 1 << 31:
        raise ValueError("annealing chunk size must be a positive int32")
    if any(
        not isinstance(a, jax.core.Tracer)
        and isinstance(a, jax.Array)
        and a.device.platform != "cpu"
        for a in arguments
    ):
        raise ValueError("the native CPU annealing schedule requires CPU arrays")
    if any(a.dtype != np.float32 for a in arguments[-2:]):
        raise TypeError("native CPU annealing requires float32 energy tables")
    n_trajectories = validate_anneal_trajectories(n_trajectories)
    return _execute_cpu(
        arguments,
        chunk_size=int(chunk_size),
        seed=int(seed),
        n_trajectories=n_trajectories,
        threads=_cpu_thread_budget(),
    )


@lru_cache(maxsize=1)
def _register_cuda_annealer():
    from tmol_jax.score._native import libraries

    library = libraries().get("cuda")
    if library is None:
        raise RuntimeError("native CUDA annealing requires the CUDA adapter")
    target = "tmol_anneal_cuda"
    try:
        symbol = getattr(library, target)
        query = library.tmol_cuda_annealer_workspace
    except AttributeError as error:
        raise RuntimeError("rebuild the matching CUDA annealing adapter") from error
    jax.ffi.register_ffi_target(target, jax.ffi.pycapsule(symbol), platform="CUDA")
    query.argtypes, query.restype = [ctypes.c_int64] * 5, ctypes.c_uint64
    return target, query


@partial(jax.jit, static_argnames=("chunk_size", "max_rotamers", "n_trajectories"))
def _execute_cuda(arguments, phase_seeds, *, chunk_size, max_rotamers, n_trajectories):
    target, query = _register_cuda_annealer()
    poses, residues = arguments[3].shape
    capacity = query(poses, residues, max_rotamers, arguments[7].size, n_trajectories)
    final = annealer_trajectory_counts(n_trajectories)[2]
    if capacity == (1 << 64) - 1:
        raise ValueError("invalid native CUDA annealing workspace dimensions")
    shape = jax.ShapeDtypeStruct
    return jax.ffi.ffi_call(
        target,
        (
            shape((poses, final), np.float32),
            shape((poses, final, residues), np.int32),
            shape((capacity,), np.uint8),
        ),
        vmap_method="sequential",
    )(
        *arguments,
        phase_seeds,
        chunk_size=np.int64(chunk_size),
        max_rotamers=np.int64(max_rotamers),
        n_trajectories=np.int64(n_trajectories),
    )[
        :2
    ]


def anneal_cuda(
    *arguments,
    chunk_size,
    phase_seeds,
    max_rotamers,
    n_trajectories=DEFAULT_ANNEAL_TRAJECTORIES,
):
    """Run the mature CUDA schedule with three explicit uint64 phase seeds.

    Graph arrays use the same layout as ``anneal_cpu``. ``phase_seeds`` is
    uint8[3,8], transporting all 64 seed bits with either JAX precision setting.
    ``max_rotamers`` is the maximum retained conformer count per pose, prepared
    outside tracing. Output width follows the common trajectory policy (80
    survivors with the default 128 high-temperature trajectories).
    Inputs are immutable; execution supports jit/vmap, with no discrete AD rule.
    The CPU and CUDA schedules differ and need not return identical assignments.
    """
    if len(arguments) != 12:
        raise ValueError("CUDA annealing requires twelve prepared graph arrays")
    if not isinstance(chunk_size, (int, np.integer)) or not 0 < chunk_size < 1 << 31:
        raise ValueError("annealing chunk size must be a positive int32")
    if (
        not isinstance(max_rotamers, (int, np.integer))
        or not 0 <= max_rotamers < 1 << 30
    ):
        raise ValueError("maximum conformer count exceeds native capacity")
    if phase_seeds.shape != (3, 8) or phase_seeds.dtype != np.uint8:
        raise TypeError("CUDA phase seeds require uint8[3,8]")
    if any(a.dtype != np.float32 for a in arguments[-2:]):
        raise TypeError("native CUDA annealing requires float32 energy tables")
    if any(
        not isinstance(a, jax.core.Tracer) and a.device.platform != "gpu"
        for a in (*arguments, phase_seeds)
    ):
        raise ValueError("native CUDA annealing requires CUDA arrays")
    return _execute_cuda(
        arguments,
        phase_seeds,
        chunk_size=int(chunk_size),
        max_rotamers=int(max_rotamers),
        n_trajectories=validate_anneal_trajectories(n_trajectories),
    )
