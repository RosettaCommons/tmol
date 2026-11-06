"""Array execution of the common CPU annealing schedule."""

from functools import lru_cache, partial

import jax
import numpy as np


@lru_cache(maxsize=1)
def _register_cpu_annealer():
    from tmol_jax.score._native import libraries

    library = libraries().get("cpu")
    if library is None:
        raise RuntimeError("native CPU annealing requires the CPU adapter")
    target = "tmol_anneal_cpu"
    try:
        symbol = getattr(library, target)
    except AttributeError as error:
        raise RuntimeError(
            "rebuild the matching native CPU annealing adapter"
        ) from error
    jax.ffi.register_ffi_target(target, jax.ffi.pycapsule(symbol), platform="cpu")
    return target


@partial(jax.jit, static_argnames=("chunk_size", "seed"))
def _execute_cpu(arguments, *, chunk_size, seed):
    poses, residues = arguments[3].shape
    rotamers = arguments[5].size
    shape = jax.ShapeDtypeStruct
    return jax.ffi.ffi_call(
        _register_cpu_annealer(),
        (
            shape((poses, 1), np.float32),
            shape((poses, 1, residues), np.int32),
            shape((poses, 1, residues), np.int32),
            shape((rotamers,), np.int32),
        ),
        vmap_method="sequential",
    )(*arguments, chunk_size=np.int64(chunk_size), seed=np.uint64(seed).view(np.int64))[
        :2
    ]


def anneal_cpu(*arguments, chunk_size, seed):
    """Run one mature CPU trajectory with explicit reproducible uint64 seed.

    This internal entry consumes prepared interaction-graph arrays: pose sizes,
    pose counts/offsets, residue counts/one-body offsets, residue per rotamer,
    CSR row offsets/neighbor IDs/edge chunk offsets/chunk offsets, and one-/two-
    body energies. CSR int64 arrays use contiguous uint8 rows of eight bytes,
    preserving their capacity independently of JAX's float64 setting. Energies
    are float32, matching the mature CPU algorithm; no precision conversion is
    performed. Outputs are scores and local residue assignments.

    Inputs are immutable; XLA owns temporary and output arrays. Native execution
    supports jit/vmap on CPU, but discrete annealing has no derivative rule.
    This is not the CUDA schedule or a complete packing frontend.
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
    return _execute_cpu(arguments, chunk_size=int(chunk_size), seed=int(seed))
