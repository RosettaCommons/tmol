"""Versioned Torch-free native registration and coordinate differentiation."""

import ctypes
from functools import lru_cache
from importlib import import_module

import jax
import jax.numpy as jnp
import numpy as np

ABI_VERSION = 1


@lru_cache(maxsize=1)
def libraries():
    """Retain loaded adapters for the lifetime of their registered symbols."""
    try:
        package = import_module("tmol_jax_native")
    except ImportError:
        return {}
    loaded = {}
    for platform in ("cpu", "cuda"):
        path = package.library_path(platform)
        if path is None:
            continue
        library = ctypes.CDLL(str(path))
        try:
            marker = library.tmol_jax_native_abi_version
        except AttributeError as error:
            raise RuntimeError(
                "tmol-jax-native has no ABI marker; rebuild it"
            ) from error
        marker.argtypes = []
        marker.restype = ctypes.c_int
        if marker() != ABI_VERSION:
            raise RuntimeError(
                "tmol-jax-native ABI mismatch; install matching artifacts"
            )
        loaded[platform] = library
    return loaded


@lru_cache(maxsize=None)
def register(target):
    registered = []
    for platform, library in libraries().items():
        try:
            symbol = getattr(library, target)
        except AttributeError as error:
            raise RuntimeError(f"tmol-jax-native lacks {target}; rebuild it") from error
        jax.ffi.register_ffi_target(
            target,
            jax.ffi.pycapsule(symbol),
            platform=platform.upper() if platform == "cuda" else platform,
        )
        registered.append(platform)
    if not registered:
        raise RuntimeError(
            f"{target} requires tmol-jax-native; build and install the native adapter"
        )
    return tuple(registered)


def coordinate_energy(target, coords, parameters, *, channels=1):
    """First coordinate derivative from the common analytic native kernels.

    Outputs are independent per pose. The native value/gradient call defines a
    linear tangent map, so JAX provides JVP and its reverse transpose without
    differentiating fixed chemistry or parameter tables. Higher derivatives
    require a native derivative-of-gradient rule and are not implemented.
    """
    if coords.dtype not in (jnp.float32, jnp.float64):
        raise TypeError(
            "native molecular scoring requires float32 or float64 coordinates"
        )
    parameters = tuple(
        jax.lax.stop_gradient(
            p.astype(coords.dtype) if jnp.issubdtype(p.dtype, jnp.floating) else p
        )
        for p in parameters
    )
    prefix = () if channels == 1 else (channels,)
    shape = jax.ShapeDtypeStruct(prefix + coords.shape[:1], coords.dtype)

    def evaluate(x, args, derivative):
        results = (
            (shape, jax.ShapeDtypeStruct(prefix + x.shape, x.dtype))
            if derivative
            else shape
        )
        return jax.ffi.ffi_call(target, results, vmap_method="sequential")(x, *args)

    @jax.custom_jvp
    def energy(x, *args):
        return evaluate(x, args, False)

    @energy.defjvp
    def energy_jvp(primals, tangents):
        x, *args = primals
        value, gradient = evaluate(x, args, True)
        return value, jnp.sum(gradient * tangents[0], axis=(-2, -1))

    return energy(coords, *parameters)


def bonded_pose_arguments(pose, pbt):
    """Runtime layout and common connection paths for whole-pose kernels."""
    from tmol_core.chemical.packed import PackedBlockTypesData

    bti = np.asarray(pose.block_type_ind)
    poses, blocks = bti.shape
    first = np.arange(poses * blocks, dtype=np.int32).reshape(poses, blocks)
    first = np.where(bti >= 0, first, -1)
    offsets = np.asarray(pose.block_coord_offset) + (
        np.arange(poses, dtype=np.int32)[:, None] * pose.coords.shape[1]
    )
    paths = PackedBlockTypesData.join_atom_paths_from_conn(pbt.active_block_types)
    if paths.shape[1] < pbt.max_n_conn:
        paths = np.pad(
            paths,
            ((0, 0), (0, pbt.max_n_conn - paths.shape[1]), (0, 0), (0, 0)),
            constant_values=-1,
        )
    return offsets.reshape(-1), first, bti, pose.inter_residue_connections, paths
