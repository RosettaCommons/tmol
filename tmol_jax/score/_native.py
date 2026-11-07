"""Versioned Torch-free native registration and coordinate differentiation."""

import ctypes
from functools import lru_cache
from importlib import import_module

import jax
import jax.numpy as jnp
import numpy as np

ABI_VERSION = 4


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


def coordinate_energy(
    target,
    coords,
    parameters,
    *,
    channels=1,
    always_gradient=False,
    attributes=None,
    workspace_bytes=None,
):
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
        results = (shape,)
        if derivative:
            results += (jax.ShapeDtypeStruct(prefix + x.shape, x.dtype),)
        if workspace_bytes is not None:
            size = (
                workspace_bytes[derivative]
                if isinstance(workspace_bytes, tuple)
                else workspace_bytes
            )
            results += (jax.ShapeDtypeStruct((size,), np.uint8),)
        outputs = jax.ffi.ffi_call(
            target, results if len(results) > 1 else shape, vmap_method="sequential"
        )(x, *args, **(attributes or {}))
        if derivative:
            return outputs[0], outputs[1]
        return outputs[0] if len(results) > 1 else outputs

    @jax.custom_jvp
    def energy(x, *args):
        result = evaluate(x, args, always_gradient)
        return result[0] if always_gradient else result

    @energy.defjvp
    def energy_jvp(primals, tangents):
        x, *args = primals
        value, gradient = evaluate(x, args, True)
        return value, jnp.sum(gradient * tangents[0], axis=(-2, -1))

    return energy(coords, *parameters)


def pose_arguments(pose):
    """Flattened coordinate offsets and pose/block indices for native kernels."""
    cached = getattr(pose, "_native_pose_arguments", None)
    if cached is not None:
        return cached
    bti = np.asarray(pose.block_type_ind)
    poses, blocks = bti.shape
    first = np.arange(poses * blocks, dtype=np.int32).reshape(poses, blocks)
    first = np.where(bti >= 0, first, -1)
    offsets = np.asarray(pose.block_coord_offset) + (
        np.arange(poses, dtype=np.int32)[:, None] * pose.coords.shape[1]
    )
    arguments = jax.device_put(
        (
            offsets.reshape(-1),
            first,
            pose.block_type_ind,
            pose.inter_residue_connections,
        ),
        device=pose.coords.device,
    )
    # A pose is immutable. Replacement and pytree reconstruction create a new
    # owner, so changed layout/device/coordinate stride cannot reuse this entry.
    object.__setattr__(pose, "_native_pose_arguments", arguments)
    return arguments


def bonded_pose_arguments(pose, pbt):
    """Runtime layout and common connection paths for whole-pose kernels."""
    from tmol_core.chemical.packed import PackedBlockTypesData

    paths = PackedBlockTypesData.join_atom_paths_from_conn(pbt.active_block_types)
    if paths.shape[1] < pbt.max_n_conn:
        paths = np.pad(
            paths,
            ((0, 0), (0, pbt.max_n_conn - paths.shape[1]), (0, 0), (0, 0)),
            constant_values=-1,
        )
    return (*pose_arguments(pose), paths)


def atom_downstream_of_conn(pbt):
    """Shared downstream atom paths padded to the frontend's connection width."""
    from tmol_core.chemical.packed import PackedBlockTypesData

    paths = PackedBlockTypesData.join_atom_downstream_of_conn(pbt.active_block_types)
    return np.pad(
        paths,
        ((0, 0), (0, pbt.max_n_conn - paths.shape[1]), (0, 0)),
        constant_values=-1,
    )


@lru_cache(maxsize=None)
def workspace_sizes(target, dimensions, bits):
    """Query the native ABI's exact value/derivative workspace sizes on the host."""
    library = libraries().get("cpu")
    if library is None:
        raise RuntimeError("native workspace preparation requires CPU support")
    try:
        query = getattr(library, target + "_workspace_bytes")
    except AttributeError as error:
        raise RuntimeError(
            f"native workspace layout missing for {target}; rebuild"
        ) from error
    query.argtypes = [ctypes.c_int64] * len(dimensions) + [ctypes.c_int, ctypes.c_int]
    query.restype = ctypes.c_uint64
    sizes = tuple(query(*dimensions, bits, derivative) for derivative in (0, 1))
    if max(sizes) == (1 << 64) - 1:
        raise ValueError(f"invalid native workspace layout for {target}")
    return sizes


def fit_bspline_coefficients(values):
    """Fit shared periodic cubic coefficients during host parameter setup."""
    values = np.asarray(values)
    if values.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError("spline values must be float32 or float64")
    if values.ndim not in (2, 3, 4) or any(size == 0 for size in values.shape):
        raise ValueError("spline tables require 2-4 nonempty dimensions")
    library = libraries().get("cpu")
    if library is None:
        raise RuntimeError("spline preparation requires tmol-jax-native CPU support")
    try:
        fit = library.tmol_bspline_coefficients
    except AttributeError as error:
        raise RuntimeError(
            "native spline preparation is missing; rebuild tmol-jax-native"
        ) from error
    fit.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    fit.restype = ctypes.c_int
    output = np.array(values, order="C", copy=True)
    shape = (ctypes.c_int64 * output.ndim)(*output.shape)
    error = ctypes.create_string_buffer(256)
    status = fit(
        output.ctypes.data,
        output.dtype.itemsize * 8,
        output.ndim,
        shape,
        error,
        len(error),
    )
    if status:
        raise ValueError(error.value.decode("utf-8", errors="replace"))
    return output
