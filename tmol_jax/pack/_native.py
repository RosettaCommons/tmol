"""Native execution for chi sampling and conformer initialization."""

import ctypes
from functools import lru_cache, partial

import jax
import numpy as np

from tmol_jax._host import invoke


@partial(jax.jit, static_argnames=("target", "sampler", "n_rows", "root_shift"))
def _conformer_dofs(
    dofs, arguments, *, target, sampler=None, n_rows=None, root_shift=0
):
    attributes = {} if sampler is None else {"sampler": np.int64(sampler)}
    if n_rows is not None:
        attributes.update(n_rows=np.int64(n_rows), root_shift=np.int64(root_shift))
    return jax.ffi.ffi_call(
        target,
        jax.ShapeDtypeStruct(dofs.shape, dofs.dtype),
        input_output_aliases={0: 0},
        vmap_method="sequential",
    )(dofs, *arguments, **attributes)


def copy_conformer_dofs(dofs, source, *metadata, sampler):
    """Transfer finite retained-atom DOFs using shared native chemistry maps.

    The input remains immutable; XLA owns the aliased output allocation. This
    initializes independent conformers and has no coordinate autodiff rule.
    """
    from tmol_jax.score._native import register

    target = "tmol_copy_conformer_dofs"
    register(target)
    return _conformer_dofs(dofs, (source, *metadata), target=target, sampler=sampler)


def assign_conformer_chi(dofs, chi, *metadata):
    """Apply sampled chi and shared ring corrections to independent DOFs."""
    from tmol_jax.score._native import register

    target = "tmol_assign_conformer_chi"
    register(target)
    return _conformer_dofs(dofs, (chi, *metadata), target=target)


def copy_conformer_rows(
    values,
    source,
    source_offsets,
    target_offsets,
    packed_offsets,
    *,
    n_rows,
    root_shift=0,
):
    """Copy complete input conformers, retaining all components including NaNs."""
    from tmol_jax.score._native import register

    target = "tmol_copy_conformer_rows"
    register(target)
    return _conformer_dofs(
        values,
        (source, source_offsets, target_offsets, packed_offsets),
        target=target,
        n_rows=n_rows,
        root_shift=root_shift,
    )


# Native argument order; table contents and fitting belong to tmol_core.
_TABLE_ARGUMENTS = (
    "rotameric_prob_tables",
    "rotprob_table_sizes",
    "rotprob_table_strides",
    "rotameric_mean_tables",
    "rotameric_sdev_tables",
    "rotmean_table_sizes",
    "rotmean_table_strides",
    "rotameric_meansdev_tableset_offsets",
    "rotameric_bb_start",
    "rotameric_bb_step",
    "rotameric_bb_periodicity",
    "rotameric_bb_source_start",
    "rotameric_bb_is_mirrored",
    "semirotameric_tables",
    "semirot_table_sizes",
    "semirot_table_strides",
    "semirot_start",
    "semirot_step",
    "semirot_periodicity",
    "rotameric_rotind2tableind",
    "semirotameric_rotind2tableind",
    "all_chi_rotind2tableind",
    "all_chi_rotind2tableind_offsets",
    "n_rotamers_for_tableset",
    "n_rotamers_for_tableset_offsets",
    "sorted_rotamer_2_rotamer",
    "nchi_for_table_set",
    "rotwells",
)
_INT64_TABLES = {
    "rotprob_table_sizes",
    "rotprob_table_strides",
    "rotmean_table_sizes",
    "rotmean_table_strides",
    "semirot_table_sizes",
    "semirot_table_strides",
    "n_rotamers_for_tableset",
    "sorted_rotamer_2_rotamer",
}


def _sampling_inputs(tables, coordinates, request):
    if isinstance(coordinates, jax.core.Tracer):
        raise TypeError("dynamic chi sampling requires eager coordinates")
    coordinates = np.asarray(coordinates)
    if coordinates.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError("sampling coordinates must be float32 or float64")
    if coordinates.dtype == np.float64 and not jax.config.x64_enabled:
        raise ValueError("float64 sampling requires JAX x64 to preserve output dtype")

    def array(value, wide=False):
        value = np.asarray(value)
        if value.dtype.kind == "f":
            return value.astype(coordinates.dtype, copy=False)
        if value.dtype.kind == "b":
            return value
        if value.dtype.kind not in "iu":
            raise TypeError("sampling indices must be integers")
        if (
            not wide
            and value.size
            and (value.min() < -(1 << 31) or value.max() >= (1 << 31))
        ):
            raise OverflowError("sampling indices exceed int32")
        return value.astype(np.int64 if wide else np.int32, copy=False)

    arguments = tuple(
        array(tables.sampling_db[name], name in _INT64_TABLES)
        for name in _TABLE_ARGUMENTS
    )
    return coordinates, arguments, tuple(array(value) for value in request)


def sample_chi_cpu(tables, coordinates, *request, max_samples_per_restype=0):
    """Eager CPU sampling of prepared requests with dynamic output cardinality."""
    device = coordinates.device if isinstance(coordinates, jax.Array) else None
    if device is not None and device.platform != "cpu":
        raise ValueError("sample_chi_cpu requires CPU coordinates; use sample_chi")
    coordinates, arguments, request = _sampling_inputs(tables, coordinates, request)
    result = invoke(12, coordinates, *arguments, *request, max_samples_per_restype)
    device = device or jax.devices("cpu")[0]
    return tuple(jax.device_put(value, device) for value in result)


@lru_cache(maxsize=1)
def _cuda_sampling():
    from tmol_jax.score._native import libraries

    library = libraries().get("cuda")
    if library is None:
        raise RuntimeError("CUDA chi sampling requires the native CUDA adapter")
    for target in ("tmol_sampling_plan", "tmol_sampling_emit"):
        try:
            symbol = getattr(library, target)
        except AttributeError as error:
            raise RuntimeError(
                "native CUDA sampler missing; rebuild matching artifacts"
            ) from error
        jax.ffi.register_ffi_target(target, jax.ffi.pycapsule(symbol), platform="CUDA")
    size = library.tmol_sampling_workspace_bytes
    size.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int, ctypes.c_int]
    size.restype = ctypes.c_uint64
    return size


def sample_chi(tables, coordinates, *request, max_samples_per_restype=0):
    """Execute common sampling on the coordinates' device, without host fallback.

    Dynamic counts require an eager device-to-host allocation barrier. Sampling
    is discrete, so jit/vmap/autodiff are not defined for this interface.
    Table int64 metadata is transported as opaque bytes even when JAX x64 is off.
    """
    if isinstance(coordinates, jax.core.Tracer):
        raise TypeError("dynamic chi sampling requires eager coordinates")
    device = coordinates.device if isinstance(coordinates, jax.Array) else None
    if device is None or device.platform == "cpu":
        return sample_chi_cpu(
            tables,
            coordinates,
            *request,
            max_samples_per_restype=max_samples_per_restype,
        )
    if device.platform != "gpu":
        raise NotImplementedError("chi sampling supports native CPU and CUDA")
    # Host request preparation is explicit; coordinates remain the device input
    # to the canonical geometry kernel, never a CPU sampling result.
    if coordinates.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError("sampling coordinates must be float32 or float64")
    dummy = np.empty((0, 3), coordinates.dtype)
    _, all_tables, prepared_request = _sampling_inputs(tables, dummy, request)
    if len(prepared_request) != 9:
        raise ValueError("native sampling requires nine prepared request arrays")
    counts, offsets, dimensions = invoke(
        13,
        prepared_request[3],
        all_tables[_TABLE_ARGUMENTS.index("n_rotamers_for_tableset")],
        prepared_request[8],
    )
    possible, width = map(int, dimensions)
    ntypes, nblocks = len(counts), len(prepared_request[0])
    size = _cuda_sampling()

    def workspace(count, emit):
        nbytes = size(count, ntypes, coordinates.dtype.itemsize * 8, emit)
        if nbytes == (1 << 64) - 1:
            raise ValueError("invalid sampling workspace extent")
        return jax.ShapeDtypeStruct((nbytes,), np.uint8)

    def put(value):
        value = np.ascontiguousarray(value)
        if value.dtype == np.int64:
            value = value.view(np.uint8).reshape(value.shape + (8,))
        return jax.device_put(value, device)

    active = all_tables[:13] + all_tables[23:27]
    inputs = (coordinates,) + tuple(map(put, active + prepared_request))
    shape = jax.ShapeDtypeStruct
    plan_shapes = (
        shape((2 * nblocks,), coordinates.dtype),
        shape((ntypes,), np.int32),
        shape((ntypes, width), np.int32),
        shape((ntypes,), np.int32),
    )
    with jax.default_device(device):
        plan = jax.ffi.ffi_call(
            "tmol_sampling_plan", plan_shapes + (workspace(possible, 0),)
        )(*inputs, put(counts), put(offsets), count=np.int64(possible))[:4]
        host_counts = np.asarray(jax.device_get(plan[3]))
        host_offsets, total = invoke(14, host_counts, max_samples_per_restype)
        count = int(total[0])
        identities, chi, _ = jax.ffi.ffi_call(
            "tmol_sampling_emit",
            (
                shape((count,), np.int32),
                shape((count, width), coordinates.dtype),
                workspace(count, 1),
            ),
        )(*inputs, *plan, put(host_offsets), count=np.int64(count))
    return plan[3], put(host_offsets), identities, chi
