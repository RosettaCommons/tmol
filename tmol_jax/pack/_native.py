"""Native execution for chi sampling and conformer initialization."""

import ctypes
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax._host import invoke
from ._ffi import stage


@partial(jax.jit, static_argnames=("operation", "outputs", "flip", "width"))
def _copy_plan_stage(arguments, *, operation, outputs, flip=False, width=0):
    target = "tmol_conformer_copy_plan"
    return stage(arguments, target, outputs, operation, flip, width)


@jax.jit
def current_conformer_copy_plan(
    original, natoms, considered, global_blocks, types, offsets, selected
):
    """Common compact source forest and selected-conformer copy segments."""
    from .dispatch import _narrow_prefix
    from .graph import prefix_counts

    wide = jax.ShapeDtypeStruct((original.size, 8), np.uint8)
    counts, _ = _copy_plan_stage((original, natoms), operation=0, outputs=(wide, wide))
    sources = prefix_counts(counts)[:-1]
    segments = _copy_plan_stage(
        (
            original,
            natoms,
            considered,
            global_blocks,
            types,
            offsets,
            selected,
            sources,
        ),
        operation=1,
        outputs=jax.ShapeDtypeStruct((3, selected.size, 8), np.uint8),
    )
    prefix = _narrow_prefix(prefix_counts(segments[0]))
    narrow = _narrow_prefix(segments.reshape(-1, 8)).reshape(3, selected.size)
    return narrow[1], narrow[2], prefix


@partial(jax.jit, static_argnames=("flip",))
def reset_opth_dofs(
    dofs, ideal, types, offsets, selected, atoms, downstream, counts, *, flip
):
    """Apply the shared NHQ reset plan; inactive slots leave DOFs untouched."""
    from .dispatch import _narrow_prefix

    if not flip or not selected.size or not downstream.shape[1]:
        return dofs
    n = selected.size * downstream.shape[1]
    indices = _copy_plan_stage(
        (types, offsets, selected, atoms, downstream, counts),
        operation=2,
        outputs=jax.ShapeDtypeStruct((2, n, 8), np.uint8),
        flip=flip,
        width=ideal.shape[1],
    )
    source, target = _narrow_prefix(indices.reshape(-1, 8)).reshape(2, n)
    values = ideal.reshape(-1, 9)[jnp.maximum(source, 0)]
    # Active downstream atoms are distinct within each disjoint conformer.
    return dofs.at[jnp.where(target >= 0, target, dofs.shape[0])].set(
        values, mode="drop", unique_indices=True
    )


@partial(jax.jit, static_argnames=("operation", "extent", "width"))
def _conformer_indices(arguments, *, operation, extent, width):
    from .dispatch import _narrow_prefix

    output = jax.ShapeDtypeStruct((2, extent, 8), np.uint8)
    indices = stage(
        arguments, "tmol_conformer_indices", output, operation, extent, width
    )
    return _narrow_prefix(indices.reshape(-1, 8)).reshape(2, extent)


@jax.custom_jvp
def _assign_chi(
    dofs, chi, types, offsets, selected, atoms, kinematic_index, corrections
):
    return _conformer_dofs(
        dofs,
        (chi, types, offsets, selected, atoms, kinematic_index, corrections),
        target="tmol_assign_conformer_chi",
    )


@_assign_chi.defjvp
def _assign_chi_jvp(primals, tangents):
    dofs, chi, types, offsets, selected, atoms, kinematic_index, corrections = primals
    dd, dchi, *_, dc = tangents
    if atoms.size:
        destination, parameter = _conformer_indices(
            (types, offsets, selected, atoms, kinematic_index),
            operation=1,
            extent=atoms.size,
            width=corrections.shape[1],
        )
        valid = destination >= 0
        updates = dchi.reshape(-1) - dc.reshape(-1)[jnp.maximum(parameter, 0)]
        # Each active chi owns one DOF. Inactive entries are dropped; they do
        # not participate in the uniqueness contract used by the transpose.
        dd = (
            dd.reshape(-1)
            .at[jnp.where(valid, destination, dd.size)]
            .set(updates, mode="drop", unique_indices=True)
            .reshape(dofs.shape)
        )
    return _assign_chi(*primals), dd


@partial(jax.custom_jvp, nondiff_argnums=(5, 6))
def _copy_rows(
    values, source, source_offsets, target_offsets, packed_offsets, n_rows, root_shift
):
    return _conformer_dofs(
        values,
        (source, source_offsets, target_offsets, packed_offsets),
        target="tmol_copy_conformer_rows",
        n_rows=n_rows,
        root_shift=root_shift,
    )


@_copy_rows.defjvp
def _copy_rows_jvp(n_rows, root_shift, primals, tangents):
    values, source, *metadata = primals
    dv, ds, *_ = tangents
    rows = _conformer_indices(
        tuple(metadata), operation=0, extent=n_rows, width=root_shift
    )
    # Source rows may repeat, but conformers occupy disjoint target rows.
    return _copy_rows(*primals, n_rows, root_shift), dv.at[rows[1]].set(
        ds[rows[0]], unique_indices=True
    )


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


@partial(jax.custom_jvp, nondiff_argnums=(3,))
def _copy_dofs(dofs, source, metadata, sampler):
    return _conformer_dofs(
        dofs, (source, *metadata), target="tmol_copy_conformer_dofs", sampler=sampler
    )


@_copy_dofs.defjvp
def _copy_dofs_jvp(sampler, primals, tangents):
    dofs, source, metadata = primals
    dd, ds, _ = tangents
    extent = metadata[0].size * metadata[5].shape[1] * 9
    if extent:
        # The common finite predicate selects source vs ideal independently
        # per scalar DOF. Refresh this discrete mask whenever source values change.
        origin, destination = _conformer_indices(
            (jax.lax.stop_gradient(dofs), jax.lax.stop_gradient(source), *metadata),
            operation=2,
            extent=extent,
            width=sampler,
        )
        updates = ds.reshape(-1)[jnp.maximum(origin, 0)]
        dd = (
            dd.reshape(-1)
            .at[jnp.where(destination >= 0, destination, dd.size)]
            .set(updates, mode="drop", unique_indices=True)
            .reshape(dofs.shape)
        )
    return _copy_dofs(*primals, sampler), dd


def copy_conformer_dofs(dofs, source, *metadata, sampler):
    """Transfer finite retained-atom DOFs using shared native chemistry maps.

    The input remains immutable; XLA owns the aliased output allocation. This
    preserves source/ideal gradients using the common finite-value selection.
    """
    from tmol_jax.score._native import register

    target = "tmol_copy_conformer_dofs"
    register(target)
    return _copy_dofs(dofs, source, metadata, sampler)


def assign_conformer_chi(dofs, chi, *metadata):
    """Apply sampled chi and shared ring corrections to independent DOFs."""
    from tmol_jax.score._native import register

    target = "tmol_assign_conformer_chi"
    register(target)
    return _assign_chi(dofs, chi, *metadata)


@partial(jax.jit, static_argnames=("operation",))
def _chi_derivative_stage(arguments, *, operation):
    target = "tmol_conformer_chi_derivatives"
    if operation == 0:
        results = (jax.ShapeDtypeStruct((5, arguments[3].size, 8), np.uint8),)
    else:
        results = tuple(
            jax.ShapeDtypeStruct(a.shape, a.dtype)
            for a in arguments[: 2 if operation == 2 else 1]
        )
    return stage(arguments, target, results, operation)


@jax.custom_vjp
def _chi_coordinate_gradient(coords, weights, indices):
    return _chi_derivative_stage((coords, weights, indices), operation=1)[0]


def _chi_coordinate_gradient_fwd(coords, weights, indices):
    return _chi_coordinate_gradient(coords, weights, indices), (
        coords,
        weights,
        indices,
    )


def _chi_coordinate_gradient_bwd(saved, direction):
    coords, weights, indices = saved
    gx, gw = _chi_derivative_stage((coords, weights, indices, direction), operation=2)
    return gx, gw, None


_chi_coordinate_gradient.defvjp(
    _chi_coordinate_gradient_fwd, _chi_coordinate_gradient_bwd
)


@jax.custom_vjp
def _correct_chi(dofs, chi, trial_coordinates, metadata):
    return _conformer_dofs(
        dofs, (chi, trial_coordinates, *metadata), target="tmol_correct_conformer_chi"
    )


def _correct_chi_fwd(dofs, chi, coords, metadata):
    indices = _chi_derivative_stage(metadata, operation=0)[0]
    result = _correct_chi(dofs, chi, coords, metadata)
    return result, (coords, chi, indices)


def _correct_chi_bwd(saved, weights):
    from .dispatch import _narrow_prefix

    coords, chi, indices = saved
    destination = _narrow_prefix(indices[0])
    gc = jnp.where(
        destination >= 0, weights.reshape(-1)[jnp.maximum(destination, 0)], 0
    )
    return (
        weights,
        gc.reshape(chi.shape),
        _chi_coordinate_gradient(coords, weights, indices),
        None,
    )


_correct_chi.defvjp(_correct_chi_fwd, _correct_chi_bwd)


def correct_conformer_chi(dofs, chi, trial_coordinates, *metadata):
    """Shared measured-chi correction, reverse gradients and reverse Hessian products.

    The wrapped-angle discontinuity has no derivative. Forward AD and third
    derivatives are not implemented by this native execution adapter.
    """
    from tmol_jax.score._native import register

    register("tmol_correct_conformer_chi")
    return _correct_chi(dofs, chi, trial_coordinates, metadata)


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
    return _copy_rows(
        values,
        source,
        source_offsets,
        target_offsets,
        packed_offsets,
        n_rows,
        root_shift,
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


def _sampling_array(value, dtype, wide=False):
    value = np.asarray(value)
    if value.dtype.kind == "f":
        return value.astype(dtype, copy=False)
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


def _sampling_buffer(value):
    value = np.ascontiguousarray(value)
    if value.dtype == np.int64:
        value = value.view(np.uint8).reshape(value.shape + (8,))
    return value


def _sampling_put(values, device):
    return jax.device_put(_sampling_buffer(values), device)


def _sampling_put_many(values, device):
    return jax.device_put(tuple(map(_sampling_buffer, values)), device)


class SamplingStorage:
    """Typed runtime buffers owned by one immutable sampler parameter snapshot."""

    def __init__(self, tables):
        self.tables = tables
        self._host = {}
        self._device = {}

    def host(self, dtype):
        dtype = np.dtype(dtype)
        if dtype not in self._host:
            arrays = tuple(
                np.ascontiguousarray(
                    _sampling_array(
                        self.tables.sampling_db[name], dtype, name in _INT64_TABLES
                    )
                )
                for name in _TABLE_ARGUMENTS
            )
            for value in arrays:
                value.setflags(write=False)
            self._host[dtype] = arrays
        return self._host[dtype]

    def device(self, dtype, device):
        key = (np.dtype(dtype), device)
        if key not in self._device:
            tables = self.host(dtype)
            self._device[key] = _sampling_put_many(tables[:13] + tables[23:27], device)
        return self._device[key]


def _sampling_inputs(tables, coordinates, request):
    if isinstance(coordinates, jax.core.Tracer):
        raise TypeError("dynamic chi sampling requires eager coordinates")
    coordinates = np.asarray(coordinates)
    if coordinates.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError("sampling coordinates must be float32 or float64")
    if coordinates.dtype == np.float64 and not jax.config.x64_enabled:
        raise ValueError("float64 sampling requires JAX x64 to preserve output dtype")

    array = partial(_sampling_array, dtype=coordinates.dtype)
    arguments = (
        tables.host(coordinates.dtype)
        if isinstance(tables, SamplingStorage)
        else tuple(
            array(tables.sampling_db[name], wide=name in _INT64_TABLES)
            for name in _TABLE_ARGUMENTS
        )
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
    return jax.device_put(result, device)


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

    put = partial(_sampling_put_many, device=device)
    active = (
        tables.device(coordinates.dtype, device)
        if isinstance(tables, SamplingStorage)
        else put(all_tables[:13] + all_tables[23:27])
    )
    request_buffers = put((*prepared_request, counts, offsets))
    inputs = (coordinates,) + active + request_buffers[:-2]
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
        )(*inputs, *request_buffers[-2:], count=np.int64(possible))[:4]
        host_counts = np.asarray(jax.device_get(plan[3]))
        host_offsets, total = invoke(14, host_counts, max_samples_per_restype)
        count = int(total[0])
        (device_offsets,) = put((host_offsets,))
        identities, chi, _ = jax.ffi.ffi_call(
            "tmol_sampling_emit",
            (
                shape((count,), np.int32),
                shape((count, width), coordinates.dtype),
                workspace(count, 1),
            ),
        )(*inputs, *plan, device_offsets, count=np.int64(count))
    return plan[3], device_offsets, identities, chi
