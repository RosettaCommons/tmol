"""Eager CPU execution of the common dynamically sized chi sampler."""

import jax
import numpy as np

from tmol_jax._host import invoke

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


def sample_chi_cpu(tables, coordinates, *request, max_samples_per_restype=0):
    """Execute prepared native inputs; pose/task preparation is a separate API.

    Sampling has dynamic output cardinality and is not differentiable or jittable.
    CUDA execution requires the shared staged workspace API; it never falls back
    silently to host execution here.
    """
    if isinstance(coordinates, jax.core.Tracer):
        raise TypeError("dynamic chi sampling requires eager coordinates")
    device = coordinates.device if isinstance(coordinates, jax.Array) else None
    if device is not None and device.platform != "cpu":
        raise NotImplementedError("native dynamic chi sampling currently requires CPU")
    coordinates = np.asarray(coordinates)
    if coordinates.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError("sampling coordinates must be float32 or float64")

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
    result = invoke(
        12,
        coordinates,
        *arguments,
        *(array(value) for value in request),
        max_samples_per_restype,
    )
    device = device or jax.devices("cpu")[0]
    return tuple(jax.device_put(value, device) for value in result)
