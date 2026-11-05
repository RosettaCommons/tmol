"""Typed host array transport; topology algorithms belong to the common core."""

import ctypes
import math
from functools import lru_cache, partial

import numpy as np

from tmol_jax.score._native import libraries


class _Array(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("shape", ctypes.c_int64 * 5),
        ("rank", ctypes.c_int32),
        ("scalar_type", ctypes.c_int32),
    ]


_OPERATIONS = {
    "get_kfo_indices_for_atoms": 0,
    "get_kfo_atom_parents": 1,
    "get_children": 2,
    "get_id_and_frame_xyz": 3,
    "calculate_ff_edge_delays": 4,
    "get_jump_atom_indices": 5,
    "get_block_parent_connectivity_from_toposort": 6,
    "get_scans2": 7,
    "create_minimizer_map": 8,
    "assemble_pose": 10,
}


@lru_cache(maxsize=1)
def _bindings():
    library = libraries().get("cpu")
    if library is None:
        raise RuntimeError("native pose preparation requires the CPU adapter")
    try:
        size = library.tmol_kinematic_host_array_size
        prepare = library.tmol_kinematic_host_prepare
        count = library.tmol_kinematic_host_result_count
        array = library.tmol_kinematic_host_result_array
        release = library.tmol_kinematic_host_release
    except AttributeError as error:
        raise RuntimeError(
            "native pose preparation is missing; rebuild tmol-jax-native"
        ) from error
    size.argtypes, size.restype = [], ctypes.c_size_t
    if size() != ctypes.sizeof(_Array):
        raise RuntimeError("native pose-preparation array ABI mismatch")
    prepare.argtypes = [
        ctypes.c_int32,
        ctypes.POINTER(_Array),
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    prepare.restype = ctypes.c_int
    count.argtypes, count.restype = [ctypes.c_void_p], ctypes.c_int64
    array.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.POINTER(_Array)]
    array.restype = ctypes.c_int
    release.argtypes, release.restype = [ctypes.c_void_p], None
    return prepare, count, array, release


def _invoke(operation, *arguments):
    arrays, scalars, records = [], [], []
    for argument in arguments:
        if isinstance(argument, (int, bool, np.integer, np.bool_)):
            value = int(argument)
            if not -(1 << 63) <= value < (1 << 63):
                raise OverflowError("native pose scalar exceeds int64")
            scalars.append(value)
            continue
        array = np.asarray(argument)
        if array.dtype.kind == "b":
            scalar_type = 2
        elif array.dtype.kind in "iu":
            needs_range_check = array.dtype.itemsize > 4 or (
                array.dtype.kind == "u" and array.dtype.itemsize == 4
            )
            if (
                needs_range_check
                and array.size
                and (array.min() < -(1 << 31) or array.max() >= (1 << 31))
            ):
                raise OverflowError("native pose indices exceed int32")
            array = array.astype(np.int32, copy=False)
            scalar_type = 1
        else:
            raise TypeError(
                "native pose metadata must contain integer or boolean arrays"
            )
        if not 1 <= array.ndim <= 5:
            raise ValueError("native pose metadata requires rank one through five")
        array = np.ascontiguousarray(array)
        arrays.append(array)  # Keep input memory alive throughout the native call.
        records.append(
            _Array(
                array.ctypes.data,
                (ctypes.c_int64 * 5)(*array.shape),
                array.ndim,
                scalar_type,
            )
        )
    prepare, count, get_array, release = _bindings()
    result = ctypes.c_void_p()
    error = ctypes.create_string_buffer(1024)
    try:
        status = prepare(
            operation,
            (_Array * len(records))(*records),
            len(records),
            (ctypes.c_int64 * len(scalars))(*scalars),
            len(scalars),
            ctypes.byref(result),
            error,
            len(error),
        )
        if status:
            raise ValueError(error.value.decode("utf-8", errors="replace"))
        size = count(result)
        if size < 1:
            raise RuntimeError("native pose preparation returned no result")
        outputs = []
        for index in range(size):
            record = _Array()
            if get_array(result, index, ctypes.byref(record)):
                raise RuntimeError("invalid native pose result descriptor")
            if record.scalar_type not in (1, 2) or not 1 <= record.rank <= 5:
                raise RuntimeError("invalid native pose result dtype or rank")
            shape = tuple(record.shape[: record.rank])
            if any(dim < 0 for dim in shape):
                raise RuntimeError("invalid native pose result extent")
            ctype = ctypes.c_int32 if record.scalar_type == 1 else ctypes.c_bool
            dtype = np.int32 if record.scalar_type == 1 else np.bool_
            elements = math.prod(shape)
            if elements > (1 << 31) - 1:
                raise RuntimeError("native pose result exceeds int32 indexing")
            if elements:
                if not record.data:
                    raise RuntimeError("native pose result has no data")
                pointer = ctypes.cast(record.data, ctypes.POINTER(ctype))
                value = (
                    np.ctypeslib.as_array(pointer, shape=(elements,))
                    .copy()
                    .reshape(shape)
                )
            else:
                value = np.empty(shape, dtype=dtype)
            outputs.append(value)
        return outputs[0] if operation in (6, 8) else tuple(outputs)
    finally:
        if result.value:
            release(result)


class NativePosePlanner:
    """Host execution protocol consumed by shared topology preparation."""

    def __getattr__(self, name):
        try:
            operation = _OPERATIONS[name]
        except KeyError as error:
            raise AttributeError(name) from error
        return partial(_invoke, operation)
