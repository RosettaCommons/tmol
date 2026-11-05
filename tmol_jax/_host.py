"""Typed host array transport; scientific algorithms belong to the common core."""

import ctypes
import math
from functools import lru_cache

import numpy as np

from tmol_jax.score._native import libraries


class _Array(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("shape", ctypes.c_int64 * 5),
        ("rank", ctypes.c_int32),
        ("scalar_type", ctypes.c_int32),
    ]


_SCALARS = {
    1: (ctypes.c_int32, np.int32),
    2: (ctypes.c_bool, np.bool_),
    3: (ctypes.c_float, np.float32),
    4: (ctypes.c_double, np.float64),
    5: (ctypes.c_int64, np.int64),
}


@lru_cache(maxsize=1)
def _bindings():
    library = libraries().get("cpu")
    if library is None:
        raise RuntimeError("native host preparation requires the CPU adapter")
    try:
        size = library.tmol_kinematic_host_array_size
        prepare = library.tmol_kinematic_host_prepare
        count = library.tmol_kinematic_host_result_count
        array = library.tmol_kinematic_host_result_array
        release = library.tmol_kinematic_host_release
    except AttributeError as error:
        raise RuntimeError(
            "native host preparation is missing; rebuild tmol-jax-native"
        ) from error
    size.argtypes, size.restype = [], ctypes.c_size_t
    if size() != ctypes.sizeof(_Array):
        raise RuntimeError("native host array ABI mismatch")
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


def _prepare_array(argument, allow_real, preserve_int64=False):
    array = np.asarray(argument)
    if array.dtype.kind == "b":
        scalar_type = 2
    elif allow_real and array.dtype in (
        np.dtype("float32"),
        np.dtype("float64"),
    ):
        scalar_type = 3 if array.dtype.itemsize == 4 else 4
    elif preserve_int64 and array.dtype == np.dtype("int64"):
        scalar_type = 5
    elif array.dtype.kind in "iu":
        needs_range_check = array.dtype.itemsize > 4 or (
            array.dtype.kind == "u" and array.dtype.itemsize == 4
        )
        if (
            needs_range_check
            and array.size
            and (array.min() < -(1 << 31) or array.max() >= (1 << 31))
        ):
            raise OverflowError("native host indices exceed int32")
        array = array.astype(np.int32, copy=False)
        scalar_type = 1
    else:
        raise TypeError("native host arrays require int32, bool, float32 or float64")
    if not 1 <= array.ndim <= 5:
        raise ValueError("native host metadata requires rank one through five")
    array = np.ascontiguousarray(array)
    return array, scalar_type


def invoke(operation, *arguments):
    arrays, scalars, records = [], [], []
    for argument in arguments:
        if isinstance(argument, (int, bool, np.integer, np.bool_)):
            value = int(argument)
            if not -(1 << 63) <= value < (1 << 63):
                raise OverflowError("native host scalar exceeds int64")
            scalars.append(value)
            continue
        array, scalar_type = _prepare_array(
            argument, operation in (11, 12), preserve_int64=operation == 12
        )
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
            raise RuntimeError("native host preparation returned no result")
        outputs = []
        for index in range(size):
            record = _Array()
            if get_array(result, index, ctypes.byref(record)):
                raise RuntimeError("invalid native host result descriptor")
            if record.scalar_type not in _SCALARS or not 1 <= record.rank <= 5:
                raise RuntimeError("invalid native host result dtype or rank")
            shape = tuple(record.shape[: record.rank])
            if any(dim < 0 for dim in shape):
                raise RuntimeError("invalid native host result extent")
            ctype, dtype = _SCALARS[record.scalar_type]
            elements = math.prod(shape)
            if elements > (1 << 31) - 1:
                raise RuntimeError("native host result exceeds int32 indexing")
            if elements:
                if not record.data:
                    raise RuntimeError("native host result has no data")
                pointer = ctypes.cast(record.data, ctypes.POINTER(ctype))
                value = (
                    np.ctypeslib.as_array(pointer, shape=(elements,))
                    .copy()
                    .reshape(shape)
                )
            else:
                value = np.empty(shape, dtype=dtype)
            outputs.append(value)
        return outputs[0] if operation in (6, 8, 11) else tuple(outputs)
    finally:
        if result.value:
            release(result)
